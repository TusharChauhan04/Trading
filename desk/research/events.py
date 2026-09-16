"""Scheduled company events, per symbol, as of a date.

THE DESIGN QUESTION THIS SETTLES
--------------------------------
`RegimeState.EventProximity` (desk/regime/state.py) holds market-wide event
proximity - RBI policy, the budget, expiry - and is documented as the only
regime dimension that can force NO TRADE on its own. Two architecture reviews
pointed out that `CorporateEvent` is per-SYMBOL while `EventProximity` is a
set of market-wide scalars, so the question Stage 4 actually asks - "does
THIS name report inside my holding window?" - had nowhere to land.

The resolution is NOT to add a per-symbol field to `EventProximity`. Regime is
a property of the market by definition; a per-symbol value inside a regime
vector would make "the regime" mean something different for every candidate,
and regime-sliced performance measurement would stop being meaningful.

So per-symbol event proximity is a STAGE 4 CONCERN, passed alongside the
regime rather than inside it. That mirrors how the risk engine already works:
`market_risk_off` is a market-wide flag AND `atr_pct`/`adv_shares` are
per-symbol arguments, on the same call.

WHY THIS GATE EXISTS AT ALL
---------------------------
A company announcing results inside your holding window turns the trade into
a coin flip on an event you have no edge on. The setup can be perfect and the
position still resolves on something the chart could not see. That is why it
belongs at Stage 4 with the other hard gates rather than as a Stage 2 score
penalty: it is not a reason to rank a name lower, it is a reason not to take
it today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from desk.research.models import CorporateEvent

__all__ = ["EventCalendar", "EventWindow"]

#: Purposes that move a price on the day. NSE's `purpose` field is free text
#: written by humans, so this matches loosely - and errs toward BLOCKING,
#: because a false block costs one skipped trade while a false clear costs a
#: position held through an earnings gap.
_PRICE_MOVING = (
    "result", "financial", "earning", "dividend", "bonus", "split",
    "buyback", "buy back", "rights", "merger", "amalgamat", "demerger",
    "fund rais", "fundrais", "preferential", "stock option",
)


@dataclass(frozen=True, slots=True)
class EventWindow:
    """What is scheduled for one symbol inside a horizon."""

    symbol: str
    days_until: int | None
    """Calendar days to the next price-moving event. None when nothing is
    scheduled OR when this symbol has no calendar data at all - `known`
    separates those two, and they are not the same fact."""
    event: CorporateEvent | None = None
    known: bool = True
    """False when the calendar holds nothing for this symbol. Not the same as
    'nothing scheduled': one is an answer, the other is the absence of one."""

    def blocks(self, horizon_days: int) -> bool:
        """Is a fresh position a coin flip on this event?

        An UNKNOWN symbol does not block. That is a deliberate choice and the
        opposite of this project's usual instinct: the calendar covers only
        the next few weeks market-wide, so 'no entry' is the normal state for
        almost every symbol on almost every day. Blocking on unknown would
        block everything, every day, and a gate that always fires is a gate
        that gets switched off. The honest handling is to report coverage -
        see EventCalendar.coverage - rather than to pretend absence is
        information.
        """
        return self.days_until is not None and 0 <= self.days_until <= horizon_days


@dataclass(slots=True)
class EventCalendar:
    """Forward-looking events, indexed by symbol.

    Built from `parse_event_calendar`. Deliberately a plain in-memory index
    rather than a store: the calendar is small (tens of rows), entirely
    forward-looking, and refetched every day - there is no history to keep
    point-in-time.
    """

    events: list[CorporateEvent] = field(default_factory=list)
    _by_symbol: dict[str, list[CorporateEvent]] = field(default_factory=dict,
                                                        repr=False)

    def __post_init__(self) -> None:
        for e in self.events:
            if e.event_date is None:
                continue
            self._by_symbol.setdefault(e.symbol.split(".")[0].upper(),
                                       []).append(e)
        for rows in self._by_symbol.values():
            rows.sort(key=lambda e: e.event_date)

    @classmethod
    def from_events(cls, events) -> "EventCalendar":
        return cls(events=list(events))

    def window(self, symbol: str, *, as_of: date,
               price_moving_only: bool = True) -> EventWindow:
        """The next scheduled event for one symbol, on or after `as_of`."""
        base = symbol.split(".")[0].upper()
        rows = self._by_symbol.get(base)
        if not rows:
            return EventWindow(symbol=base, days_until=None, known=False)

        for e in rows:
            if e.event_date < as_of:
                continue
            if price_moving_only and not _is_price_moving(e):
                continue
            return EventWindow(symbol=base,
                               days_until=(e.event_date - as_of).days,
                               event=e, known=True)
        return EventWindow(symbol=base, days_until=None, known=True)

    @property
    def symbols(self) -> list[str]:
        return sorted(self._by_symbol)

    def coverage(self, symbols) -> tuple[int, int]:
        """(symbols with calendar data, total asked about).

        The honest counterpart to `blocks()` not firing on unknowns: a caller
        can report "the event gate could only see 12 of 1,598 names" rather
        than implying every other name was checked and cleared.
        """
        asked = [s.split(".")[0].upper() for s in symbols]
        return sum(1 for s in asked if s in self._by_symbol), len(asked)


def _is_price_moving(e: CorporateEvent) -> bool:
    text = f"{e.purpose} {e.description}".lower()
    return any(k in text for k in _PRICE_MOVING)
