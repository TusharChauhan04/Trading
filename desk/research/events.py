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


# ===========================================================================
# Persistence
# ===========================================================================
#
# The calendar is market-wide and forward-looking, so unlike filings there is
# no point-in-time history to preserve: one snapshot, refetched daily,
# overwritten in place.
#
# WHAT MATTERS HERE IS STALENESS, and it is the whole reason this is not just
# json.dump. A calendar fetched two weeks ago will answer "nothing scheduled"
# for every company that has announced a board meeting since. That is a FALSE
# CLEAR - the gate reports it checked and found nothing, and a position goes
# on through an earnings print.
#
# A stale calendar is therefore WORSE THAN NO CALENDAR. No calendar is
# declared unavailable and the caller says the check did not run; a stale one
# lies quietly. So `load_calendar` reports age and staleness as data, and the
# funnel treats a stale file as absent.

import json as _json
from dataclasses import dataclass as _dataclass
from datetime import datetime as _datetime, timedelta as _timedelta, timezone as _timezone
from pathlib import Path as _Path

_IST_TZ = _timezone(_timedelta(hours=5, minutes=30))

#: Trading days, not calendar days. NSE publishes board-meeting intimations
#: continuously and companies give as little as two clear days' notice, so a
#: file older than this can already be missing an event inside a one-week
#: holding window.
DEFAULT_MAX_AGE_DAYS = 3


@_dataclass(frozen=True, slots=True)
class StoredCalendar:
    calendar: EventCalendar
    fetched_at: _datetime
    age_days: int
    stale: bool
    """True when the snapshot cannot be trusted for this `as_of`. A caller
    must treat it as NO CALENDAR rather than as a calendar - see above."""

    @property
    def from_the_future(self) -> bool:
        """Fetched AFTER the date being scanned.

        Not an error case - it is the normal situation when re-running a
        past day, and it is a LOOK-AHEAD violation rather than a staleness
        one. The file holds board-meeting intimations published between
        `as_of` and the fetch, so the gate would be screening that day's
        candidates using announcements the market had not yet seen. Same
        guarantee FilingStore enforces by filename.
        """
        return self.age_days < 0

    @property
    def caveat(self) -> str | None:
        if not self.stale:
            return None
        if self.from_the_future:
            return (f"the event calendar on file was fetched "
                    f"{-self.age_days} day(s) AFTER the date being scanned, "
                    f"so using it would mean screening with announcements "
                    f"the market had not yet seen. The earnings gate was NOT "
                    f"applied - this is look-ahead protection, not a fault.")
        return (f"the event calendar on file was fetched {self.age_days} "
                f"day(s) ago and is too old to trust - the earnings gate was "
                f"NOT applied. Refresh it before relying on today's plan.")


def save_calendar(events, path, *, fetched_at: _datetime | None = None) -> int:
    """Write the market-wide calendar snapshot atomically.

    Atomic because a half-written calendar read by tomorrow's pre-open run
    would parse as a SHORTER calendar - fewer events, more names reported as
    having nothing scheduled. A truncated file that still parses is the worst
    shape this data can take.
    """
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"symbol": e.symbol,
             "event_date": e.event_date.isoformat() if e.event_date else None,
             "purpose": e.purpose, "company": e.company,
             "description": e.description}
            for e in events if e.event_date is not None]
    payload = {
        "fetched_at": (fetched_at or _datetime.now(_IST_TZ)).astimezone(
            _IST_TZ).isoformat(),
        "events": rows,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    return len(rows)


def load_calendar(path, *, as_of: date,
                  max_age_days: int = DEFAULT_MAX_AGE_DAYS
                  ) -> StoredCalendar | None:
    """Read the snapshot, or None when there is no usable file.

    None means "no calendar" - the caller declares the check unavailable.
    A StoredCalendar with stale=True means "a calendar too old to believe",
    which the caller must also treat as unavailable, but with a different
    caveat: one says nobody fetched it, the other says somebody did and then
    stopped.
    """
    path = _Path(path)
    if not path.is_file():
        return None
    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
        fetched_at = _datetime.fromisoformat(payload["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError):
        # A corrupt calendar is an ABSENT calendar, never a partial one.
        return None

    events = []
    for row in payload.get("events", []):
        if not isinstance(row, dict):
            continue
        raw = row.get("event_date")
        try:
            when = date.fromisoformat(raw) if raw else None
        except (TypeError, ValueError):
            continue
        if when is None:
            continue
        events.append(CorporateEvent(
            symbol=str(row.get("symbol", "")), event_date=when,
            purpose=str(row.get("purpose", "")),
            company=str(row.get("company", "")),
            description=str(row.get("description", ""))))

    age = (as_of - fetched_at.astimezone(_IST_TZ).date()).days
    return StoredCalendar(calendar=EventCalendar.from_events(events),
                          fetched_at=fetched_at, age_days=age,
                          stale=age > max_age_days or age < 0)
