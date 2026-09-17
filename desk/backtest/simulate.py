"""Walking one proposed trade forward through daily bars.

THE AMBIGUITY THIS MODULE IS REALLY ABOUT
------------------------------------------
On a day where the low reaches the stop AND the high reaches the target,
daily bars CANNOT say which came first. Open, high, low, close is four
numbers; the path between them is gone.

Most backtests resolve this silently, and almost always in their own
favour - check the target first, book 2.5R, move on. On a strategy with a
2 ATR stop and a 2.5R target this is not a rounding error: those bars are
the volatile ones, they cluster exactly where the edge is supposed to live,
and assuming the good outcome converts a losing system into a winning one
on paper.

So here:

  1. THE STOP WINS. Always. It is the pessimistic reading and the only
     defensible default, because the loss is the outcome you must be able
     to survive.
  2. EVERY SUCH BAR IS COUNTED. `ExitSimulation.ambiguous` says how many
     trades were decided this way, and a result where that number is large
     is a result that rests on the assumption rather than on the data.
     Reporting it is what makes the pessimism honest instead of merely
     conservative.

GAPS ARE NOT THE SAME AS TOUCHES
--------------------------------
A stop at 180 does not fill at 180 when the session opens at 172. From the
second session onward the open is checked BEFORE the intrabar range, and
the fill is the OPEN, not the level. Filling gapped exits at the intended
price is the other standard way a backtest quietly invents money, and on
Indian mid-caps around results it is not rare.

A gap on the ENTRY session is a different thing again, and it invalidates
the setup rather than resolving it - see the comment in simulate_trade.

ENTRY IS THE NEXT SESSION'S OPEN
--------------------------------
The decision is made pre-open from the previous close's data, so the first
price it could actually transact at is the next open. The plan's `entry`
level is a REFERENCE the risk engine sized against, not a promised fill -
and because the real fill differs, `entry_slip_pct` is reported so a result
can show how much of its outcome came from entering somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from desk.journal.models import ExitReason

__all__ = ["ExitSimulation", "SimulatedTrade", "simulate_trade"]


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    symbol: str
    entry_date: date
    entry_price: float
    """The actual fill - next session's open - not the planned level."""
    planned_entry: float | None
    exit_date: date
    exit_price: float
    exit_reason: str
    qty: int
    bars_held: int
    stop: float
    target: float | None
    ambiguous: bool = False
    """True when the deciding bar touched BOTH stop and target. Resolved as
    a stop; counted so the result can say how much rests on that."""
    gapped: bool = False
    """True when the exit filled at an open beyond the intended level."""

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop)

    @property
    def r_multiple(self) -> float | None:
        """Result in units of the risk ACTUALLY taken at the fill.

        Against the realised entry, not the planned one: a trade entered
        2% above its planned level took more risk than the plan intended,
        and measuring it against the plan would hide that.
        """
        r = self.risk_per_share
        if not r:
            return None
        return (self.exit_price - self.entry_price) / r

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def entry_slip_pct(self) -> float | None:
        if not self.planned_entry:
            return None
        return 100.0 * (self.entry_price - self.planned_entry) / self.planned_entry


@dataclass(frozen=True, slots=True)
class ExitSimulation:
    trade: SimulatedTrade | None
    reason_not_taken: str | None = None
    """Set when no trade happened at all - no bar to enter on, no price.
    Distinct from a trade that was taken and lost."""


def simulate_trade(bars: pd.DataFrame, *, symbol: str, decided_on: date,
                   planned_entry: float | None, stop: float,
                   target: float | None, qty: int,
                   horizon_days: int = 5) -> ExitSimulation:
    """Enter on the session after `decided_on`; walk forward to an exit.

    `bars` is one symbol's daily OHLC indexed by date, and it MUST NOT be
    trimmed to the decision date - this function needs the future to
    simulate it. That is the one place in this codebase where forward data
    is legitimate, and it is safe precisely because nothing here feeds back
    into the decision: the plan was already fixed before this ran.

    Returns an ExitSimulation whose `trade` is None when the position could
    never be opened, with the reason. A trade that could not be entered is
    not a trade that broke even.
    """
    if qty <= 0:
        return ExitSimulation(None, "quantity was zero")
    if bars is None or bars.empty:
        return ExitSimulation(None, "no price history for this symbol")

    frame = bars.sort_index()
    idx = [_as_date(d) for d in frame.index]
    forward = [i for i, d in enumerate(idx) if d > decided_on]
    if not forward:
        return ExitSimulation(None, "no session after the decision date")

    first = forward[0]
    entry_row = frame.iloc[first]
    entry_price = float(entry_row["open"])
    if not entry_price or entry_price <= 0:
        return ExitSimulation(None, "no usable open on the entry session")

    # THE GAP THAT INVALIDATES THE SETUP, rather than instantly resolving it.
    # If the entry session opens beyond a level, the trade is NOT TAKEN.
    #
    # An earlier version entered at the open and then checked that same open
    # against the stop, so a session opening at 172 under a 180 stop
    # "entered" at 172 and "stopped out" at 172 for a flat 0R. That is not
    # pessimistic or optimistic, it is incoherent: nobody buys at 172 with
    # the stop ABOVE at 180, because the trade is inverted before it exists.
    # The mirror case is an open already past the target - the reward is
    # gone before the position is on.
    #
    # Recording these as not-taken also keeps a real property of the desk
    # visible: gap-ups through the entry are exactly how a momentum
    # shortlist loses its best-looking names before it can act on them, and
    # booking them at a tidy 0R hides that entirely.
    if entry_price <= stop:
        return ExitSimulation(
            None, "gapped below the stop before entry - setup invalidated")
    if target is not None and entry_price >= target:
        return ExitSimulation(
            None, "gapped past the target before entry - setup invalidated")

    # The horizon counts sessions HELD, and the entry bar can still resolve
    # the trade INTRABAR - entering at a valid open and then breaking the
    # stop that same session is a real same-day loss, and pretending a
    # position must survive its first day would hide the worst of them.
    window = forward[:horizon_days]
    last_i = window[-1]

    for n, i in enumerate(window, start=1):
        row = frame.iloc[i]
        day = idx[i]
        o, h, l = float(row["open"]), float(row["high"]), float(row["low"])

        # 1. THE OPEN FIRST, from the SECOND session onward. A gap through
        #    a level fills at the open, not at the level, because the open
        #    is what happened first in that session.
        #
        #    Skipped on the entry bar: that open IS the entry price and was
        #    already validated above. Re-testing it there is what produced
        #    the incoherent same-price entry-and-exit.
        if n > 1 and o <= stop:
            return _exit(symbol, idx[first], entry_price, planned_entry, day,
                         o, ExitReason.STOP, qty, n, stop, target,
                         gapped=True)
        if n > 1 and target is not None and o >= target:
            return _exit(symbol, idx[first], entry_price, planned_entry, day,
                         o, ExitReason.TARGET, qty, n, stop, target,
                         gapped=True)

        hit_stop = l <= stop
        hit_target = target is not None and h >= target

        # 2. BOTH TOUCHED: unresolvable from daily bars. Stop wins, and the
        #    trade is flagged so the result can report how many of its
        #    outcomes were decided by this rule rather than by the data.
        if hit_stop and hit_target:
            return _exit(symbol, idx[first], entry_price, planned_entry, day,
                         stop, ExitReason.STOP, qty, n, stop, target,
                         ambiguous=True)
        if hit_stop:
            return _exit(symbol, idx[first], entry_price, planned_entry, day,
                         stop, ExitReason.STOP, qty, n, stop, target)
        if hit_target:
            return _exit(symbol, idx[first], entry_price, planned_entry, day,
                         target, ExitReason.TARGET, qty, n, stop, target)

    # 3. Neither level reached inside the horizon: out at the last close.
    return _exit(symbol, idx[first], entry_price, planned_entry, idx[last_i],
                 float(frame.iloc[last_i]["close"]), ExitReason.TIME, qty,
                 len(window), stop, target)


def _exit(symbol, entry_date, entry_price, planned_entry, exit_date,
          exit_price, reason, qty, bars_held, stop, target, *,
          ambiguous=False, gapped=False) -> ExitSimulation:
    return ExitSimulation(SimulatedTrade(
        symbol=symbol, entry_date=entry_date, entry_price=entry_price,
        planned_entry=planned_entry, exit_date=exit_date,
        exit_price=float(exit_price), exit_reason=reason, qty=qty,
        bars_held=bars_held, stop=stop, target=target,
        ambiguous=ambiguous, gapped=gapped))


def _as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()
