"""Is a 1:2 target reachable inside one session? Measured: no, and why.

THIS ANSWERS THE PROJECT'S MAIN GOAL
------------------------------------
The desk exists to produce a daily 1:2 trade - "Risk 500, potential profit
1000". On DAILY bars that failed for a reason the project measured: at the 12%
stop the structural stops actually imply, a 2R target is 24% away, and only
26.6% of trades reached it even over 60 sessions. The salvage map's step 2 was
therefore the obvious next question - intraday moves are small, so the target
is close - and it called that the main goal.

MEASURED ON 84,090 TRADES. 40 most liquid NSE names, 31 sessions of 5-minute
bars, entry at EVERY bar, each trade confined to its own session:

    stop%   cost_R   win%   stopped  timed out   breakeven win    net R
     0.25    1.689   26.8%    63.1%      10.1%           89.6%   -1.783
     0.50    0.844   16.2%    47.4%      36.5%           61.5%   -0.994
     0.75    0.563    9.9%    33.4%      56.6%           52.1%   -0.698
     1.00    0.422    6.9%    23.3%      69.8%           47.4%   -0.518
     1.50    0.281    3.7%    12.5%      83.8%           42.7%   -0.332
     2.00    0.211    1.9%     7.4%      90.7%           40.4%   -0.248

THE ANSWER IS NOT ABOUT COSTS, which is the part worth understanding. GROSS
expectancy is negative at every stop width - -0.150R at a 0.5% stop, -0.096R at
1.0%, -0.037R at 2.0% - so the round-trip cost required to break even comes out
NEGATIVE. You would have to be paid to trade. The 0.300% slippage assumption
that dominates the daily cost model is irrelevant here; the payoff structure
loses money before anyone charges for it.

WHY, IN ONE SENTENCE: the median session range is 2.18% of the open, and a 1:2
structure needs the price to travel twice the stop in favour while never
travelling once the stop against - two requirements that do not both fit inside
2.18%.

Spelled out, because each half is independently fatal:
  - THE STOP MUST CLEAR THE NOISE. The median 5-minute bar range is 0.166% of
    price, so a 0.5% stop sits about three bar-ranges away and ordinary
    oscillation takes it out. At 0.25% the stop is inside a single bad bar:
    63.1% of trades stop out.
  - THE TARGET MUST FIT IN THE SESSION. Widen the stop to clear the noise and
    the target moves to 2x it. At a 2% stop the 4% target exceeds the entire
    median session range, so 90.7% of trades simply run out of session.
  - AND THE TIME LIMIT TRUNCATES WINNERS, NOT LOSERS. The winner has to travel
    twice as far as the loser, so it takes longer, so the session boundary cuts
    it off disproportionately. At a 2% stop the resolved trades split 1.9% win
    to 7.4% stop - a ratio of 1:3.9, worse than the 1:2 the geometry alone
    would give.

ROBUST TO THE ONE CONVENTION THAT COULD FLATTER IT. A bar touching both the
stop and the target is unknowable without ticks; this module checks the STOP
first, which is the conservative reading. Re-run optimistically - target first -
the 0.5% cell moves from 16.2% to 16.3% and net R from -0.994 to -0.992. The
convention is not what makes the answer negative.

WHAT THIS DOES NOT SAY. Entry is UNCONDITIONAL - every bar - so this measures
the payoff STRUCTURE, not any particular signal. A signal could in principle
select bars with favourable drift. But the bar it would have to clear is now
quantified: at a 0.5% stop it must add more than +0.150R of gross edge merely
to reach zero, and then 0.844R more to cover costs. Nothing this project has
measured on daily bars comes close to +1.0R of selection.

Nor does it rule out other intraday structures. 1:1 needs a 50% win rate
against a 2x-closer target; the same lattice can measure it, and `rr` is a
parameter here for that reason. What is settled is the 1:2 version, which is
the one the desk is for.

THE SAMPLE IS SMALL AND CANNOT BE GROWN BY ASKING. Yahoo serves at most ~31
sessions of 5-minute history and refuses older windows outright, so this rests
on 31 sessions - see desk/marketdata/sources/yahoo.py. 84,090 trades is a lot
of entries drawn from a little calendar, and a regime where intraday ranges are
wider would move these numbers. The daily refresh accumulates what the source
will not hand over, so re-run this as the history deepens; `session_outcomes`
exists to be re-run rather than quoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["STATUTORY_ROUND_TRIP_PCT", "SessionOutcomes", "breakeven_win_rate",
           "session_outcomes"]

#: Statutory round-trip cost in percent - STT, exchange charges, GST, stamp
#: duty. Irreducible: it is tax, not execution quality, so no amount of broker
#: shopping or patience removes it. Slippage sits on top.
STATUTORY_ROUND_TRIP_PCT = 0.1222


@dataclass(frozen=True, slots=True)
class SessionOutcomes:
    """Target-first-or-stop-first counts over a set of intraday entries."""

    target: int
    stopped: int
    timed_out: int
    stop_pct: float
    rr: float
    max_bars: int

    @property
    def trades(self) -> int:
        return self.target + self.stopped + self.timed_out

    @property
    def win_rate(self) -> float | None:
        return self.target / self.trades if self.trades else None

    @property
    def gross_r(self) -> float | None:
        """Expectancy in R before costs. A timeout is scored 0R.

        SCORING A TIMEOUT AT ZERO IS GENEROUS and that is deliberate: the real
        exit is the session's last close, which after costs is usually slightly
        negative. Being generous here means a negative result cannot be blamed
        on the convention.
        """
        if not self.trades:
            return None
        return (self.target * self.rr - self.stopped) / self.trades

    def cost_r(self, *, slippage_pct: float = 0.300) -> float:
        """Round-trip cost as a share of risk.

        cost_R = round_trip% / stop%, so the quantity cancels and only stop
        WIDTH matters. This is why a tight intraday stop is so expensive: at
        0.25% the round trip is 1.7x the stop.
        """
        return (STATUTORY_ROUND_TRIP_PCT + slippage_pct) / self.stop_pct

    def net_r(self, *, slippage_pct: float = 0.300) -> float | None:
        g = self.gross_r
        return None if g is None else g - self.cost_r(slippage_pct=slippage_pct)

    def required_round_trip_pct(self) -> float:
        """The round-trip cost at which this cell would break even.

        NEGATIVE WHEN GROSS IS NEGATIVE, which is the headline finding: a
        negative requirement means no cost structure makes the cell viable,
        because it loses before costs. Compare against
        STATUTORY_ROUND_TRIP_PCT - anything below that is unreachable even
        with a zero-commission broker and perfect fills.
        """
        g = self.gross_r or 0.0
        return g * self.stop_pct


def breakeven_win_rate(*, stop_pct: float, rr: float = 2.0,
                       slippage_pct: float = 0.300) -> float:
    """The win rate a cell needs to break even, given its cost.

    For a payoff of `rr` against 1, net zero requires
    p*rr - (1-p) - cost_R = 0, so p = (1 + cost_R) / (1 + rr).
    """
    cost_r = (STATUTORY_ROUND_TRIP_PCT + slippage_pct) / stop_pct
    return (1.0 + cost_r) / (1.0 + rr)


def session_outcomes(bars: pd.DataFrame, *, stop_pct: float, rr: float = 2.0,
                     max_bars: int = 73, optimistic: bool = False,
                     symbol_col: str = "symbol",
                     session_col: str = "session") -> SessionOutcomes:
    """Walk every bar as an entry and record what happened first.

    `bars` is long-form with symbol, session, high, low, close. Each trade is
    confined to its own session: holding across the overnight gap is a
    different trade with different risk, and an intraday rule that cannot close
    by the session's last reliable bar is not the rule being tested.

    `optimistic` checks the TARGET before the stop on a bar touching both. The
    default is the conservative reading. Measured, the two differ by about 0.1
    percentage points of win rate, so this exists to demonstrate that rather
    than to be switched on.
    """
    if bars.empty:
        return SessionOutcomes(0, 0, 0, stop_pct, rr, max_bars)
    need = {symbol_col, session_col, "high", "low", "close"}
    missing = need - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing {sorted(missing)}")

    tgt = stopped = timed = 0
    for _, grp in bars.sort_values([symbol_col, session_col]).groupby(
            [symbol_col, session_col], sort=False):
        c = grp["close"].to_numpy(float)
        h = grp["high"].to_numpy(float)
        lo = grp["low"].to_numpy(float)
        n = len(c)
        for i in range(n - 1):
            entry = c[i]
            if not np.isfinite(entry) or entry <= 0:
                continue
            s_lvl = entry * (1.0 - stop_pct / 100.0)
            t_lvl = entry * (1.0 + rr * stop_pct / 100.0)
            last = min(i + max_bars, n - 1)
            hit = ""
            for j in range(i + 1, last + 1):
                if optimistic:
                    if h[j] >= t_lvl:
                        hit = "t"
                        break
                    if lo[j] <= s_lvl:
                        hit = "s"
                        break
                else:
                    if lo[j] <= s_lvl:
                        hit = "s"
                        break
                    if h[j] >= t_lvl:
                        hit = "t"
                        break
            if hit == "t":
                tgt += 1
            elif hit == "s":
                stopped += 1
            else:
                timed += 1
    return SessionOutcomes(tgt, stopped, timed, stop_pct, rr, max_bars)
