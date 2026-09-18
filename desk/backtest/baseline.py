"""The null hypothesis: what does RANDOM selection earn?

WHY A SINGLE BACKTEST NUMBER CANNOT BE READ ON ITS OWN
-------------------------------------------------------
The funnel measured -0.35R per trade. That is a fact about the whole
pipeline - Stage 0's universe, Stage 1's features, Stage 2's ranking,
Stage 4's sizing, the exit rules, and the costs - and it says nothing
about which of them is responsible.

The decisive question is what the SAME machinery earns when the only
thing removed is the ranking. Stage 0 picks the same survivors, Stage 1
computes the same features, Stage 4 applies the same gates and sizes
against the same capital, the exits and costs are identical - and the
names are drawn at random.

    random near 0R minus costs, strategy -0.35R
        -> the ranking is ACTIVELY HARMFUL. It is selecting names that go
           on to do worse than a coin flip, which is a real (invertible)
           signal, not an absent one.

    random also around -0.35R
        -> the ranking is irrelevant and the loss lives in the EXITS,
           the horizon, or the costs. Tuning factors would be wasted
           effort.

Those two point at completely different work, and without this you cannot
tell them apart.

ONE RANDOM RUN IS NOT A BASELINE
--------------------------------
A single shuffled draw is one sample from a wide distribution, and
comparing one number against another number is how noise gets promoted to
a finding. `run_baseline` runs many seeded trials and reports the spread,
so the strategy can be placed against a DISTRIBUTION - inside it means
the ranking has not demonstrated anything.

The seeds are explicit and sequential so a result is reproducible: an
unseeded baseline that moves every run cannot be argued with.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

from desk.backtest.costs import CostModel
from desk.backtest.engine import BacktestResult, run_backtest
from desk.contracts.enums import Regime

__all__ = ["BaselineResult", "run_baseline"]


@dataclass(slots=True)
class BaselineResult:
    """Many random runs, plus the strategy measured the same way."""

    trials: list[BacktestResult] = field(default_factory=list)
    strategy: BacktestResult | None = None

    @property
    def expectancies(self) -> list[float]:
        return [t.expectancy_r for t in self.trials
                if t.expectancy_r is not None]

    @property
    def mean(self) -> float | None:
        xs = self.expectancies
        return statistics.fmean(xs) if xs else None

    @property
    def stdev(self) -> float | None:
        xs = self.expectancies
        return statistics.stdev(xs) if len(xs) > 1 else None

    @property
    def spread(self) -> tuple[float, float] | None:
        xs = self.expectancies
        return (min(xs), max(xs)) if xs else None

    @property
    def percentile_of_strategy(self) -> float | None:
        """Where the strategy sits inside the random distribution.

        50 means indistinguishable from chance. Low means WORSE than
        random - which is informative rather than merely bad, because a
        reliably wrong signal can be inverted.
        """
        xs = self.expectancies
        if not xs or self.strategy is None or self.strategy.expectancy_r is None:
            return None
        below = sum(1 for x in xs if x < self.strategy.expectancy_r)
        return 100.0 * below / len(xs)

    def verdict(self) -> str:
        """What the comparison actually licenses saying. Deliberately
        conservative: with a handful of trials over a few hundred trades,
        'inside the random range' is the honest default."""
        if self.strategy is None or self.strategy.expectancy_r is None:
            return "the strategy produced no trades - nothing to compare"
        if not self.expectancies:
            return "no random trial produced trades - nothing to compare"

        pct = self.percentile_of_strategy
        lo, hi = self.spread
        s = self.strategy.expectancy_r
        if lo <= s <= hi:
            return (f"the strategy ({s:+.3f}R) sits INSIDE the random range "
                    f"({lo:+.3f}R to {hi:+.3f}R, {pct:.0f}th percentile). The "
                    f"ranking has not demonstrated an edge either way - and "
                    f"since random selection loses too, the loss is coming "
                    f"from the exits, the horizon or the costs rather than "
                    f"from which names are picked.")
        if s > hi:
            return (f"the strategy ({s:+.3f}R) beats EVERY random trial "
                    f"(best {hi:+.3f}R). The ranking is adding something - "
                    f"on this sample, in this regime.")
        return (f"the strategy ({s:+.3f}R) is WORSE than every random trial "
                f"(worst {lo:+.3f}R). That is a real signal pointing the "
                f"wrong way, not an absent one - worth testing inverted "
                f"before discarding.")

    def report(self) -> str:
        lines = [f"Random baseline: {len(self.trials)} trial(s)"]
        if self.expectancies:
            lo, hi = self.spread
            lines += [
                f"  random expectancy   mean {self.mean:+.3f}R"
                + (f", sd {self.stdev:.3f}" if self.stdev else ""),
                f"  random range        {lo:+.3f}R to {hi:+.3f}R",
                f"  random trades/trial "
                f"{statistics.fmean(len(t.trades) for t in self.trials):.0f}",
            ]
        if self.strategy is not None:
            lines.append(f"  STRATEGY            "
                         f"{self.strategy.expectancy_r:+.3f}R over "
                         f"{len(self.strategy.trades)} trades")
        lines += ["", "  " + self.verdict()]
        return "\n".join(lines)


def run_baseline(store, *, start: date, end: date, trials: int = 10,
                 seed: int = 0, capital: float = 1_000_000,
                 max_trades: int = 3, lookback: int = 120,
                 holding_days: int = 5, target_r: float | None = None,
                 regime: Regime = Regime.UNKNOWN,
                 costs: CostModel | None = None,
                 progress=None) -> BaselineResult:
    """Run the strategy once and random selection `trials` times.

    Everything except the SELECTION is held identical - same universe,
    features, gates, sizing, exits and costs - so the difference between
    them is attributable to the ranking and to nothing else.
    """
    common = dict(start=start, end=end, capital=capital,
                  max_trades=max_trades, lookback=lookback,
                  holding_days=holding_days, target_r=target_r,
                  regime=regime, costs=costs)

    if progress:
        progress("strategy", 0, trials)
    out = BaselineResult(strategy=run_backtest(store, **common))

    for i in range(trials):
        if progress:
            progress("random", i + 1, trials)
        # Sequential seeds, so the whole comparison is reproducible.
        out.trials.append(run_backtest(store, shuffle=seed + i, **common))
    return out
