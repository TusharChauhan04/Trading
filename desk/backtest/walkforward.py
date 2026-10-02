"""Measuring a catalogued strategy on the same yardstick as the funnel.

B3 of the roadmap, and the machine the `Maturity` ladder depends on.
`StrategySpec.trusted` is False below WALK_FORWARD, so no catalogued
strategy can ever be sized until something runs it out of sample. Nothing
did: `run_backtest` replays `_run_funnel`, which is the Stage 0-4 factor
path and cannot reach an adapter.

SAME YARDSTICK, and that is the entire point of reusing pieces here rather
than writing a second harness. The universe is Stage 0's survivors, the
features are Stage 1's, sizing goes through `size_position` with every gate
it owns, exits go through `simulate_trade`, and the result is a
`BacktestResult` so costs and net R are computed by the code that already
computes them for the funnel. A strategy scoring -0.05R here and the funnel
scoring -0.082R are comparable because nothing in between differs.

THE REGIME GATE IS ENFORCED PER SESSION. A strategy is asked only on days
`eligible()` permits it, exactly as the live endpoint does. Measuring
donchian_breakout across range days would produce a number the live desk
would never experience, and it would be a WORSE number - which is the
dangerous direction, because it would retire a strategy on evidence from
days it was never going to trade.

WHAT THIS DOES NOT SIMULATE, declared rather than left implicit:

  - A STRATEGY'S OWN EXIT. donchian_breakout carries an `exit_lookback`
    channel exit and it is not modelled; every trade here exits on the
    proposal's stop, its target, or the horizon, through the shared
    simulator. That keeps strategies comparable to each other and to the
    funnel, and it means these numbers describe the SETUP rather than the
    complete strategy. A Donchian result is therefore a lower bound on a
    rule whose stated edge includes cutting failures early.
  - PORTFOLIO INTERACTION ACROSS STRATEGIES. Each runs against its own
    empty portfolio, so heat and correlation caps bind within a strategy
    and not between them. Combining them is the coordinator's problem (B4)
    and combining them HERE would measure the coordinator, not the rule.

WALK-FORWARD, HONESTLY LABELLED. These strategies have no fitted
parameters - 20/10/2.0 for Donchian and 20/2.0/14/30 for Bollinger are
catalog constants, not values optimised on this data - so splitting the
history buys no protection against overfitting, because no fitting
happened. What it does buy is a test of STABILITY: a rule that earns its
whole result in one window and loses in the others has not demonstrated an
edge, it has demonstrated one good quarter. `WalkForwardResult.verdict()`
reports consistency and refuses to promote on the aggregate alone.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:          # the robustness package imports from backtest, so
    from desk.backtest.permutation import PermutationResult   # these stay
    from desk.robustness import RobustnessReport              # lazy

from desk.backtest.costs import CostModel
from desk.backtest.engine import (BacktestResult, DayResult,
                                  min_risk_pct_for)
from desk.backtest.simulate import simulate_trade
from desk.marketdata.sectors import SectorMap
from desk.regime.engine import compute_regime
from desk.risk.engine import Portfolio, RiskConfig, size_position
from desk.scanner.stage0 import run_stage0
from desk.scanner.stage1 import REQUIRED_BARS, run_stage1
from desk.strategies.adapters import ADAPTERS, propose
from desk.strategies.catalog import BY_KEY

#: The sibling harness, re-exported so there is ONE place to look for "walk
#: something forward" rather than two that must be known about separately.
#:
#: WHICH ONE TO USE. `walk_forward` here drives a CATALOGUED strategy through
#: the desk's whole funnel - Stage 0, Stage 1, regime routing, the risk engine,
#: structural stops. `walk_forward_preset` drives a fixed-percentage preset cell
#: over a price panel, because that is what the surviving configuration actually
#: is, and forcing it through the funnel would change the thing being tested.
#:
#: The preset version also asks MORE per window: it runs the permutation test
#: inside each window, which is what caught the 12%/20-bar cell. Its full-sample
#: permutation p was 0.0033, yet window by window it was significant in only 2
#: of 4 - the edge sat almost entirely in Sep 2023 to Jun 2024. A whole-sample
#: permutation cannot see that, because shuffling timing across the whole sample
#: still loses to a concentrated edge.
from desk.backtest.preset_wf import (            # noqa: E402
    PresetWalkForward, PresetWindow, walk_forward_preset,
)

__all__ = ["PresetWalkForward", "PresetWindow", "run_strategy_backtest",
           "walk_forward", "walk_forward_preset", "WalkForwardResult",
           "WindowResult"]


@dataclass(slots=True)
class WindowResult:
    """One out-of-sample window."""
    start: date
    end: date
    result: BacktestResult
    sessions_eligible: int = 0
    sessions_silenced: int = 0

    @property
    def trades(self) -> int:
        return len(self.result.trades)

    @property
    def strategy_exits(self) -> int:
        """How many trades left on the STRATEGY'S OWN signal.

        Reported because "simulated" and "actually fired" are different
        facts, and the gap between them is where a rule gets credit for
        machinery that never ran. Measured on donchian_breakout: zero, on
        every trade, because its catalogued 2.0x ATR stop sits ABOVE its
        10-session channel on 100% of setups (median 16.77% of entry
        price above it). Price has to cross the stop before it can close
        under the channel, so the exit is unreachable as specified.
        """
        return sum(1 for x in self.result.trades
                   if x.exit_reason == "strategy_exit")

    @property
    def net_r(self) -> float | None:
        return self.result.expectancy_r_net

    @property
    def median_net_r(self) -> float | None:
        """The ROBUST centre, and it is reported beside the mean on purpose.

        Per-trade cost in R is cost divided by the rupees at risk, and an
        entry that gaps to sit very close to its stop makes that
        denominator tiny - one such trade in 1,251 carried a 251R cost and
        moved the MEAN cost per trade from 0.035R to 0.242R. The mean is
        the right definition of expectancy and the wrong summary of a
        distribution with that tail, so both are shown and disagreement
        between them is itself the signal.
        """
        rs = self.result.net_r_multiples
        return statistics.median(rs) if rs else None


@dataclass(slots=True)
class WalkForwardResult:
    key: str
    windows: list[WindowResult] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def scored(self) -> list[WindowResult]:
        """Windows that produced enough trades to say anything about."""
        return [w for w in self.windows if w.trades >= 10]

    @property
    def total_trades(self) -> int:
        return sum(w.trades for w in self.windows)

    @property
    def total_strategy_exits(self) -> int:
        return sum(w.strategy_exits for w in self.windows)

    @property
    def pooled_net_r(self) -> float | None:
        """Every trade from every window, pooled.

        Pooled rather than an average of window means: windows differ in
        trade count, and averaging their averages would weight a 12-trade
        window equally with a 200-trade one.
        """
        rs = [r for w in self.windows for r in w.result.net_r_multiples]
        return (sum(rs) / len(rs)) if rs else None

    @property
    def pooled_median_net_r(self) -> float | None:
        rs = [r for w in self.windows for r in w.result.net_r_multiples]
        return statistics.median(rs) if rs else None

    @property
    def windows_positive(self) -> int:
        return sum(1 for w in self.scored
                   if w.net_r is not None and w.net_r > 0)

    def robustness(self, *, num_trials: int) -> "RobustnessReport":
        """How much of the POOLED result survives admitting the search.

        Pooled window-by-window means as the series: a window is the unit this
        harness judges on, so it is the unit the deflated statistics should see
        too. Feeding individual trades instead would claim far more
        observations than there are independent periods.

        `num_trials` is required for the same reason it is required everywhere
        else in this project - a silent default of 1 asserts that exactly one
        thing was tried, which is almost never true of anything worth
        deflating. Count the strategies times the parameter cells times the
        regimes that were tried before this one was reported.
        """
        from desk.robustness import deflated_sharpe

        series = [w.net_r for w in self.scored if w.net_r is not None]
        # A window is roughly a quarter when four of them span three years, so
        # ~4 a year. Passing 252 would annualise a quarterly figure as a daily
        # one and inflate the Sharpe by sqrt(63).
        return deflated_sharpe(series, num_trials=num_trials,
                               periods_per_year=4)

    def timing_test(self, signals, lattice, exit_rows, *, cost_r: float = 0.0,
                    n_permutations: int = 500) -> "PermutationResult":
        """Did the entry TIMING carry information, independent of the search?

        A complement to `robustness`, not a substitute. The deflated statistics
        need observations and will report "insufficient" on the handful of
        windows a walk-forward produces; a permutation test builds its null
        from this data and is exact at any sample size. A strategy seeking
        promotion should pass both - DSR says the result is not an artefact of
        trying many things, the permutation says it is not an artefact of a
        rising window.

        See desk.backtest.permutation for why a SIGN-FLIP null would be wrong
        for R-multiples from a stop-and-target system.
        """
        from desk.robustness import signal_timing_test

        return signal_timing_test(signals, lattice, exit_rows, cost_r=cost_r,
                                  n_permutations=n_permutations)

    def verdict(self) -> str:
        """What this evidence licenses saying about the maturity ladder.

        Deliberately conservative, and it will not promote on an aggregate.
        The ladder's own docstring says never skip a rung, and the rung
        above AUDITED is earned by surviving out-of-sample windows - plural.
        """
        n = len(self.scored)
        if n == 0:
            return (f"{self.key}: no window produced 10+ trades. Nothing is "
                    f"demonstrated either way - this is a sample-size "
                    f"result, not a verdict on the rule.")
        pooled = self.pooled_net_r
        med = self.pooled_median_net_r
        pos = self.windows_positive

        if pooled is not None and med is not None and pooled > 0 and med > 0 \
                and pos == n and n >= 3:
            return (f"{self.key}: positive in ALL {n} windows "
                    f"(pooled {pooled:+.4f}R, median {med:+.4f}R over "
                    f"{self.total_trades} trades). This is what REVALIDATED "
                    f"-> WALK_FORWARD is supposed to look like; promoting it "
                    f"is a judgement for a human, not this function.")
        if pooled is not None and pooled > 0 and pos < n:
            return (f"{self.key}: pooled {pooled:+.4f}R but positive in only "
                    f"{pos} of {n} windows. An edge concentrated in some "
                    f"periods and absent in others has not been "
                    f"demonstrated - it stays at AUDITED.")
        return (f"{self.key}: pooled {pooled:+.4f}R across {n} windows, "
                f"positive in {pos}. No case for promotion."
                if pooled is not None else
                f"{self.key}: no R multiples to pool.")

    def report(self) -> str:
        lines = [f"Walk-forward: {self.key}  ({BY_KEY[self.key].name})",
                 f"  {'window':<26}{'sess':>6}{'trades':>8}"
                 f"{'net R':>10}{'median':>10}{'win%':>8}{'sExit':>7}"]
        for w in self.windows:
            nr = f"{w.net_r:+.4f}" if w.net_r is not None else "     -"
            md = f"{w.median_net_r:+.4f}" if w.median_net_r is not None else "     -"
            wr = (f"{w.result.win_rate:.1f}"
                  if w.result.r_multiples else "   -")
            lines.append(
                f"  {str(w.start) + ' - ' + str(w.end):<26}"
                f"{w.sessions_eligible:>6}{w.trades:>8}{nr:>10}{md:>10}"
                f"{wr:>8}{w.strategy_exits:>7}")
        p, m = self.pooled_net_r, self.pooled_median_net_r
        lines.append(f"  {'POOLED':<26}{'':>6}{self.total_trades:>8}"
                     f"{(f'{p:+.4f}' if p is not None else '-'):>10}"
                     f"{(f'{m:+.4f}' if m is not None else '-'):>10}")
        for c in self.caveats:
            lines.append(f"  CAVEAT: {c}")
        lines += ["", "  " + self.verdict()]
        return "\n".join(lines)


def run_strategy_backtest(store, key: str, *, start: date, end: date,
                          capital: float = 1_000_000, max_trades: int = 3,
                          lookback: int = 120, holding_days: int = 10,
                          costs: CostModel | None = None,
                          enforce_regime: bool = True,
                          cfg: RiskConfig | None = None,
                          use_strategy_exit: bool = False,
                          progress=None) -> tuple[BacktestResult, int, int]:
    """Replay ONE catalogued strategy over [start, end].

    Returns (result, sessions_eligible, sessions_silenced). The two counts
    are returned rather than folded away because "the strategy took no
    trades" and "the strategy was never allowed to speak" are different
    facts about a window, and a report that cannot separate them will read
    a correctly-silenced strategy as a failed one.
    """
    if key not in ADAPTERS:
        raise NotImplementedError(
            f"{key!r} has no adapter - implemented: {sorted(ADAPTERS)}")
    spec = BY_KEY[key]
    result = BacktestResult(start=start, end=end, costs=costs or CostModel())
    risk = cfg or RiskConfig(capital=capital)

    available = store.available_days()
    sessions = [d for d in available if start <= d <= end]
    if not sessions:
        return result, 0, 0

    # Both frames loaded ONCE, same reasoning as run_backtest: per-session
    # reads re-scan the same parquet bytes up to `lookback` times each.
    forward = store.history(as_of=available[-1], start=sessions[0],
                            columns=["open", "high", "low", "close"])
    lookback_start = available[max(0, available.index(sessions[0]) - lookback)]
    prefetched = store.history(as_of=sessions[-1], start=lookback_start,
                               columns=list(REQUIRED_BARS))
    sectors = SectorMap.load(store.root.parent / "sectors.json") \
        if (store.root.parent / "sectors.json").is_file() else None

    # THE STRATEGY'S OWN EXIT, built once for every symbol at once rather
    # than per trade. donchian_breakout leaves on a close below the lowest
    # low of the previous `exit_lookback` sessions, so the level is
    # low.shift(1).rolling(k).min() - shifted first, for the same reason
    # the entry channel is: a window containing the current bar includes
    # that bar's own low, and the rule would fire on a level it helped set.
    #
    # Reading the forward frame here is legitimate for the same reason
    # simulate.py may: the decision is already fixed before this is touched.
    exit_matrix = None
    if use_strategy_exit:
        k = spec.params.get("exit_lookback")
        if k is not None:
            lows = forward.wide("low")
            exit_matrix = lows.shift(1).rolling(int(k),
                                                min_periods=int(k)).min()

    floor_pct = min_risk_pct_for(result.costs, risk)
    eligible_days = silenced_days = 0
    for i, day in enumerate(sessions, 1):
        if progress:
            progress(i, len(sessions), day)
        try:
            snapshot = store.load_day(day)
        except Exception as exc:                      # noqa: BLE001
            result.days.append(DayResult(as_of=day, error=str(exc)[:200]))
            continue

        stage0 = run_stage0(snapshot)
        survivors = stage0.survivors["symbol"].tolist()
        if not survivors:
            result.days.append(DayResult(as_of=day))
            continue
        history = prefetched.window(as_of=day, lookback=lookback,
                                    symbols=survivors)
        if history.frame.empty:
            result.days.append(DayResult(as_of=day))
            continue

        if enforce_regime:
            state = compute_regime(history, as_of=day, sectors=sectors)
            if state.label in spec.hostile_regimes:
                silenced_days += 1
                result.days.append(DayResult(
                    as_of=day,
                    no_trade_reason=f"silenced: {state.label.value} is "
                                    f"hostile to {key}"))
                continue
        eligible_days += 1

        stage1 = run_stage1(history, as_of=day)
        ideas = propose(key, stage1, limit=max_trades)
        day_row = DayResult(as_of=day, considered=len(stage1.features),
                            proposed=len(ideas))

        portfolio = Portfolio()
        for idea in ideas:
            # THE SAME GATE THE FUNNEL USES. Nothing about being a
            # catalogued strategy buys a setup past the risk engine, and
            # `today=day` is not cosmetic - against the wall clock every
            # historical session is stale and the whole replay silently
            # returns nothing.
            sized = size_position(
                symbol=idea.symbol, entry=idea.entry, stop=idea.stop,
                target=idea.target, cfg=risk, portfolio=portfolio,
                atr_pct=idea.metadata.get("atr_pct"),
                data_as_of=day, today=day)
            if not sized.approved or sized.qty <= 0:
                why = (sized.reasons[0].value if sized.reasons
                       else "no size")
                result.not_taken[why] = result.not_taken.get(why, 0) + 1
                continue
            bars = forward.series(idea.symbol)
            levels = None
            if exit_matrix is not None and idea.symbol in exit_matrix.columns:
                levels = exit_matrix[idea.symbol]
            sim = simulate_trade(bars, symbol=idea.symbol, decided_on=day,
                                 planned_entry=sized.entry, stop=sized.stop,
                                 target=sized.target, qty=sized.qty,
                                 horizon_days=holding_days,
                                 min_risk_pct=floor_pct,
                                 exit_levels=levels)
            if sim.trade is None:
                why = sim.reason_not_taken or "unknown"
                result.not_taken[why] = result.not_taken.get(why, 0) + 1
                continue
            result.trades.append(sim.trade)
        result.days.append(day_row)

    return result, eligible_days, silenced_days


def walk_forward(store, key: str, *, start: date, end: date, windows: int = 4,
                 **kwargs) -> WalkForwardResult:
    """Split [start, end] into `windows` disjoint periods and score each.

    Disjoint and contiguous, not rolling-with-overlap: overlapping windows
    share trades, so "positive in 3 of 4" would count the same trades more
    than once and read as corroboration when it is repetition.
    """
    out = WalkForwardResult(key=key)
    available = [d for d in store.available_days() if start <= d <= end]
    if len(available) < windows * 2:
        out.caveats.append(
            f"only {len(available)} sessions in range - too few to split "
            f"into {windows} windows that could each say anything")
        return out

    size = len(available) // windows
    for w in range(windows):
        lo = available[w * size]
        hi = available[(w + 1) * size - 1] if w < windows - 1 else available[-1]
        res, elig, sil = run_strategy_backtest(store, key, start=lo, end=hi,
                                               **kwargs)
        out.windows.append(WindowResult(start=lo, end=hi, result=res,
                                        sessions_eligible=elig,
                                        sessions_silenced=sil))

    spec = BY_KEY[key]
    if kwargs.get("use_strategy_exit") and spec.params.get("exit_lookback"):
        fired = out.total_strategy_exits
        if fired:
            out.caveats.append(
                f"the strategy's own exit IS simulated "
                f"({int(spec.params['exit_lookback'])}-session channel) and "
                f"fired on {fired} of {out.total_trades} trades, so this "
                f"measures the complete rule rather than the setup alone")
        else:
            # The case that would otherwise read as "the complete rule was
            # measured" while the exit never ran. It is not a harness fault:
            # the two catalogued parameters are incompatible.
            out.caveats.append(
                f"the strategy's own exit was simulated and fired ZERO "
                f"times in {out.total_trades} trades - its "
                f"{int(spec.params['exit_lookback'])}-session channel sits "
                f"BELOW the catalogued "
                f"{spec.params.get('atr_stop_mult')}x ATR stop on every "
                f"setup, so price must cross the stop before it can close "
                f"under the channel. The exit is unreachable as specified, "
                f"and the result is the complete rule only because the "
                f"exit half of it can never happen")
    else:
        out.caveats.append(
            "the strategy's own exit is NOT simulated - every trade exits on "
            "the proposal's stop, target or the horizon, so this measures the "
            "SETUP rather than the complete rule")
    if spec.defects:
        out.caveats.append(
            f"{len(spec.defects)} catalogued defect(s) still stand: "
            + "; ".join(d.split(":")[0] for d in spec.defects))
    total_sil = sum(w.sessions_silenced for w in out.windows)
    if total_sil:
        out.caveats.append(
            f"{total_sil} session(s) skipped because the measured regime "
            f"was hostile - correct behaviour, and it means the trade count "
            f"is lower than the calendar suggests")
    return out
