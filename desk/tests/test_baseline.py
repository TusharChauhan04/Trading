"""The random baseline - the null hypothesis the ranking has to beat.

A single backtest number cannot be read on its own. -0.35R is a fact
about the WHOLE pipeline and says nothing about which part is
responsible. The comparison that separates them is what the same
machinery earns when only the selection is randomised.

The property tested hardest: the comparison must be REPRODUCIBLE and must
report a DISTRIBUTION. One shuffled draw against one strategy number is
how noise gets promoted to a finding.
"""

from __future__ import annotations

from datetime import date

import pytest

from desk.backtest import BacktestResult, SimulatedTrade
from desk.backtest.baseline import BaselineResult
from desk.journal.models import ExitReason

DAY = date(2026, 9, 11)


def _trade(r, entry=100.0, stop=90.0):
    return SimulatedTrade(
        symbol="X.NS", entry_date=DAY, entry_price=entry, planned_entry=entry,
        exit_date=DAY, exit_price=entry + r * abs(entry - stop),
        exit_reason=ExitReason.TARGET if r > 0 else ExitReason.STOP,
        qty=10, bars_held=1, stop=stop, target=None)


def _result(rs):
    return BacktestResult(start=DAY, end=DAY, trades=[_trade(r) for r in rs])


# --- the distribution ----------------------------------------------------

def test_the_baseline_reports_a_spread_not_a_single_number():
    """Comparing one number against another number is how noise becomes a
    finding."""
    b = BaselineResult(trials=[_result([1.0, -1.0]), _result([2.0, -1.0]),
                               _result([-1.0, -1.0])])
    assert b.mean is not None
    assert b.stdev is not None
    lo, hi = b.spread
    assert lo < hi


def test_a_single_trial_reports_no_standard_deviation():
    """Honest: one sample has no spread, and inventing one would imply a
    confidence the data does not support."""
    b = BaselineResult(trials=[_result([1.0, -1.0])])
    assert b.stdev is None
    assert b.mean is not None


def test_no_trials_reports_nothing_rather_than_zero():
    b = BaselineResult(trials=[])
    assert b.mean is None and b.spread is None
    assert "nothing to compare" in b.verdict()


# --- the verdict, which is the point -------------------------------------

def test_a_strategy_inside_the_random_range_has_proved_nothing():
    """And critically, it says WHERE the loss is instead: if random loses
    too, the exits or costs are responsible, not the ranking."""
    b = BaselineResult(
        trials=[_result([-1.0, 0.0]), _result([0.0, -1.0]), _result([-1.0, 1.0])],
        strategy=_result([-1.0, 0.0]))
    v = b.verdict()
    assert "INSIDE the random range" in v
    assert "exits" in v and "costs" in v


def test_a_strategy_beating_every_trial_is_credited_but_hedged():
    b = BaselineResult(trials=[_result([-1.0]), _result([-1.0])],
                       strategy=_result([2.0]))
    v = b.verdict()
    assert "beats EVERY random trial" in v
    assert "this sample" in v and "regime" in v, \
        "a win on one window must not be reported as a general result"


def test_a_strategy_worse_than_every_trial_is_called_a_signal():
    """A reliably wrong signal is invertible, which is information. Saying
    only 'bad' would throw that away."""
    b = BaselineResult(trials=[_result([0.0]), _result([1.0])],
                       strategy=_result([-1.0]))
    v = b.verdict()
    assert "WORSE than every random trial" in v
    assert "inverted" in v


def test_the_percentile_places_the_strategy_in_the_distribution():
    b = BaselineResult(
        trials=[_result([-2.0]), _result([-1.0]), _result([0.0]), _result([1.0])],
        strategy=_result([-1.5]))
    # One of four trials is below -1.5.
    assert b.percentile_of_strategy == pytest.approx(25.0)


def test_a_strategy_with_no_trades_is_not_scored():
    b = BaselineResult(trials=[_result([1.0])],
                       strategy=BacktestResult(start=DAY, end=DAY))
    assert b.percentile_of_strategy is None
    assert "no trades" in b.verdict()


def test_the_report_names_both_sides():
    b = BaselineResult(trials=[_result([-1.0]), _result([0.0])],
                       strategy=_result([-0.5]))
    r = b.report()
    assert "Random baseline" in r and "STRATEGY" in r
    assert "random range" in r


# --- the control that makes it possible ----------------------------------

def test_the_funnel_exposes_a_seeded_shuffle():
    """Without it there is no way to hold everything constant except the
    selection, and the -0.35R number cannot be attributed to anything."""
    import inspect

    from desk.api import main as api
    sig = inspect.signature(api._run_funnel)
    assert "shuffle" in sig.parameters
    assert sig.parameters["shuffle"].default is None, \
        "the research control must be inert by default"


def test_the_backtest_forwards_the_shuffle():
    import inspect

    from desk.backtest import engine
    code = [ln.split("#")[0]
            for ln in inspect.getsource(engine.run_backtest).splitlines()]
    assert any("shuffle=shuffle" in ln for ln in code)


def test_run_baseline_uses_sequential_seeds():
    """An unseeded baseline that moves every run cannot be argued with."""
    import inspect

    from desk.backtest import baseline
    assert "seed + i" in inspect.getsource(baseline.run_baseline)


def test_the_baseline_holds_everything_but_selection_constant():
    """If any other parameter differed between the strategy run and the
    random runs, the comparison would attribute that difference to the
    ranking."""
    import inspect

    src = inspect.getsource(
        __import__("desk.backtest.baseline", fromlist=["x"]).run_baseline)
    # One `common` dict is built and passed to both.
    assert "common = dict(" in src
    assert src.count("**common") == 2
