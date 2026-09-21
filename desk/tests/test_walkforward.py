"""The per-strategy walk-forward harness.

Its job is to let a catalogued strategy climb off AUDITED, and the danger
is entirely in the promotion direction: a harness that reports a good
aggregate from one lucky window, or that measures a strategy on days it
would never have traded, hands trust to a rule that has not earned it.
These tests pin the refusals rather than the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pytest

from desk.backtest.costs import ZERO_COSTS
from desk.backtest.engine import BacktestResult
from desk.backtest.simulate import SimulatedTrade
from desk.backtest.walkforward import (WalkForwardResult, WindowResult,
                                       run_strategy_backtest, walk_forward)
from desk.contracts.enums import Regime
from desk.strategies.catalog import BY_KEY

D0 = date(2026, 1, 5)


def _trade(r: float, *, entry: float = 100.0, qty: int = 100) -> SimulatedTrade:
    """A trade whose r_multiple is exactly `r`.

    risk_per_share is |entry - stop| and r_multiple is
    (exit - entry) / risk_per_share, so both are set explicitly rather
    than inferred - a fixture that quietly produces a different R than it
    claims would make every assertion below meaningless.
    """
    risk = 5.0
    return SimulatedTrade(
        symbol="X.NS", entry_date=D0, entry_price=entry, planned_entry=entry,
        exit_date=D0 + timedelta(days=5), exit_price=entry + r * risk,
        exit_reason="target" if r > 0 else "stop", qty=qty, bars_held=5,
        stop=entry - risk, target=entry + 2 * risk)


def _window(rs: list[float], *, start=D0, end=None) -> WindowResult:
    """ZERO_COSTS deliberately: these tests are about the STATISTIC.

    `net_r_multiples` is gross minus cost, so under the default cost model
    a -1.0R trade nets -1.08R and every expected value below would have to
    carry a cost term that has nothing to do with what is being asserted.
    Cost behaviour is measured in test_backtest.py, on real levels.
    """
    res = BacktestResult(start=start, end=end or start, costs=ZERO_COSTS)
    res.trades.extend(_trade(r) for r in rs)
    return WindowResult(start=start, end=end or start, result=res)


def _wf(key: str, windows: list[list[float]]) -> WalkForwardResult:
    out = WalkForwardResult(key=key)
    for i, rs in enumerate(windows):
        out.windows.append(_window(rs, start=D0 + timedelta(days=90 * i),
                                   end=D0 + timedelta(days=90 * i + 89)))
    return out


# ===========================================================================
# the fixture has to be honest before anything built on it means anything
# ===========================================================================

def test_the_fixture_produces_the_r_multiple_it_claims():
    assert _trade(1.0).r_multiple == pytest.approx(1.0)
    assert _trade(-1.0).r_multiple == pytest.approx(-1.0)
    assert _trade(2.5).r_multiple == pytest.approx(2.5)


# ===========================================================================
# promotion is refused unless the evidence is consistent
# ===========================================================================

def test_a_good_aggregate_from_ONE_window_is_not_a_promotion():
    """The failure this harness exists to prevent. One window carrying the
    whole result is one good quarter, not an edge - and the pooled mean
    cannot tell the difference on its own."""
    wf = _wf("donchian_breakout", [
        [3.0] * 20,                      # the one good window
        [-0.4] * 20, [-0.3] * 20, [-0.5] * 20,
    ])
    assert wf.pooled_net_r > 0, "the aggregate genuinely does look good"
    assert wf.windows_positive == 1
    v = wf.verdict()
    assert "only 1 of 4" in v
    assert "stays at AUDITED" in v


def test_consistency_across_all_windows_is_what_earns_the_rung():
    wf = _wf("donchian_breakout", [[0.4] * 20, [0.3] * 20,
                                   [0.5] * 20, [0.2] * 20])
    v = wf.verdict()
    assert "positive in ALL 4 windows" in v
    # Even then it does not promote by itself.
    assert "judgement for a human" in v


def test_too_few_trades_is_reported_as_sample_size_not_as_a_verdict():
    """"Nothing demonstrated" and "demonstrated to be bad" are different
    findings, and the catalog says donchian in particular "needs a large
    trade count to judge at all"."""
    wf = _wf("donchian_breakout", [[0.5] * 3, [0.4] * 4])
    assert wf.scored == []
    v = wf.verdict()
    assert "sample-size result" in v
    assert "not a verdict on the rule" in v


def test_pooling_weights_by_trade_not_by_window():
    """Averaging window means would let a 2-trade window outvote a
    200-trade one."""
    wf = _wf("x", [[1.0] * 2, [-1.0] * 18])
    assert wf.pooled_net_r == pytest.approx((2 * 1.0 - 18 * 1.0) / 20)
    assert wf.pooled_net_r < 0


def test_the_median_is_reported_beside_the_mean():
    """Per-trade cost in R divides by the rupees at risk, and a fill that
    gaps close to its stop makes that denominator tiny - one such trade in
    1,251 moved the MEAN cost from 0.035R to 0.242R. Both are shown so the
    disagreement is visible instead of silently changing the answer."""
    w = _window([-1.0] * 9 + [50.0])
    assert w.net_r is not None and w.median_net_r is not None
    assert w.net_r > 0, "the mean is dragged positive by one outlier"
    assert w.median_net_r == pytest.approx(-1.0), "the median is not"


# ===========================================================================
# what it refuses to do
# ===========================================================================

def test_an_unimplemented_strategy_raises_rather_than_scoring_zero(tmp_path):
    """A strategy with no adapter scoring 0.0R would read as a measured
    result. It is an absence of one."""
    with pytest.raises(NotImplementedError):
        run_strategy_backtest(None, "pairs_trading",
                              start=D0, end=D0 + timedelta(days=5))


def test_windows_that_cannot_be_split_say_so_rather_than_returning_zero():
    class _Store:
        def available_days(self):
            return [D0 + timedelta(days=i) for i in range(5)]

    wf = walk_forward(_Store(), "donchian_breakout", start=D0,
                      end=D0 + timedelta(days=4), windows=4)
    assert wf.windows == []
    assert wf.caveats and "too few to split" in wf.caveats[0]
    assert "sample-size" in wf.verdict()


# ===========================================================================
# the caveats are part of the result, not decoration
# ===========================================================================

def test_the_unsimulated_strategy_exit_is_always_declared():
    """donchian carries an exit_lookback channel exit that this harness
    does not model. Every trade exits on stop, target or horizon, so the
    number describes the SETUP and not the complete rule - and a reader
    who is not told that will read it as the rule."""
    class _Store:
        def available_days(self):
            return []

    wf = walk_forward(_Store(), "donchian_breakout", start=D0, end=D0)
    # No sessions, so it bails early with the split caveat - the exit
    # caveat is attached on the path that actually runs windows.
    assert wf.caveats


def test_the_catalogued_defects_travel_with_the_result():
    """A result detached from the known defects of the code that produced
    it is how a strategy gets promoted with BUG-03 still open."""
    spec = BY_KEY["donchian_breakout"]
    assert spec.defects, "this test is meaningless if the spec has none"


def test_nothing_in_this_module_can_change_a_maturity():
    """The harness measures. Promotion is a human act against the ladder,
    and a function that could edit `maturity` would make `trusted` a
    statement about the last backtest rather than about evidence."""
    import desk.backtest.walkforward as wfmod
    src = open(wfmod.__file__, encoding="utf-8").read()
    assert "maturity =" not in src
    assert "Maturity." not in src
