"""Drawdown must never be understated, and the block bootstrap is why.

A risk model's failure mode is one-directional. Overstating drawdown costs
opportunity, which is recoverable. Understating it sends real money into a path
the person cannot sit through, which is not. So every test here that could go
either way asserts the conservative side.

The specific thing being guarded: OpenTerminalUI's monte_carlo resamples IID,
which breaks losing streaks apart. On the real donchian sequence it reported a
p95 drawdown of 16.8% against the block bootstrap's 33.2%, and a 1.7% chance of
exceeding 20% against a real 32.6%. That is a 19x understatement of the number
that actually decides whether someone keeps trading.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from desk.backtest.ruin import (
    compare_iid, simulate_paths, trades_to_significance,
)


def _streaky(n: int = 600, seed: int = 7) -> list[float]:
    """R multiples with CLUSTERED losses - runs, not coin flips.

    Built deliberately so the two bootstraps must disagree: the same values in
    the same proportions, but arranged in runs. An IID resample cannot see the
    arrangement, which is the whole point.
    """
    rng = np.random.default_rng(seed)
    out: list[float] = []
    while len(out) < n:
        if rng.random() < 0.35:
            out += [-1.0] * int(rng.integers(5, 12))      # a losing streak
        else:
            out += [2.0] * int(rng.integers(2, 5))        # a winning run
    return out[:n]


# -- the headline property ---------------------------------------------------

def test_the_block_bootstrap_reports_deeper_drawdown_than_iid() -> None:
    """THE reason the borrowed module was not imported.

    Same values, same proportions. The IID resample destroys the runs and
    reports a kinder worst case - in the one direction a risk model must not
    err.
    """
    gap = compare_iid(_streaky(), n_trades=150, n_paths=400,
                      risk_per_trade_pct=1.0, block=20, seed=7)
    assert gap["block_p95_dd_pct"] > gap["iid_p95_dd_pct"], (
        "IID matched or exceeded the block bootstrap - the block sampling is "
        "not preserving streaks"
    )
    assert gap["block_prob_dd_over_20"] >= gap["iid_prob_dd_over_20"]


def test_a_longer_block_preserves_more_clustering() -> None:
    """Monotone in the right direction: a block of 1 IS the IID bootstrap, so
    growing the block can only reveal more streak risk."""
    rs = _streaky()
    short = simulate_paths(rs, n_trades=150, n_paths=400, block=1, seed=7)
    long_ = simulate_paths(rs, n_trades=150, n_paths=400, block=30, seed=7)
    assert long_.p95_max_drawdown_pct > short.p95_max_drawdown_pct


# -- the shape of the answer -------------------------------------------------

def test_bigger_risk_per_trade_means_bigger_drawdown() -> None:
    rs = _streaky()
    half = simulate_paths(rs, n_trades=200, n_paths=400,
                          risk_per_trade_pct=0.5, seed=7)
    double = simulate_paths(rs, n_trades=200, n_paths=400,
                            risk_per_trade_pct=2.0, seed=7)
    assert double.p95_max_drawdown_pct > half.p95_max_drawdown_pct
    assert double.prob_drawdown_over_35 >= half.prob_drawdown_over_35


def test_a_losing_sequence_mostly_loses() -> None:
    """Sanity in the other direction: a negative-expectancy book must not come
    back with a reassuring median."""
    rep = simulate_paths([-1.0] * 60 + [2.0] * 20, n_trades=200, n_paths=400,
                         risk_per_trade_pct=1.0, seed=7)
    assert rep.prob_loss > 0.9
    assert rep.median_return_pct < 0


def test_drawdown_is_reported_as_a_positive_percentage() -> None:
    rep = simulate_paths(_streaky(), n_trades=100, n_paths=200, seed=7)
    assert rep.median_max_drawdown_pct >= 0
    assert rep.p95_max_drawdown_pct >= rep.median_max_drawdown_pct


def test_percentiles_are_ordered() -> None:
    rep = simulate_paths(_streaky(), n_trades=150, n_paths=500, seed=7)
    assert rep.p5_return_pct <= rep.median_return_pct <= rep.p95_return_pct


def test_equity_cannot_go_negative() -> None:
    """A pathological R must not drive equity below zero, which would make the
    drawdown arithmetic meaningless rather than merely bad."""
    rep = simulate_paths([-50.0, 2.0, -1.0], n_trades=100, n_paths=200,
                         risk_per_trade_pct=5.0, seed=7)
    assert rep.p95_max_drawdown_pct <= 100.0


# -- guards ------------------------------------------------------------------

def test_too_few_r_multiples_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        simulate_paths([0.5])


def test_nan_r_multiples_are_dropped_not_propagated() -> None:
    rep = simulate_paths([1.0, float("nan"), -1.0, 2.0, None],  # type: ignore
                         n_trades=50, n_paths=100, seed=7)
    assert math.isfinite(rep.median_return_pct)


def test_an_out_of_range_risk_raises() -> None:
    with pytest.raises(ValueError, match="risk_per_trade_pct"):
        simulate_paths(_streaky(), risk_per_trade_pct=0.0)
    with pytest.raises(ValueError, match="risk_per_trade_pct"):
        simulate_paths(_streaky(), risk_per_trade_pct=101.0)


def test_the_block_is_clamped_to_the_sample() -> None:
    """A block longer than the data must not loop forever or index out."""
    rep = simulate_paths([1.0, -1.0, 2.0], n_trades=50, n_paths=50,
                         block=999, seed=7)
    assert math.isfinite(rep.median_return_pct)


# -- time to significance ---------------------------------------------------

def test_trades_to_significance_is_quadratic_in_the_mean() -> None:
    """Halving the edge QUADRUPLES the trades needed. That is the fact that
    makes a thin edge hard to confirm rather than merely slow."""
    a = trades_to_significance(0.08, 0.94)
    b = trades_to_significance(0.04, 0.94)
    assert b == pytest.approx(4 * a, rel=1e-9)


def test_the_real_donchian_figure() -> None:
    """Pinned to the measured sequence so the roadmap's claim is checkable:
    mean +0.082R against sd 0.938R needs ~523 trades for t=2."""
    assert trades_to_significance(0.0820, 0.9383) == pytest.approx(523, abs=3)


def test_a_non_edge_never_becomes_significant() -> None:
    assert math.isinf(trades_to_significance(0.0, 1.0))
    assert math.isinf(trades_to_significance(-0.05, 1.0))
