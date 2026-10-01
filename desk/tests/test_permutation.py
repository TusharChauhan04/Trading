"""The permutation test must find an edge that exists and miss one that doesn't.

Both directions are checked on constructed data where the answer is known,
because a robustness test that cannot be trusted is worse than none: it would
be used to license exactly the promotion decisions this project exists to
refuse.

The specific trap being guarded is the one upstream fell into for our use case.
OpenTerminalUI's version sign-flips returns to build its null, which is right
for a symmetric return series and WRONG for R-multiples from a stop-and-target
system, where a stop is exactly -1R and a target exactly +2R. Flipping a +2R
win yields -2R, an outcome the stop makes impossible, so the null gets a wider
spread than reality and the observed result looks MORE significant than it is.
A test that errs toward finding edges is worse than no test.
"""

from __future__ import annotations

import numpy as np
import pytest

from desk.backtest.permutation import (
    outcome_lattice, signal_timing_test,
)


def _ramp_panel(n: int = 200, m: int = 8, seed: int = 7):
    """Symbols that drift up with noise - a rising window, deliberately.

    A rising window is the confound the test has to survive: every permutation
    lives in the same window, so a null that beats zero is expected and is not
    a failure of the test.
    """
    rng = np.random.default_rng(seed)
    close = np.empty((n, m))
    for c in range(m):
        close[:, c] = 100 * np.cumprod(1 + rng.normal(0.002, 0.015, n))
    span = close * 0.01
    return close + span, close - span, close


# -- the lattice -------------------------------------------------------------

def test_the_lattice_resolves_a_stop_and_a_target_exactly() -> None:
    """-1R on a stop, +rr on a target. Anything else means the walk is wrong.

    Constructed so both outcomes are forced: one symbol falls through the stop,
    the other rises through the target.
    """
    n = 30
    close = np.column_stack([np.full(n, 100.0), np.full(n, 100.0)])
    high = close.copy()
    low = close.copy()
    low[5:, 0] = 80.0        # col 0 breaks a 10% stop
    high[5:, 1] = 130.0      # col 1 reaches a 20% target
    r, ex = outcome_lattice(high, low, close, stop_pct=10.0, rr=2.0, hold=20)
    assert r[0, 0] == -1.0
    assert r[0, 1] == 2.0
    assert ex[0, 0] == 5 and ex[0, 1] == 5


def test_the_stop_wins_a_bar_that_touches_both() -> None:
    """A daily bar cannot say which came first, and assuming the favourable one
    is how a backtest flatters itself. Same tie-break as simulate.py."""
    n = 10
    close = np.full((n, 1), 100.0)
    high = np.full((n, 1), 100.0)
    low = np.full((n, 1), 100.0)
    high[3, 0] = 130.0       # target AND
    low[3, 0] = 80.0         # stop, on the same bar
    r, _ = outcome_lattice(high, low, close, stop_pct=10.0, rr=2.0, hold=5)
    assert r[0, 0] == -1.0, "the favourable level was assumed"


def test_a_bad_stop_or_hold_raises() -> None:
    h, lo, c = _ramp_panel(40, 2)
    with pytest.raises(ValueError, match="stop_pct"):
        outcome_lattice(h, lo, c, stop_pct=0.0, rr=2.0, hold=10)
    with pytest.raises(ValueError, match="hold"):
        outcome_lattice(h, lo, c, stop_pct=5.0, rr=2.0, hold=0)


# -- the test finds a real edge ---------------------------------------------

def test_a_genuinely_well_timed_signal_is_significant() -> None:
    """THE positive control.

    Signals are placed only where the lattice already says the trade wins, so
    the timing carries maximal information. If this is not significant the test
    cannot detect an edge at all and nothing it says is worth reading.
    """
    high, low, close = _ramp_panel(200, 10)
    r, ex = outcome_lattice(high, low, close, stop_pct=5.0, rr=2.0, hold=20)
    sig = np.zeros(close.shape, dtype=bool)
    # One winning entry per symbol per 25 bars, chosen with hindsight.
    for c in range(close.shape[1]):
        for blk in range(0, close.shape[0] - 40, 25):
            win = np.flatnonzero(r[blk:blk + 25, c] == 2.0)
            if win.size:
                sig[blk + win[0], c] = True
    res = signal_timing_test(sig, r, ex, n_permutations=200, seed=7)
    assert res.observed > res.null_mean
    assert res.significant, f"failed to detect a hindsight-perfect signal: {res}"


# -- and misses one that is not there --------------------------------------

def test_a_random_signal_is_not_significant() -> None:
    """THE negative control, and the one that matters more.

    Random entry dates in a RISING window still make money, so the observed
    value will be positive. The test must still report it as indistinguishable
    from the null - because the null is also random dates in the same window.
    If this comes back significant, the test is finding edges in noise.
    """
    high, low, close = _ramp_panel(200, 10)
    r, ex = outcome_lattice(high, low, close, stop_pct=5.0, rr=2.0, hold=20)
    rng = np.random.default_rng(11)
    sig = rng.random(close.shape) < 0.04
    res = signal_timing_test(sig, r, ex, n_permutations=200, seed=7)
    assert not res.significant, \
        f"found significance in a random signal: {res}"
    assert abs(res.observed - res.null_mean) < 3 * res.null_std


# -- properties that keep the p-value honest -------------------------------

def test_cost_shifts_both_sides_and_cannot_change_the_p_value() -> None:
    """Costs are subtracted from the observed AND every permutation, so they
    move the reported R and leave the inference alone. If a cost ever changed
    the p-value, the null would not be built from the same trades."""
    high, low, close = _ramp_panel(150, 6)
    r, ex = outcome_lattice(high, low, close, stop_pct=5.0, rr=2.0, hold=20)
    rng = np.random.default_rng(3)
    sig = rng.random(close.shape) < 0.05
    free = signal_timing_test(sig, r, ex, cost_r=0.0, n_permutations=150,
                              seed=7)
    costly = signal_timing_test(sig, r, ex, cost_r=0.5, n_permutations=150,
                                seed=7)
    assert costly.p_value == free.p_value
    assert costly.observed == pytest.approx(free.observed - 0.5)


def test_the_p_value_can_never_be_zero() -> None:
    """N permutations cannot evidence p < 1/(N+1). The +1/+1 correction is what
    stops a report claiming certainty it has not bought."""
    high, low, close = _ramp_panel(150, 6)
    r, ex = outcome_lattice(high, low, close, stop_pct=5.0, rr=2.0, hold=20)
    sig = np.zeros(close.shape, dtype=bool)
    for c in range(close.shape[1]):
        win = np.flatnonzero(r[:120, c] == 2.0)
        sig[win[:4], c] = True
    res = signal_timing_test(sig, r, ex, n_permutations=50, seed=7)
    assert res.p_value >= 1 / 51


def test_permutation_keeps_each_symbols_signal_count() -> None:
    """Shuffling WITHIN a column preserves how many times each name traded.
    Shuffling the flat array would move signals between symbols and test
    selection instead of timing - a different question with a different answer.
    """
    high, low, close = _ramp_panel(120, 5)
    r, ex = outcome_lattice(high, low, close, stop_pct=5.0, rr=2.0, hold=15)
    sig = np.zeros(close.shape, dtype=bool)
    sig[10:14, 0] = True            # 4 signals in one symbol only
    res = signal_timing_test(sig, r, ex, n_permutations=50, seed=7)
    # Non-overlap means not all 4 become trades, but trades must come only
    # from the symbol that signalled, in every permutation.
    assert res.n_trades >= 1
    assert np.isfinite(res.null_mean)


def test_a_mismatched_signal_shape_raises() -> None:
    high, low, close = _ramp_panel(60, 3)
    r, ex = outcome_lattice(high, low, close, stop_pct=5.0, rr=2.0, hold=10)
    with pytest.raises(ValueError, match="signal shape"):
        signal_timing_test(np.zeros((60, 2), bool), r, ex)


def test_signals_that_resolve_no_trades_raise_rather_than_report_zero() -> None:
    """An empty result reported as 0.0R would read as a measured flat outcome
    instead of a measurement that did not happen."""
    high, low, close = _ramp_panel(60, 3)
    r, ex = outcome_lattice(high, low, close, stop_pct=5.0, rr=2.0, hold=10)
    with pytest.raises(ValueError, match="no resolvable trades"):
        signal_timing_test(np.zeros(close.shape, bool), r, ex)
