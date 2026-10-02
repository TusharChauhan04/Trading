"""The walk-forward must refuse to promote, and must refuse for stated reasons.

A promotion gate that passes things is not a gate. The failure that matters here
is a harness that reports what it was built to confirm, so the tests drive it
with constructed data where the right answer is known in both directions: a
hindsight-perfect signal must clear every hurdle, and a random one must not.

Also checked: the windows are genuinely DISJOINT. Rolling windows share trades,
so "positive in 3 of 4" can be one good period counted three times - which is
the exact failure the original harness's docstring warns about.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from desk.backtest.preset_wf import (
    MIN_TRADES_PER_WINDOW, walk_forward_preset,
)


def _panel(n: int = 400, m: int = 80, seed: int = 7):
    """A rising-with-noise panel, indexed by date objects like the bar store."""
    rng = np.random.default_rng(seed)
    idx = [date(2023, 9, 1) + timedelta(days=i) for i in range(n)]
    close = np.empty((n, m))
    for c in range(m):
        close[:, c] = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    cl = pd.DataFrame(close, index=idx)
    span = cl * 0.012
    return {"close": cl, "open": cl, "high": cl + span, "low": cl - span,
            "volume": pd.DataFrame(1e6, index=idx, columns=cl.columns)}


def _perfect_signals(panel, stop_pct=10.0, hold=20):
    """Entries only where the lattice already says the trade wins."""
    from desk.backtest.permutation import outcome_lattice

    r, _ex = outcome_lattice(panel["high"].to_numpy(float),
                             panel["low"].to_numpy(float),
                             panel["close"].to_numpy(float),
                             stop_pct=stop_pct, rr=2.0, hold=hold)
    sig = np.zeros(panel["close"].shape, dtype=bool)
    for c in range(sig.shape[1]):
        wins = np.flatnonzero(r[:, c] == 2.0)
        # EVERY winning bar, not every hold-th one. The harness's own
        # sequential rule thins overlaps via `busy`, so subsampling here
        # only starved the later windows of trades.
        sig[wins, c] = True
    return sig


# -- the gate refuses what it should ----------------------------------------

def test_a_random_signal_is_not_promoted() -> None:
    """THE negative control. Random entries in a RISING panel still make money,
    so the pooled figure may be positive - and the gate must still refuse,
    because the permutation null is also random entries in the same panel."""
    p = _panel()
    rng = np.random.default_rng(11)
    sig = rng.random(p["close"].shape) < 0.05
    wf = walk_forward_preset(p, sig, label="random", stop_pct=10.0,
                             hold_bars=20, cost_r=0.04, windows=4,
                             num_trials=39, n_permutations=60)
    v = wf.verdict()
    assert "STAYS AT DRAFT" in v, v


def test_the_verdict_names_every_hurdle_it_failed() -> None:
    """"Not promoted" without a reason is not a finding."""
    p = _panel()
    rng = np.random.default_rng(3)
    sig = rng.random(p["close"].shape) < 0.05
    wf = walk_forward_preset(p, sig, label="random", stop_pct=10.0,
                             hold_bars=20, cost_r=0.04, windows=4,
                             num_trials=39, n_permutations=60)
    v = wf.verdict()
    assert "windows positive" in v or "pooled net R" in v or "DSR" in v


def test_a_hindsight_perfect_signal_clears_the_hurdles_it_can() -> None:
    """THE positive control. If a signal built from the answer cannot even be
    positive in every window with significant timing, the harness is broken and
    nothing it reports means anything.

    DSR is NOT asserted: four window observations is far below the 60 the
    deflated estimators want, so it is expected to stay underpowered no matter
    how good the signal - which is itself the honest finding.
    """
    p = _panel()
    wf = walk_forward_preset(p, _perfect_signals(p), label="perfect",
                             stop_pct=10.0, hold_bars=20, cost_r=0.04,
                             windows=4, num_trials=1, n_permutations=60)
    assert len(wf.scored) >= 3, wf.report()
    assert wf.windows_positive == len(wf.scored), wf.report()
    assert wf.pooled_net_r is not None and wf.pooled_net_r > 0


# -- disjointness, which is what the whole design buys ----------------------

def test_no_trade_crosses_a_window_boundary() -> None:
    """Rolling windows share trades, so "positive in 3 of 4" can be one good
    period counted three times. Verified by construction: the window trade
    counts must sum to LESS than an un-windowed run, because entries within
    one hold of each boundary are dropped."""
    p = _panel()
    sig = _perfect_signals(p)
    four = walk_forward_preset(p, sig, label="w4", stop_pct=10.0, hold_bars=20,
                               cost_r=0.0, windows=4, num_trials=1,
                               n_permutations=10)
    one = walk_forward_preset(p, sig, label="w2", stop_pct=10.0, hold_bars=20,
                              cost_r=0.0, windows=2, num_trials=1,
                              n_permutations=10)
    assert sum(w.trades for w in four.windows) < sum(w.trades
                                                     for w in one.windows), (
        "more windows did not drop more boundary trades - the windows may be "
        "overlapping")


def test_windows_tile_the_panel_without_gaps_or_overlap() -> None:
    p = _panel()
    wf = walk_forward_preset(p, _perfect_signals(p), label="x", stop_pct=10.0,
                             hold_bars=20, cost_r=0.0, windows=4,
                             num_trials=1, n_permutations=10)
    assert len(wf.windows) == 4
    for earlier, later in zip(wf.windows, wf.windows[1:]):
        assert earlier.end < later.start, "windows overlap"


# -- scoring discipline -----------------------------------------------------

def test_a_thin_window_is_reported_but_not_scored() -> None:
    """A nine-trade window is not evidence either way, and averaging it in
    would let it outvote a nine-hundred-trade one."""
    p = _panel()
    sig = np.zeros(p["close"].shape, dtype=bool)
    sig[50, 0] = True                      # exactly one trade, in window 0
    wf = walk_forward_preset(p, sig, label="thin", stop_pct=10.0, hold_bars=20,
                             cost_r=0.0, windows=4, num_trials=1,
                             n_permutations=10)
    assert any(w.trades > 0 for w in wf.windows)
    assert not any(w.scored for w in wf.windows)
    assert "sample-size result" in wf.verdict()


def test_the_report_marks_unscored_windows() -> None:
    p = _panel()
    sig = np.zeros(p["close"].shape, dtype=bool)
    sig[50, 0] = True
    out = walk_forward_preset(p, sig, label="thin", stop_pct=10.0,
                              hold_bars=20, cost_r=0.0, windows=4,
                              num_trials=1, n_permutations=10).report()
    assert "(unscored)" in out


def test_min_trades_is_a_named_constant() -> None:
    """So the threshold is arguable rather than buried."""
    assert MIN_TRADES_PER_WINDOW >= 10


# -- guards -----------------------------------------------------------------

def test_mismatched_signal_shape_raises() -> None:
    p = _panel(100, 4)
    with pytest.raises(ValueError, match="signals"):
        walk_forward_preset(p, np.zeros((100, 3), bool), label="x",
                            stop_pct=10.0, hold_bars=10)


def test_one_window_is_not_a_walk_forward() -> None:
    p = _panel(100, 4)
    with pytest.raises(ValueError, match="at least 2 windows"):
        walk_forward_preset(p, np.zeros(p["close"].shape, bool), label="x",
                            stop_pct=10.0, hold_bars=10, windows=1)


def test_the_disjointness_cost_is_declared_as_a_caveat() -> None:
    """Dropping boundary trades is a real cost and must not be silent."""
    p = _panel()
    wf = walk_forward_preset(p, _perfect_signals(p), label="x", stop_pct=10.0,
                             hold_bars=20, cost_r=0.0, windows=4,
                             num_trials=1, n_permutations=10)
    assert any("DISJOINT" in c for c in wf.caveats)
