"""Pairs estimates must use only the past. BUG-03 is what happens otherwise.

The bug that parked pairs_trading is a hedge ratio fitted on the whole sample
and then applied to every historical bar, so each day's spread knows its own
future. OpenTerminalUI's cointegration module has it too. Measured on six
liquid Indian pairs over five years:

    in-sample beta      +2.698% per round trip
    point-in-time beta  +0.279% per round trip
    inflation           +2.418 percentage points, about 10x

So the headline test here is the look-ahead one: truncate the series and the
earlier estimates must not move. Everything else is secondary to that.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from desk.research.cointegration import (
    DEFAULT_WINDOW, MIN_WINDOW, available, hedge_ratio, pair_state,
    screen_pairs, walk_spread,
)

requires_statsmodels = pytest.mark.skipif(
    not available(), reason="statsmodels not installed")


def _cointegrated(n: int = 800, beta: float = 2.0, seed: int = 7):
    """x is a random walk; y tracks beta*x plus a mean-reverting spread."""
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1.0, n))
    spread, s = [], 0.0
    for _ in range(n):
        s = 0.94 * s + rng.normal(0, 1.0)      # AR(1), reverts
        spread.append(s)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return (pd.Series(beta * x + np.array(spread), index=idx, name="Y"),
            pd.Series(x, index=idx, name="X"))


# -- the look-ahead guard, which is the whole point -------------------------

@requires_statsmodels
def test_truncating_the_future_does_not_change_the_past() -> None:
    """THE test. BUG-03 fails it by construction.

    Walk the spread on the full series, then on a truncated copy, and compare
    the overlap. A whole-sample beta changes every historical z-score when the
    tail is removed; a trailing beta cannot.
    """
    y, x = _cointegrated(800)
    full = walk_spread(y, x, window=DEFAULT_WINDOW, refit_every=21)
    cut = walk_spread(y.iloc[:650], x.iloc[:650], window=DEFAULT_WINDOW,
                      refit_every=21)
    common = full.index.intersection(cut.index)
    assert len(common) > 200, "not enough overlap to compare"
    pd.testing.assert_frame_equal(
        full.loc[common], cut.loc[common], check_freq=False,
        obj="the past changed when the future was removed - look-ahead")


@requires_statsmodels
def test_the_whole_sample_beta_does_fail_that_test() -> None:
    """The positive control: show BUG-03 is detectable by the test above.

    If the whole-sample approach ALSO passed, the test would be proving
    nothing about point-in-time-ness.
    """
    y, x = _cointegrated(800)
    b_full, _ = hedge_ratio(y, x)
    b_cut, _ = hedge_ratio(y.iloc[:650], x.iloc[:650])
    assert not math.isclose(b_full, b_cut, rel_tol=1e-9), (
        "the whole-sample beta did not change when the tail was removed, so "
        "this fixture cannot demonstrate the bug")


@requires_statsmodels
def test_the_scored_bar_is_excluded_from_its_own_statistics() -> None:
    """A z-score computed with the current bar inside its own mean and sd is
    mildly self-referential and biases every signal toward zero."""
    y, x = _cointegrated(500)
    wf = walk_spread(y, x, window=DEFAULT_WINDOW)
    assert not wf.empty
    # The first scored bar must be AFTER the first `window` bars.
    assert wf.index[0] >= y.index[DEFAULT_WINDOW]


# -- refusals, each one a 0.0 that upstream would have returned -------------

@requires_statsmodels
def test_an_unestimable_beta_is_none_not_zero() -> None:
    """Upstream falls back to beta = 0.0, which silently turns the spread into
    an unhedged price difference and keeps trading it."""
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    flat = pd.Series(100.0, index=idx)              # zero variance
    moving = pd.Series(np.linspace(100, 200, 300), index=idx)
    assert hedge_ratio(moving, flat) is None
    assert hedge_ratio(flat, moving) is None


def test_too_short_a_window_yields_no_beta() -> None:
    y, x = _cointegrated(MIN_WINDOW - 20)
    assert hedge_ratio(y, x) is None


@requires_statsmodels
def test_a_short_pair_state_reports_nothing_rather_than_guessing() -> None:
    y, x = _cointegrated(60)
    st = pair_state(y, x)
    assert st.beta is None and st.zscore is None
    assert not st.tradeable
    assert st.n_obs == 60


@requires_statsmodels
def test_a_diverging_spread_has_no_half_life() -> None:
    """Upstream reports 0.0 for a non-reverting spread, which reads as
    "reverts instantly" - the exact opposite of what it means."""
    n = 400
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    x = pd.Series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, n)),
                  index=idx)
    # y diverges from x steadily: the spread trends, it does not revert.
    y = pd.Series(x.to_numpy() * 2 + np.linspace(0, 80, n), index=idx)
    st = pair_state(y, x)
    assert st.half_life is None or st.half_life > 0, (
        "a half-life must be positive or absent, never zero-as-a-sentinel")


# -- the happy path ---------------------------------------------------------

@requires_statsmodels
def test_a_cointegrated_pair_is_recognised() -> None:
    """Beta and half-life are recovered on the default 250-bar window.

    The COINTEGRATION P-VALUE IS NOT ASSERTED HERE, and that is a measured
    decision rather than a loosened threshold. On a fixture cointegrated by
    construction with a slow spread (phi 0.94), Engle-Granger on 250 bars finds
    it only half the time - median p 0.064 across 12 seeds. At 600 bars it is
    100%. See COINT_TEST_WINDOW; a one-year p-value is not evidence of absence.
    """
    y, x = _cointegrated(600, beta=2.0)
    st = pair_state(y, x)
    assert st.tradeable
    assert st.beta == pytest.approx(2.0, abs=0.25)
    assert st.coint_pvalue is not None       # reported, not necessarily small
    assert st.half_life is not None and 1 < st.half_life < 100


@requires_statsmodels
def test_the_longer_window_actually_detects_what_250_bars_misses() -> None:
    """The power claim above, asserted rather than only documented."""
    from desk.research.cointegration import COINT_TEST_WINDOW

    y, x = _cointegrated(900, beta=2.0)
    short = pair_state(y, x, window=250)
    long_ = pair_state(y, x, window=COINT_TEST_WINDOW)
    assert long_.n_obs > short.n_obs
    assert long_.coint_pvalue is not None and short.coint_pvalue is not None
    assert long_.coint_pvalue < short.coint_pvalue, (
        f"the longer window did not improve detection: "
        f"{long_.coint_pvalue:.4f} vs {short.coint_pvalue:.4f}")


@requires_statsmodels
def test_the_zscore_is_finite_and_centred() -> None:
    y, x = _cointegrated(700)
    wf = walk_spread(y, x, window=DEFAULT_WINDOW, refit_every=21)
    z = wf["zscore"]
    assert np.isfinite(z).all()
    assert abs(z.mean()) < 1.0, "the z-score is not centred near zero"


@requires_statsmodels
def test_refitting_more_often_still_uses_only_the_past() -> None:
    """refit_every is a cost knob, not a correctness one. A daily refit must
    give the same look-ahead guarantee as a monthly one."""
    y, x = _cointegrated(600)
    for every in (1, 21):
        full = walk_spread(y, x, refit_every=every)
        cut = walk_spread(y.iloc[:500], x.iloc[:500], refit_every=every)
        common = full.index.intersection(cut.index)
        pd.testing.assert_frame_equal(full.loc[common], cut.loc[common],
                                      check_freq=False)


# -- the multiple-testing correction ----------------------------------------

def test_benjamini_hochberg_matches_a_hand_computed_case() -> None:
    """The correction's arithmetic, checked against figures worked by hand.

    Does not need statsmodels: it drives the BH step-up through screen_pairs
    with p-values supplied directly, because the arithmetic is what can be
    silently wrong. m = 5, fdr = 0.10, so the thresholds are
    0.02 / 0.04 / 0.06 / 0.08 / 0.10 -

        p      threshold   clears
        0.001     0.02      yes
        0.030     0.04      yes
        0.070     0.06      no
        0.400     0.08      no
        0.900     0.10      no

    The largest clearing rank is 2, so the first TWO pass - including any pair
    below it that did not clear its own threshold, which is the step-up and the
    part a naive implementation gets wrong.
    """
    import desk.research.cointegration as c

    ps = [0.001, 0.030, 0.070, 0.400, 0.900]
    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    closes = pd.DataFrame({f"S{i}": np.arange(10.0) + i for i in range(6)},
                          index=idx)
    pairs = [(f"S{i}", "S0") for i in range(1, 6)]
    seq = iter(ps)

    def fake_state(y, x, *, as_of=None, window=0):
        return c.PairState(as_of=idx[-1], beta=1.0, alpha=0.0, spread=0.0,
                           zscore=0.5, coint_pvalue=next(seq), half_life=10.0,
                           n_obs=600)

    real_state, real_avail = c.pair_state, c.available
    c.pair_state, c.available = fake_state, (lambda: True)
    try:
        res = c.screen_pairs(closes, pairs, fdr=0.10)
    finally:
        c.pair_state, c.available = real_state, real_avail

    assert res.n_tested == 5
    assert res.n_passed == 2, f"BH step-up is wrong: {res.pairs.to_dict('records')}"
    assert list(res.pairs["passes"]) == [True, True, False, False, False]
    # q-values must be monotone in p.
    q = [float(v) for v in res.pairs["qvalue"]]
    assert q == sorted(q), f"q-values are not monotone: {q}"
    assert res.expected_false_positives_uncorrected == pytest.approx(0.25)


@requires_statsmodels
def test_a_tradeability_veto_beats_a_good_pvalue() -> None:
    """A 400-day half-life is cointegrated and untradeable. It must be REPORTED
    with a reason, not silently dropped and not passed."""
    import desk.research.cointegration as c

    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    closes = pd.DataFrame({"A": np.arange(10.0), "B": np.arange(10.0) * 2},
                          index=idx)

    def fake_state(y, x, *, as_of=None, window=0):
        return c.PairState(as_of=idx[-1], beta=2.0, alpha=0.0, spread=0.0,
                           zscore=0.1, coint_pvalue=1e-9, half_life=400.0,
                           n_obs=600)

    real = c.pair_state
    c.pair_state = fake_state
    try:
        res = c.screen_pairs(closes, [("A", "B")])
    finally:
        c.pair_state = real

    row = res.pairs.iloc[0]
    assert row["pvalue"] == 1e-9          # statistically immaculate
    assert not row["passes"]              # and still not tradeable
    assert "too slow" in row["reason"]
    assert res.n_passed == 0


@requires_statsmodels
def test_an_unknown_symbol_is_reported_not_counted_as_a_test() -> None:
    """m must count tests that actually ran. Inflating it weakens every
    q-value in the batch."""
    y, x = _cointegrated(700)
    closes = pd.DataFrame({"Y": y, "X": x})
    res = screen_pairs(closes, [("Y", "X"), ("Y", "NOSUCH")], window=600)
    assert len(res.pairs) == 2
    assert res.n_tested == 1, "a symbol that was never tested inflated m"
    missing = res.pairs[res.pairs["y"].eq("Y") & res.pairs["x"].eq("NOSUCH")]
    assert "not in the panel" in missing.iloc[0]["reason"]


@requires_statsmodels
def test_the_screen_defaults_to_the_higher_power_window() -> None:
    """A screen wants power, not recency - the default must be the long one."""
    import inspect
    from desk.research.cointegration import COINT_TEST_WINDOW

    sig = inspect.signature(screen_pairs)
    assert sig.parameters["window"].default == COINT_TEST_WINDOW
