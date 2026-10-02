"""Mean reversion must be measured against a null, not a guessed threshold.

The defect this file exists to prevent: upstream's `core/statlab/stationarity`
calls H < 0.45 mean-reverting, and the lagged-variance estimator is biased
downward enough that 60% of PURE RANDOM WALKS score below 0.45 at n=250. The
headline test here is therefore the calibration one - feed the estimator
random walks and the false-positive rate must come out at the advertised 5%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.research.stationarity import (
    MIN_OBSERVATIONS, available, hurst, stationarity_report,
)

requires_statsmodels = pytest.mark.skipif(
    not available(), reason="statsmodels not installed")


def _walk(n: int, seed: int = 3, sigma: float = 0.015) -> pd.Series:
    """A GEOMETRIC random walk - real prices whose log is a walk, true H 0.5."""
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, sigma, n))))


def _ar1(n: int, phi: float, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    s, out = 0.0, []
    for _ in range(n):
        s = phi * s + rng.normal(0, 0.015)
        out.append(s)
    return pd.Series(100 * np.exp(np.array(out)))


# -- the calibration, which is the whole point ------------------------------

def test_a_random_walk_is_rarely_called_mean_reverting() -> None:
    """THE test. Upstream gets 60% here; the advertised rate is 5%.

    If a refactor breaks the null table, the log-price transform, or the
    quantile interpolation, this is what notices - nothing else would.
    """
    rng = np.random.default_rng(99)
    flagged = 0
    trials = 400
    for _ in range(trials):
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 250))))
        if hurst(px).mean_reverting:
            flagged += 1
    rate = flagged / trials
    # 400 trials puts the standard error at ~1.1 points, so this band is
    # roughly +/-4 sigma and a real miscalibration cannot hide in it.
    assert 0.015 < rate < 0.11, (
        f"false-positive rate {rate:.3f} is not the advertised 0.05 - the "
        f"null calibration is broken")


def test_the_pvalue_is_uniform_under_the_null() -> None:
    """A stronger statement than the 5% rate: the whole distribution must be
    right, or thresholds other than 0.05 would be wrong."""
    rng = np.random.default_rng(1234)
    ps = []
    for _ in range(400):
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 500))))
        e = hurst(px)
        if e.pvalue is not None:
            ps.append(e.pvalue)
    arr = np.array(ps)
    for level in (0.10, 0.25, 0.50):
        got = float((arr < level).mean())
        assert abs(got - level) < 0.09, (
            f"P(p < {level}) came out {got:.3f}; the null is not uniform")


def test_fast_mean_reversion_is_detected() -> None:
    """The estimator must have real power, or a correct null is useless."""
    hits = sum(1 for seed in range(40)
               if hurst(_ar1(250, 0.85, seed=seed)).mean_reverting)
    assert hits >= 30, (
        f"only {hits}/40 fast mean-reverting series were detected; measured "
        f"power at phi=0.85 is 96%")


def test_the_null_mean_is_below_one_half_and_rises_with_length() -> None:
    """The finite-sample bias that makes a fixed 0.45 threshold wrong. If the
    table were ever replaced by a flat 0.5, this fails."""
    from desk.research.stationarity import _interpolate_null

    means = [_interpolate_null(n)[0] for n in (120, 250, 500, 1250, 2500)]
    assert all(m < 0.5 for m in means), "the null is not biased downward"
    assert means == sorted(means), "the bias does not shrink with length"
    assert means[0] < 0.42 and means[-1] > 0.47


def test_upstream_threshold_is_inside_the_null() -> None:
    """The measured reason this module exists, asserted so it cannot be
    quietly undone: 0.45 is above the null MEDIAN at realistic lengths, so
    most random walks fall below it."""
    from desk.research.stationarity import (
        UPSTREAM_MEAN_REVERT, _interpolate_null,
    )

    for n in (120, 250, 500):
        _, _, qs = _interpolate_null(n)
        median = qs[6]
        assert UPSTREAM_MEAN_REVERT > median, (
            f"at n={n} upstream's {UPSTREAM_MEAN_REVERT} is below the null "
            f"median {median:.3f}, which would contradict the docstring")


# -- refusals, each one a number upstream would have invented ---------------

def test_a_frozen_series_is_refused_not_called_mean_reverting() -> None:
    """THE WORST UPSTREAM BUG. A constant price gives tau = 0, log(0) = -inf,
    a garbage polyfit, then _safe_float turns NaN into 0.0 - which reads as the
    most strongly mean-reverting series on the screen. A circuit-locked stock
    would sort to the top of a mean-reversion scan.
    """
    e = hurst(pd.Series([250.0] * 300))
    assert e.h is None
    assert not e.mean_reverting
    assert e.verdict == "UNAVAILABLE"
    assert "frozen" in e.why or "variation" in e.why


def test_a_short_series_is_refused() -> None:
    e = hurst(_walk(MIN_OBSERVATIONS - 20))
    assert e.h is None and e.pvalue is None
    assert str(MIN_OBSERVATIONS) in e.why


def test_a_non_positive_price_is_refused_not_logged() -> None:
    s = _walk(200)
    s.iloc[50] = -1.0
    e = hurst(s)
    assert e.h is None
    assert "non-positive" in e.why


def test_an_unavailable_estimate_is_never_one_half() -> None:
    """Upstream returns 0.5 on failure and its own interpretation reads 0.5 as
    "random walk", so a crash becomes a finding."""
    for bad in (pd.Series([100.0] * 300), _walk(30),
                pd.Series([], dtype=float)):
        e = hurst(bad)
        assert e.h is None, f"got {e.h} instead of a refusal"


def test_there_is_no_trending_verdict() -> None:
    """Measured: a walk with drift 0.20/bar against sd 1.0 scores the same as
    a driftless one, so this estimator cannot support a trend label."""
    rng = np.random.default_rng(7)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.004, 0.015, 600))))
    e = hurst(px)
    assert e.verdict in ("MEAN-REVERTING", "INDETERMINATE", "UNAVAILABLE")
    assert "TREND" not in e.verdict


# -- ADF and KPSS, whose nulls are opposite ---------------------------------

@requires_statsmodels
def test_adf_and_kpss_nulls_are_recorded_and_opposite() -> None:
    """The commonest way to misread these two together. The direction is
    invisible in the number, so it travels with it."""
    rep = stationarity_report(_walk(400))
    assert "unit root" in rep.adf.null_hypothesis
    assert "stationary" in rep.kpss.null_hypothesis
    assert "unit root" not in rep.kpss.null_hypothesis


@requires_statsmodels
def test_a_random_walk_reads_as_non_stationary_in_price() -> None:
    rep = stationarity_report(_walk(500))
    assert rep.adf.says_stationary is False
    assert rep.agreement in ("NON-STATIONARY", "INCONCLUSIVE")


@requires_statsmodels
def test_returns_of_a_walk_are_stationary() -> None:
    rep = stationarity_report(_walk(500))
    assert rep.returns_adf.says_stationary is True


@requires_statsmodels
def test_a_failed_test_is_none_not_a_pvalue() -> None:
    """Upstream falls back to ADF p=1.0 and KPSS p=0.0, both of which assert
    non-stationarity from a crash."""
    rep = stationarity_report(pd.Series([100.0] * 300))
    for t in (rep.adf, rep.kpss):
        if t.rejects_null is None:
            assert t.pvalue is None and t.statistic is None
            assert t.why
            assert t.says_stationary is None


@requires_statsmodels
def test_the_summary_states_the_null_it_was_judged_against() -> None:
    rep = stationarity_report(_walk(600))
    s = rep.summary
    assert "random-walk null" in s or "Hurst unavailable" in s


def test_the_module_does_not_silence_warnings_globally() -> None:
    """Upstream calls warnings.filterwarnings("ignore") at import, which
    reconfigures the host process."""
    import warnings

    import desk.research.stationarity  # noqa: F401

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("canary", UserWarning)
    assert caught, "a warning raised after import was swallowed"
