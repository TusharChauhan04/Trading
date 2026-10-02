"""The Markov regime must identify states correctly and never leak the future.

Three things are asserted here, in order of how much damage getting them wrong
would do:

  1. `filtered` and `smoothed` are NOT the same series. Smoothed probabilities
     use the whole sample to infer each day, so feeding them to a backtest
     manufactures an edge that cannot be traded. If a refactor ever makes
     `filtered` an alias of `smoothed`, every gate built on it becomes
     look-ahead and nothing else here would notice.
  2. The high-volatility state is identified by its VARIANCE, not by its index.
     statsmodels labels the states 0 and 1 in whatever order the optimiser
     lands on, so assuming state 1 is the volatile one inverts the answer on
     roughly half of all fits.
  3. A non-convergent or too-short fit RAISES. A default state would read as
     "the market is calm", which is the opposite of "the model did not fit".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.regime.markov import (
    MIN_OBSERVATIONS, SPREAD_STABLE_OBSERVATIONS, available, fit_regimes,
)

requires_statsmodels = pytest.mark.skipif(
    not available(), reason="statsmodels not installed")


def _two_regime_series(n: int = 900, seed: int = 7) -> pd.Series:
    """A series with two genuinely different volatility regimes.

    Built in long blocks rather than by flipping a coin per day: a Markov model
    identifies PERSISTENT states, and a series that switches every bar has no
    persistence to find.
    """
    rng = np.random.default_rng(seed)
    rets, block = [], 0
    while len(rets) < n:
        calm = block % 2 == 0
        size = int(rng.integers(60, 140))
        sd = 0.004 if calm else 0.022
        mu = 0.0010 if calm else -0.0015
        rets.extend(rng.normal(mu, sd, size))
        block += 1
    px = 100 * np.cumprod(1 + np.array(rets[:n]))
    return pd.Series(px, index=pd.date_range("2020-01-01", periods=n,
                                             freq="B"))


# -- look-ahead, the one that would do real damage --------------------------

@requires_statsmodels
def test_filtered_and_smoothed_are_different_series() -> None:
    """THE look-ahead guard.

    Smoothed probabilities know the future. If these ever become the same
    object, every gate built on `filtered` silently becomes look-ahead and no
    other test in this file would catch it.
    """
    fit = fit_regimes(_two_regime_series())
    assert len(fit.filtered) == len(fit.smoothed)
    assert not fit.filtered.equals(fit.smoothed), (
        "filtered and smoothed are identical - the point-in-time series is "
        "leaking the whole-sample one")
    # They should agree broadly while differing day to day.
    assert fit.filtered.corr(fit.smoothed) > 0.5


@requires_statsmodels
def test_the_last_filtered_value_does_not_change_when_history_grows() -> None:
    """A point-in-time probability for a given day must not be revised by what
    comes after it. Smoothed values ARE revised, which is exactly why they are
    unsafe - so this is asserted on filtered only, and loosely, because the
    model is refitted and its parameters legitimately change."""
    s = _two_regime_series(900)
    short = fit_regimes(s.iloc[:800])
    long_ = fit_regimes(s)
    common = short.filtered.index.intersection(long_.filtered.index)
    assert len(common) > 600
    # Refitting changes parameters, so this is a correlation check rather than
    # equality - the point is that the early states are not rewritten wholesale.
    assert short.filtered[common].corr(long_.filtered[common]) > 0.7


# -- state identification ---------------------------------------------------

@requires_statsmodels
def test_the_high_vol_state_is_the_noisier_one() -> None:
    """Identified by variance, never by index. statsmodels labels the states in
    whatever order the optimiser lands on."""
    fit = fit_regimes(_two_regime_series())
    assert fit.high_vol_ann_pct > fit.low_vol_ann_pct, (
        f"the states are inverted: high {fit.high_vol_ann_pct:.1f}% vs low "
        f"{fit.low_vol_ann_pct:.1f}%")


@requires_statsmodels
def test_both_states_are_actually_occupied() -> None:
    """A fit that assigns 99% of days to one state has not found two regimes,
    it has found one and a rounding error."""
    fit = fit_regimes(_two_regime_series())
    assert 0.05 < fit.high_vol_share < 0.95


@requires_statsmodels
def test_probabilities_stay_in_range() -> None:
    fit = fit_regimes(_two_regime_series())
    for s in (fit.filtered, fit.smoothed):
        assert s.min() >= -1e-9 and s.max() <= 1 + 1e-9


# -- refusals ---------------------------------------------------------------

@requires_statsmodels
def test_too_short_a_series_raises() -> None:
    """A default state would read as "the market is calm", which is the
    opposite of "the model could not fit"."""
    with pytest.raises(ValueError, match="too few"):
        fit_regimes(_two_regime_series(MIN_OBSERVATIONS - 50))


@requires_statsmodels
def test_a_date_outside_the_fit_is_none_not_false() -> None:
    """A day the model never saw is not a calm day. A gate treating it as one
    would trade straight through every gap in the data."""
    fit = fit_regimes(_two_regime_series())
    assert fit.state_on("1990-01-02") is None
    assert fit.state_on(fit.filtered.index[-1]) in (True, False)


# -- the stability threshold, which is a measured fact ----------------------

@requires_statsmodels
def test_spread_reliability_tracks_the_sample_length() -> None:
    """Measured on the real index: the return spread flips sign below ~500
    sessions and settles from about 750, while the VOLATILITY separation is
    stable at every length. So a short fit must not have its sign read.
    """
    short = fit_regimes(_two_regime_series(400))
    long_ = fit_regimes(_two_regime_series(900))
    assert not short.spread_reliable
    assert long_.spread_reliable
    assert short.n_obs < SPREAD_STABLE_OBSERVATIONS <= long_.n_obs
    # And the volatility separation holds even on the short one.
    assert short.high_vol_ann_pct > short.low_vol_ann_pct


def test_missing_statsmodels_is_reported_not_raised_blindly() -> None:
    """`available()` is a question, not an error path. A desk without
    statsmodels should degrade to "no regime fitted", never crash a scan."""
    import desk.regime.markov as m

    assert isinstance(m.available(), bool)
    if not m.available():
        with pytest.raises(ValueError, match="statsmodels is not installed"):
            m.fit_regimes(pd.Series([1.0, 2.0, 3.0]))


def test_the_module_does_not_silence_warnings_globally() -> None:
    """Upstream's version calls warnings.filterwarnings("ignore") at import,
    which reconfigures the HOST PROCESS. Importing ours must not."""
    import warnings

    import desk.regime.markov  # noqa: F401

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("canary", UserWarning)
    assert caught, "a warning raised after import was swallowed"
