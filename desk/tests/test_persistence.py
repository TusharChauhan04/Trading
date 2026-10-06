"""Volatility persistence, and why significance is not the test.

The options work established that a both-sides spread book is a LONG
VOLATILITY position - R squared 0.742 against the ABSOLUTE market move. So the
open question stopped being "which way" and became "how far", and this module
asks whether "how far" is forecastable at all before anyone models it.

The trap these tests exist for: at n=1,718 the Ljung-Box test calls BOTH
returns and |returns| significant, so a verdict built on p-values says "both
persist" about a series whose magnitude autocorrelation is five times its
return autocorrelation. Effect size is the test; significance is cheap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.research.persistence import (
    MIN_OBSERVATIONS, available, persistence_report,
)

requires_statsmodels = pytest.mark.skipif(
    not available(), reason="statsmodels not installed")


def _garch(n=1500, seed=5, omega=0.05, alpha=0.10, beta=0.85):
    """Volatility clusters; direction does not. The stylised fact, simulated."""
    rng = np.random.default_rng(seed)
    e = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for i in range(n):
        z = rng.normal()
        e[i] = np.sqrt(s2) * z
        s2 = omega + alpha * e[i] ** 2 + beta * s2
    return pd.Series(100 * np.exp(np.cumsum(e / 100)))


def _iid(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))))


@requires_statsmodels
def test_garch_shows_magnitude_persistence_without_direction() -> None:
    """THE positive control. If this fails the diagnostic cannot detect the
    thing it exists to detect."""
    r = persistence_report(_garch())
    assert r.volatility_is_forecastable is True
    # >2.0, not >3.0: a simulated GARCH of this length produces a ratio around
    # 2.6 where the REAL index gives 5.45, because a finite sample's return ACF
    # wanders above zero by chance and inflates the denominator. Pinning the
    # fixture's exact number would be tuning the test to the simulation.
    assert r.persistence_ratio is not None and r.persistence_ratio > 2.0
    assert "persists" in r.verdict.lower()
    assert "more than direction" in r.verdict.lower()


@requires_statsmodels
def test_iid_returns_show_neither() -> None:
    """The negative control - a plain random walk has no clustering."""
    r = persistence_report(_iid())
    assert r.volatility_is_forecastable is False
    assert "NO VOLATILITY PERSISTENCE" in r.verdict


@requires_statsmodels
def test_the_verdict_is_judged_on_effect_size_not_significance() -> None:
    """The bug this replaced. A verdict reading only p-values said "BOTH
    PERSIST" about a series with a 5.4x ratio, which is true and useless."""
    r = persistence_report(_garch(n=3000))
    # Both may well be "significant" at this sample size...
    assert r.volatility_is_forecastable is True
    # ...and the verdict must still separate them by magnitude.
    assert r.persistence_ratio > 2.0
    assert "more than direction" in r.verdict.lower()


@requires_statsmodels
def test_magnitude_acf_decays_and_has_a_half_life() -> None:
    """A volatility forecast is only useful over a horizon shorter than this.
    Measured at 13 days on the real index, against a 28-day option cycle."""
    r = persistence_report(_garch())
    assert r.half_life_days is not None
    assert 1 < r.half_life_days < 60


@requires_statsmodels
def test_a_gap_is_not_padded_into_a_run_of_zero_returns() -> None:
    """pandas pct_change pads by default, and a run of identical zeros IS
    autocorrelation - it would manufacture exactly the persistence being
    tested for."""
    s = _iid(800)
    s.iloc[200:260] = np.nan
    r = persistence_report(s)
    assert r.n_obs < 800
    # Still no clustering - the holes did not create any.
    assert r.volatility_is_forecastable is False


def test_a_short_series_reports_why_rather_than_a_number() -> None:
    r = persistence_report(_iid(MIN_OBSERVATIONS - 30))
    assert r.acf_returns == ()
    assert r.volatility_is_forecastable is None
    assert r.verdict == "UNAVAILABLE"
    assert str(MIN_OBSERVATIONS) in r.why


def test_unavailable_is_distinct_from_no_persistence() -> None:
    """Upstream collapses "could not measure" into an empty result inside the
    same shape as a success. These are different claims."""
    short = persistence_report(_iid(50))
    assert short.volatility_is_forecastable is None      # not False
    assert short.verdict == "UNAVAILABLE"


def test_the_module_does_not_silence_warnings_globally() -> None:
    import warnings

    import desk.research.persistence  # noqa: F401

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("canary", UserWarning)
    assert caught
