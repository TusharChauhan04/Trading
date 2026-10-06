"""Is volatility forecastable? The diagnostic that must precede trying.

WHY THIS EXISTS AND WHY NOW
---------------------------
Every signal this project has tested forecasts DIRECTION - breakouts, mean
reversion, cointegration, three regime gates. All failed. The options work then
landed on a different question: a book of both call and put spreads is a LONG
VOLATILITY position that pays when the market moves more than about 4% in a
month and bleeds otherwise, and over 12 expiries it netted +0.0441R at t 1.15 -
nothing. But the mechanism was clean: regressing per-expiry return on the
ABSOLUTE market move gave R squared 0.742.

So the structure does not need to know WHICH WAY the market goes. It needs to
know HOW FAR. That is a volatility forecast, and this desk has never attempted
one.

Before attempting it, the honest step is to ask whether volatility is
forecastable AT ALL on this data. This module answers that, and it is a
diagnostic rather than a forecast.

WHAT IT MEASURES, AND WHY THE CONTRAST IS THE WHOLE POINT
---------------------------------------------------------
Two autocorrelation functions on the same series:

  - the ACF of RETURNS. If returns were autocorrelated, direction would be
    forecastable. The stylised fact across every liquid market is that they are
    not, and this desk's own failures are consistent with that.
  - the ACF of ABSOLUTE or SQUARED returns. This is volatility clustering, and
    the same stylised fact says it IS strongly autocorrelated and persistent
    over weeks.

If that contrast holds here - returns uncorrelated, |returns| strongly
correlated - then volatility is forecastable where direction is not, and the
options structure has something to aim at. If it does NOT hold, the avenue
closes before any model is built.

Ljung-Box is reported for both, because a single lag's ACF value is one draw
and the joint test over the first k lags is the one that answers "is there any
structure here".

WHAT UPSTREAM GOT WRONG
-----------------------
`core/statlab/autocorrelation.py` computes the right quantities and then:

  - calls `warnings.filterwarnings("ignore")` AT MODULE IMPORT, reconfiguring
    the host process - the same disqualifier as their regimes and stationarity
    modules.
  - `_safe_float` returns 0.0 on NaN or inf. An ACF of 0.0 means "no
    autocorrelation at this lag", which is a finding; a failed computation
    returning it is that finding fabricated.
  - wraps everything in one `except Exception` that returns empty lists and the
    string "Autocorrelation analysis failed" inside the SAME dict shape as a
    success, so a caller reading `acf` gets `[]` and no error.
  - only ever looks at the series as given. It has no notion of testing
    |returns| separately, which is the one thing that distinguishes a
    forecastable quantity from an unforecastable one here.

So the statsmodels calls are the same and nothing else is.

A NOTE ON WHAT A SIGNIFICANT ACF DOES NOT MEAN. The 1.96/sqrt(n) band is the
null for a SINGLE lag under white noise. Scanning 30 lags and reporting the
ones that cross it will find about 1.5 by chance, and upstream's
`significant` flag does exactly that. The Ljung-Box statistic is the joint
test and is what should be read; the per-lag flags are kept only because the
shape of the decay is informative to look at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MIN_OBSERVATIONS", "PersistenceReport", "available",
           "persistence_report"]

#: Below this the ACF's own standard error swamps the values being measured.
#: 1.96/sqrt(60) is 0.253, so on 60 points nothing short of an enormous
#: autocorrelation is distinguishable from noise. Upstream accepts 40.
MIN_OBSERVATIONS = 120


def available() -> bool:
    try:
        import statsmodels.tsa.stattools  # noqa: F401
    except Exception:                                   # noqa: BLE001
        return False
    return True


@dataclass(frozen=True, slots=True)
class PersistenceReport:
    """Autocorrelation of returns against autocorrelation of their magnitude."""

    n_obs: int
    conf_band: float
    acf_returns: tuple[float, ...]
    acf_abs_returns: tuple[float, ...]
    acf_squared_returns: tuple[float, ...]
    ljung_returns_p: dict[int, float]
    ljung_abs_p: dict[int, float]
    half_life_days: float | None
    why: str = ""

    @property
    def direction_is_forecastable(self) -> bool | None:
        """Does the RETURN series carry autocorrelation? Almost never true.

        None when the test could not run - which is not the same as False, and
        conflating them would turn a missing diagnostic into a finding.
        """
        p = self.ljung_returns_p.get(10)
        return None if p is None else bool(p < 0.05)

    @property
    def volatility_is_forecastable(self) -> bool | None:
        """Does the MAGNITUDE of returns carry autocorrelation?

        This is the question the options work raised. A True here does not say
        a forecast will be profitable - only that there is something to
        forecast, which is the precondition.
        """
        p = self.ljung_abs_p.get(10)
        return None if p is None else bool(p < 0.05)

    @property
    def persistence_ratio(self) -> float | None:
        """How much stronger magnitude persistence is than return persistence.

        Mean |ACF| over lags 1-5 for |returns| divided by the same for returns.
        THE NUMBER TO READ, not the two p-values. On 1,718 observations the
        Ljung-Box test calls BOTH series significant - the return ACF is 0.031
        and that clears a 0.047 band jointly over ten lags - but 0.031 explains
        0.1% of variance while 0.256 explains 6.7% and keeps doing so for
        weeks. Significance is cheap at this sample size; the ratio is not.
        """
        if len(self.acf_returns) < 6 or len(self.acf_abs_returns) < 6:
            return None
        den = float(np.mean(np.abs(self.acf_returns[1:6])))
        if den <= 0:
            return None
        return float(np.mean(np.abs(self.acf_abs_returns[1:6]))) / den

    @property
    def verdict(self) -> str:
        """Judged on EFFECT SIZE, not on which p-values clear 0.05.

        An earlier version compared only significance and reported "BOTH
        PERSIST" on a series whose magnitude autocorrelation was eight times
        its return autocorrelation - literally true and badly misleading. At
        n=1,718 almost anything is significant.
        """
        d, v = self.direction_is_forecastable, self.volatility_is_forecastable
        ratio = self.persistence_ratio
        if d is None or v is None or ratio is None:
            return "UNAVAILABLE"
        if not v:
            return "NO VOLATILITY PERSISTENCE - nothing to forecast"
        if ratio >= 3.0:
            return (f"VOLATILITY PERSISTS {ratio:.1f}x MORE THAN DIRECTION - "
                    f"forecast magnitude, not sign")
        if ratio >= 1.5:
            return f"volatility persists {ratio:.1f}x more than direction"
        return ("direction and magnitude persist comparably - the usual "
                "stylised fact does NOT hold here, check the data")


def _half_life(acf_vals: np.ndarray) -> float | None:
    """Lags until the magnitude ACF decays below half its lag-1 value.

    A crude but honest persistence measure: a volatility forecast is only
    useful over a horizon shorter than this. None when it never decays within
    the lags computed, which means the window is too short to say.
    """
    if len(acf_vals) < 3 or not np.isfinite(acf_vals[1]) or acf_vals[1] <= 0:
        return None
    target = acf_vals[1] / 2.0
    for lag in range(2, len(acf_vals)):
        if acf_vals[lag] < target:
            return float(lag)
    return None


def persistence_report(prices, *, nlags: int = 30) -> PersistenceReport:
    """ACF and Ljung-Box for returns and for their magnitude.

    Raises nothing on a short series - returns a report with `why` set and
    every test None, because "could not measure" and "measured no structure"
    are different states and upstream collapses them.
    """
    empty = {"ljung_returns_p": {}, "ljung_abs_p": {}}
    if not available():
        return PersistenceReport(
            n_obs=0, conf_band=float("nan"), acf_returns=(),
            acf_abs_returns=(), acf_squared_returns=(), half_life_days=None,
            why="statsmodels is not installed; `pip install statsmodels`",
            **empty)

    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    s = pd.Series(prices).dropna()
    # fill_method=None: pandas pads gaps into fabricated zero returns, and a
    # run of zeros is itself autocorrelation - it would manufacture exactly the
    # persistence this module is testing for.
    r = s.pct_change(fill_method=None).dropna()
    n = len(r)
    if n < MIN_OBSERVATIONS:
        return PersistenceReport(
            n_obs=n, conf_band=float("nan"), acf_returns=(),
            acf_abs_returns=(), acf_squared_returns=(), half_life_days=None,
            why=f"{n} returns is below the {MIN_OBSERVATIONS} floor; the ACF's "
                f"own error band would swamp the values", **empty)

    lags = max(5, min(int(nlags), n // 2 - 1))
    rv = r.to_numpy(float)
    a_ret = np.asarray(acf(rv, nlags=lags, fft=True), dtype=float)
    a_abs = np.asarray(acf(np.abs(rv), nlags=lags, fft=True), dtype=float)
    a_sq = np.asarray(acf(rv ** 2, nlags=lags, fft=True), dtype=float)

    want = [x for x in (5, 10, 20) if x <= lags] or [lags]
    lb_ret = acorr_ljungbox(rv, lags=want, return_df=True)
    lb_abs = acorr_ljungbox(np.abs(rv), lags=want, return_df=True)

    return PersistenceReport(
        n_obs=n,
        conf_band=1.96 / np.sqrt(n),
        acf_returns=tuple(float(x) for x in a_ret),
        acf_abs_returns=tuple(float(x) for x in a_abs),
        acf_squared_returns=tuple(float(x) for x in a_sq),
        ljung_returns_p={int(k): float(lb_ret.loc[k, "lb_pvalue"])
                         for k in want},
        ljung_abs_p={int(k): float(lb_abs.loc[k, "lb_pvalue"]) for k in want},
        half_life_days=_half_life(a_abs),
    )
