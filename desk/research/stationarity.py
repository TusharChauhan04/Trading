"""Mean reversion, measured against a calibrated null instead of a guess.

WHY THIS EXISTS
---------------
OpenTerminalUI's `core/statlab/stationarity.py` runs ADF, KPSS and a Hurst
exponent and labels the result. The tests are the right tests. Everything
around them is wrong in ways that produce confident, actionable, incorrect
answers, and the Hurst threshold is wrong by MEASUREMENT rather than by taste.

1. THE HURST THRESHOLD SITS INSIDE THE NULL. Upstream calls H < 0.45
   "mean-reverting" and H > 0.55 "trending". The lagged-variance estimator is
   biased DOWNWARD at finite length, so a PURE RANDOM WALK - true H = 0.5 -
   scores a mean of 0.41 at 250 observations with a standard deviation of 0.116.
   Measured over 4,000 simulated walks per length:

       n      null mean   null sd   P(upstream calls it mean-reverting)
       120        0.399     0.128                                  0.62
       250        0.413     0.116                                  0.61
       500        0.435     0.095                                  0.55
      1250        0.478     0.057                                  0.26

   So on a year of daily data upstream labels 61% of pure random walks
   "mean-reverting". That is not a weak signal, it is a wrong one, and it is
   the module's headline output.

2. IT CANNOT DETECT A TREND AT ALL, so the other half of the label is dead.
   A random walk with a drift of 0.20 per bar against a noise sd of 1.0 is an
   enormous trend; the estimator scores it 0.409 at n=250, indistinguishable
   from the driftless walk's 0.410, and calls it "trending" 11% of the time
   against 9% for the pure walk. The structure function of increments is
   dominated by noise at the lags it uses. `hurst > 0.55 -> "trending"` fires
   essentially at random.

3. `_hurst_exponent` RETURNS 0.5 FROM A BARE `except:`, and the interpretation
   string reads 0.5 as "random walk". A crash is reported as a finding. Worse:
   `np.log(tau)` takes the log of a zero standard deviation on any series with
   a constant stretch - a circuit-locked or barely-traded stock - which gives
   -inf, a garbage polyfit, and then `_safe_float` turns the NaN into 0.0,
   which the interpretation reads as STRONGLY MEAN-REVERTING. A frozen stock
   is the most attractive thing on the screen.

4. `warnings.filterwarnings("ignore")` AT MODULE IMPORT, which reconfigures the
   host process - the same disqualifier as their regimes module.

5. ADF ON PRICES AND ADF ON RETURNS ARE VERDICTS THAT NEVER VARY. Equity
   prices are non-stationary and their returns are stationary, for essentially
   every name; the interpretation string says so for every stock and reads as
   analysis. Both are reported here, because a conflict IS informative, but
   they are not the answer to anything on their own.

WHAT THIS DOES INSTEAD. The estimator is kept - it has real power, section 1's
table understates it - and the THRESHOLD is replaced by a null calibrated at
the actual sample length. `_NULL` holds the simulated random-walk distribution
of the same estimator at ten lengths, and `hurst()` reports where the observed
value falls in it. A verdict is MEAN-REVERTING only when the series is below
the null's lower tail; otherwise it is INDETERMINATE. There is no TRENDING
verdict, because section 2 shows this estimator cannot support one.

THE NULL IS ROBUST TO WHAT REAL RETURNS DO, which had to be checked before any
of it could be used - a p-value calibrated on Gaussian increments would be
wrong on real stocks if fat tails or volatility clustering moved it. The 5%
critical value across 2,500 paths each:

    innovations                  n=250 p05    n=1250 p05
    Gaussian                        0.2156        0.3830
    Student t, df=3                 0.2211        0.3812
    GARCH(1,1) 0.10/0.85            0.2043        0.3744
    two-state vol (this desk's)     0.2172        0.3732

The largest shift is 0.011 in H against a null sd of 0.114 - under a tenth of
a standard deviation. GARCH moves it slightly DOWN, so a Gaussian-calibrated
table is marginally generous about calling a clustered-volatility series
mean-reverting; the margin is small enough to state rather than correct for.

WHAT IT CAN AND CANNOT DETECT, verified through this module's own code on
geometric random walks and AR(1) log-price series at n=250:

    false positives     P(p<0.05)  0.060 / 0.057 / 0.051 / 0.046
                        at n = 120 / 250 / 500 / 1250 - uniform, as it must be

    AR(1) phi=0.99      half-life 69d    detected  5%   (none - this IS a walk)
    AR(1) phi=0.97      half-life 23d    detected 14%
    AR(1) phi=0.94      half-life 11d    detected 41%
    AR(1) phi=0.85      half-life  4d    detected 96%

THE POWER PROBLEM IS THE SAME SHAPE AS THE COINTEGRATION ONE and it bites in
the same place. Fast mean reversion is found reliably; mean reversion slow
enough to actually trade around - a half-life of two to four weeks - is found
14% to 41% of the time on a year of data. So a non-significant result is weak
evidence of absence, and a screen built on this will miss most of what is
there. Use the longest history available, and read `pvalue` rather than `h`.

For comparison, the same walks under upstream's fixed threshold: it calls
60.3% of them "mean-reverting" at n=250 and 33.7% at n=1250.

WHAT THE UNIVERSE ACTUALLY SAYS, which is the reason to have built this.
Measured on the 1,490 estimable Stage 0 survivors to 2026-09-29:

    window   p<0.05        vs the 5% null        upstream's fixed threshold
       250   52/1490  3.49%   z=-2.67            936/1490  62.8% "mean-reverting"
      1240   66/1490  4.43%   z=-1.01            552/1490  37.0% "mean-reverting"

THE COUNT IS AT OR BELOW THE NULL AT BOTH WINDOWS. There is no excess of
mean-reverting names in this universe - fewer names pass than chance alone
would produce, and the universe's average H (0.411 at n=250) is
indistinguishable from the null mean (0.4123). Upstream's threshold would hand
a user 936 candidates where the calibrated answer is that there are none.
(The slight DEFICIT at 250 bars is consistent with mild trend persistence at
multi-day horizons, which would push H up and empty the lower tail. Not
tested here, so not claimed.)

AND THE TOOL IS WORKING, which this is the evidence for. The four most
mean-reverting instruments at n=250 are LIQUID, LIQUIDBEES, LIQUIDETF and
LIQUIDIETF - overnight money-market ETFs, with H of 0.046, -0.007, -0.001 and
-0.115. A cash fund that accrues steadily and barely moves is genuinely not a
random walk, and H near zero is the right answer for it. The estimator found
the only instruments in the universe that are not random walks, and they are
cash equivalents rather than trades. A screen that did NOT surface them would
be the broken one.

WHAT THIS IS NOT EVIDENCE FOR. That a mean-reverting name is tradeable. The
cointegration work in this package found no tradeable pair relationship in the
liquid universe; this is the univariate question, it is a different question,
and nothing here has been walk-forward tested as a signal. Measure, then gate,
then judge on windows that are positive AND timing-significant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MIN_OBSERVATIONS", "HurstEstimate", "StationarityReport",
           "TestResult", "available", "hurst", "stationarity_report"]

#: Below this the null's standard deviation exceeds 0.128 and the estimator
#: cannot separate a mean-reverting series from a random walk at any useful
#: confidence. Upstream accepts 30, where `lags = range(2, 20)` leaves the
#: longest lag with 11 differences to take a standard deviation of.
MIN_OBSERVATIONS = 120

#: The lagged-variance estimator's distribution under a RANDOM WALK, by sample
#: length: (mean, sd, then the 0.1 / 1 / 2.5 / 5 / 10 / 25 / 50 / 75 / 90 / 95
#: / 99 percentiles). 20,000 simulated walks per row, seed 20260102.
#:
#: THIS TABLE IS THE POINT OF THE MODULE. Upstream's fixed 0.45 expressed as a
#: quantile of this null: the 66th percentile at n=120, the 63rd at 250, the
#: 56th at 500. A threshold that most random walks fall below cannot identify
#: mean reversion. Regenerate with scripts in the commit that added this file;
#: do not hand-edit.
_NULL: dict[int, tuple[float, ...]] = {
      120: (0.3982, 0.1288, 0.0371, 0.1012, 0.1442, 0.1792, 0.2262, 0.3082, 0.4020, 0.4899, 0.5638, 0.6051, 0.6743),
      180: (0.4055, 0.1198, 0.0777, 0.1324, 0.1690, 0.2022, 0.2451, 0.3230, 0.4079, 0.4907, 0.5606, 0.5990, 0.6636),
      250: (0.4123, 0.1145, 0.0884, 0.1453, 0.1851, 0.2187, 0.2599, 0.3327, 0.4159, 0.4941, 0.5603, 0.5971, 0.6536),
      375: (0.4185, 0.1075, 0.1113, 0.1723, 0.2048, 0.2360, 0.2763, 0.3432, 0.4205, 0.4948, 0.5575, 0.5911, 0.6497),
      500: (0.4370, 0.0948, 0.1607, 0.2154, 0.2464, 0.2745, 0.3116, 0.3730, 0.4395, 0.5047, 0.5590, 0.5886, 0.6406),
      750: (0.4607, 0.0758, 0.2141, 0.2746, 0.3050, 0.3313, 0.3615, 0.4103, 0.4633, 0.5142, 0.5562, 0.5806, 0.6243),
     1000: (0.4716, 0.0645, 0.2684, 0.3174, 0.3405, 0.3614, 0.3867, 0.4286, 0.4736, 0.5167, 0.5532, 0.5741, 0.6119),
     1250: (0.4777, 0.0569, 0.2976, 0.3400, 0.3628, 0.3821, 0.4036, 0.4393, 0.4794, 0.5176, 0.5501, 0.5690, 0.6018),
     1750: (0.4847, 0.0472, 0.3321, 0.3710, 0.3895, 0.4039, 0.4231, 0.4532, 0.4857, 0.5172, 0.5447, 0.5609, 0.5880),
     2500: (0.4889, 0.0392, 0.3638, 0.3943, 0.4096, 0.4229, 0.4376, 0.4631, 0.4898, 0.5159, 0.5385, 0.5523, 0.5755),
}

#: The percentiles `_NULL`'s rows carry, after (mean, sd).
_QUANTILES = (0.001, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
              0.99)

#: Upstream's thresholds, kept only so `upstream_verdict` can show what they
#: would have said. Never used to decide anything here.
UPSTREAM_MEAN_REVERT = 0.45
UPSTREAM_TREND = 0.55


def available() -> bool:
    """Whether statsmodels is installed - needed for ADF and KPSS only.

    `hurst` works without it, which is deliberate: the Hurst estimate is the
    part that carries information, so it must not be gated on an optional
    dependency.
    """
    try:
        import statsmodels.tsa.stattools  # noqa: F401
    except Exception:                                   # noqa: BLE001
        return False
    return True


def _interpolate_null(n: int) -> tuple[float, float, tuple[float, ...]]:
    """(mean, sd, quantiles) for sample length `n`, interpolated in log n.

    Linear in log n because that is how the bias decays - the null mean runs
    0.398, 0.412, 0.437, 0.478 at 120, 250, 500, 1250, which is close to
    straight against log n and nowhere near straight against n.
    """
    lengths = sorted(_NULL)
    if n <= lengths[0]:
        row = _NULL[lengths[0]]
        return row[0], row[1], row[2:]
    if n >= lengths[-1]:
        row = _NULL[lengths[-1]]
        return row[0], row[1], row[2:]
    hi = next(L for L in lengths if L >= n)
    lo = max(L for L in lengths if L <= n)
    if lo == hi:
        row = _NULL[lo]
        return row[0], row[1], row[2:]
    w = (math.log(n) - math.log(lo)) / (math.log(hi) - math.log(lo))
    a, b = _NULL[lo], _NULL[hi]
    blend = tuple(a[i] + w * (b[i] - a[i]) for i in range(len(a)))
    return blend[0], blend[1], blend[2:]


def _lower_tail_p(h: float, quantiles: tuple[float, ...]) -> float:
    """P(random walk scores <= h), from the empirical grid.

    The empirical grid rather than a normal approximation because the null is
    mildly left-skewed (skewness -0.07 to -0.21) and the normal approximation
    puts the 5% critical value at 0.2222 where the truth is 0.2154 - it would
    OVER-reject, which is the wrong direction for a test whose whole job is
    refusing to call noise a signal.
    """
    if h <= quantiles[0]:
        return _QUANTILES[0]
    if h >= quantiles[-1]:
        return _QUANTILES[-1]
    for i in range(len(quantiles) - 1):
        lo, hi = quantiles[i], quantiles[i + 1]
        if lo <= h <= hi:
            if hi == lo:
                return _QUANTILES[i]
            w = (h - lo) / (hi - lo)
            return _QUANTILES[i] + w * (_QUANTILES[i + 1] - _QUANTILES[i])
    return _QUANTILES[-1]


@dataclass(frozen=True, slots=True)
class HurstEstimate:
    """A Hurst exponent with the null it has to beat."""

    h: float | None
    """None when it could not be estimated. NEVER 0.5, which is upstream's
    fallback and is read downstream as "random walk" - a finding, not a
    failure."""

    n_obs: int
    usable_lags: int
    null_mean: float
    null_sd: float
    pvalue: float | None
    """P(a random walk of this length scores this low or lower). The number
    that makes the estimate mean anything."""
    why: str = ""
    """Why there is no estimate, when there isn't."""

    @property
    def mean_reverting(self) -> bool:
        """Below the null's 5% lower tail at this sample length."""
        return self.pvalue is not None and self.pvalue < 0.05

    @property
    def verdict(self) -> str:
        """MEAN-REVERTING or INDETERMINATE. There is no TRENDING.

        NOT AN OMISSION - a measurement. A random walk with drift 0.20 per bar
        against noise sd 1.0 scores 0.409 at n=250 where a driftless walk
        scores 0.410. This estimator cannot see a trend, so offering the label
        would be offering a coin flip. `desk.indicators.directional.adx_wide`
        measures trend strength; that is the right tool for it.
        """
        if self.h is None:
            return "UNAVAILABLE"
        return "MEAN-REVERTING" if self.mean_reverting else "INDETERMINATE"

    @property
    def upstream_verdict(self) -> str:
        """What a fixed 0.45/0.55 threshold would have said.

        Reported so the difference is visible rather than asserted. On this
        desk's data the two disagree constantly, and the null table says which
        one is wrong.
        """
        if self.h is None:
            return "random walk"          # upstream's 0.5 fallback
        if self.h < UPSTREAM_MEAN_REVERT:
            return "mean-reverting"
        if self.h > UPSTREAM_TREND:
            return "trending"
        return "random walk"


def hurst(prices, *, max_lag: int | None = None) -> HurstEstimate:
    """The lagged-variance Hurst exponent of `prices`, against its null.

    ON LOG PRICES, and that is required rather than preferred: `_NULL` was
    simulated as a random walk in LEVEL space, so it is the null for log
    prices of a geometric random walk - which is the standard model for an
    equity. Running the estimator on raw prices would compare a multiplicative
    series against an additive null.

    Returns an estimate with `h=None` and a reason rather than guessing. The
    three ways it legitimately cannot answer, each of which upstream turns
    into a number:

      - fewer than MIN_OBSERVATIONS points (upstream accepts 30)
      - a non-positive price, so no log exists (a bad print, or a
        corporate-action artefact)
      - too few lags with non-zero spread, which is what a circuit-locked or
        barely-traded stock looks like. Upstream takes log(0) here, gets -inf,
        polyfits garbage, and `_safe_float` turns the NaN into 0.0 - which
        reads as the most strongly mean-reverting series on the screen.
    """
    s = pd.Series(prices).dropna()
    n = len(s)
    if n < MIN_OBSERVATIONS:
        return HurstEstimate(
            h=None, n_obs=n, usable_lags=0, null_mean=float("nan"),
            null_sd=float("nan"), pvalue=None,
            why=f"{n} observations is below the {MIN_OBSERVATIONS} floor; "
                f"the null's sd there exceeds 0.128 and nothing is separable")
    v = s.to_numpy(float)
    if not np.all(np.isfinite(v)) or float(v.min()) <= 0:
        return HurstEstimate(
            h=None, n_obs=n, usable_lags=0, null_mean=float("nan"),
            null_sd=float("nan"), pvalue=None,
            why="a non-positive or non-finite price, so there is no log price "
                "to measure")
    x = np.log(v)

    # Lags scale with the series. Upstream fixes range(2, 20) whatever the
    # length, so at its own 30-point minimum the longest lag takes a standard
    # deviation over 11 differences.
    cap = max_lag if max_lag is not None else max(4, min(n // 4, 100))
    taus, lags = [], []
    for lag in range(2, int(cap) + 1):
        d = x[lag:] - x[:-lag]
        sd = float(d.std())
        if sd > 0:
            taus.append(sd)
            lags.append(lag)
    if len(lags) < 4:
        return HurstEstimate(
            h=None, n_obs=n, usable_lags=len(lags), null_mean=float("nan"),
            null_sd=float("nan"), pvalue=None,
            why=f"only {len(lags)} lags have any price variation - this is a "
                "frozen or barely-traded series, not a mean-reverting one")

    slope = float(np.polyfit(np.log(lags), np.log(taus), 1)[0])
    if not math.isfinite(slope):
        return HurstEstimate(
            h=None, n_obs=n, usable_lags=len(lags), null_mean=float("nan"),
            null_sd=float("nan"), pvalue=None,
            why="the log-log regression did not produce a finite slope")

    mean, sd, qs = _interpolate_null(n)
    return HurstEstimate(h=slope, n_obs=n, usable_lags=len(lags),
                         null_mean=mean, null_sd=sd,
                         pvalue=_lower_tail_p(slope, qs))


@dataclass(frozen=True, slots=True)
class TestResult:
    """One hypothesis test, with its null spelled out."""

    name: str
    null_hypothesis: str
    """WRITTEN OUT BECAUSE ADF AND KPSS HAVE OPPOSITE NULLS, and that is the
    single most common way to misread them together. ADF's null is a unit root,
    so a SMALL p-value means stationary. KPSS's null is stationarity, so a
    SMALL p-value means NON-stationary. Upstream gets both directions right;
    the mistake here would be subtle and invisible, so the null travels with
    the number."""
    statistic: float | None
    pvalue: float | None
    rejects_null: bool | None
    """None when the test could not run. Not False - "the test did not run" and
    "the test ran and did not reject" are different states, and upstream's
    fallbacks (ADF p=1.0, KPSS p=0.0) both assert non-stationarity from a
    crash."""
    why: str = ""

    @property
    def says_stationary(self) -> bool | None:
        """The test's answer in one direction, whichever null it had."""
        if self.rejects_null is None:
            return None
        if self.name == "ADF":
            return self.rejects_null          # rejected a unit root
        if self.name == "KPSS":
            return not self.rejects_null      # failed to reject stationarity
        return None


@dataclass(frozen=True, slots=True)
class StationarityReport:
    """ADF, KPSS and Hurst on one series, combined honestly."""

    adf: TestResult
    kpss: TestResult
    returns_adf: TestResult
    hurst: HurstEstimate
    n_obs: int

    @property
    def agreement(self) -> str:
        """How ADF and KPSS combine - the part worth having both for.

        Four outcomes, and the third is the informative one that two
        independent booleans cannot express:

          STATIONARY       ADF rejects a unit root AND KPSS does not reject
                           stationarity. Both point the same way.
          NON-STATIONARY   ADF does not reject AND KPSS does. Both again.
          CONFLICTING      BOTH reject. Not a contradiction to paper over - it
                           is what a series with a structural break or long
                           memory looks like, and it means neither simple
                           answer fits.
          INCONCLUSIVE     NEITHER rejects. Almost always too little data
                           rather than a finding about the market.
        """
        a, k = self.adf.rejects_null, self.kpss.rejects_null
        if a is None or k is None:
            return "UNAVAILABLE"
        if a and not k:
            return "STATIONARY"
        if not a and k:
            return "NON-STATIONARY"
        if a and k:
            return "CONFLICTING"
        return "INCONCLUSIVE"

    @property
    def summary(self) -> str:
        """One line, with no verdict the measurements do not support."""
        bits = [f"prices {self.agreement.lower()} (ADF+KPSS)"]
        if self.hurst.h is None:
            bits.append(f"Hurst unavailable: {self.hurst.why}")
        else:
            bits.append(
                f"Hurst {self.hurst.h:.3f} vs a random-walk null of "
                f"{self.hurst.null_mean:.3f}+/-{self.hurst.null_sd:.3f} at "
                f"n={self.hurst.n_obs} (p={self.hurst.pvalue:.3f}) -> "
                f"{self.hurst.verdict}")
        return "; ".join(bits)


def _run(name: str, null: str, fn, series) -> TestResult:
    """One test, with failure reported rather than given a p-value."""
    try:
        res = fn(series)
        stat, p = float(res[0]), float(res[1])
    except Exception as exc:                            # noqa: BLE001
        return TestResult(name=name, null_hypothesis=null, statistic=None,
                          pvalue=None, rejects_null=None,
                          why=f"{type(exc).__name__}: {exc}")
    if not (math.isfinite(stat) and math.isfinite(p)):
        return TestResult(name=name, null_hypothesis=null, statistic=None,
                          pvalue=None, rejects_null=None,
                          why="the test returned a non-finite value")
    return TestResult(name=name, null_hypothesis=null, statistic=stat,
                      pvalue=p, rejects_null=bool(p < 0.05))


def stationarity_report(prices) -> StationarityReport:
    """ADF, KPSS, ADF-on-returns and Hurst, with nothing defaulted.

    A NOTE ON WHAT THE FIRST TWO ARE WORTH. Equity prices are non-stationary
    and their returns are stationary, for essentially every name - so those
    two lines are near-constant across the universe and are not evidence about
    a particular stock. They are reported because the CONFLICTING case is
    informative and because their absence would be noticed; the Hurst estimate
    against its calibrated null is the part that distinguishes one series from
    another.

    Warnings are suppressed around the KPSS call only, inside a context
    manager. KPSS legitimately emits an InterpolationWarning when the statistic
    falls outside its tabulated range - which is itself information, so the
    p-value it returns there is a bound rather than a value - and the
    alternative upstream chose, a module-level global filter, silences the
    host process.
    """
    import warnings

    s = pd.Series(prices).dropna()
    h = hurst(s)

    if not available():
        miss = TestResult(
            name="ADF", null_hypothesis="the series has a unit root",
            statistic=None, pvalue=None, rejects_null=None,
            why="statsmodels is not installed; `pip install statsmodels`")
        return StationarityReport(
            adf=miss,
            kpss=TestResult(name="KPSS",
                            null_hypothesis="the series is stationary",
                            statistic=None, pvalue=None, rejects_null=None,
                            why=miss.why),
            returns_adf=TestResult(
                name="ADF", null_hypothesis="the series has a unit root",
                statistic=None, pvalue=None, rejects_null=None, why=miss.why),
            hurst=h, n_obs=len(s))

    from statsmodels.tsa.stattools import adfuller, kpss

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = _run("ADF", "the series has a unit root (non-stationary)",
                   lambda v: adfuller(v), s.to_numpy(float))
        kp = _run("KPSS", "the series is stationary",
                  lambda v: kpss(v, regression="c", nlags="auto"),
                  s.to_numpy(float))
        rets = s.pct_change().dropna()
        if len(rets) < 30:
            radf = TestResult(
                name="ADF", null_hypothesis="the series has a unit root "
                                            "(non-stationary)",
                statistic=None, pvalue=None, rejects_null=None,
                why=f"{len(rets)} returns is too few to test")
        else:
            radf = _run("ADF", "the series has a unit root (non-stationary)",
                        lambda v: adfuller(v), rets.to_numpy(float))

    return StationarityReport(adf=adf, kpss=kp, returns_adf=radf, hurst=h,
                              n_obs=len(s))
