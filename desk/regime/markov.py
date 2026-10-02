"""A two-state Markov volatility regime, fitted on the cross-section.

WHAT THIS IS AND WHY IT IS NOT AN IMPORT
----------------------------------------
OpenTerminalUI's `core/statlab/regimes.py` fits a Markov switching regression
with switching variance and labels the states HIGH-VOL and LOW-VOL. The
modelling is sound and it converges on this desk's data. Three things made it
unusable as-is, and the first two are not style objections:

1. IT CAPS THE OUTPUT AT 300 POINTS. `series` is built from
   `high_vol_prob.tail(300)`, which is a chart's worth of history. A gate needs
   the state on every bar of the backtest - on our 1,240-session panel their
   function answers for the last 300 days and six of eight walk-forward windows
   come back empty. The model computes the full series; the function discards it.

2. IT CALLS `warnings.filterwarnings("ignore")` AT MODULE IMPORT. Importing it
   anywhere in the desk would silence warnings PROCESS-WIDE - every pandas
   FutureWarning, every numpy RuntimeWarning, in every other module. A library
   that reconfigures the host process on import cannot go in the funnel.

3. `_safe_float` RETURNS 0.0 ON NaN. A high-vol probability of 0.0 means
   "certainly calm", and that is also what a failed computation returns. Same
   missing-becomes-meaningful trap as their fundamental scores, and here it
   would read a broken fit as a guarantee of calm.

So statsmodels is called directly and the full series is returned.

WHAT THE MODEL ACTUALLY SAYS ON THIS DESK'S DATA, measured on the
equal-weighted close of the Stage 0 universe, 2021-09 to 2026-09:

    HIGH-VOL   annualised vol 21.3%   mean -0.2068%/day   37.3% of sessions
    LOW-VOL    annualised vol  8.3%   mean +0.1390%/day   62.7% of sessions

High volatility IS the falling market here - the leverage effect, and a large
one: the two states differ by a third of a percent per day. That makes the state
worth knowing regardless of whether it gates any particular rule.

WHAT IT IS NOT EVIDENCE FOR. A previous attempt to gate the surviving breakout
cell on a trend proxy failed because a window-level association did not transfer
to the day level. This is a different measure but the same risk applies, and
H8's window-average volatility did NOT separate the working walk-forward windows
from the failing ones (12.8% against 15.2%, overlapping). So this module exists
to make the per-day state available for testing, not because the test has
passed. Fit it, gate with it, and judge on the count of windows that are both
positive and timing-significant - never on the pooled mean, which rises whenever
losing periods are deleted.

LOOK-AHEAD, STATED PLAINLY. `smoothed` probabilities use the WHOLE sample to
infer each day's state, so they are not available in real time and must never
feed a backtest signal. `filtered` probabilities use only data up to each day
and are the point-in-time answer. Both are returned, `filtered` is the default
for gating, and `MarkovRegimeFit.smoothed` carries a docstring saying so.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MIN_OBSERVATIONS", "SPREAD_STABLE_OBSERVATIONS",
           "MarkovRegimeFit", "available", "fit_regimes"]

#: Below this the model does not have enough data to identify two states, and
#: upstream raises at the same figure. A short series produces a fit that
#: converges on noise.
MIN_OBSERVATIONS = 250

#: Below this the VOLATILITY states are still identified correctly but their
#: DIRECTIONAL meaning is not stable. Measured by fitting the same model to
#: progressively longer windows of the same index:
#:
#:     sessions   hi ann%  lo ann%   hi mean   lo mean    spread
#:          300      26.0     10.2   +0.0239   -0.0564   -0.0803
#:          500      21.5      9.7   -0.0692   -0.0145   +0.0546
#:          750      24.1      8.9   -0.2319   +0.1140   +0.3460
#:         1000      21.7      7.9   -0.1924   +0.1207   +0.3131
#:         1241      21.3      8.3   -0.2068   +0.1390   +0.3457
#:
#: The volatility separation is stable at every length - high-vol is always the
#: noisier state, which is what the model identifies. The return spread FLIPS
#: SIGN below 500 sessions and settles only from about 750. So a fit on a short
#: window may report that volatile days pay BETTER, which is an artefact of the
#: window rather than a fact about the market.
#:
#: `MarkovRegimeFit.spread_reliable` is False below this, and a caller must not
#: read the spread's sign when it is.
SPREAD_STABLE_OBSERVATIONS = 750


def available() -> bool:
    """Whether statsmodels is installed. False is a real answer, not an error."""
    try:
        import statsmodels.tsa.regime_switching.markov_regression  # noqa: F401
    except Exception:                                   # noqa: BLE001
        return False
    return True


@dataclass(frozen=True, slots=True)
class MarkovRegimeFit:
    """A fitted two-state volatility regime over one series."""

    filtered: pd.Series
    """P(high-vol) on each day using ONLY data up to that day.

    THE ONE SAFE FOR A BACKTEST. Use this for any gate, any signal, anything
    whose output feeds a trade.
    """

    smoothed: pd.Series
    """P(high-vol) on each day using the WHOLE sample.

    LOOK-AHEAD BY CONSTRUCTION. A smoothed probability for January knows what
    happened in December of the same year. It is the right thing for describing
    history on a chart and the wrong thing for deciding a trade, and feeding it
    to a backtest would manufacture an edge that cannot be traded.
    """

    high_vol_ann_pct: float
    low_vol_ann_pct: float
    high_vol_mean_pct: float
    low_vol_mean_pct: float
    high_vol_share: float
    n_obs: int

    @property
    def spread_pct(self) -> float:
        """Daily mean return difference between the two states.

        The number that says whether the regime matters at all. Measured at
        about 0.35 percentage points a day on this desk's full universe, which
        is large; a spread near zero would mean the states differ in volatility
        without differing in outcome.

        CHECK `spread_reliable` BEFORE READING THE SIGN. On fewer than about
        750 sessions this number flips - see SPREAD_STABLE_OBSERVATIONS for the
        measurement.
        """
        return self.low_vol_mean_pct - self.high_vol_mean_pct

    @property
    def spread_reliable(self) -> bool:
        """Whether the sample is long enough for `spread_pct` to mean anything.

        The volatility states themselves are identified correctly well below
        this threshold; it is only their DIRECTIONAL reading that needs the
        longer window.
        """
        return self.n_obs >= SPREAD_STABLE_OBSERVATIONS

    def state_on(self, when, *, threshold: float = 0.5,
                 use_smoothed: bool = False) -> bool | None:
        """Was it a high-vol day? None when the date is outside the fit.

        None rather than False: a date the model never saw is not a calm day,
        and a gate that treats it as one would silently trade through every gap.
        """
        s = self.smoothed if use_smoothed else self.filtered
        key = pd.Timestamp(when)
        if key not in s.index:
            return None
        v = s.loc[key]
        return None if not np.isfinite(v) else bool(v > threshold)


def fit_regimes(prices: pd.Series) -> MarkovRegimeFit:
    """Fit a two-state Markov switching model to the returns of `prices`.

    Raises ValueError on too little data or non-convergence - never returns a
    fit that did not converge, because its probabilities would be noise wearing
    a model's clothes.

    Warnings are suppressed only around the fit itself, in a context manager.
    statsmodels is legitimately noisy during optimisation, and the alternative
    upstream chose - a module-level global filter - silences the host process.
    """
    if not available():
        raise ValueError(
            "statsmodels is not installed, so no regime can be fitted. "
            "`pip install statsmodels` - it accepts pandas 2.3.3 and does not "
            "move the pin (verified by dry run).")

    from statsmodels.tsa.regime_switching.markov_regression import (
        MarkovRegression,
    )

    s = pd.Series(prices).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime([str(x) for x in s.index])
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    returns = s.pct_change().dropna() * 100.0
    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(
            f"{len(returns)} returns is too few to identify two regimes "
            f"(minimum {MIN_OBSERVATIONS}); a shorter series fits noise.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MarkovRegression(returns, k_regimes=2, trend="c",
                                   switching_variance=True).fit()
    except Exception as exc:                            # noqa: BLE001
        raise ValueError(
            f"the Markov regime model did not converge: {exc}") from exc

    # WHICH STATE IS WHICH IS NOT FIXED. The optimiser labels them 0 and 1 in
    # whatever order it lands on, so the high-vol state must be identified by
    # its variance on every fit. Assuming state 1 is the volatile one inverts
    # the gate on roughly half of all fits.
    v0 = float(res.params.get("sigma2[0]", np.nan))
    v1 = float(res.params.get("sigma2[1]", np.nan))
    if not (np.isfinite(v0) and np.isfinite(v1)):
        raise ValueError("the fit returned no state variances to compare")
    hi = 1 if v1 > v0 else 0

    filtered = res.filtered_marginal_probabilities.iloc[:, hi]
    smoothed = res.smoothed_marginal_probabilities.iloc[:, hi]

    states = res.smoothed_marginal_probabilities.idxmax(axis=1)
    hi_mask = states == hi

    def _mean(mask) -> float:
        sub = returns[mask]
        return float(sub.mean()) if len(sub) else float("nan")

    def _vol(mask) -> float:
        sub = returns[mask]
        return float(sub.std() * np.sqrt(252)) if len(sub) > 1 else float("nan")

    return MarkovRegimeFit(
        filtered=filtered, smoothed=smoothed,
        high_vol_ann_pct=_vol(hi_mask), low_vol_ann_pct=_vol(~hi_mask),
        high_vol_mean_pct=_mean(hi_mask), low_vol_mean_pct=_mean(~hi_mask),
        high_vol_share=float(hi_mask.mean()), n_obs=int(len(returns)),
    )
