"""Pairs cointegration, estimated point-in-time. BUG-03's fix.

WHY THIS EXISTS
---------------
`pairs_trading` has been parked at P2 for the whole project with one blocker:
BUG-03, "the hedge ratio is fitted on the whole sample, so every backtested
spread is look-ahead". OpenTerminalUI's `core/statlab/cointegration.py` looked
like the fix and is in fact the same bug, which is worth recording precisely
because it is the single easiest mistake to make here:

    model = sm.OLS(s1, sm.add_constant(s2)).fit()      # the WHOLE series
    beta  = model.params.iloc[1]
    spread = s1 - beta * s2 - alpha                    # applied to every day

Day one's spread is computed with a beta estimated from days one to N. The
spread therefore knows its own future, and a mean-reversion signal built on it
reverts to a mean that was calculated using the reversion.

THEIR VERSION ADDS TWO MORE LOOK-AHEADS, both subtler:

  - the z-score uses a 60-observation ROLLING mean and standard deviation,
    which is correct - and then fills the first 60 NaNs with
    `(spread - spread.mean()) / spread.std()`, whole-sample. The warm-up period
    is scored against statistics from the entire history.
  - `coint(s1, s2)` and the half-life regression are both whole-sample, which
    is fine for describing a pair today and wrong inside a backtest.

And three bare `except:` clauses swallow everything, with `beta = 0.0` as a
fallback - which silently turns the spread into `s1 - s2`, an unhedged
difference and a completely different trade.

WHAT THIS DOES INSTEAD. Every quantity is estimated on a TRAILING window ending
at the bar being scored, and re-estimated as the walk proceeds: the hedge ratio,
the spread mean and standard deviation, the cointegration p-value and the
half-life. Nothing is filled from the full sample. The cost is that the first
`window` bars produce no signal, which is correct - there was nothing to know
then.

THE MULTIPLE-TESTING PROBLEM, which is not a detail for pairs. Screening N
symbols means testing N(N-1)/2 pairs, and at p < 0.05 roughly one in twenty
unrelated pairs passes by chance. On the desk's ~1,000-name universe that is
499,500 tests and about 25,000 false positives. `screen_pairs` therefore
requires a correction and reports how many tests were run, so the p-value a
caller sees is one they can act on.

WHAT THE FIXED CODE THEN SAID, WHICH IS THE POINT OF HAVING FIXED IT
--------------------------------------------------------------------
BUG-03 is now fixed, so the question it was blocking could finally be asked:
are there cointegrated pairs in this universe? Measured on 278 within-industry
pairs - the six most liquid names in each of the classified industries, 1,240
sessions to 2026-09-29, at the window where the test was shown above to have
full power:

    window   tested   passed (FDR 10%)   min p    p<0.05    P(>= that | null)
       300      278                  0   0.00604   19/278               0.106
       600      278                  1   0.00021   15/278               0.419
      1200      278                  2   0.00026   18/278               0.160

THE ANSWER IS NO, and it does not rest on the pass counts. Three things say it:

  - THE P-VALUE DISTRIBUTION IS THE NULL. Median p is 0.49-0.54 against a null
    0.50, and the number of pairs under 0.05 is what chance alone produces at
    every window (last column; none significant). If even a handful of these
    pairs were genuinely cointegrated there would be an excess of small
    p-values. There is no excess.
  - THE SURVIVORS ARE NOT STABLE. M&M/MARUTI passes at 600 sessions and fails
    at 1,200; the two that pass at 1,200 (ANANTRAJ/OBEROIRLTY,
    TATACHEM/NAVINFLUOR) both fail at 600. A real long-run equilibrium gets
    MORE significant with more data, it does not change identity.
  - THE HEDGE RATIOS ARE NOT ECONOMIC. 0.175 between M&M and Maruti is barely
    a hedge, and TATACHEM/NAVINFLUOR comes out NEGATIVE at -0.081, which means
    going long both legs - a leveraged directional bet wearing a pair trade's
    clothes.

The seductive part, and the reason this needed a correction rather than a
glance: at 300 sessions the best raw p-values are exactly the pairs a human
would name - ULTRACEMCO/AMBUJACEM and ULTRACEMCO/ACC (cement), M&M/MARUTI,
LT/NCC, DRREDDY/DIVISLAB. The list looks like domain knowledge confirming
itself. The minimum p-value across those 278 tests was 0.0060 where chance
alone gives 0.0036, so the top pair was LESS extreme than noise would predict.

WHAT IS NOT CLAIMED. 278 pairs is not 499,500 - this says nothing about
cross-sector pairs or illiquid names, which were never tested. Engle-Granger is
one specification; Johansen may differ. And cointegration is not the only basis
for a pairs trade (distance and ratio-z-score methods do not require it). The
claim is narrow and it is the one the roadmap needed: among the liquid
within-industry pairs a human would actually trade, there is no cointegration
to trade, and the five-year window is long enough to say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["COINT_TEST_WINDOW", "DEFAULT_WINDOW", "MIN_WINDOW",
           "PairState", "ScreenResult", "available",
           "hedge_ratio", "pair_state", "screen_pairs", "walk_spread"]

#: The trailing window every estimate uses. 250 sessions is about a year - long
#: enough for a stable beta, short enough that a structural break in the
#: relationship ages out rather than being averaged over forever.
#:
#: BUT THE BETA AND THE COINTEGRATION TEST WANT OPPOSITE THINGS, and that
#: tension is real rather than a tuning preference. The beta wants RECENCY: on
#: HDFCBANK/ICICIBANK the trailing estimate ranged -0.708 to +6.553 while the
#: whole-sample figure was -0.339, so a long window averages relationships that
#: never coexisted. The Engle-Granger test wants LENGTH: measured on a fixture
#: that is cointegrated by construction, with 12 seeds per cell -
#:
#:     window   spread phi   median p   detected at p<0.05
#:        250          0.94      0.064          50%
#:        400          0.94      0.009          67%
#:        600          0.94      0.001         100%
#:        250          0.85      0.000         100%
#:
#: At 250 bars a SLOW-reverting spread (phi 0.94, half-life ~11 days) is missed
#: half the time. A fast one (phi 0.85) is always found. So a non-significant
#: p-value from a one-year window is NOT evidence of absence, and
#: `COINT_TEST_WINDOW` exists for callers that need the test rather than the
#: beta.
DEFAULT_WINDOW = 250

#: For the cointegration TEST only, where power matters more than recency. At
#: this length the test found every pair in the fixture above.
COINT_TEST_WINDOW = 600

#: Below this an OLS beta on two price series is noise. Upstream accepts 30,
#: which is enough to fit a line and nowhere near enough to trust one.
MIN_WINDOW = 120


def available() -> bool:
    try:
        import statsmodels.api  # noqa: F401
    except Exception:                                   # noqa: BLE001
        return False
    return True


@dataclass(frozen=True, slots=True)
class PairState:
    """One pair, assessed using only data up to `as_of`."""

    as_of: pd.Timestamp
    beta: float | None
    alpha: float | None
    spread: float | None
    zscore: float | None
    coint_pvalue: float | None
    half_life: float | None
    n_obs: int

    @property
    def tradeable(self) -> bool:
        """Enough was estimable to act on. NOT a view on whether to trade.

        A pair can be perfectly estimable and a terrible trade; this only says
        the numbers exist.
        """
        return (self.beta is not None and self.zscore is not None
                and math.isfinite(self.zscore))


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float] | None:
    """OLS beta and intercept of `y` on `x`. None when it cannot be estimated.

    None rather than 0.0 - which is upstream's fallback and is far worse than
    nothing, because a beta of zero turns the spread into an unhedged
    difference between two prices and keeps trading it.
    """
    import statsmodels.api as sm

    pair = pd.concat([y, x], axis=1).dropna()
    if len(pair) < MIN_WINDOW:
        return None
    yv, xv = pair.iloc[:, 0], pair.iloc[:, 1]
    if xv.std() == 0 or yv.std() == 0:
        # A constant leg has no relationship to estimate. Circuit-locked or
        # untraded names reach here.
        return None
    try:
        res = sm.OLS(yv.to_numpy(float),
                     sm.add_constant(xv.to_numpy(float))).fit()
    except Exception:                                   # noqa: BLE001
        return None
    a, b = float(res.params[0]), float(res.params[1])
    return (b, a) if math.isfinite(a) and math.isfinite(b) else None


def pair_state(y: pd.Series, x: pd.Series, *, as_of=None,
               window: int = DEFAULT_WINDOW) -> PairState:
    """Assess one pair using ONLY the `window` bars ending at `as_of`.

    Everything - beta, spread mean and sd, the cointegration p-value, the
    half-life - comes from that trailing window. Nothing is filled from the
    full sample, which is the whole difference between this and upstream.
    """
    if not available():
        raise ValueError("statsmodels is not installed, so no pair can be "
                         "assessed. `pip install statsmodels`.")
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint

    pair = pd.concat([y, x], axis=1).dropna()
    pair.columns = ["y", "x"]
    if as_of is not None:
        pair = pair.loc[:pd.Timestamp(as_of)] if isinstance(
            pair.index, pd.DatetimeIndex) else pair.loc[:as_of]
    trail = pair.tail(window)
    stamp = pd.Timestamp(trail.index[-1]) if len(trail) else pd.Timestamp(
        as_of or "1970-01-01")

    if len(trail) < MIN_WINDOW:
        return PairState(as_of=stamp, beta=None, alpha=None, spread=None,
                         zscore=None, coint_pvalue=None, half_life=None,
                         n_obs=len(trail))

    hr = hedge_ratio(trail["y"], trail["x"])
    if hr is None:
        return PairState(as_of=stamp, beta=None, alpha=None, spread=None,
                         zscore=None, coint_pvalue=None, half_life=None,
                         n_obs=len(trail))
    beta, alpha = hr

    spread = trail["y"] - beta * trail["x"] - alpha
    sd = float(spread.std())
    # A zero-variance spread cannot be z-scored. None, never inf: inf survives
    # isna() and would read as an extreme signal.
    z = (float((spread.iloc[-1] - spread.mean()) / sd)
         if sd > 0 and math.isfinite(sd) else None)

    try:
        pval = float(coint(trail["y"], trail["x"])[1])
    except Exception:                                   # noqa: BLE001
        # None, not 1.0. Upstream's 1.0 means "definitely not cointegrated",
        # which is a claim; a failed test is an absence of one.
        pval = None

    half = None
    try:
        d = spread.diff().dropna()
        lag = spread.shift(1).loc[d.index]
        res = sm.OLS(d.to_numpy(float),
                     sm.add_constant(lag.to_numpy(float))).fit()
        k = float(res.params[1])
        # Only a MEAN-REVERTING spread has a half-life. k >= 0 means the spread
        # is diverging, and upstream reports 0.0 there - which reads as
        # "reverts instantly", the exact opposite of what it means.
        half = -math.log(2) / k if k < 0 and math.isfinite(k) else None
    except Exception:                                   # noqa: BLE001
        half = None

    return PairState(as_of=stamp, beta=beta, alpha=alpha,
                     spread=float(spread.iloc[-1]), zscore=z,
                     coint_pvalue=pval, half_life=half, n_obs=len(trail))


def walk_spread(y: pd.Series, x: pd.Series, *, window: int = DEFAULT_WINDOW,
                step: int = 1, refit_every: int = 21) -> pd.DataFrame:
    """The point-in-time z-score on every bar, beta re-estimated as we go.

    THE COMPARISON THIS EXISTS FOR. Running it against a whole-sample beta on
    the same pair shows how much BUG-03 flatters a backtest, which is the
    number that decides whether pairs trading is worth unparking.

    `refit_every` re-estimates the hedge ratio periodically rather than on
    every bar - a daily refit is ~250x the work for a beta that barely moves,
    and the spread between refits still uses only past data. Set it to 1 for a
    strict daily refit.
    """
    pair = pd.concat([y, x], axis=1).dropna()
    pair.columns = ["y", "x"]
    rows = []
    beta = alpha = None
    for i in range(window, len(pair), step):
        trail = pair.iloc[i - window:i]          # EXCLUDES bar i
        if beta is None or (i - window) % refit_every == 0:
            hr = hedge_ratio(trail["y"], trail["x"])
            if hr is None:
                continue
            beta, alpha = hr
        sp = trail["y"] - beta * trail["x"] - alpha
        sd = float(sp.std())
        if not (sd > 0 and math.isfinite(sd)):
            continue
        # Bar i's own spread, scored against statistics from BEFORE it.
        now = float(pair["y"].iloc[i] - beta * pair["x"].iloc[i] - alpha)
        rows.append({"date": pair.index[i], "beta": beta,
                     "spread": now,
                     "zscore": (now - float(sp.mean())) / sd})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["beta", "spread", "zscore"])


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """The outcome of screening many pairs at once, correction included."""

    pairs: pd.DataFrame
    """One row per pair tested, with `pvalue`, `qvalue` and `passes`.

    Sorted by raw p-value. `qvalue` is the Benjamini-Hochberg adjusted figure
    and `passes` is the decision AFTER correction - read that column, not
    `pvalue`, when deciding what to trade.
    """

    n_tested: int
    """How many pairs were tested. The number that makes `qvalue` meaningful."""

    n_passed: int
    fdr: float

    @property
    def expected_false_positives_uncorrected(self) -> float:
        """How many pairs would have passed on noise alone at p < 0.05.

        Reported so the correction's size is visible rather than implied.
        """
        return 0.05 * self.n_tested


def screen_pairs(closes: pd.DataFrame, candidates, *, as_of=None,
                 window: int = COINT_TEST_WINDOW, fdr: float = 0.10,
                 min_half_life: float = 2.0,
                 max_half_life: float = 60.0) -> ScreenResult:
    """Test many pairs and correct for having tested many pairs.

    THE PROBLEM THIS SOLVES, because it is the difference between a pairs book
    that works and one that looks like it works. Engle-Granger at p < 0.05
    passes about one in twenty UNRELATED pairs. Screening 100 symbols means
    4,950 tests and roughly 248 pairs that pass on noise; the desk's full
    universe gives about 25,000. Picking the smallest p-values from that set
    selects for luck, and the selected pairs then fail out of sample in exactly
    the way a real relationship breaking down would look.

    So every p-value here is adjusted by Benjamini-Hochberg across the whole
    batch, and `passes` reflects the adjusted figure. BH controls the false
    DISCOVERY rate - of the pairs reported, at most `fdr` of them are expected
    to be noise - which is the right target for a screen whose output is a
    shortlist. Bonferroni would control the family-wise rate instead and reject
    nearly everything at this test count.

    `window` defaults to COINT_TEST_WINDOW rather than DEFAULT_WINDOW: a screen
    wants the test's POWER, and 250 bars finds a slow-reverting pair only half
    the time. The beta a caller trades on should still come from the shorter,
    more recent window via `pair_state`.

    HALF-LIFE BOUNDS ARE A TRADEABILITY FILTER, not a statistical one. A spread
    with a 400-day half-life is cointegrated and untradeable - the capital is
    tied up past any horizon this desk works on - and one with a half-life under
    two days reverts inside the cost of trading it. Pairs outside the bounds are
    REPORTED with `passes` False and a reason, never dropped silently.
    """
    if not available():
        raise ValueError("statsmodels is not installed, so no pair can be "
                         "screened. `pip install statsmodels`.")

    rows = []
    for a, b in candidates:
        if a not in closes.columns or b not in closes.columns:
            rows.append({"y": a, "x": b, "pvalue": None, "beta": None,
                         "half_life": None, "n_obs": 0,
                         "reason": "symbol not in the panel"})
            continue
        st = pair_state(closes[a], closes[b], as_of=as_of, window=window)
        reason = ""
        if st.beta is None:
            reason = f"not estimable on {st.n_obs} bars"
        elif st.coint_pvalue is None:
            reason = "the cointegration test failed"
        elif st.half_life is None:
            reason = "the spread does not revert"
        elif st.half_life < min_half_life:
            reason = f"half-life {st.half_life:.1f}d is inside trading costs"
        elif st.half_life > max_half_life:
            reason = f"half-life {st.half_life:.1f}d is too slow to trade"
        rows.append({"y": a, "x": b, "pvalue": st.coint_pvalue,
                     "beta": st.beta, "half_life": st.half_life,
                     "zscore": st.zscore, "n_obs": st.n_obs,
                     "reason": reason})

    out = pd.DataFrame(rows)
    if out.empty:
        return ScreenResult(pairs=out, n_tested=0, n_passed=0, fdr=fdr)

    # BENJAMINI-HOCHBERG. Rank the p-values, compare each to (rank/m)*fdr, and
    # take everything up to the LARGEST rank that clears its own threshold -
    # the step-up, which is what gives BH more power than testing each
    # independently. Pairs with no p-value are excluded from m rather than
    # counted as tests that happened.
    out = out.sort_values("pvalue", na_position="last").reset_index(drop=True)
    have = out["pvalue"].notna()
    m = int(have.sum())
    out["qvalue"] = None
    out["passes"] = False
    if m:
        pv = out.loc[have, "pvalue"].to_numpy(float)
        ranks = np.arange(1, m + 1)
        # Monotone q-values: the running minimum from the largest p downwards,
        # so a small p never gets a bigger q than a larger one.
        q = np.minimum.accumulate((pv * m / ranks)[::-1])[::-1].clip(max=1.0)
        out.loc[have, "qvalue"] = q
        cleared = ranks[pv <= ranks / m * fdr]
        cut = int(cleared.max()) if len(cleared) else 0
        idx = out.index[have][:cut]
        out.loc[idx, "passes"] = True
    # A tradeability reason vetoes the statistical pass.
    out.loc[out["reason"].astype(bool), "passes"] = False

    return ScreenResult(pairs=out, n_tested=m,
                        n_passed=int(out["passes"].sum()), fdr=fdr)
