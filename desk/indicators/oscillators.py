"""The eleven indicators from upstream's registry that the desk did not have.

WHY THESE ARE REWRITTEN RATHER THAN IMPORTED
--------------------------------------------
`services/indicators.py` offers thirteen behind a `compute(name, **kwargs)`
registry. Only ADX and the directional indices were taken (see
`desk/indicators/directional.py`); these are the other eleven. The reason for
rewriting is the same one that applied to ADX, and it is not a style
preference:

ITS `_fillna_local` IS `ffill().bfill()`, AND `bfill` FILLS EARLIER NaNs FROM
LATER VALUES. On a 14-period indicator, bars 0-13 come back carrying the value
computed at bar 14 - a number that did not exist until two weeks after the bar
it is attached to. Nine of its thirteen indicators do this. A backtest reading
the warmup period would be reading the future, and the error is invisible
because the series simply looks complete.

So every function here returns NaN until its window is genuinely full. A short
series yields NaN, never a number, and `min_periods` equals the period
everywhere.

TWO SMALLER CORRECTIONS, both the same shape as the ADX one:
  - Wilder smoothing where Wilder specified it (alpha = 1/period), not a plain
    rolling mean. A plain mean puts crossings on different bars.
  - `std_deviation` annualises with sqrt(252) only when asked. Upstream
    reports `volatility` and `volatility_annual` from one call, and the
    annualised figure of a 14-bar window is a number with a very wide error
    bar presented as a property of the stock.

MATRIX-NATIVE, like the rest of this package: every function takes wide
date x symbol frames and returns the same shape, so Stage 1 can compute the
whole universe in one pass rather than looping symbols.

WHAT THIS IS NOT. It is not a claim that any of these predicts anything. The
desk has tested its existing features extensively and found no validated edge;
adding eleven more is adding eleven more things that must be counted in
`num_trials` before any of them is believed. They are here because the
registry's absence was a stated gap, and because a feature that does not exist
cannot be ruled out either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from desk.indicators.directional import true_range_wide

__all__ = ["aroon", "awesome_oscillator", "cci", "ichimoku", "momentum",
           "parabolic_sar", "roc", "std_deviation", "stochastic",
           "ultimate_oscillator", "williams_r"]


def _rolling_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).max()


def _rolling_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).min()


def stochastic(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, *,
               k_period: int = 14,
               d_period: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stochastic oscillator (%K, %D). 0-100, where the close sits in range.

    A FLAT RANGE HAS NO ANSWER, and that is the one judgement call here. When
    the `k_period` high equals the low - a circuit-locked or untraded stock -
    the denominator is zero. Upstream's division yields inf or NaN and then
    `_safe_float` turns it into 0.0, which reads as "at the very bottom of its
    range", the strongest oversold signal the indicator can give. A frozen
    stock would top an oversold screen. Here it stays NaN.
    """
    hh = _rolling_max(high, k_period)
    ll = _rolling_min(low, k_period)
    span = hh - ll
    k = (close - ll) / span.where(span > 0) * 100.0
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def williams_r(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, *,
               period: int = 14) -> pd.DataFrame:
    """Williams %R. -100 to 0; the stochastic's mirror image.

    Same zero-range refusal as `stochastic`, for the same reason.
    """
    hh = _rolling_max(high, period)
    ll = _rolling_min(low, period)
    span = hh - ll
    return (hh - close) / span.where(span > 0) * -100.0


def aroon(high: pd.DataFrame, low: pd.DataFrame, *,
          period: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aroon up and down. How recently the window's extreme occurred.

    Measured in bars since the high/low as a percentage of the window, so 100
    means the extreme is the current bar and 0 means it is the oldest bar in
    the window.

    USES `argmax` ON THE WINDOW, which needs care: pandas' rolling.apply with
    raw=True is the only way to get a positional argmax per window, and it is
    slow. The loop below is over COLUMNS, not bars, so a 1,100-name panel runs
    1,100 vectorised passes rather than 1.2 million scalar ones.
    """
    def _since(frame: pd.DataFrame, want_max: bool) -> pd.DataFrame:
        out = {}
        for col in frame.columns:
            s = frame[col]
            fn = (lambda w: float(np.argmax(w))) if want_max else (
                lambda w: float(np.argmin(w)))
            # min_periods=period, so no partial window produces a number.
            idx = s.rolling(period, min_periods=period).apply(fn, raw=True)
            out[col] = idx
        pos = pd.DataFrame(out, index=frame.index, columns=frame.columns)
        # `pos` is the window POSITION of the extreme: 0 at the oldest bar,
        # period-1 at the current one. Aroon is defined on RECENCY, so that
        # maps straight across - the extreme being the current bar gives 100,
        # the oldest bar gives 0. Do not "correct" this to (period-1-pos),
        # which is bars-since and inverts the indicator.
        return (pos / (period - 1)) * 100.0

    return _since(high, True), _since(low, False)


def cci(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, *,
        period: int = 20,
        constant: float = 0.015) -> pd.DataFrame:
    """Commodity Channel Index.

    Uses MEAN ABSOLUTE DEVIATION, not standard deviation, which is Lambert's
    original definition and what the 0.015 constant is scaled for. Substituting
    sd - a common error, and what a careless rewrite does because pandas offers
    `.std()` and not `.mad()` - changes the scale by roughly 1.25x and makes the
    conventional +/-100 thresholds mean something else.
    """
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    # pandas removed DataFrame.mad(); this is the same quantity, per window.
    mad = tp.rolling(period, min_periods=period).apply(
        lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    return (tp - sma) / (constant * mad.where(mad > 0))


def roc(close: pd.DataFrame, *, period: int = 12) -> pd.DataFrame:
    """Rate of change, as a percentage.

    Guards a zero or negative prior price rather than dividing by it: a bad
    print of 0.00 would otherwise produce inf, and inf survives `isna()` so it
    would pass every downstream finite check that only looks for NaN.
    """
    prior = close.shift(period)
    return (close / prior.where(prior > 0) - 1.0) * 100.0


def momentum(close: pd.DataFrame, *, period: int = 10) -> pd.DataFrame:
    """Momentum: the raw price difference over `period` bars.

    IN RUPEES, NOT A RATIO, which makes it unusable as a cross-sectional
    feature on its own - a 50-rupee move means something different on a
    200-rupee stock and a 5,000-rupee one. Upstream reports it as-is and so
    does this; `roc` is the comparable version. Kept because the registry has
    it and because dividing it silently would be a different indicator wearing
    this one's name.
    """
    return close - close.shift(period)


def std_deviation(close: pd.DataFrame, *, period: int = 20,
                  annualise: bool = False,
                  periods_per_year: int = 252) -> pd.DataFrame:
    """Rolling standard deviation of RETURNS, optionally annualised.

    ON RETURNS, not on price. Upstream's `std_dev` is the standard deviation of
    the price level, which scales with the price - so it ranks expensive stocks
    as volatile and is not comparable across the cross-section. The return
    version is.

    `annualise` is off by default and that is deliberate: scaling a 20-bar
    estimate by sqrt(252) produces a confident-looking annual figure from a
    month of data. Upstream returns both from one call, which invites reading
    the annualised one as a property of the stock.
    """
    rets = close.pct_change(fill_method=None)
    sd = rets.rolling(period, min_periods=period).std()
    return sd * np.sqrt(periods_per_year) if annualise else sd


def awesome_oscillator(high: pd.DataFrame, low: pd.DataFrame, *,
                       fast: int = 5, slow: int = 34) -> pd.DataFrame:
    """Awesome Oscillator: 5-period minus 34-period SMA of the median price."""
    median = (high + low) / 2.0
    return (median.rolling(fast, min_periods=fast).mean()
            - median.rolling(slow, min_periods=slow).mean())


def ultimate_oscillator(high: pd.DataFrame, low: pd.DataFrame,
                        close: pd.DataFrame, *, short: int = 7,
                        medium: int = 14, long: int = 28) -> pd.DataFrame:
    """Williams' Ultimate Oscillator, 0-100, weighted 4:2:1.

    Uses TRUE RANGE with the prior close, not the bar's own high-low range -
    which is what makes it an Ultimate Oscillator rather than a three-period
    stochastic. The gap between yesterday's close and today's low is part of
    the move.
    """
    prior_close = close.shift(1)
    true_low = pd.concat([low, prior_close]).groupby(level=0).min()
    true_low = true_low.reindex(index=low.index, columns=low.columns)
    bp = close - true_low
    tr = true_range_wide(high, low, close)

    def _avg(n: int) -> pd.DataFrame:
        num = bp.rolling(n, min_periods=n).sum()
        den = tr.rolling(n, min_periods=n).sum()
        return num / den.where(den > 0)

    a, b, c = _avg(short), _avg(medium), _avg(long)
    return 100.0 * (4.0 * a + 2.0 * b + c) / 7.0


def ichimoku(high: pd.DataFrame, low: pd.DataFrame, *, tenkan: int = 9,
             kijun: int = 26, senkou_b: int = 52) -> dict[str, pd.DataFrame]:
    """Ichimoku lines. Returns tenkan_sen, kijun_sen, senkou_span_a/b.

    THE SENKOU SPANS ARE DELIBERATELY NOT SHIFTED FORWARD, and this is the one
    place where the conventional drawing is a look-ahead trap. A chart plots
    them `kijun` bars into the FUTURE, which is why the cloud appears ahead of
    price. Shifting a series forward means bar t carries a value computed from
    bars t-26..t, which is fine - but a naive implementation shifts the other
    way, or a caller reads the plotted cloud at bar t as if it were known at
    bar t, when the plotted value at t was computed at t-26 and the value
    computed AT t is drawn at t+26.

    Returned unshifted, each line carrying the value computed from data up to
    and including its own bar. A caller that wants the chart's appearance
    shifts them forward itself and accepts that the forward portion is not yet
    knowable.
    """
    def _mid(n: int) -> pd.DataFrame:
        return (_rolling_max(high, n) + _rolling_min(low, n)) / 2.0

    t = _mid(tenkan)
    k = _mid(kijun)
    return {
        "tenkan_sen": t,
        "kijun_sen": k,
        "senkou_span_a": (t + k) / 2.0,
        "senkou_span_b": _mid(senkou_b),
    }


def parabolic_sar(high: pd.DataFrame, low: pd.DataFrame, *,
                  step: float = 0.02,
                  max_step: float = 0.20) -> pd.DataFrame:
    """Wilder's Parabolic SAR.

    SEQUENTIAL BY CONSTRUCTION and therefore the one indicator here that
    cannot be vectorised: each bar's stop depends on the previous bar's stop,
    the current trend direction and an acceleration factor that resets on every
    reversal. Computed per column in a loop, which is O(bars) per symbol.

    Starts flat-footed rather than guessing: the first bar has no prior
    extreme, so the series begins at NaN and the first real value appears on
    bar 1. Upstream seeds it from the first bar's low and reports a value on
    bar 0, which is a stop level derived from a single bar.
    """
    out = pd.DataFrame(np.nan, index=high.index, columns=high.columns)
    for col in high.columns:
        h = high[col].to_numpy(float)
        l = low[col].to_numpy(float)
        n = len(h)
        if n < 2:
            continue
        sar = np.full(n, np.nan)
        # Direction seeded from the first two bars rather than assumed.
        rising = h[1] >= h[0]
        af = step
        ep = h[1] if rising else l[1]
        sar[1] = l[0] if rising else h[0]
        for i in range(2, n):
            if not (np.isfinite(h[i]) and np.isfinite(l[i])):
                sar[i] = np.nan
                continue
            prev = sar[i - 1]
            if not np.isfinite(prev):
                sar[i] = np.nan
                continue
            nxt = prev + af * (ep - prev)
            if rising:
                # The stop may never move above the last two lows.
                nxt = min(nxt, l[i - 1], l[i - 2])
                if l[i] < nxt:                      # reversal
                    rising = False
                    nxt = ep
                    ep = l[i]
                    af = step
                elif h[i] > ep:
                    ep = h[i]
                    af = min(af + step, max_step)
            else:
                nxt = max(nxt, h[i - 1], h[i - 2])
                if h[i] > nxt:
                    rising = True
                    nxt = ep
                    ep = h[i]
                    af = step
                elif l[i] < ep:
                    ep = l[i]
                    af = min(af + step, max_step)
            sar[i] = nxt
        out[col] = sar
    return out
