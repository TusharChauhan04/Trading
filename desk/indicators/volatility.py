"""ATR, Bollinger Bands, and volatility compression/expansion.

Compression detection matters for a specific, named reason: the master plan's
Stage 1 scanner explicitly wants "volatility compression" as a filter - a
range-bound stock whose bands have narrowed is a coiled spring, and several
breakout strategies (including two of the six shared ones) are only
meaningful once compression has been confirmed rather than assumed.
"""

from __future__ import annotations

import pandas as pd

from desk.indicators.trend import sma, wilder


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14, smoothing: str = "wilder") -> pd.Series:
    tr = true_range(df)
    return wilder(tr, period) if smoothing == "wilder" else sma(tr, period)


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a % of close - comparable ACROSS symbols at different price
    levels, which raw ATR is not. This is what the risk engine's
    max_instrument_atr_pct gate and the scanner's compression filter both
    want, expressed the same way."""
    return 100.0 * atr(df, period) / df["close"]


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0
                    ) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period).std()
    return pd.DataFrame({
        "mid": mid,
        "upper": mid + num_std * std,
        "lower": mid - num_std * std,
    })


def bb_width_pct(close: pd.Series, period: int = 20, num_std: float = 2.0
                 ) -> pd.Series:
    """Band width as a % of the midline - the standard normalized "squeeze"
    measure. Narrow relative to its own recent history, not an absolute
    threshold, since a quiet stock and a volatile one have different normal
    widths."""
    bb = bollinger_bands(close, period, num_std)
    return 100.0 * (bb["upper"] - bb["lower"]) / bb["mid"]


def is_compressed(close: pd.Series, period: int = 20, num_std: float = 2.0,
                  lookback: int = 120, percentile: float = 20.0) -> pd.Series:
    """True where today's band width sits in the bottom `percentile` of its
    own trailing `lookback` history - a squeeze relative to this stock's own
    recent behaviour, not a fixed number that means something different for
    a large-cap and a small-cap.
    """
    width = bb_width_pct(close, period, num_std)
    threshold = width.rolling(lookback).quantile(percentile / 100.0)
    return width <= threshold
