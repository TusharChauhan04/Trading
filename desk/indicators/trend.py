"""Moving averages and MACD.

Wilder smoothing (alpha = 1/period) is the TradingView/Chartink-compatible
form and is used throughout this package wherever "smoothed" is ambiguous -
see strategies/india/strategy_lib.py's own docstring for why SMA-smoothing an
RSI/ADX-style indicator puts signals on different bars than the chart a
retail trader is actually looking at. The same reasoning applies here.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """EMA with alpha = 1/period. Not the same curve as ema(series, period) -
    a Wilder-smoothed value lags less initially and more later; the two are
    genuinely different indicators that happen to share a shape."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram - the classic 12/26/9.

    Returns a DataFrame with columns `macd`, `signal`, `hist` so a caller
    never has to remember which tuple position is which. `hist` is what most
    screens actually plot as the momentum bars; `macd` crossing `signal` is
    the classic entry trigger, and `hist` crossing zero is the same event
    seen one derivative earlier.
    """
    if fast >= slow:
        raise ValueError(f"fast period ({fast}) must be < slow period ({slow})")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "hist": macd_line - signal_line,
    })
