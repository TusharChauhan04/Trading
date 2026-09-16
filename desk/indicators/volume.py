"""VWAP and relative volume.

VWAP was one of the plainly-named gaps: master prompt section 9 lists it
explicitly and nothing in the project computed it before this module. Two
variants are genuinely different things and are named separately rather than
overloaded behind one flag: session VWAP resets every trading day (the
intraday reference every execution desk actually watches) and rolling VWAP
does not (a longer-horizon fair-value reference some swing strategies use).
Silently picking one would make "VWAP" ambiguous in exactly the way this
project's other modules go out of their way to avoid.
"""

from __future__ import annotations

import pandas as pd


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Resets at the start of each calendar day in the index. For a daily-bar
    frame (one row per day) this is identical to the typical price itself,
    since each "session" is one bar - it becomes genuinely cumulative only
    once fed intraday bars, which is the case it exists for.
    """
    tp = _typical_price(df)
    pv = tp * df["volume"]
    day = pd.Series(df.index.date, index=df.index)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_vol.replace(0, pd.NA)


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """A volume-weighted average over a trailing window, never resetting.
    Useful as a longer-horizon fair-value reference on daily bars, where
    session_vwap degenerates to the typical price."""
    tp = _typical_price(df)
    pv = (tp * df["volume"]).rolling(period).sum()
    vol = df["volume"].rolling(period).sum()
    return pv / vol.replace(0, pd.NA)


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Today's volume as a multiple of its own trailing average - the
    standard "unusual volume" screen named in the master prompt's Stage 1
    filters. >1 means busier than usual; the average EXCLUDES today's own bar
    so a huge print does not inflate its own baseline."""
    baseline = volume.shift(1).rolling(period).mean()
    return volume / baseline.replace(0, pd.NA)
