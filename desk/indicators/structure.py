"""Gaps and market structure (swing highs/lows, HH/HL/LH/LL).

Gap detection here is a SCANNER SIGNAL and must not be confused with
desk.marketdata.quality's gap check, which flags moves beyond the circuit
band (20%) as a likely missing corporate action - a DATA ERROR. This module's
gaps are the everyday, much smaller moves (a default 2%) that are themselves
a tradeable event, not evidence the data is wrong. Same underlying
arithmetic, different question, so it lives in a different module on
purpose.

Market structure (higher-highs/higher-lows vs lower-highs/lower-lows) is
named explicitly in the master prompt's technical-analysis section and had no
implementation anywhere in the project before this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gap_pct(df: pd.DataFrame) -> pd.Series:
    """Today's open vs yesterday's close, as a percentage."""
    prev_close = df["close"].shift(1)
    return 100.0 * (df["open"] / prev_close - 1.0)


def classify_gap(df: pd.DataFrame, threshold_pct: float = 2.0) -> pd.Series:
    """"up" / "down" / "none" per bar. threshold_pct is deliberately far
    below quality.py's 20% circuit-band threshold - this is an everyday
    screening signal, not a data-integrity check."""
    g = gap_pct(df)
    return pd.Series(
        np.select([g >= threshold_pct, g <= -threshold_pct], ["up", "down"],
                 default="none"),
        index=df.index,
    )


def swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Fractal swing highs/lows: a bar is a swing high if its `high` is the
    STRICT maximum within `window` bars on both sides (swing low analogously
    on `low`). window=3 is the common "3-bar fractal" default; a larger
    window finds fewer, more significant swings.

    Returns a frame aligned to df.index with boolean swing_high/swing_low
    columns - most bars are neither. The first and last `window` bars can
    never qualify (there is no full window around them) and are always False.
    """
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(df)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)

    for i in range(window, n - window):
        seg_h = hi[i - window:i + window + 1]
        if hi[i] == seg_h.max() and (seg_h == hi[i]).sum() == 1:
            is_high[i] = True
        seg_l = lo[i - window:i + window + 1]
        if lo[i] == seg_l.min() and (seg_l == lo[i]).sum() == 1:
            is_low[i] = True

    return pd.DataFrame({"swing_high": is_high, "swing_low": is_low}, index=df.index)


def market_structure(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Classifies each swing point against the immediately PRECEDING swing
    of the same type: a swing high is HH (higher high) or LH (lower high); a
    swing low is HL (higher low) or LL (lower low). The first swing of each
    type has nothing to compare against and stays unlabeled - there is no
    such thing as a "higher high" with no prior high to be higher than.

    high_label and low_label are kept as SEPARATE columns rather than one
    merged label, because a bar can in principle be a swing high and a swing
    low simultaneously (a flat-range edge case) and merging would let one
    silently overwrite the other.
    """
    sw = swing_points(df, window)
    high_label = pd.Series(pd.NA, index=df.index, dtype="object")
    low_label = pd.Series(pd.NA, index=df.index, dtype="object")

    prev_high = None
    for idx in df.index[sw["swing_high"]]:
        h = df.loc[idx, "high"]
        if prev_high is not None:
            high_label.loc[idx] = "HH" if h > prev_high else "LH"
        prev_high = h

    prev_low = None
    for idx in df.index[sw["swing_low"]]:
        low = df.loc[idx, "low"]
        if prev_low is not None:
            low_label.loc[idx] = "HL" if low > prev_low else "LL"
        prev_low = low

    return pd.DataFrame({
        "swing_high": sw["swing_high"], "swing_low": sw["swing_low"],
        "high_label": high_label, "low_label": low_label,
    })


def latest_structure_bias(ms: pd.DataFrame) -> str:
    """One word from the most recently labeled swing pair:
    "uptrend" (HH and HL), "downtrend" (LH and LL), "mixed" (disagreement -
    e.g. a higher high but a lower low, a genuinely ambiguous structure), or
    "unknown" (fewer than two swings of either type exist yet to compare)."""
    highs = ms["high_label"].dropna()
    lows = ms["low_label"].dropna()
    if highs.empty or lows.empty:
        return "unknown"
    last_high, last_low = highs.iloc[-1], lows.iloc[-1]
    if last_high == "HH" and last_low == "HL":
        return "uptrend"
    if last_high == "LH" and last_low == "LL":
        return "downtrend"
    return "mixed"
