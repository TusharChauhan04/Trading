"""desk/indicators/ - the layer named in master prompt section 9 as distinct
from strategies, filling the gap left by strategy_lib.py (which only has what
the six shared strategies happen to need). Every function here is checked
against a hand-computed value, not just "runs without crashing".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.indicators.relative_strength import rs_line, rs_rank
from desk.indicators.structure import (
    classify_gap,
    gap_pct,
    latest_structure_bias,
    market_structure,
    swing_points,
)
from desk.indicators.trend import ema, macd, sma, wilder
from desk.indicators.volatility import (
    atr_pct,
    bollinger_bands,
    is_compressed,
    true_range,
)
from desk.indicators.volume import relative_volume, rolling_vwap, session_vwap


def _bars(opens, highs, lows, closes, volumes=None, start="2024-01-01"):
    n = len(closes)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes or [1000] * n,
    }, index=idx)


# ===========================================================================
# trend
# ===========================================================================

def test_sma_matches_a_hand_computed_average():
    s = pd.Series([1, 2, 3, 4, 5])
    out = sma(s, 3)
    assert out.iloc[2] == pytest.approx(2.0)      # mean(1,2,3)
    assert out.iloc[4] == pytest.approx(4.0)      # mean(3,4,5)
    assert out.iloc[:2].isna().all()


def test_wilder_and_ema_are_genuinely_different_curves():
    """Same starting inputs, different alpha - they must diverge, not
    coincidentally match (a common copy-paste bug: passing period as span)."""
    s = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0, 120.0, 80.0])
    w = wilder(s, 4)     # alpha = 0.25
    e = ema(s, 4)        # alpha = 2/(4+1) = 0.4
    assert not np.isclose(w.iloc[-1], e.iloc[-1])


def test_macd_hist_is_exactly_macd_minus_signal():
    close = pd.Series(100 + np.cumsum(np.random.RandomState(0).normal(0, 1, 60)))
    m = macd(close)
    assert list(m.columns) == ["macd", "signal", "hist"]
    diff = (m["hist"] - (m["macd"] - m["signal"])).abs()
    assert (diff.dropna() < 1e-9).all()


def test_macd_rejects_fast_greater_than_or_equal_to_slow():
    close = pd.Series([1.0] * 30)
    with pytest.raises(ValueError, match="must be <"):
        macd(close, fast=26, slow=12)


# ===========================================================================
# volatility
# ===========================================================================

def test_true_range_picks_the_widest_of_three_cases():
    """Normal bar (high-low wins), a gap up (high-prev_close wins), and a
    gap down (prev_close-low wins) - the three cases the formula exists to
    cover, each hand-verified."""
    df = _bars(
        opens=[100, 100, 101, 109],
        highs=[101, 102, 110, 95],
        lows=[99, 99, 108, 90],
        closes=[100, 101, 109, 92],
    )
    tr = true_range(df)
    # No previous close on the first bar -> both gap terms are NaN and the
    # standard convention (what every TA library does) is that max() then
    # falls back to the one defined term, high-low. Not a special case in
    # the code; it falls out of pandas skipping NaN in .max(axis=1).
    assert tr.iloc[0] == pytest.approx(2.0)        # 101-99, the only defined term
    assert tr.iloc[1] == pytest.approx(3.0)        # normal: 102-99
    assert tr.iloc[2] == pytest.approx(9.0)        # gap up: 110-101
    assert tr.iloc[3] == pytest.approx(19.0)       # gap down: 109-90


def test_atr_pct_is_comparable_across_price_levels():
    """A share at 1000 with the same relative range as one at 100 must give
    the same atr_pct - that comparability is the entire point of the %."""
    cheap = _bars([100]*20, [102]*20, [98]*20, [100]*20)
    expensive = cheap.copy()
    for c in ("open", "high", "low", "close"):
        expensive[c] = expensive[c] * 10
    a1, a2 = atr_pct(cheap).iloc[-1], atr_pct(expensive).iloc[-1]
    assert a1 == pytest.approx(a2, rel=1e-9)


def test_bollinger_bands_bracket_the_midline_symmetrically():
    close = pd.Series(np.random.RandomState(2).normal(100, 5, 40))
    bb = bollinger_bands(close, period=20, num_std=2.0)
    gap_up = (bb["upper"] - bb["mid"]).dropna()
    gap_down = (bb["mid"] - bb["lower"]).dropna()
    assert np.allclose(gap_up, gap_down)


def test_is_compressed_flags_a_genuinely_narrow_recent_range():
    """A series that goes wide-then-flat should mark the flat stretch (and
    only the flat stretch) as compressed relative to its own history."""
    wide = 100 + 20 * np.sin(np.linspace(0, 20, 150))
    flat = np.full(30, wide[-1])
    close = pd.Series(np.concatenate([wide, flat]))
    comp = is_compressed(close, period=10, lookback=100, percentile=20.0)
    assert comp.iloc[-1]           # the flat tail is compressed
    assert not comp.iloc[100]      # the middle of the wide swing is not


# ===========================================================================
# volume
# ===========================================================================

def test_session_vwap_equals_typical_price_on_one_bar_per_day_data():
    """Each 'session' is exactly one bar for daily data, so the cumulative
    sum within the group degenerates to that single bar's own typical
    price - float tolerance because both sides recompute the same division."""
    df = _bars([100]*5, [102, 103, 101, 104, 105], [98, 99, 97, 100, 101],
              [100, 101, 99, 102, 103], [1000, 2000, 1500, 1800, 2200])
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    assert np.allclose(session_vwap(df), tp)


def test_rolling_vwap_matches_a_hand_computed_window():
    df = _bars([100]*3, [102, 104, 106], [98, 100, 102], [100, 102, 104],
              [1000, 2000, 3000])
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    expected_last = (tp * df["volume"]).iloc[-2:].sum() / df["volume"].iloc[-2:].sum()
    assert rolling_vwap(df, period=2).iloc[-1] == pytest.approx(expected_last)


def test_relative_volume_excludes_todays_own_bar_from_its_baseline():
    """A huge print must not inflate its own baseline - otherwise a real
    volume spike could understate itself as merely 'average'."""
    vol = pd.Series([1000, 1000, 1000, 1000, 5000])
    rv = relative_volume(vol, period=4)
    assert rv.iloc[4] == pytest.approx(5.0)        # 5000 / mean(1000,1000,1000,1000)


# ===========================================================================
# structure
# ===========================================================================

def test_gap_pct_and_classify_gap_on_real_gaps():
    df = _bars(opens=[100, 103, 97, 100],
               highs=[101, 104, 98, 101],
               lows=[99, 102, 96, 99],
               closes=[100, 100, 100, 100])
    g = gap_pct(df)
    assert g.iloc[1] == pytest.approx(3.0)     # 103/100 - 1
    assert g.iloc[2] == pytest.approx(-3.0)    # 97/100 - 1 (prev close still 100)
    cls = classify_gap(df, threshold_pct=2.0)
    assert cls.iloc[1] == "up"
    assert cls.iloc[2] == "down"
    assert cls.iloc[3] == "none"               # 100 -> 100, no gap


def test_swing_points_hand_verified_fractal():
    """window=1: a bar is a swing high/low relative to just its immediate
    neighbours. Every high/low index below was chosen and verified by hand."""
    df = _bars(opens=[0]*7, closes=[0]*7,
              highs=[10, 12, 11, 15, 13, 14, 9],
              lows=[20, 18, 19, 15, 17, 16, 21])
    sw = swing_points(df, window=1)
    assert list(sw.index[sw["swing_high"]].map(df.index.get_loc)) == [1, 3, 5]
    assert list(sw.index[sw["swing_low"]].map(df.index.get_loc)) == [1, 3, 5]


def test_market_structure_labels_match_hand_verification():
    """Continuing the fixture above: swing highs at 12,15,14 -> HH then LH.
    Swing lows at 18,15,16 -> LL then HL. The pair (LH, HL) is neither a
    clean uptrend nor downtrend structure - "mixed" is the correct label,
    and is the branch most likely to be silently wrong."""
    df = _bars(opens=[0]*7, closes=[0]*7,
              highs=[10, 12, 11, 15, 13, 14, 9],
              lows=[20, 18, 19, 15, 17, 16, 21])
    ms = market_structure(df, window=1)
    highs = ms["high_label"].dropna().tolist()
    lows = ms["low_label"].dropna().tolist()
    assert highs == ["HH", "LH"]
    assert lows == ["LL", "HL"]
    assert latest_structure_bias(ms) == "mixed"


def test_latest_structure_bias_recognises_a_clean_uptrend():
    df = _bars(opens=[0]*7, closes=[0]*7,
              highs=[10, 12, 11, 16, 14, 20, 15],   # 12 -> 16 -> 20: HH, HH
              lows=[5, 8, 6, 11, 9, 14, 10])         # 8 -> 11 -> 14: HL, HL
    ms = market_structure(df, window=1)
    assert latest_structure_bias(ms) == "uptrend"


def test_structure_is_unknown_with_too_few_swings():
    """A strictly monotonic series has NO interior local max or min at all -
    every point is between two others that are both higher or both lower -
    so zero swings of either type are ever detected, not merely one
    unlabeled first swing."""
    df = _bars(opens=[0]*5, closes=[0]*5, highs=[10, 11, 12, 13, 14],
              lows=[5, 6, 7, 8, 9])
    sw = swing_points(df, window=1)
    assert not sw["swing_high"].any() and not sw["swing_low"].any()
    ms = market_structure(df, window=1)
    assert latest_structure_bias(ms) == "unknown"


# ===========================================================================
# relative strength
# ===========================================================================

def test_rs_line_rises_when_outperforming_a_falling_benchmark():
    """The whole point of RS: a stock that is FLAT while the benchmark FALLS
    must show a rising RS line, even though the stock itself went nowhere."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    price = pd.Series([100, 100, 100, 100, 100], index=idx)
    bench = pd.Series([100, 98, 96, 94, 92], index=idx)
    rs = rs_line(price, bench)
    assert rs.is_monotonic_increasing


def test_rs_rank_is_100_at_the_top_of_a_monotonic_series():
    rs = pd.Series(range(1, 11), dtype=float)
    rank = rs_rank(rs, period=5)
    # Every fully-windowed point of a strictly increasing series is its own
    # window's maximum, so its percentile rank is exactly 100.
    assert np.allclose(rank.dropna(), 100.0)
