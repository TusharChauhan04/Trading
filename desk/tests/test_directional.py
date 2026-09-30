"""ADX must be Wilder's, must be directionless, and must not see the future.

The reason this file is thorough is that ADX was going to be imported from
OpenTerminalUI until its `_fillna_local` turned out to be `ffill().bfill()`.
A backfilled warmup produces numbers that look entirely reasonable and are
built from bars that had not happened yet. So the property that matters most
here is not "does it match a reference" but "can an early value possibly
depend on a later one" - and that is testable directly.
"""

from __future__ import annotations

import importlib.util as iu
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desk.indicators.directional import (
    DEFAULT_ADX_PERIOD, adx_wide, directional_movement, true_range_wide,
)

_UP = Path("agents/openterminal_ui/upstream")


def _bars(n: int = 120, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Two symbols of plausible OHLC: one trending, one chopping."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    out = {}
    trend = 100 * np.cumprod(1 + rng.normal(0.004, 0.012, n))
    chop = 250 + np.cumsum(rng.normal(0, 1.2, n))
    close = pd.DataFrame({"TREND": trend, "CHOP": chop}, index=idx)
    span = close * rng.uniform(0.004, 0.02, close.shape)
    out["close"] = close
    out["high"] = close + span
    out["low"] = close - span
    return out


# -- the property that made this file necessary ------------------------------

def test_an_early_value_cannot_depend_on_a_later_bar() -> None:
    """THE look-ahead test, done by construction rather than by inspection.

    Compute on the full series, then on a truncated copy, and compare the
    overlap. If any early reading is built from later bars - a backfill, a
    negative shift, a centred window - truncating the tail must change it.
    Nothing else in this file would catch that.
    """
    b = _bars(120)
    full = adx_wide(b["high"], b["low"], b["close"])
    cut = 80
    part = adx_wide(b["high"].iloc[:cut], b["low"].iloc[:cut],
                    b["close"].iloc[:cut])
    for key in ("adx", "plus_di", "minus_di"):
        pd.testing.assert_frame_equal(
            full[key].iloc[:cut], part[key],
            check_freq=False,
            obj=f"{key} changed when future bars were removed",
        )


def test_the_warmup_is_nan_not_a_borrowed_number() -> None:
    """Upstream returns a filled warmup; we return the truth.

    dx is undefined until there is directional movement to smooth, so the
    first row must be NaN rather than a value carried back from later.
    """
    b = _bars(60)
    out = adx_wide(b["high"], b["low"], b["close"])
    assert out["adx"].iloc[0].isna().all()
    # And by the end it is a real reading.
    assert out["adx"].iloc[-1].notna().all()


# -- Wilder's arithmetic, not a simple mean ----------------------------------

def test_smoothing_is_wilder_not_a_rolling_mean() -> None:
    """The specific thing upstream's ADX gets wrong.

    A simple moving average and alpha=1/period smoothing put crossings on
    different bars. If these ever agree to 6 decimals, the implementation has
    silently become the SMA version.
    """
    b = _bars(200)
    ours = adx_wide(b["high"], b["low"], b["close"])["adx"]

    tr = true_range_wide(b["high"], b["low"], b["close"])
    plus_dm, minus_dm = directional_movement(b["high"], b["low"])
    p = DEFAULT_ADX_PERIOD
    atr_sma = tr.rolling(p).mean()
    pdi = 100.0 * plus_dm.rolling(p).mean() / atr_sma
    mdi = 100.0 * minus_dm.rolling(p).mean() / atr_sma
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi)
    sma_adx = dx.rolling(p).mean()

    diff = (ours - sma_adx).abs().iloc[-1]
    assert (diff > 1e-6).all(), \
        "ADX matches the SMA-smoothed form - Wilder smoothing was lost"


def test_adx_is_computed_against_a_hand_worked_wilder_series() -> None:
    """Pin the arithmetic to an independent recomputation.

    Written out longhand rather than calling the module's own helpers, so a
    change inside adx_wide cannot silently redefine what "correct" means.
    """
    b = _bars(90, seed=11)
    high, low, close = b["high"], b["low"], b["close"]
    p = 14
    a = 1.0 / p

    prev = close.shift(1)
    tr = np.fmax(np.fmax(high - low, (high - prev).abs()), (low - prev).abs())
    up, dn = high.diff(), -low.diff()
    known = up.notna() & dn.notna()
    pdm = up.where((up > dn) & (up > 0), 0.0).where(known)
    mdm = dn.where((dn > up) & (dn > 0), 0.0).where(known)
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100.0 * pdm.ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100.0 * mdm.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi)
    want = dx.ewm(alpha=a, adjust=False).mean()

    got = adx_wide(high, low, close, period=p)["adx"]
    pd.testing.assert_frame_equal(got, want, check_freq=False, atol=1e-12)


# -- Wilder's outside-bar rule ----------------------------------------------

def test_an_outside_bar_signals_neither_direction() -> None:
    """Higher high AND lower low moved both ways, so it signals neither.

    The naive `max(up, 0)` / `max(down, 0)` pair reports both directions at
    once on precisely the bars where direction is least clear.
    """
    high = pd.DataFrame({"X": [100.0, 105.0]})
    low = pd.DataFrame({"X": [99.0, 94.0]})       # outside bar: +5 up, +5 down
    plus, minus = directional_movement(high, low)
    assert plus.iloc[1, 0] == 0.0
    assert minus.iloc[1, 0] == 0.0


def test_only_one_direction_is_ever_non_zero() -> None:
    b = _bars(150, seed=3)
    plus, minus = directional_movement(b["high"], b["low"])
    both = (plus > 0) & (minus > 0)
    assert not both.to_numpy().any(), \
        "a bar reported movement in both directions"


def test_the_first_bar_has_no_directional_movement() -> None:
    """NaN, not zero. There is nothing to measure yet, and calling that zero
    is an invented reading rather than a missing one."""
    b = _bars(30)
    plus, minus = directional_movement(b["high"], b["low"])
    assert plus.iloc[0].isna().all()
    assert minus.iloc[0].isna().all()


# -- what ADX does and does not say -----------------------------------------

def test_a_strong_trend_reads_higher_than_chop() -> None:
    """The one behavioural claim worth asserting, on constructed data where
    the answer is known rather than on random bars."""
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    up = pd.Series(np.linspace(100, 200, n), index=idx)
    flat = pd.Series(150 + np.tile([0.0, 1.0], n // 2), index=idx)
    close = pd.DataFrame({"UP": up, "FLAT": flat})
    out = adx_wide(close + 0.5, close - 0.5, close)
    assert out["adx"]["UP"].iloc[-1] > out["adx"]["FLAT"].iloc[-1]


def test_a_downtrend_also_reads_high_because_adx_is_directionless() -> None:
    """A high ADX means "trending hard", not "going up".

    Using ADX alone to pick a side is a category error, and this is the test
    that documents it.
    """
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    dn = pd.DataFrame({"DOWN": np.linspace(200, 100, n)}, index=idx)
    out = adx_wide(dn + 0.5, dn - 0.5, dn)
    assert out["adx"]["DOWN"].iloc[-1] > 20
    assert out["minus_di"]["DOWN"].iloc[-1] > out["plus_di"]["DOWN"].iloc[-1]


# -- degenerate inputs ------------------------------------------------------

def test_a_near_flat_grinder_pegs_adx_and_that_is_correct() -> None:
    """TRIPWIRE. ADX is a ratio, so a microscopic consistent drift is a
    PERFECT trend and must read near 100.

    This is deliberately asserted rather than suppressed. On the real
    universe (2026-09-29, 995 names) the six highest ADX readings were all
    money-market ETFs - LIQUIDCASE at adx 100.0 with atr 0.029%, against a
    universe median atr of 3.36%. Twelve names of 995 sat under a 1% ATR
    floor and those twelve held the whole top of the ranking.

    If someone later "fixes" this by clamping ADX on low-range names, this
    test fails and points them at the real fix: gate on atr_pct alongside
    adx_14. ADX is not broken; ADX alone is not a selection rule.
    """
    n = 120
    # A paisa a day on a Rs 1,000 instrument, with a one-paisa range.
    close = pd.DataFrame({"LIQUIDLIKE": 1000 + np.arange(n) * 0.01})
    out = adx_wide(close + 0.005, close - 0.005, close)
    assert out["adx"]["LIQUIDLIKE"].iloc[-1] > 90, \
        "a perfectly consistent drift must peg ADX - gate on ATR instead"
    assert out["plus_di"]["LIQUIDLIKE"].iloc[-1] > \
        out["minus_di"]["LIQUIDLIKE"].iloc[-1]


def test_a_flat_series_yields_nan_not_inf() -> None:
    """inf is the one bad value that passes isna() and still sorts first.

    A circuit-locked or untraded name has zero true range, and direction is
    undefined there rather than infinite.
    """
    flat = pd.DataFrame({"LOCKED": [500.0] * 40})
    out = adx_wide(flat, flat, flat)
    for key in ("adx", "plus_di", "minus_di"):
        vals = out[key].to_numpy()
        assert not np.isinf(vals).any(), f"{key} produced inf"


def test_a_bad_period_raises() -> None:
    b = _bars(30)
    with pytest.raises(ValueError, match="period must be"):
        adx_wide(b["high"], b["low"], b["close"], period=0)


def test_output_is_aligned_to_the_input() -> None:
    b = _bars(77)
    out = adx_wide(b["high"], b["low"], b["close"])
    for f in out.values():
        assert f.shape == b["close"].shape
        assert list(f.columns) == list(b["close"].columns)
        assert f.index.equals(b["close"].index)


# -- the comparison against upstream, recorded rather than asserted ---------

@pytest.mark.skipif(not (_UP / "backend/services/indicators.py").is_file(),
                    reason="openterminal_ui upstream absent - gitignored")
def test_we_disagree_with_upstream_exactly_where_expected() -> None:
    """Upstream is a valid reference AFTER warmup and wrong inside it.

    This asserts the disagreement rather than parity, because the two are
    genuinely different indicators: upstream SMA-smooths and backfills. If
    this test ever fails, upstream changed and the reasoning in
    directional.py's docstring needs re-checking rather than patching.
    """
    p = str(_UP)
    if p not in sys.path:
        sys.path.insert(0, p)
    mod_path = _UP / "backend/services/indicators.py"
    spec = iu.spec_from_file_location("otui_ind_cmp", mod_path)
    up = iu.module_from_spec(spec)
    sys.modules["otui_ind_cmp"] = up
    spec.loader.exec_module(up)

    b = _bars(120, seed=5)
    col = "TREND"
    theirs = up.adx(b["high"][col].to_numpy(), b["low"][col].to_numpy(),
                    b["close"][col].to_numpy(), period=14)["adx"]
    # THE BUG, demonstrated: their first bar carries a number, ours is NaN.
    assert not pd.isna(theirs[0]), \
        "upstream stopped backfilling - re-read directional.py's docstring"
    ours = adx_wide(b["high"], b["low"], b["close"])["adx"][col]
    assert pd.isna(ours.iloc[0])
