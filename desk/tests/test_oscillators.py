"""The eleven borrowed indicators, and the warmup leak they must not have.

THE headline test is the no-backfill one. Upstream's `_fillna_local` is
`ffill().bfill()`, and `bfill` fills EARLIER NaNs from LATER values - so on a
14-period indicator, bars 0-13 come back carrying the value computed at bar 14.
Nine of its thirteen indicators do this. The series looks complete, which is
exactly why nothing notices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.indicators.oscillators import (
    aroon, awesome_oscillator, cci, ichimoku, momentum, parabolic_sar, roc,
    std_deviation, stochastic, ultimate_oscillator, williams_r,
)


def _ohlc(n: int = 160, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    c = pd.DataFrame(
        {"AAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n))),
         "BBB": 400 * np.exp(np.cumsum(rng.normal(0, 0.020, n)))}, index=idx)
    h = c * (1 + np.abs(rng.normal(0, 0.006, (n, 2))))
    l = c * (1 - np.abs(rng.normal(0, 0.006, (n, 2))))
    return h, l, c


# -- the leak ---------------------------------------------------------------

def test_no_indicator_produces_a_value_before_its_window_is_full() -> None:
    """THE test. A number inside the warmup can only have come from the
    future."""
    h, l, c = _ohlc()
    k, d = stochastic(h, l, c, k_period=14, d_period=3)
    checks = {
        "stochastic_k": (k, 14),
        "stochastic_d": (d, 14 + 3 - 1),
        "williams_r": (williams_r(h, l, c, period=14), 14),
        "cci": (cci(h, l, c, period=20), 20),
        "roc": (roc(c, period=12), 12),
        "momentum": (momentum(c, period=10), 10),
        "std_deviation": (std_deviation(c, period=20), 21),
        "awesome": (awesome_oscillator(h, l, fast=5, slow=34), 34),
        "ultimate": (ultimate_oscillator(h, l, c, long=28), 28),
        "aroon_up": (aroon(h, l, period=25)[0], 25),
    }
    for name, (frame, warm) in checks.items():
        head = frame.iloc[:warm - 1]
        assert head.isna().all().all(), (
            f"{name} produced a value inside its {warm}-bar warmup - the only "
            f"place that number could come from is a later bar")
        assert frame.iloc[warm:].notna().any().any(), (
            f"{name} never produced a value at all")


def test_ichimoku_lines_start_only_when_their_own_window_is_full() -> None:
    h, l, _ = _ohlc()
    lines = ichimoku(h, l, tenkan=9, kijun=26, senkou_b=52)
    for name, warm in (("tenkan_sen", 9), ("kijun_sen", 26),
                       ("senkou_span_b", 52)):
        assert lines[name].iloc[:warm - 1].isna().all().all(), name


def test_the_senkou_spans_are_not_shifted_into_the_future() -> None:
    """A chart draws the cloud `kijun` bars ahead. Returning it pre-shifted
    would hand a caller a value at bar t that was computed at t-26, or worse
    one computed from bars after t."""
    h, l, _ = _ohlc()
    lines = ichimoku(h, l, tenkan=9, kijun=26)
    a = lines["senkou_span_a"]
    expect = (lines["tenkan_sen"] + lines["kijun_sen"]) / 2.0
    pd.testing.assert_frame_equal(a, expect)


def test_truncating_the_future_does_not_change_the_past() -> None:
    """The general point-in-time guard, across every function at once."""
    h, l, c = _ohlc(200)
    cut = 150
    for name, full, short in (
        ("stoch", stochastic(h, l, c)[0], stochastic(h[:cut], l[:cut], c[:cut])[0]),
        ("willr", williams_r(h, l, c), williams_r(h[:cut], l[:cut], c[:cut])),
        ("cci", cci(h, l, c), cci(h[:cut], l[:cut], c[:cut])),
        ("roc", roc(c), roc(c[:cut])),
        ("uo", ultimate_oscillator(h, l, c),
         ultimate_oscillator(h[:cut], l[:cut], c[:cut])),
        ("sar", parabolic_sar(h, l), parabolic_sar(h[:cut], l[:cut])),
    ):
        pd.testing.assert_frame_equal(
            full.iloc[:cut], short, obj=f"{name} changed when the tail was cut")


# -- the refusals -----------------------------------------------------------

def test_a_flat_range_is_nan_not_zero() -> None:
    """Upstream's _safe_float turns the zero-denominator case into 0.0, which
    for a stochastic reads as "at the very bottom of its range" - the
    strongest oversold reading available. A circuit-locked stock would top an
    oversold screen."""
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    flat = pd.DataFrame({"FROZEN": [250.0] * n}, index=idx)
    k, _ = stochastic(flat, flat, flat, k_period=14)
    assert k["FROZEN"].iloc[20:].isna().all()
    w = williams_r(flat, flat, flat, period=14)
    assert w["FROZEN"].iloc[20:].isna().all()
    assert not (k.fillna(-1) == 0).any().any()


def test_roc_refuses_a_zero_prior_price_rather_than_returning_inf() -> None:
    """inf survives isna(), so it passes any downstream check that only looks
    for NaN - and then ranks first on a momentum sort."""
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    c = pd.DataFrame({"X": np.linspace(100, 140, n)}, index=idx)
    c.iloc[5, 0] = 0.0                      # a bad print
    out = roc(c, period=12)
    assert np.isfinite(out["X"].dropna()).all()
    assert pd.isna(out["X"].iloc[17])       # 5 + 12


def test_cci_uses_mean_absolute_deviation_not_standard_deviation() -> None:
    """Lambert's 0.015 constant is scaled for MAD. Using sd changes the scale
    by about 1.25x and moves the conventional +/-100 thresholds."""
    h, l, c = _ohlc(120)
    period = 20
    tp = (h + l + c) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    expect = (tp - sma) / (0.015 * mad)
    pd.testing.assert_frame_equal(cci(h, l, c, period=period), expect)
    # And it is genuinely different from the sd version.
    sd_version = (tp - sma) / (0.015 * tp.rolling(period,
                                                  min_periods=period).std())
    assert not np.allclose(cci(h, l, c, period=period).dropna().to_numpy(),
                           sd_version.dropna().to_numpy())


def test_std_deviation_is_on_returns_and_not_annualised_by_default() -> None:
    """Upstream's std_dev is the sd of the PRICE, which scales with price and
    so ranks expensive stocks as volatile. And its annualised figure comes
    from the same call, inviting a 20-bar estimate to be read as an annual
    property."""
    _, _, c = _ohlc(120)
    plain = std_deviation(c, period=20)
    ann = std_deviation(c, period=20, annualise=True)
    # On returns: both names are ~0.015-0.02 daily despite a 4x price gap.
    assert plain.iloc[-1].max() < 0.1
    assert np.allclose((ann / plain).dropna().to_numpy(), np.sqrt(252))


# -- Parabolic SAR, the sequential one -------------------------------------

def test_sar_does_not_report_a_value_on_the_first_bar() -> None:
    """A stop level from a single bar is not a stop level."""
    h, l, _ = _ohlc(80)
    s = parabolic_sar(h, l)
    assert s.iloc[0].isna().all()
    assert s.iloc[5:].notna().any().any()


def test_sar_stays_inside_the_price_range_it_trails() -> None:
    """A parabolic stop that sits outside the recent range would fire
    immediately or never."""
    h, l, _ = _ohlc(200)
    s = parabolic_sar(h, l)
    for col in s.columns:
        v = s[col].dropna()
        assert v.min() >= l[col].min() * 0.5
        assert v.max() <= h[col].max() * 1.5


def test_aroon_reads_100_when_the_extreme_is_the_current_bar() -> None:
    """Pins the direction. The inverse convention (bars-since) would make a
    fresh high read 0, silently inverting any rule built on it."""
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rising = pd.DataFrame({"UP": np.linspace(100, 200, n)}, index=idx)
    up, down = aroon(rising, rising, period=25)
    assert float(up["UP"].iloc[-1]) == pytest.approx(100.0)
    assert float(down["UP"].iloc[-1]) == pytest.approx(0.0)


def test_every_function_preserves_shape_and_columns() -> None:
    h, l, c = _ohlc(120)
    outs = [stochastic(h, l, c)[0], stochastic(h, l, c)[1],
            williams_r(h, l, c), cci(h, l, c), roc(c), momentum(c),
            std_deviation(c), awesome_oscillator(h, l),
            ultimate_oscillator(h, l, c), aroon(h, l)[0], aroon(h, l)[1],
            parabolic_sar(h, l)]
    outs.extend(ichimoku(h, l).values())
    for o in outs:
        assert o.shape == c.shape
        assert list(o.columns) == list(c.columns)
        assert o.index.equals(c.index)
