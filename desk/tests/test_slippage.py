"""Slippage must scale with order size, and must never charge nothing.

A cost model's failure mode is asymmetric. Charging too much hides a real edge,
which is recoverable - the edge is still there to find later. Charging too
little invents an edge that does not exist and sends real money after it. So
every test here that could go either way is written to fail on the optimistic
side.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from desk.backtest.slippage import SlippageModel, participation, retail_bound


# -- the floor: size-independent cost must survive any order size ------------

def test_a_tiny_order_still_pays_the_base() -> None:
    """THE most important test here.

    A model that charges nothing for a small order makes every strategy look
    profitable at retail size. The spread does not care how small the order is
    - crossing it costs the same half-spread on one share as on a thousand.
    """
    m = SlippageModel(base_bps=2.0)
    assert m.bps(1e-9) >= 2.0
    assert m.bps(0.0) >= 2.0


def test_missing_volume_data_is_not_free() -> None:
    """NaN participation means the session could not be measured, which is a
    reason for caution rather than a discount."""
    m = SlippageModel(base_bps=2.0)
    assert m.bps(float("nan")) >= 2.0
    assert m.bps(float("-inf")) >= 2.0


def test_base_is_not_zero_by_default() -> None:
    """Defaults are what most callers get, so the default must be the
    conservative choice rather than the convenient one."""
    assert SlippageModel().base_bps > 0


# -- monotonicity and shape -------------------------------------------------

def test_slippage_rises_with_participation() -> None:
    m = SlippageModel()
    vals = [m.bps(p) for p in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    for a, b in zip(vals, vals[1:]):
        assert b > a, f"slippage did not rise with size: {vals}"


def test_the_sqrt_form_is_sublinear() -> None:
    """Impact grows with the square root of participation, so doubling the
    order LESS than doubles the impact. If this ever becomes linear, large
    orders are being overcharged and small ones undercharged."""
    m = SlippageModel(base_bps=0.0, impact_coefficient_bps=35.0)
    assert m.bps(0.04) == pytest.approx(2 * m.bps(0.01))      # sqrt(4x) = 2x


def test_the_linear_form_is_strictly_more_pessimistic_at_small_size() -> None:
    """Kept precisely so a marginal result can be re-checked against the
    harsher assumption. At p < 1 the linear form charges LESS impact than
    sqrt, so the pessimism runs the other way - asserted here so the
    direction is recorded rather than assumed."""
    sq = SlippageModel(base_bps=0.0, form="sqrt")
    li = SlippageModel(base_bps=0.0, form="linear")
    assert li.bps(0.01) < sq.bps(0.01)
    assert li.bps(1.0) == pytest.approx(sq.bps(1.0))


def test_participation_is_capped_at_one() -> None:
    """You cannot take 300% of a session's volume. Without the cap the sqrt
    keeps growing and produces a number with no meaning."""
    m = SlippageModel()
    assert m.bps(3.0) == m.bps(1.0)


def test_the_cap_binds() -> None:
    m = SlippageModel(base_bps=0.0, impact_coefficient_bps=10_000.0,
                      max_bps=500.0)
    assert m.bps(1.0) == 500.0


def test_a_bad_form_raises() -> None:
    with pytest.raises(ValueError, match="sqrt or linear"):
        SlippageModel(form="magic")


def test_negative_coefficients_raise() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        SlippageModel(base_bps=-1.0)


# -- participation ----------------------------------------------------------

def test_participation_is_a_fraction_of_turnover() -> None:
    assert participation(1_000_000, 100_000_000) == pytest.approx(0.01)


def test_zero_turnover_is_nan_not_zero() -> None:
    """A symbol that did not trade cannot be sized against. NaN forces the
    caller to decide; zero would silently read as a free fill."""
    assert math.isnan(participation(1000, 0))
    assert math.isnan(participation(1000, float("nan")))


# -- round trip -------------------------------------------------------------

def test_round_trip_is_both_sides_as_a_percent() -> None:
    """costs.py works in percent of notional, so the unit must match or the
    two models silently differ by 100x."""
    m = SlippageModel(base_bps=2.0, impact_coefficient_bps=0.0)
    assert m.round_trip_pct(1e-6) == pytest.approx(0.04)     # 2bps x2 = 4bps


# -- the retail bound, on a realistic turnover distribution -----------------

def _turnover() -> pd.Series:
    """Shaped like the real post-Stage-0 universe: median ~Rs 34 crore,
    floor Rs 5 crore, a long right tail."""
    rng = np.random.default_rng(7)
    return pd.Series(5e7 * np.exp(rng.normal(1.9, 1.2, 1100)))


def test_retail_size_is_far_below_the_flat_fifteen_bps() -> None:
    """The finding that moved several preset cells from negative to positive.

    costs.py charges a flat 15bps a side and labels it a guess. At retail size
    on this universe the modelled figure is a small single digit, because
    participation is a few hundredths of one percent.
    """
    b = retail_bound(100_000, _turnover())
    assert b["median_participation_pct"] < 0.1
    assert b["median_bps"] < 6.0, \
        "retail-size slippage came out near institutional levels - check the " \
        "turnover units (bhavcopy is in LACS, not rupees)"


def test_a_bigger_position_costs_more_on_the_same_universe() -> None:
    t = _turnover()
    small = retail_bound(100_000, t)["median_bps"]
    large = retail_bound(10_000_000, t)["median_bps"]
    assert large > small


def test_the_pessimistic_tail_is_reported_not_just_the_median() -> None:
    """The mean of a participation distribution is dominated by the thinnest
    names; the median is what a typical fill looks like. Both are needed, so
    both are returned."""
    b = retail_bound(1_000_000, _turnover())
    assert b["p95_bps"] >= b["median_bps"]
    assert b["p95_participation_pct"] >= b["median_participation_pct"]


def test_an_empty_universe_returns_nothing_rather_than_zero() -> None:
    assert retail_bound(100_000, pd.Series([], dtype=float)) == {}
    assert retail_bound(100_000, pd.Series([0.0, -1.0])) == {}
