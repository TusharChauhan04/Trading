"""The cost model, and the one charge that does not scale with position size.

Every line in `CostModel` is a percentage of turnover except one: the DP
charge, a flat fee per scrip on the sell side. That made the round trip a
constant 0.422% at any size - which is correct for a large position and badly
wrong for a small one, where a single flat fee can exceed the entire measured
edge. These tests pin that behaviour, because a backtest that omits it reports
a strategy that does not exist.
"""

from __future__ import annotations

import pytest

# -- the flat DP charge, which decides whether a small account can trade ----

def test_the_dp_charge_dominates_a_small_position() -> None:
    """THE finding. Every other charge scales with turnover, so the round trip
    was a constant 0.422% at any size. The DP charge is flat, so at Rs 500 it
    IS the cost: 3.19 percentage points of a 3.61% round trip."""
    from desk.backtest.costs import CostModel

    c = CostModel()
    assert c.round_trip_pct(500) == pytest.approx(3.608, abs=0.01)
    assert c.round_trip_pct(10_000) == pytest.approx(0.582, abs=0.01)
    assert c.round_trip_pct(100_000) == pytest.approx(0.438, abs=0.01)
    # Seven times the cost at Rs 500 versus Rs 1 lakh, for the same trade.
    assert c.round_trip_pct(500) > 7 * c.round_trip_pct(100_000)


def test_omitting_the_position_gives_the_old_percentage_only_figure() -> None:
    """Every walk-forward before 2026-10-06 used this number, so it must stay
    reachable - and callers who care about small accounts must pass a size."""
    from desk.backtest.costs import CostModel

    assert CostModel().round_trip_pct() == pytest.approx(0.422, abs=0.01)


def test_break_even_position_is_about_two_thousand_rupees() -> None:
    """Below this a single flat fee eats the measured +0.08R edge before the
    market does anything."""
    from desk.backtest.costs import CostModel

    be = CostModel().breakeven_position(stop_pct=15.0, gross_edge_r=0.08)
    assert be == pytest.approx(2048, rel=0.02)


def test_a_tight_stop_cannot_be_rescued_by_any_position_size() -> None:
    """At a 4% stop the PERCENTAGE charges alone (0.422%) exceed the edge
    (0.08 x 4% = 0.32%), so there is no size that works and None says so
    rather than returning a misleading number."""
    from desk.backtest.costs import CostModel

    assert CostModel().breakeven_position(stop_pct=4.0, gross_edge_r=0.08) is None


def test_cost_r_rises_as_the_position_shrinks() -> None:
    """cost_R = round_trip / stop. Quantity cancels for the percentage lines,
    which is why only stop WIDTH used to matter - the DP charge is where
    position size re-enters, and why a small account cannot widen its way
    out."""
    from desk.backtest.costs import CostModel

    c = CostModel()
    small = c.cost_r(stop_pct=15.0, position_value=500)
    large = c.cost_r(stop_pct=15.0, position_value=100_000)
    assert small > 8 * large
    assert large == pytest.approx(0.0292, abs=0.002)


def test_cost_r_refuses_a_non_positive_stop() -> None:
    from desk.backtest.costs import CostModel

    with pytest.raises(ValueError, match="stop_pct must be positive"):
        CostModel().cost_r(stop_pct=0.0, position_value=10_000)


def test_zero_dp_charge_restores_the_scale_free_behaviour() -> None:
    """An account where DP charges genuinely do not apply."""
    from desk.backtest.costs import CostModel

    c = CostModel(dp_charge_rupees=0.0)
    assert c.round_trip_pct(500) == pytest.approx(c.round_trip_pct(100_000))
