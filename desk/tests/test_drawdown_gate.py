"""The peak-to-trough drawdown breaker, and why the daily cap is not one.

`max_daily_loss_pct` reads only `realised_pnl_today`, so it resets every
morning. Losing 2% a day for ten sessions is an 18% drawdown that never trips
a 3% daily cap. That is the hole this gate fills.

The cap has NO default on purpose. From RiskConfig.drawdown_outlook on the
donchian 12%/20-session sequence, a 20% cap at the default 1% risk fires about
one run in three while the strategy behaves exactly as measured - a coin flip
that halts trading during the ordinary drawdown before a recovery. The same cap
at 0.5% risk fires 2.5% of the time. The cap and risk_pct are one decision.
"""

from __future__ import annotations

import pytest

from desk.contracts.enums import RejectReason
from desk.risk.engine import Portfolio, Position, RiskConfig, size_position


def _sized(cfg: RiskConfig, pf: Portfolio):
    return size_position(symbol="TEST", entry=100.0, stop=90.0, target=120.0,
                         cfg=cfg, portfolio=pf)


def test_a_drawdown_past_the_cap_stops_new_positions() -> None:
    cfg = RiskConfig(capital=1_000_000.0, max_drawdown_pct=15.0)
    pf = Portfolio(equity=820_000.0, peak_equity=1_000_000.0)
    out = _sized(cfg, pf)
    assert not out.approved
    assert RejectReason.DRAWDOWN_CAP in out.reasons
    assert any("18.0%" in n for n in out.notes), out.notes


def test_a_drawdown_inside_the_cap_is_allowed() -> None:
    cfg = RiskConfig(capital=1_000_000.0, max_drawdown_pct=15.0)
    pf = Portfolio(equity=900_000.0, peak_equity=1_000_000.0)
    out = _sized(cfg, pf)
    assert out.approved, (out.reasons, out.notes)


def test_the_slow_bleed_the_daily_cap_cannot_see() -> None:
    """THE reason this exists. Ten sessions of -2% is -18% from peak, and the
    daily cap sees only today's realised P&L, which is well inside 3%."""
    cfg = RiskConfig(capital=1_000_000.0, max_daily_loss_pct=3.0,
                     max_drawdown_pct=15.0)
    bleeding = Portfolio(equity=820_000.0, peak_equity=1_000_000.0,
                         realised_pnl_today=-20_000.0)   # -2%, inside the cap
    # The daily cap alone would approve this.
    no_dd = RiskConfig(capital=1_000_000.0, max_daily_loss_pct=3.0)
    assert _sized(no_dd, bleeding).approved, (
        "the daily cap rejected a 2% day, so this fixture cannot show the gap")
    # With the drawdown cap it is refused.
    out = _sized(cfg, bleeding)
    assert not out.approved
    assert RejectReason.DRAWDOWN_CAP in out.reasons


# -- the third state, which is the house rule ------------------------------

def test_no_cap_configured_is_recorded_as_skipped_not_passed() -> None:
    cfg = RiskConfig(capital=1_000_000.0)
    out = _sized(cfg, Portfolio(equity=500_000.0, peak_equity=1_000_000.0))
    assert out.approved
    assert any("drawdown" in c for c in out.checks_skipped), out.checks_skipped


def test_a_cap_without_equity_state_is_skipped_not_passed() -> None:
    """A configured cap the engine cannot evaluate must say so. Treating a
    missing equity figure as "no drawdown" would be the exact
    missing-becomes-meaningful trap, at the one gate meant to stop a losing
    run."""
    cfg = RiskConfig(capital=1_000_000.0, max_drawdown_pct=15.0)
    out = _sized(cfg, Portfolio())
    assert out.approved
    assert any("equity/peak_equity" in c for c in out.checks_skipped), (
        out.checks_skipped)


def test_a_missing_figure_gives_no_drawdown_rather_than_zero() -> None:
    """0.0 would read as "at an equity high", which is the opposite of
    "unknown"."""
    assert Portfolio().drawdown_pct is None
    assert Portfolio(equity=900_000.0).drawdown_pct is None
    assert Portfolio(peak_equity=1_000_000.0).drawdown_pct is None
    assert Portfolio(equity=900_000.0, peak_equity=0.0).drawdown_pct is None


def test_equity_above_peak_is_zero_not_negative() -> None:
    """A caller whose peak is stale must not get a negative drawdown, which
    would compare as safely below any cap and read as a bug when printed."""
    pf = Portfolio(equity=1_100_000.0, peak_equity=1_000_000.0)
    assert pf.drawdown_pct == 0.0


def test_the_cap_has_no_default() -> None:
    """Pinned: the measured table in the field docstring shows a 20% cap at 1%
    risk fires one run in three. Any default would be inventing the pairing
    between this number and risk_pct."""
    assert RiskConfig(capital=1_000_000.0).max_drawdown_pct is None


def test_the_reason_is_distinct_from_the_daily_cap() -> None:
    """Two different failures must not share a name - a trader reading
    daily_loss_cap would look at today's trades for a cause that is months
    old."""
    assert RejectReason.DRAWDOWN_CAP != RejectReason.DAILY_LOSS_CAP
    assert RejectReason.DRAWDOWN_CAP.value == "drawdown_cap"
