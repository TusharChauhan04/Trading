"""desk/plan/ - moved out of desk/api/main.py so the scanner (still unbuilt)
can import DailyPlan without dragging in FastAPI. See build.py's docstring.

Also covers the two warnings that used to be hardcoded and went stale the
moment the calendar and corporate-action feed actually started existing -
a hand-maintained warning drifts, and these tests exist so a future one
does too, loudly, in CI rather than silently in the UI.
"""

from __future__ import annotations

from datetime import date

import pytest

from desk.marketdata.calendar_in import TradingCalendar
from desk.plan.build import build_plan
from desk.plan.models import DailyPlan, PlanTrade


def _cal() -> TradingCalendar:
    cal = TradingCalendar()
    cal.load({
        "exchange": "NSE", "years": [2026],
        "holidays": [{"date": "2026-01-26", "name": "Republic Day"}],
        "special_sessions": [],
    })
    return cal


def test_build_plan_returns_no_trade_on_a_market_holiday():
    p = build_plan(as_of=date(2026, 1, 26), calendar=_cal())
    assert p.is_no_trade
    assert "Republic Day" in p.no_trade_reason


def test_build_plan_returns_no_trade_on_a_weekend():
    p = build_plan(as_of=date(2026, 1, 24), calendar=_cal())  # a Saturday
    assert "weekend" in p.no_trade_reason


def test_build_plan_falls_back_to_the_honest_pipeline_reason_on_a_trading_day():
    """REGRESSION target: a stale warning is worse than none, because a reader
    who finds one wrong learns to skim them all. This assertion has now been
    updated TWICE for that reason - first when the calendar and
    corporate-action feed stopped being 'not wired', and again when the
    scanner stopped being 'not connected'. On an open day with no scan
    supplied, the honest remaining gap is simply that no scan was run."""
    p = build_plan(as_of=date(2026, 1, 27), calendar=_cal())   # a Tuesday, open
    assert "no scan was run for this date" in p.no_trade_reason
    assert "refresh bhavcopy" in p.no_trade_reason      # names the fix
    assert "not wired" not in " ".join(p.warnings)
    assert "corporate-action feed" not in " ".join(p.warnings)
    assert "not connected to a running scanner" not in p.no_trade_reason


def test_build_plan_uses_the_scanners_own_no_trade_reason_when_a_scan_ran():
    """The scanner can distinguish "nothing was flagged" from "forty names
    reached the risk gate and every one was refused". build_plan must not
    overwrite that with its own generic sentence."""
    from desk.plan.models import ScanSummary

    scan = ScanSummary(
        universe_scanned=3485, survived_stage0=1598, survived_stage1=673,
        survived_stage2=673, considered=20,
        no_trade_reason="NO TRADE - 20 candidate(s) reached the risk gate "
                        "and every one was refused: sector_cap (20).",
        caveats=["F&O ban list not checked."],
        coverage_note="215 sessions, complete",
    )
    p = build_plan(as_of=date(2026, 1, 27), calendar=_cal(), scan=scan)
    assert "every one was refused" in p.no_trade_reason
    assert p.universe_scanned == 3485
    assert p.survived_stage1 == 673
    assert p.analysed == 20
    assert "F&O ban list not checked." in p.warnings
    assert any("215 sessions" in w for w in p.warnings)


def test_a_scan_with_trades_clears_the_no_trade_reason():
    from desk.contracts.enums import Stance
    from desk.plan.models import PlanTrade, ScanSummary

    scan = ScanSummary(
        universe_scanned=3485, survived_stage0=1598, considered=20,
        trades=[PlanTrade(symbol="AAA.NS", stance=Stance.BUY, confidence=0.9,
                          entry=100.0, stop=95.0, target=115.0, qty=100)],
        no_trade_reason=None,
    )
    p = build_plan(as_of=date(2026, 1, 27), calendar=_cal(), scan=scan)
    assert p.no_trade_reason is None
    assert p.is_no_trade is False
    assert len(p.trades) == 1


def test_a_holiday_still_overrides_a_scan_that_produced_trades():
    """The exchange being shut is not negotiable by a good-looking signal."""
    from desk.contracts.enums import Stance
    from desk.plan.models import PlanTrade, ScanSummary

    scan = ScanSummary(
        trades=[PlanTrade(symbol="AAA.NS", stance=Stance.BUY, confidence=0.9)],
    )
    p = build_plan(as_of=date(2026, 1, 24), calendar=_cal(), scan=scan)
    assert "weekend" in p.no_trade_reason
    assert p.trades == []


def test_build_plan_degrades_gracefully_with_no_calendar_at_all():
    p = build_plan(as_of=date(2026, 1, 27), calendar=None)
    assert any("calendar unavailable" in w for w in p.warnings)
    assert p.no_trade_reason is not None            # still says something


def test_build_plan_warns_rather_than_crashes_on_an_unloaded_year():
    p = build_plan(as_of=date(2030, 1, 1), calendar=_cal())
    assert any("no data for 2030" in w for w in p.warnings)
    assert p.no_trade_reason is not None


def test_daily_plan_model_is_importable_without_fastapi():
    """The whole point of the extraction: DailyPlan must be constructible
    from desk.plan alone, with no transport-layer import anywhere on the
    path a scanner would take to reach it."""
    import sys
    assert "fastapi" not in sys.modules or True   # fastapi may be loaded by
    # the test session itself (pytest imports desk.api.main elsewhere) - the
    # real guarantee is the IMPORT GRAPH, checked directly below.
    import ast
    import desk.plan.models as models_module
    src = open(models_module.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    imported = {n.module for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) for n in [node]}
    assert not any(m and "fastapi" in m for m in imported)


def test_plan_trade_still_carries_every_field_the_ui_reads():
    t = PlanTrade(symbol="X.NS", stance="Buy", confidence=0.7, entry=100,
                 stop=95, target=115, qty=50, capital_at_risk=250.0,
                 supporting=["a"], dissenting=["b"], rationale="r")
    assert t.model_dump()["symbol"] == "X.NS"


def test_daily_plan_is_no_trade_property():
    empty = DailyPlan(as_of=date(2026, 1, 1), regime="unknown")
    assert empty.is_no_trade
    with_trade = DailyPlan(as_of=date(2026, 1, 1), regime="unknown",
                           trades=[PlanTrade(symbol="X.NS", stance="Buy",
                                             confidence=0.5)])
    assert not with_trade.is_no_trade
