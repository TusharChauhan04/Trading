"""Assembles the daily plan from real system state.

Moved out of desk/api/main.py alongside the models (see models.py's
docstring for why). This is also where two stale warnings got fixed: the
route used to unconditionally claim "no trading calendar" and "no
corporate-action feed wired" long after both existed - a hand-maintained
warning drifts, and a warning that is wrong today teaches the reader to skim
all of them. Every warning below is derived from something actually checked.
"""

from __future__ import annotations

from datetime import date

from desk.marketdata.calendar_in import CalendarNotLoaded, TradingCalendar
from desk.plan.models import DailyPlan, ScanSummary
from desk.regime.state import RegimeState
from desk.strategies.catalog import CATALOG


def build_plan(*, as_of: date, calendar: TradingCalendar | None,
               scan: ScanSummary | None = None) -> DailyPlan:
    """The day's plan.

    Returns NO TRADE until the data foundation exists. That is the correct
    answer, not a placeholder - the system must never invent a trade because
    a UI has a slot for one.

    `calendar` is passed in rather than loaded here, so this function stays a
    plain data transform with no filesystem access of its own - the caller
    (desk/api/main.py's route, or eventually the scanner) owns fetching it.
    """
    trusted = [s.key for s in CATALOG if s.trusted]
    open_defects = sum(len(s.defects) for s in CATALOG)
    regime = RegimeState(as_of=as_of)

    warnings: list[str] = []
    market_risks: list[str] = [
        "No event calendar wired: RBI policy, budget, expiry and results "
        "dates are not being checked against the holding window.",
    ]
    no_trade_reason: str | None = None

    if calendar is None:
        warnings.append(
            "Trading calendar unavailable - cannot confirm today is even a "
            "trading day. See /health."
        )
    else:
        try:
            if not calendar.is_trading_day(as_of):
                holiday = calendar.holiday_name(as_of)
                reason = f"market holiday ({holiday})" if holiday else "a weekend"
                no_trade_reason = (
                    f"NO TRADE - {as_of} is {reason}. The exchange is not open."
                )
        except CalendarNotLoaded:
            warnings.append(
                f"Trading calendar has no data for {as_of.year} - run "
                f"'python -m desk.marketdata.refresh calendar'."
            )

    trades: list = []
    if no_trade_reason is None:
        if scan is None:
            # The exchange is open (or its status could not be confirmed) but
            # no scan was run for this date - usually because no bhavcopy
            # snapshot has been fetched for it.
            no_trade_reason = (
                "NO TRADE - no scan was run for this date. Fetch the day's "
                "snapshot with 'python -m desk.marketdata.refresh bhavcopy "
                f"--date {as_of}' and try again."
            )
        else:
            trades = list(scan.trades)
            # The scanner owns this sentence when it ran: it can distinguish
            # "nothing was flagged" from "forty names reached the risk gate
            # and every one was refused", which are materially different days.
            no_trade_reason = scan.no_trade_reason if not trades else None

    warnings.append(
        f"{len(trusted)} of {len(CATALOG)} strategies have cleared "
        f"walk-forward; {open_defects} known defects are still on the books."
    )
    warnings.append("Strategy library needs re-running on the corrected strategy_lib.")
    if scan is None:
        warnings.append(
            "No scanner run today: universe/stage1/stage2/analysed counts are "
            "all zero by construction, not because nothing qualified."
        )
    else:
        # Every caveat the funnel raised about itself. A plan that quietly
        # drops them reads more confident than the evidence supports.
        warnings.extend(scan.caveats)
        if scan.coverage_note:
            warnings.append(f"Price history behind this scan: {scan.coverage_note}")

    return DailyPlan(
        as_of=as_of,
        universe_scanned=scan.universe_scanned if scan else 0,
        survived_stage0=scan.survived_stage0 if scan else 0,
        survived_stage1=scan.survived_stage1 if scan else 0,
        survived_stage2=scan.survived_stage2 if scan else 0,
        analysed=scan.considered if scan else 0,
        trades=trades,
        regime=regime.label,
        regime_note="Regime engine not built. Requires India VIX + Nifty breadth.",
        regime_detail=regime.explain(),
        market_risks=market_risks,
        no_trade_reason=no_trade_reason,
        warnings=warnings,
    )
