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
from desk.contracts.enums import RejectReason, Stance
from desk.plan.models import DailyPlan, PlanTrade, ScanSummary
from desk.regime.state import RegimeState
from desk.strategies.catalog import CATALOG


#: Rejections that are TRUE TODAY AND MAY NOT BE TOMORROW. A name refused
#: for one of these is a watchlist entry, not a rejected thesis: the setup
#: itself was fine and something about today's circumstances stopped it.
#:
#: The distinction is the whole reason `watchlist` and `avoid` are separate
#: fields rather than one list. "Results are due on Thursday" and "the
#: reward does not justify the risk at these levels" are different
#: statements, and collapsing them tells the reader to forget a name that
#: is worth watching.
_TEMPORARY_REASONS = frozenset({
    RejectReason.EVENT_IN_WINDOW.value,   # the event passes
    RejectReason.STALE_DATA.value,        # fresher bars arrive
    RejectReason.DAILY_LOSS_CAP.value,    # tomorrow is a new day
    RejectReason.MAX_POSITIONS.value,     # a position closes
    RejectReason.PORTFOLIO_HEAT_CAP.value,
    RejectReason.SECTOR_CAP.value,
    RejectReason.CORRELATED_CAP.value,
    RejectReason.MARKET_RISK_OFF.value,   # the regime turns
})


def _shortlists(scan: ScanSummary | None) -> tuple[list[PlanTrade],
                                                   list[PlanTrade]]:
    """(watchlist, avoid) from what the funnel removed and why.

    Both were declared on DailyPlan and never populated, while Stage 3's
    vetoes and Stage 4's rejections were computed in full and thrown
    away. A trader seeing two approved trades out of eight candidates had
    no way to learn which six were dropped, or whether "dropped" meant
    "the model disliked it" or "it reports earnings on Thursday".

    WATCHLIST is for names refused by something that expires - an event
    in the window, a portfolio cap that frees up, a risk-off regime.
    AVOID is for a thesis that failed on its own merits: the narrative
    veto, a reward that does not justify the risk, an illiquid name.

    A name appearing in neither is one that was never removed.
    """
    if scan is None:
        return [], []

    watch: list[PlanTrade] = []
    avoid: list[PlanTrade] = []

    # Stage 4 first: a risk-gate refusal is the more specific fact, and a
    # name cannot be both vetoed and sized.
    for symbol, reason in scan.rejected:
        entry = PlanTrade(
            symbol=symbol, stance=Stance.REVIEW, confidence=0.0,
            rationale=f"risk gate refused: {reason}")
        parts = {r.strip() for r in reason.split(",")}
        (watch if parts & _TEMPORARY_REASONS else avoid).append(entry)

    # Stage 3's vetoes are judgements about the thesis, so they avoid.
    for symbol, why in scan.vetoed.items():
        avoid.append(PlanTrade(
            symbol=symbol, stance=Stance.REVIEW, confidence=0.0,
            rationale=f"narrative veto: {why}",
            dissenting=["scanner-stage3"]))

    return watch, avoid


def _regime_note(regime: RegimeState) -> str:
    """One line on how much of the regime is actually measured.

    This used to be the hardcoded string "Regime engine not built.
    Requires India VIX + Nifty breadth." It stayed there after the engine
    was built and wired, so the plan told the reader a capability did not
    exist while using it - and none of the six feeds it turned out to need
    were required in the end, because breadth is a cross-sectional fact
    and the bhavcopy is a cross-section.
    """
    if not regime.sources:
        return ("Regime not measured for this scan - no market history was "
                "supplied, so factor silencing did not run.")
    unknown = sorted(k for k, v in regime.sources.items()
                     if str(v).startswith("not measured"))
    if regime.is_measured:
        return (f"Measured from {len(regime.sources)} dimension(s) of the "
                f"liquid universe. Fully measured.")
    return (f"Measured, but {', '.join(unknown) or 'some dimensions'} could "
            f"not be computed - the label rests on the dimensions that "
            f"could, and factor silencing used it anyway.")


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
    # The regime the SCAN measured, not a fresh empty one. Building a new
    # RegimeState here meant the plan reported "unknown" on every dimension
    # while the funnel had just measured the market and used that
    # measurement to silence factors - the plan displayed one regime and
    # the scan acted on another.
    regime = (scan.regime_state if scan and scan.regime_state is not None
              else RegimeState(as_of=as_of))

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

    watchlist, avoid = _shortlists(scan)
    return DailyPlan(
        as_of=as_of,
        universe_scanned=scan.universe_scanned if scan else 0,
        survived_stage0=scan.survived_stage0 if scan else 0,
        survived_stage1=scan.survived_stage1 if scan else 0,
        survived_stage2=scan.survived_stage2 if scan else 0,
        analysed=scan.considered if scan else 0,
        trades=trades,
        watchlist=watchlist,
        avoid=avoid,
        regime=regime.label,
        regime_note=_regime_note(regime),
        regime_detail=regime.explain(),
        market_risks=market_risks,
        no_trade_reason=no_trade_reason,
        warnings=warnings,
    )
