"""Watchlist and avoid: what the funnel removed, and why.

Both fields were declared on DailyPlan and NEVER POPULATED, while Stage
3's vetoes and Stage 4's rejections were computed in full and discarded
before the API. A trader seeing two approved trades out of eight
candidates could not learn which six were dropped, nor whether "dropped"
meant "the model disliked the story" or "it reports earnings on
Thursday".

The split between the two lists is the point. A name refused because
results are due is worth watching; a name refused because the reward does
not justify the risk is not the same statement, and one list would tell
the reader to forget both.
"""

from __future__ import annotations

from datetime import date

from desk.contracts.enums import RejectReason, Regime, Stance
from desk.plan.build import _shortlists, build_plan
from desk.plan.models import ScanSummary

DAY = date(2026, 9, 17)


def _scan(**kw) -> ScanSummary:
    return ScanSummary(universe_scanned=3483, survived_stage0=1548,
                       survived_stage1=502, survived_stage2=502, **kw)


def test_nothing_removed_means_both_lists_empty():
    watch, avoid = _shortlists(_scan())
    assert watch == [] and avoid == []


def test_no_scan_at_all_is_not_an_error():
    assert _shortlists(None) == ([], [])


# --- the split -----------------------------------------------------------

def test_an_event_in_the_window_is_a_watchlist_name_not_an_avoid():
    """The setup was fine; the calendar stopped it. That expires."""
    watch, avoid = _shortlists(_scan(
        rejected=[("RELIANCE.NS", RejectReason.EVENT_IN_WINDOW.value)]))
    assert [t.symbol for t in watch] == ["RELIANCE.NS"]
    assert avoid == []
    assert "event_in_window" in watch[0].rationale


def test_a_poor_reward_to_risk_is_an_avoid_not_a_watchlist_name():
    """Nothing about tomorrow changes this one at these levels."""
    watch, avoid = _shortlists(_scan(
        rejected=[("TCS.NS", RejectReason.RR_TOO_LOW.value)]))
    assert [t.symbol for t in avoid] == ["TCS.NS"]
    assert watch == []


def test_a_portfolio_cap_frees_up_so_it_is_a_watchlist_name():
    watch, avoid = _shortlists(_scan(
        rejected=[("A.NS", RejectReason.MAX_POSITIONS.value),
                  ("B.NS", RejectReason.PORTFOLIO_HEAT_CAP.value),
                  ("C.NS", RejectReason.SECTOR_CAP.value)]))
    assert {t.symbol for t in watch} == {"A.NS", "B.NS", "C.NS"}
    assert avoid == []


def test_an_illiquid_name_is_an_avoid():
    watch, avoid = _shortlists(_scan(
        rejected=[("PENNY.NS", RejectReason.ILLIQUID.value)]))
    assert [t.symbol for t in avoid] == ["PENNY.NS"]


def test_a_risk_off_regime_is_temporary():
    watch, _ = _shortlists(_scan(
        rejected=[("X.NS", RejectReason.MARKET_RISK_OFF.value)]))
    assert [t.symbol for t in watch] == ["X.NS"]


def test_several_reasons_at_once_watchlist_if_any_is_temporary():
    """A name blocked by an event AND a thin reward still becomes
    actionable when the event passes and the levels move - so the
    temporary reason wins. Erring toward keeping a name visible costs a
    line on a screen; erring the other way loses it."""
    watch, avoid = _shortlists(_scan(rejected=[
        ("X.NS", f"{RejectReason.RR_TOO_LOW.value}, "
                 f"{RejectReason.EVENT_IN_WINDOW.value}")]))
    assert [t.symbol for t in watch] == ["X.NS"]
    assert avoid == []


# --- Stage 3's vetoes ----------------------------------------------------

def test_a_narrative_veto_lands_in_avoid_with_its_reason():
    watch, avoid = _shortlists(_scan(
        vetoed={"ADANIENT.NS": "Sell: leverage concerns into a weak tape"}))
    assert [t.symbol for t in avoid] == ["ADANIENT.NS"]
    assert "leverage concerns" in avoid[0].rationale
    assert avoid[0].dissenting == ["scanner-stage3"], \
        "the plan must say WHICH agent disagreed"


def test_a_veto_is_marked_review_not_a_tradeable_stance():
    """These are not recommendations in either direction - they are
    names that were considered and removed."""
    _, avoid = _shortlists(_scan(vetoed={"X.NS": "Hold: no catalyst"}))
    assert avoid[0].stance is Stance.REVIEW
    assert avoid[0].confidence == 0.0


def test_vetoes_and_rejections_appear_together():
    watch, avoid = _shortlists(_scan(
        vetoed={"V.NS": "Sell: deteriorating margins"},
        rejected=[("R.NS", RejectReason.EVENT_IN_WINDOW.value),
                  ("S.NS", RejectReason.STOP_TOO_WIDE.value)]))
    assert [t.symbol for t in watch] == ["R.NS"]
    assert {t.symbol for t in avoid} == {"S.NS", "V.NS"}


# --- it actually reaches the plan ----------------------------------------

def test_build_plan_populates_both_lists():
    """They were declared and always empty - that is the whole bug."""
    plan = build_plan(as_of=DAY, calendar=None, scan=_scan(
        vetoed={"V.NS": "Hold: no edge today"},
        rejected=[("R.NS", RejectReason.EVENT_IN_WINDOW.value)]))
    assert [t.symbol for t in plan.watchlist] == ["R.NS"]
    assert [t.symbol for t in plan.avoid] == ["V.NS"]


def test_a_no_trade_day_can_still_explain_itself():
    """The most important case: nothing was approved, and the reader
    needs to know it was considered and refused rather than never
    looked at."""
    plan = build_plan(as_of=DAY, calendar=None, scan=_scan(
        considered=3,
        no_trade_reason="3 candidates reached the risk gate and all failed",
        rejected=[("A.NS", RejectReason.RR_TOO_LOW.value),
                  ("B.NS", RejectReason.EVENT_IN_WINDOW.value),
                  ("C.NS", RejectReason.ILLIQUID.value)]))
    assert plan.is_no_trade
    assert len(plan.watchlist) + len(plan.avoid) == 3
    assert [t.symbol for t in plan.watchlist] == ["B.NS"]


def test_the_funnel_forwards_what_it_removed():
    """Wired, not merely available."""
    import inspect
    from desk.api import main as api
    src = inspect.getsource(api._run_funnel)
    assert "vetoed=dict(stage3.vetoed)" in src
    assert "stage4.rejected" in src
