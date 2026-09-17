"""The decision journal.

One property dominates: A RECORDED DECISION IS NEVER OVERWRITTEN. Everything
else here is in service of that, because a journal that can be rewritten
proves nothing - and the rewrite is almost never malicious. A scheduled job
runs twice, someone re-runs the morning scan after lunch with more data, a
bug reprocesses last week. Each silently replaces what the desk actually
decided with what it would decide now, knowing more.

The second property is that NO TRADE is a record rather than an absence. A
journal holding only trades cannot say how often the system correctly stayed
out, and the surviving sample is every day it chose to act - exactly the
population that flatters a bad strategy.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from desk.journal import (
    Decision, ExitReason, JournalError, JournalStore, Outcome, TradeRecord,
)
from desk.journal.models import IST
from desk.journal.store import decision_from_plan

DAY = date(2026, 9, 17)
NOON = datetime(2026, 9, 17, 9, 0, tzinfo=IST)


def _trade(symbol="RIR.NS", entry=200.0, stop=180.0, target=250.0, qty=100):
    return TradeRecord(symbol=symbol, stance="Buy", entry=entry, stop=stop,
                       target=target, qty=qty, capital_at_risk=2000.0,
                       rationale="momentum + volume")


def _decision(day=DAY, trades=None, **kw):
    return Decision(as_of=day, recorded_at=NOON, regime="trending_up",
                    trades=tuple(trades if trades is not None else [_trade()]),
                    universe_scanned=3483, survived_stage0=1548,
                    survived_stage1=438, survived_stage2=438, considered=3,
                    **kw)


# --- the rule ------------------------------------------------------------

def test_a_recorded_decision_cannot_be_overwritten(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    with pytest.raises(JournalError, match="already recorded"):
        st.record(_decision())


def test_the_refusal_names_the_existing_record(tmp_path):
    """So the operator can check what is already there before deciding
    whether an amendment is even warranted."""
    st = JournalStore(tmp_path)
    first = _decision()
    st.record(first)
    with pytest.raises(JournalError) as exc:
        st.record(_decision(trades=[_trade(qty=999)]))
    assert first.digest() in str(exc.value)
    assert "amend()" in str(exc.value)


def test_a_refused_write_leaves_the_original_untouched(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    with pytest.raises(JournalError):
        st.record(_decision(trades=[_trade(qty=999)]))
    assert st.latest(DAY).trades[0].qty == 100


def test_two_different_days_do_not_collide(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision(day=date(2026, 9, 16)))
    st.record(_decision(day=DAY))
    assert st.days() == [date(2026, 9, 16), DAY]


# --- amendments add, they do not replace ---------------------------------

def test_an_amendment_keeps_the_original_on_disk(tmp_path):
    st = JournalStore(tmp_path)
    original = _decision()
    st.record(original)
    st.amend(_decision(trades=[_trade(qty=250)]), reason="qty was wrong")

    versions = st.history(DAY)
    assert len(versions) == 2
    assert versions[0].trades[0].qty == 100, "the original must survive"
    assert versions[1].trades[0].qty == 250


def test_an_amendment_points_at_what_it_corrects(tmp_path):
    st = JournalStore(tmp_path)
    original = _decision()
    st.record(original)
    st.amend(_decision(trades=[_trade(qty=250)]), reason="qty was wrong")
    assert st.latest(DAY).amends == original.digest()


def test_an_amendment_must_state_a_reason(tmp_path):
    """Without one it is indistinguishable from an overwrite six months on."""
    st = JournalStore(tmp_path)
    st.record(_decision())
    with pytest.raises(JournalError, match="must say why"):
        st.amend(_decision(), reason="   ")


def test_the_reason_is_stored_in_the_record(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    st.amend(_decision(), reason="typo in the rationale")
    assert "typo in the rationale" in st.latest(DAY).note
    assert "AMENDMENT" in st.latest(DAY).note


def test_amending_nothing_is_refused(tmp_path):
    st = JournalStore(tmp_path)
    with pytest.raises(JournalError, match="use record"):
        st.amend(_decision(), reason="there is nothing here yet")


def test_amendments_chain(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    st.amend(_decision(trades=[_trade(qty=2)]), reason="first correction")
    st.amend(_decision(trades=[_trade(qty=3)]), reason="second correction")
    assert len(st.history(DAY)) == 3
    assert st.latest(DAY).trades[0].qty == 3


# --- NO TRADE is a record ------------------------------------------------

def test_a_no_trade_day_is_recorded(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision(trades=[],
                        no_trade_reason="every candidate failed the R:R floor"))
    got = st.latest(DAY)
    assert got.is_no_trade
    assert "R:R floor" in got.no_trade_reason
    assert DAY in st.days(), "a quiet day must still appear in the journal"


def test_a_no_trade_day_needs_no_outcomes(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision(trades=[], no_trade_reason="nothing ranked"))
    assert st.unrecorded() == []


def test_the_summary_says_no_trade_and_why(tmp_path):
    d = _decision(trades=[], no_trade_reason="all 8 refused: stale_data")
    assert "NO TRADE" in d.summary()
    assert "stale_data" in d.summary()


# --- what gets kept ------------------------------------------------------

def test_caveats_are_part_of_the_decision(tmp_path):
    """'Why did it pick that' and 'what did it not know' are the same
    question later. A record with the reasoning but not the blind spots
    will be misread."""
    st = JournalStore(tmp_path)
    st.record(_decision(caveats=("event calendar not applied",
                                 "438 of 438 candidates have no filing")))
    got = st.latest(DAY)
    assert len(got.caveats) == 2
    assert any("no filing" in c for c in got.caveats)


def test_the_funnel_counts_survive(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    got = st.latest(DAY)
    assert (got.universe_scanned, got.survived_stage0, got.considered) == (
        3483, 1548, 3)


def test_levels_round_trip_exactly(tmp_path):
    """The only way to judge later whether the stop was hit before the
    target."""
    st = JournalStore(tmp_path)
    st.record(_decision(trades=[_trade(entry=197.63, stop=173.79,
                                       target=257.22, qty=419)]))
    t = st.latest(DAY).trades[0]
    assert (t.entry, t.stop, t.target, t.qty) == (197.63, 173.79, 257.22, 419)


def test_recorded_at_is_separate_from_as_of(tmp_path):
    """A decision written days later is a reconstruction, and the gap is
    the evidence of that."""
    late = Decision(as_of=DAY, recorded_at=datetime(2026, 9, 25, 11, tzinfo=IST))
    st = JournalStore(tmp_path)
    st.record(late)
    got = st.latest(DAY)
    assert got.as_of == DAY
    assert got.recorded_at.date() == date(2026, 9, 25)


def test_reward_to_risk_is_derived_not_stored():
    t = _trade(entry=200.0, stop=180.0, target=250.0)
    assert t.risk_per_share == 20.0
    assert t.reward_to_risk == pytest.approx(2.5)


def test_a_trade_with_no_levels_has_no_ratio():
    t = TradeRecord(symbol="X.NS", stance="Buy")
    assert t.risk_per_share is None and t.reward_to_risk is None


# --- tamper evidence -----------------------------------------------------

def test_the_digest_changes_when_the_content_does():
    a = _decision()
    b = _decision(trades=[_trade(qty=101)])
    assert a.digest() != b.digest()


def test_the_digest_is_stable_across_a_round_trip(tmp_path):
    st = JournalStore(tmp_path)
    original = _decision()
    st.record(original)
    assert st.latest(DAY).digest() == original.digest()


def test_an_edited_file_no_longer_matches_its_digest(tmp_path):
    """Tamper EVIDENCE, not tamper-proofing: anyone with write access can
    change both. It catches the failure this project has actually had - an
    accidental rewrite or a file restored from the wrong backup."""
    st = JournalStore(tmp_path)
    original = _decision()
    st.record(original)

    path = tmp_path / "decisions" / f"{DAY.isoformat()}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["trades"][0]["qty"] = 9999
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert st.latest(DAY).digest() != original.digest()


# --- corruption is absence, never a guess --------------------------------

def test_a_corrupt_record_reads_as_missing_not_as_a_partial_plan(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    path = tmp_path / "decisions" / f"{DAY.isoformat()}.json"
    path.write_text('{"as_of": "2026-09-17", "trades": [', encoding="utf-8")
    assert st.latest(DAY) is None, "half a decision is not a decision"


def test_a_missing_day_is_none(tmp_path):
    assert JournalStore(tmp_path).latest(DAY) is None
    assert JournalStore(tmp_path).days() == []


def test_writing_leaves_no_temp_file(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    assert not list((tmp_path / "decisions").glob("*.tmp"))


# --- outcomes ------------------------------------------------------------

def test_an_outcome_is_recorded_against_the_original_levels(tmp_path):
    """R is computed from the risk ACCEPTED at the time, not from a stop
    moved afterwards."""
    o = Outcome(decision_date=DAY, symbol="RIR.NS", observed_at=NOON)
    o.close(price=250.0, on=date(2026, 9, 24), reason=ExitReason.TARGET,
            entry=200.0, stop=180.0, qty=100)
    assert o.status == "closed"
    assert o.r_multiple == pytest.approx(2.5)
    assert o.pnl == pytest.approx(5000.0)


def test_a_stopped_trade_is_minus_one_r():
    o = Outcome(decision_date=DAY, symbol="X.NS", observed_at=NOON)
    o.close(price=180.0, on=DAY, reason=ExitReason.STOP,
            entry=200.0, stop=180.0, qty=10)
    assert o.r_multiple == pytest.approx(-1.0)


def test_an_unknown_exit_reason_is_refused():
    o = Outcome(decision_date=DAY, symbol="X.NS", observed_at=NOON)
    with pytest.raises(ValueError, match="unknown exit reason"):
        o.close(price=1.0, on=DAY, reason="vibes", entry=1.0, stop=1.0, qty=1)


def test_a_plan_trade_never_taken_is_still_recorded():
    """A plan whose trades are routinely skipped is a fact about the
    system. Deleting those makes the journal describe a desk that does not
    exist."""
    o = Outcome(decision_date=DAY, symbol="X.NS", observed_at=NOON)
    o.close(price=0.0, on=DAY, reason=ExitReason.NOT_TAKEN,
            entry=None, stop=None, qty=0)
    assert o.exit_reason == ExitReason.NOT_TAKEN
    assert o.r_multiple is None


def test_outcomes_round_trip(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    o = Outcome(decision_date=DAY, symbol="RIR.NS", observed_at=NOON)
    o.close(price=250.0, on=date(2026, 9, 24), reason=ExitReason.TARGET,
            entry=200.0, stop=180.0, qty=100)
    st.record_outcomes(DAY, [o])

    back = st.outcomes(DAY)
    assert len(back) == 1
    assert back[0].exit_reason == ExitReason.TARGET
    assert back[0].r_multiple == pytest.approx(2.5)


def test_outcomes_are_replaceable_unlike_decisions(tmp_path):
    """An outcome is an observation that completes as a position runs."""
    st = JournalStore(tmp_path)
    st.record(_decision())
    o = Outcome(decision_date=DAY, symbol="RIR.NS", observed_at=NOON)
    st.record_outcomes(DAY, [o])
    o.close(price=250.0, on=date(2026, 9, 24), reason=ExitReason.TARGET,
            entry=200.0, stop=180.0, qty=100)
    st.record_outcomes(DAY, [o])
    assert st.outcomes(DAY)[0].status == "closed"


def test_open_positions_are_listed(tmp_path):
    st = JournalStore(tmp_path)
    st.record(_decision())
    st.record_outcomes(DAY, [Outcome(decision_date=DAY, symbol="RIR.NS",
                                     observed_at=NOON)])
    assert [o.symbol for o in st.open_positions()] == ["RIR.NS"]


def test_unrecorded_days_are_the_journals_own_todo_list(tmp_path):
    """Otherwise 'we have no losing trades' and 'nobody wrote down how the
    trades went' look identical - and the flattering reading wins."""
    st = JournalStore(tmp_path)
    st.record(_decision(trades=[_trade("A.NS"), _trade("B.NS")]))
    assert st.unrecorded() == [DAY]

    st.record_outcomes(DAY, [Outcome(decision_date=DAY, symbol="A.NS",
                                     observed_at=NOON)])
    assert st.unrecorded() == [DAY], "one of two is still incomplete"

    st.record_outcomes(DAY, [
        Outcome(decision_date=DAY, symbol="A.NS", observed_at=NOON),
        Outcome(decision_date=DAY, symbol="B.NS", observed_at=NOON)])
    assert st.unrecorded() == []


# --- building one from a plan --------------------------------------------

def test_a_decision_can_be_built_from_a_daily_plan():
    from desk.contracts.enums import Regime, Stance
    from desk.plan.models import DailyPlan, PlanTrade

    plan = DailyPlan(
        as_of=DAY, regime=Regime.TRENDING_UP,
        universe_scanned=3483, survived_stage0=1548, survived_stage1=438,
        survived_stage2=438, analysed=3,
        trades=[PlanTrade(symbol="RIR.NS", stance=Stance.BUY, confidence=0.7,
                          entry=197.63, stop=173.79, target=257.22, qty=419,
                          rationale="momentum")],
        warnings=["event gate not applied"],
        market_risks=["results season"],
    )
    d = decision_from_plan(plan, capital=1_000_000, now=NOON)

    assert d.as_of == DAY
    assert d.regime == "trending_up"
    assert d.trades[0].symbol == "RIR.NS"
    assert d.trades[0].stance == "Buy"
    assert d.trades[0].qty == 419
    assert "event gate not applied" in d.caveats
    assert "results season" in d.caveats, "market risks are caveats too"
    assert d.capital == 1_000_000


def test_a_no_trade_plan_becomes_a_no_trade_decision():
    from desk.contracts.enums import Regime
    from desk.plan.models import DailyPlan

    plan = DailyPlan(as_of=DAY, regime=Regime.RANGE,
                     no_trade_reason="nothing cleared the risk gate")
    d = decision_from_plan(plan, now=NOON)
    assert d.is_no_trade
    assert d.no_trade_reason == "nothing cleared the risk gate"
