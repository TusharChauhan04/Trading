"""Scanner Stage 4 - setup proposal and the risk gate.

The danger at this stage is the opposite of the earlier ones: not a wrong
number, but a wrong PERMISSION. A high Stage 2 score must buy a name nothing
except the right to be considered, and nothing here may loosen a limit,
retry a refusal with a wider stop, or turn a rejection into a warning.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from desk.contracts.enums import (Regime, RejectReason,
                                  Stance)
from desk.risk.engine import Portfolio, Position, RiskConfig
from desk.scanner.stage1 import FEATURE_COLUMNS, Stage1Result
from desk.scanner.stage2 import run_stage2
from desk.scanner.stage4 import propose_setup, run_stage4
from desk.store import Coverage

AS_OF = date(2026, 9, 11)
FLAGS = ("compressed", "unusual_volume", "unusual_move", "near_52w_high",
         "extended")


def _stage1(rows: dict[str, dict]) -> Stage1Result:
    df = pd.DataFrame(index=pd.Index(list(rows), name="symbol"),
                      columns=list(FEATURE_COLUMNS), dtype="float64")
    for flag in FLAGS:
        df[flag] = [bool(rows[s].get(flag, False)) for s in df.index]
    for sym, vals in rows.items():
        for k, v in vals.items():
            if k not in FLAGS:
                df.loc[sym, k] = v
    df["flag_count"] = df["flag_count"].fillna(1).astype("int64")
    df["bars"] = df["bars"].fillna(250).astype("int64")
    return Stage1Result(
        as_of=AS_OF, features=df,
        coverage=Coverage(start=None, end=AS_OF, days_loaded=250,
                          sessions_expected=250, calendar_checked=True),
        universe_in=len(df), universe_out=len(df),
    )


def _candidate(i: int, **over) -> dict:
    base = {
        "close": 1000.0, "volume": 5_000_000.0, "atr_pct": 2.0,
        "ret_20d_pct": float(i), "dist_sma50_pct": float(i),
        "pos_52w_pct": float(50 + i), "rel_volume": 1.0 + i * 0.1,
        "atr_pct_rank": float(90 - i * 5),
    }
    base.update(over)
    return base


def _pipeline(rows, regime=Regime.TRENDING_UP):
    s1 = _stage1(rows)
    s2 = run_stage2(s1, regime=regime, flagged_only=False)
    return s1, s2


# ===========================================================================
# setup proposal
# ===========================================================================

def test_stop_and_target_are_atr_derived_and_hand_checkable():
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0, "volume": 1e6})
    s = propose_setup("AAA.NS", row, score=90.0, stop_atrs=2.0, target_r=2.5)
    assert s is not None
    assert s.entry == pytest.approx(1000.0)
    assert s.stop == pytest.approx(960.0)          # 2 x 2% of 1000
    assert s.target == pytest.approx(1100.0)       # 2.5 x the 40-rupee risk
    assert s.stance is Stance.BUY


def test_no_setup_is_proposed_without_a_usable_atr():
    """A stop needs a volatility estimate. Inventing one produces a setup the
    risk engine will size happily and every number downstream looks
    legitimate - which is worse than proposing nothing."""
    for bad in ({"close": 1000.0, "atr_pct": float("nan")},
                {"close": 1000.0, "atr_pct": 0.0},
                {"close": float("nan"), "atr_pct": 2.0},
                {"close": 0.0, "atr_pct": 2.0}):
        assert propose_setup("AAA.NS", pd.Series(bad), score=90.0) is None


def test_an_atr_wide_enough_to_push_the_stop_below_zero_is_refused():
    """Not merely a risky trade - a volatility estimate that is nonsense for
    this name."""
    row = pd.Series({"close": 100.0, "atr_pct": 60.0, "volume": 1e6})
    assert propose_setup("AAA.NS", row, score=90.0, stop_atrs=2.0) is None


def test_the_rationale_names_the_flags_that_earned_the_look():
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0, "volume": 1e6,
                     "unusual_volume": True, "near_52w_high": True,
                     "compressed": False})
    s = propose_setup("AAA.NS", row, score=88.0)
    assert "unusual volume" in s.rationale
    assert "near 52-week high" in s.rationale
    assert "volatility compressed" not in s.rationale


# ===========================================================================
# the stop goes where the trade is WRONG
#
# The master spec asks for a stop "derived from trade invalidation and risk
# structure - not an arbitrary percentage". A fixed multiple of ATR is an
# arbitrary percentage wearing a volatility costume: it says how far this
# stock usually travels, which is a fact about the stock and not about the
# setup. These tests pin the level to the structure.
# ===========================================================================

def test_the_stop_sits_below_the_last_swing_low_not_at_a_fixed_atr():
    """The headline behaviour. ATR sets only the CUSHION under the level."""
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0, "volume": 1e6,
                     "swing_low": 940.0, "swing_low_age": 7.0,
                     "low_20": 900.0})
    s = propose_setup("AAA.NS", row, score=90.0, stop_atrs=2.0, target_r=2.0,
                      stop_buffer_atrs=0.25)
    # 940 - 0.25 x (2% of 1000) = 940 - 5 = 935, NOT the 960 that 2x ATR gives
    assert s.stop == pytest.approx(935.0)
    assert s.stop_basis == "swing_low"
    assert s.target == pytest.approx(1000.0 + 2.0 * 65.0)
    assert "swing low at 940.00" in s.invalidation
    assert "7 sessions ago" in s.invalidation


def test_the_nearer_level_wins_because_it_is_the_one_that_breaks_first():
    """Not the more significant level - the first one price reaches. A stop
    at the lower of the two sits through the whole of the evidence that the
    setup has failed before acting on any of it."""
    near_is_low20 = pd.Series({"close": 1000.0, "atr_pct": 2.0,
                               "swing_low": 800.0, "low_20": 950.0})
    s = propose_setup("AAA.NS", near_is_low20, score=90.0)
    assert s.stop_basis == "low_20"
    assert s.stop == pytest.approx(945.0)          # 950 - 0.25 x 20

    near_is_swing = pd.Series({"close": 1000.0, "atr_pct": 2.0,
                               "swing_low": 950.0, "low_20": 800.0})
    assert propose_setup("AAA.NS", near_is_swing, score=90.0).stop_basis \
        == "swing_low"


def test_a_tie_between_the_two_levels_goes_to_the_swing_low():
    """Same price, better description of it."""
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0,
                     "swing_low": 900.0, "low_20": 900.0})
    assert propose_setup("AAA.NS", row, score=90.0).stop_basis == "swing_low"


def test_a_pivot_ABOVE_the_price_is_overhead_resistance_not_a_stop():
    """611 of 1,508 real names on 2026-09-17 sat below their own last pivot.
    Using it would put the "stop" above the entry - an instant exit at a
    guaranteed loss - so it falls through to the level that is actually
    below."""
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0,
                     "swing_low": 1100.0, "low_20": 950.0})
    s = propose_setup("AAA.NS", row, score=90.0)
    assert s.stop_basis == "low_20"
    assert s.stop < s.entry


def test_a_level_inside_the_daily_noise_is_widened_and_SAYS_SO():
    """A pivot 0.3 ATR below entry is a real level, but a stop there is hit
    by ordinary drift rather than by the thesis failing. Widening is right;
    widening silently and still calling it a structural stop is not."""
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0, "swing_low": 994.0})
    s = propose_setup("AAA.NS", row, score=90.0, min_stop_atrs=1.5)
    assert s.stop == pytest.approx(970.0)          # 1000 - 1.5 x 20
    assert s.stop_basis == "noise_floor"
    assert "WIDENED" in s.invalidation
    assert "994.00" in s.invalidation, "must still name the level it left"


def test_with_no_structure_at_all_the_atr_stop_is_labelled_a_FALLBACK():
    """The old behaviour survives as the fallback, and the difference between
    "this is where the setup fails" and "this is how far the stock moves" is
    stated rather than left for the reader to assume."""
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0, "volume": 1e6})
    s = propose_setup("AAA.NS", row, score=90.0, stop_atrs=2.0)
    assert s.stop == pytest.approx(960.0)
    assert s.stop_basis == "atr"
    assert "VOLATILITY stop" in s.invalidation
    assert "not an invalidation level" in s.invalidation


def test_a_corrupt_or_zero_level_is_ignored_rather_than_used():
    """0 and a negative are not levels, and NaN means the feature could not
    be computed. None of the three may become a stop."""
    for bad in (0.0, -5.0, float("nan")):
        row = pd.Series({"close": 1000.0, "atr_pct": 2.0,
                         "swing_low": bad, "low_20": bad})
        s = propose_setup("AAA.NS", row, score=90.0)
        assert s.stop_basis == "atr", f"{bad} was treated as a level"


def test_a_distant_level_is_NOT_pulled_in_here_the_engine_refuses_it():
    """The one temptation this module must resist. Trimming a 20% structural
    stop to fit the risk gate would place it somewhere the thesis is still
    intact, which is exactly the arbitrary stop being replaced. Stage 4
    proposes it honestly and `size_position` says no."""
    row = pd.Series({"close": 1000.0, "atr_pct": 2.0,
                     "swing_low": 800.0, "volume": 1e6})
    s = propose_setup("AAA.NS", row, score=90.0)
    assert s.stop == pytest.approx(795.0)
    assert (s.entry - s.stop) / s.entry > 0.15     # past the 15% ceiling

    from desk.risk.engine import size_position
    sized = size_position(symbol="AAA.NS", entry=s.entry, stop=s.stop,
                          target=s.target, cfg=RiskConfig(capital=1_000_000))
    assert not sized.approved
    assert RejectReason.STOP_TOO_WIDE in sized.reasons


def test_the_basis_and_invalidation_reach_the_approved_trade():
    """Computed and discarded is the recurring failure in this codebase. The
    reader of the plan is the one who needs the invalidation level."""
    rows = {"AAA.NS": _candidate(9, close=1000.0, atr_pct=2.0,
                                 swing_low=940.0, swing_low_age=5.0,
                                 low_20=900.0)}
    s1, s2 = _pipeline(rows)
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=1_000_000),
                   portfolio=Portfolio(), today=AS_OF)
    assert r.approved, r.no_trade_reason()
    t = r.approved[0]
    assert t.stop_basis == "swing_low"
    assert "940.00" in t.invalidation
    assert t.stop == pytest.approx(935.0), "the sized stop is the proposed one"

# ===========================================================================
# the risk gate has veto power
# ===========================================================================

def test_the_top_ranked_name_is_still_refused_when_the_engine_says_no():
    """The whole point. A 100/100 Stage 2 score must not buy an exemption."""
    s1, s2 = _pipeline({"AAA.NS": _candidate(9, atr_pct=40.0)})
    cfg = RiskConfig(capital=1_000_000, max_instrument_atr_pct=12.0)
    r = run_stage4(s2, s1, cfg=cfg, today=AS_OF)
    assert r.approved == []
    assert r.is_no_trade
    assert r.rejected and r.rejected[0].reasons


def test_a_setup_below_the_risk_reward_floor_is_refused():
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    cfg = RiskConfig(capital=1_000_000, min_risk_reward=5.0)
    r = run_stage4(s2, s1, cfg=cfg, target_r=2.0, today=AS_OF)
    assert r.approved == []
    assert any("risk_reward" in x.rejected_because or "reward" in x.rejected_because
               for x in r.rejected)


def test_max_trades_bounds_the_output(_=None):
    rows = {f"SYM{i}.NS": _candidate(i) for i in range(10)}
    s1, s2 = _pipeline(rows)
    cfg = RiskConfig(capital=100_000_000, max_open_positions=50,
                     max_sector_pct=100.0, max_open_risk_pct=20.0)
    r = run_stage4(s2, s1, cfg=cfg, max_trades=2, candidates=10, today=AS_OF)
    assert len(r.approved) == 2


def test_candidates_looks_deeper_than_max_trades():
    """Considering only `max_trades` names would return nothing on a day when
    the top three fail and the fourth is perfectly tradeable."""
    rows = {"BAD1.NS": _candidate(9, atr_pct=40.0),
            "BAD2.NS": _candidate(8, atr_pct=40.0),
            "BAD3.NS": _candidate(7, atr_pct=40.0),
            "GOOD.NS": _candidate(6)}
    s1, s2 = _pipeline(rows)
    cfg = RiskConfig(capital=10_000_000, max_instrument_atr_pct=12.0)
    r = run_stage4(s2, s1, cfg=cfg, max_trades=3, candidates=10, today=AS_OF)
    assert [t.symbol for t in r.approved] == ["GOOD.NS"]
    assert r.considered == 4


def test_each_approval_is_sized_against_the_book_the_previous_one_created():
    """Sizing every candidate against the same empty portfolio would let
    three trades each pass the heat limit alone and breach it together."""
    rows = {f"SYM{i}.NS": _candidate(i) for i in range(6)}
    s1, s2 = _pipeline(rows)
    # Hand-worked: capital 1,000,000 at 1% risk and a 40-rupee stop sizes
    # 250 shares = 250,000, which the 15% position cap trims to 150 shares.
    # That is 6,000 at risk, 0.6% of capital - so one position fits under a
    # 1.0% heat cap and a second cannot.
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0, max_open_risk_pct=1.0,
                     max_sector_pct=100.0)
    r = run_stage4(s2, s1, cfg=cfg, max_trades=5, candidates=6, today=AS_OF)
    assert len(r.approved) == 1
    assert any("heat" in x.rejected_because or "open_risk" in x.rejected_because
               for x in r.rejected)


def test_an_existing_portfolio_is_respected():
    rows = {"AAA.NS": _candidate(9)}
    s1, s2 = _pipeline(rows)
    cfg = RiskConfig(capital=1_000_000, max_open_positions=1)
    held = Portfolio(positions=[Position(symbol="ZZZ.NS", qty=10,
                                         entry=100.0, stop=90.0)])
    r = run_stage4(s2, s1, cfg=cfg, portfolio=held, today=AS_OF)
    assert r.approved == []
    assert r.rejected


def test_market_risk_off_is_passed_through_to_the_engine():
    rows = {"AAA.NS": _candidate(9)}
    s1, s2 = _pipeline(rows)
    cfg = RiskConfig(capital=1_000_000)
    r = run_stage4(s2, s1, cfg=cfg, market_risk_off=True, today=AS_OF)
    assert r.approved == []
    assert any("risk_off" in x.rejected_because for x in r.rejected)


# ===========================================================================
# NO TRADE is an outcome, with a reason
# ===========================================================================

def test_no_trade_distinguishes_nothing_reached_the_gate_from_everything_refused():
    """"We looked at 40 names and every one failed the R:R floor" is a
    materially different day from "nothing was flagged". A plan that cannot
    tell them apart is not explainable."""
    s1_empty, s2_empty = _pipeline({})
    empty = run_stage4(s2_empty, s1_empty, cfg=RiskConfig(capital=1_000_000),
                       today=AS_OF)
    assert empty.considered == 0
    assert "nothing reached the risk gate" in empty.no_trade_reason()

    s1, s2 = _pipeline({"AAA.NS": _candidate(9, atr_pct=40.0)})
    refused = run_stage4(s2, s1, cfg=RiskConfig(capital=1_000_000,
                                                max_instrument_atr_pct=12.0),
                         today=AS_OF)
    reason = refused.no_trade_reason()
    assert "reached the risk gate and every one was refused" in reason
    assert "1 candidate" in reason


def test_no_trade_reason_is_none_when_there_is_something_to_do():
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    assert r.approved
    assert r.no_trade_reason() is None


def test_rejections_are_kept_not_discarded():
    rows = {f"SYM{i}.NS": _candidate(i, atr_pct=40.0) for i in range(5)}
    s1, s2 = _pipeline(rows)
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=1_000_000,
                                          max_instrument_atr_pct=12.0),
                   candidates=5, today=AS_OF)
    assert len(r.rejected) == 5
    assert "extreme" in r.summary() or "volatility" in r.summary()


def test_a_name_without_features_is_skipped_with_a_counted_reason():
    s1, s2 = _pipeline({"AAA.NS": _candidate(9), "BBB.NS": _candidate(8)})
    s1.features = s1.features.drop(index=["AAA.NS"])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    assert r.skipped["no Stage 1 features"] == 1


def test_a_name_with_no_usable_atr_is_skipped_before_the_gate():
    rows = {"AAA.NS": _candidate(9, atr_pct=float("nan")),
            "BBB.NS": _candidate(8)}
    s1, s2 = _pipeline(rows)
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    assert r.skipped["no usable ATR or price"] == 1
    assert [t.symbol for t in r.approved] == ["BBB.NS"]


# ===========================================================================
# honesty of the output
# ===========================================================================

def test_an_approval_states_the_checks_that_could_not_run():
    """An approval with skipped checks is weaker than one without, and the
    plan must say so rather than presenting both as equally vetted."""
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    assert "NOT CHECKED" in r.approved[0].rationale


def test_confidence_is_the_stage2_score_and_is_labelled_uncalibrated():
    """Nothing has measured whether a 0.8 wins 80% of the time. The number is
    a ranking, and the module says so - if this is ever presented as a
    probability the journal has to exist first."""
    s1, s2 = _pipeline({f"SYM{i}.NS": _candidate(i) for i in range(4)})
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=100_000_000,
                                          max_sector_pct=100.0,
                                          max_open_risk_pct=20.0),
                   max_trades=1, today=AS_OF)
    top = r.approved[0]
    expected = s2.ranked.loc[top.symbol, "score"] / 100.0
    assert top.confidence == pytest.approx(round(expected, 3))
    assert 0.0 <= top.confidence <= 1.0


def test_every_run_declares_the_caps_it_cannot_actually_enforce():
    """Sector and correlated-cluster caps are structurally unenforceable
    today - bhavcopy has no sector and no correlation matrix exists. Sizing
    every name as sector UNKNOWN makes the sector cap bind against unrelated
    names, which is visible in real runs and must not be silent."""
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    joined = " ".join(r.unavailable)
    assert "sector" in joined.lower()
    assert "Correlated-cluster" in joined
    assert "structural" in joined


def test_approved_trades_carry_the_engine_prices_not_the_proposed_ones():
    """The engine may trim quantity; the plan must report what the engine
    returned, never what was asked for."""
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    t = r.approved[0]
    assert t.qty > 0
    assert t.entry is not None and t.stop is not None
    assert t.stop < t.entry < t.target
    assert t.capital_at_risk > 0


def test_a_crisis_ranking_of_nothing_produces_no_trade():
    rows = {f"SYM{i}.NS": _candidate(i) for i in range(10)}
    s1, s2 = _pipeline(rows, regime=Regime.CRISIS)
    assert s2.ranked.empty
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    assert r.is_no_trade
    assert r.considered == 0


# ===========================================================================
# R4 - the event gate
# ===========================================================================

def _calendar(rows):
    from desk.research.events import EventCalendar
    from desk.research.models import CorporateEvent
    return EventCalendar.from_events(
        [CorporateEvent(symbol=s, event_date=d, purpose=p)
         for s, d, p in rows])


def test_a_name_reporting_inside_the_holding_window_is_refused():
    """The whole point of R4. A company announcing results inside the holding
    window turns the trade into a coin flip on something the chart cannot
    see - however good the setup is."""
    from desk.contracts.enums import RejectReason

    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    cal = _calendar([("AAA", date(2026, 9, 13), "Financial Results")])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF,
                   events=cal, holding_days=5)
    assert r.approved == []
    assert r.rejected[0].reasons == [RejectReason.EVENT_IN_WINDOW]
    assert "Financial Results" in r.rejected[0].notes[0]
    assert "2d" in r.rejected[0].notes[0]      # 11 Sep -> 13 Sep


def test_an_event_beyond_the_holding_window_does_not_block():
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    cal = _calendar([("AAA", date(2026, 10, 30), "Financial Results")])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF,
                   events=cal, holding_days=5)
    assert [t.symbol for t in r.approved] == ["AAA.NS"]


def test_the_gate_runs_before_sizing():
    """There is no quantity that makes holding through an earnings print
    acceptable, so the check belongs ahead of the arithmetic."""
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    cal = _calendar([("AAA", date(2026, 9, 12), "Financial Results")])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF,
                   events=cal, holding_days=5)
    rejected = r.rejected[0]
    assert rejected.qty == 0
    assert rejected.capital_at_risk == 0.0


def test_a_non_price_moving_event_does_not_block():
    """NSE's calendar carries administrative entries too. Blocking on every
    one of them makes the gate fire constantly and get switched off."""
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    cal = _calendar([("AAA", date(2026, 9, 12), "Change of Registered Office")])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF,
                   events=cal, holding_days=5)
    assert [t.symbol for t in r.approved] == ["AAA.NS"]


def test_omitting_the_calendar_is_declared_not_silently_skipped():
    """A scan that did not check for earnings must not read like one that
    checked and found none."""
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF)
    assert any("Event proximity not checked" in u for u in r.unavailable)


def test_partial_calendar_coverage_is_declared():
    """NSE's forward calendar covers a few weeks - 41 names on a typical day.
    Reporting "the event gate ran" while it could see 1 of 3 candidates is the
    same lie as reporting a skipped check as a passed one."""
    rows = {f"SYM{i}.NS": _candidate(i) for i in range(3)}
    s1, s2 = _pipeline(rows)
    cal = _calendar([("SYM2", date(2026, 11, 30), "Financial Results")])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=100_000_000,
                                          max_sector_pct=100.0,
                                          max_open_risk_pct=20.0),
                   today=AS_OF, events=cal, holding_days=5)
    note = next(u for u in r.unavailable if "Event proximity known" in u)
    assert "1 of 3" in note
    assert "was NOT cleared, it was not checked" in note


def test_a_symbol_absent_from_the_calendar_does_not_block():
    """Deliberate, and the opposite of this project's usual instinct. The
    calendar covers a few weeks market-wide, so 'no entry' is the normal state
    for almost every name on almost every day - blocking on unknown would
    block everything, and a gate that always fires gets switched off. The
    honest handling is to report coverage, not to treat absence as
    information."""
    from desk.research.events import EventCalendar

    empty = EventCalendar.from_events([])
    w = empty.window("AAA", as_of=AS_OF)
    assert w.known is False and w.days_until is None
    assert w.blocks(5) is False


def test_an_event_today_blocks():
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    cal = _calendar([("AAA", AS_OF, "Financial Results")])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF,
                   events=cal, holding_days=5)
    assert r.approved == []


def test_a_past_event_is_ignored():
    s1, s2 = _pipeline({"AAA.NS": _candidate(9)})
    cal = _calendar([("AAA", date(2026, 9, 1), "Financial Results")])
    r = run_stage4(s2, s1, cfg=RiskConfig(capital=10_000_000), today=AS_OF,
                   events=cal, holding_days=5)
    assert [t.symbol for t in r.approved] == ["AAA.NS"]


def test_the_event_gate_works_against_the_real_nse_calendar():
    """Pinned against the captured market-wide calendar rather than a
    constructed one."""
    import json as _json
    from pathlib import Path as _Path

    from desk.research.events import EventCalendar
    from desk.research.sources.nse import parse_event_calendar

    raw = _json.loads((_Path(__file__).parent / "fixtures" /
                       "nse_eventcalendar.json").read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("data", [])
    cal = EventCalendar.from_events(parse_event_calendar(rows))
    assert len(cal.symbols) == 41

    blocked = [s for s in cal.symbols
               if cal.window(s, as_of=date(2026, 9, 16)).blocks(5)]
    assert len(blocked) == 30
    assert "RELIANCE" not in cal.symbols
    assert cal.window("RELIANCE", as_of=date(2026, 9, 16)).known is False
