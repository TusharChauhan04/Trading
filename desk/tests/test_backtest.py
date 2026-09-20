"""The backtest: exit simulation, costs, and the honesty of both.

Most of this file is about the two ways a daily-bar backtest quietly
invents money:

1. RESOLVING AN AMBIGUOUS BAR IN ITS OWN FAVOUR. When the low reaches the
   stop and the high reaches the target on the same day, the bars cannot
   say which came first. Checking the target first books 2.5R on exactly
   the volatile days where the edge is supposed to live, and turns a losing
   system into a winning one on paper.

2. FILLING A GAPPED EXIT AT THE INTENDED LEVEL. A stop at 180 does not
   fill at 180 when the session opens at 172.

Both are tested directly, and the ambiguous case is also tested for being
COUNTED - pessimism that is not reported is just a different silent
assumption.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from desk.backtest import (
    BacktestResult, CostModel, SimulatedTrade, ZERO_COSTS, simulate_trade,
)
from desk.journal.models import ExitReason

DECIDED = date(2026, 9, 17)


def _bars(rows):
    """rows: list of (iso date, open, high, low, close)."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows]},
        index=idx)


def _run(rows, *, entry=200.0, stop=180.0, target=250.0, qty=10, horizon=5):
    return simulate_trade(_bars(rows), symbol="X.NS", decided_on=DECIDED,
                          planned_entry=entry, stop=stop, target=target,
                          qty=qty, horizon_days=horizon)


# --- the ambiguity, which is the whole point -----------------------------

def test_a_bar_touching_both_levels_resolves_as_a_stop():
    """Daily bars cannot say which came first. The loss is the outcome you
    have to survive, so it is the one assumed."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 200, 260, 175, 240)])
    assert sim.trade.exit_reason == ExitReason.STOP
    assert sim.trade.exit_price == 180.0
    assert sim.trade.r_multiple == pytest.approx(-1.0)


def test_an_ambiguous_exit_is_flagged():
    """Pessimism that is not reported is just another silent assumption."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 200, 260, 175, 240)])
    assert sim.trade.ambiguous is True


def test_an_unambiguous_target_is_not_flagged():
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 200, 260, 195, 255)])
    assert sim.trade.exit_reason == ExitReason.TARGET
    assert sim.trade.ambiguous is False
    assert sim.trade.r_multiple == pytest.approx(2.5)


def test_an_unambiguous_stop_is_not_flagged():
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 200, 205, 175, 185)])
    assert sim.trade.exit_reason == ExitReason.STOP
    assert sim.trade.ambiguous is False


def test_the_result_reports_how_many_exits_were_assumptions():
    amb = SimulatedTrade(symbol="A.NS", entry_date=DECIDED, entry_price=200.0,
                         planned_entry=200.0, exit_date=DECIDED,
                         exit_price=180.0, exit_reason=ExitReason.STOP, qty=10,
                         bars_held=1, stop=180.0, target=250.0, ambiguous=True)
    clean = SimulatedTrade(symbol="B.NS", entry_date=DECIDED, entry_price=200.0,
                           planned_entry=200.0, exit_date=DECIDED,
                           exit_price=250.0, exit_reason=ExitReason.TARGET,
                           qty=10, bars_held=1, stop=180.0, target=250.0)
    res = BacktestResult(start=DECIDED, end=DECIDED, trades=[amb, clean])
    assert res.ambiguous == 1
    assert "AMBIGUOUS" in res.report()
    assert "assumption, not a measurement" in res.report()


# --- gaps ----------------------------------------------------------------

def test_a_gap_below_the_stop_before_entry_is_not_a_trade():
    """REGRESSION. Entering at a 172 open under a 180 stop and then testing
    that same open against the stop booked a flat 0R. Nobody buys at 172
    with the stop ABOVE at 180 - the trade is inverted before it exists, so
    the setup is invalidated and never taken."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 172, 175, 170, 174)])
    assert sim.trade is None
    assert "gapped below the stop" in sim.reason_not_taken


def test_a_gap_through_the_stop_AFTER_entry_fills_at_the_open():
    """Filling at the intended level is the other standard way a backtest
    invents money, and on Indian mid-caps around results it is not rare."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 204, 206, 202, 205),
                ("2026-09-21", 172, 175, 170, 174)])
    assert sim.trade.exit_reason == ExitReason.STOP
    assert sim.trade.exit_price == 172.0, "filled at the open, not at 180"
    assert sim.trade.gapped is True
    assert sim.trade.r_multiple < -1.0, "a gap loses MORE than 1R"


def test_a_gap_past_the_target_before_entry_is_not_a_trade():
    """Gap-ups through the entry are exactly how a momentum shortlist loses
    its best-looking names before it can act, and booking them at a tidy 0R
    hides that."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 262, 265, 260, 263)])
    assert sim.trade is None
    assert "gapped past the target" in sim.reason_not_taken


def test_a_gap_through_the_target_after_entry_fills_at_the_open():
    """The rule cuts both ways - it is about the open being what actually
    happened first, not about pessimism for its own sake."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 204, 206, 202, 205),
                ("2026-09-21", 262, 265, 260, 263)])
    assert sim.trade.exit_reason == ExitReason.TARGET
    assert sim.trade.exit_price == 262.0
    assert sim.trade.gapped is True


def test_the_entry_bar_can_still_resolve_the_trade_intraday():
    """Entering at a valid open and then breaking the stop within the same
    session is a real same-day loss. Pretending a position must survive its
    first session would hide the worst of them."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 199, 201, 175, 178)])
    assert sim.trade is not None
    assert sim.trade.bars_held == 1
    assert sim.trade.exit_reason == ExitReason.STOP
    assert sim.trade.gapped is False, "the open was valid; this was intrabar"


# --- entry ---------------------------------------------------------------

def test_entry_is_the_next_sessions_open_not_the_planned_level():
    """The decision is made pre-open from the prior close, so the first
    price it could transact at is the next open."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 204, 210, 203, 209),
                ("2026-09-19", 209, 260, 208, 255)])
    assert sim.trade.entry_date == date(2026, 9, 18)
    assert sim.trade.entry_price == 204.0


def test_the_entry_slip_against_the_plan_is_reported():
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 204, 210, 203, 209)], entry=200.0, horizon=1)
    assert sim.trade.entry_slip_pct == pytest.approx(2.0)


def test_r_is_measured_against_the_realised_entry():
    """A trade entered 2% above its planned level took MORE risk than the
    plan intended; measuring against the plan would hide that."""
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 204, 205, 175, 180)], entry=200.0, stop=180.0)
    # Risk is 204-180 = 24, not the planned 20.
    assert sim.trade.risk_per_share == pytest.approx(24.0)
    assert sim.trade.r_multiple == pytest.approx(-1.0)


def test_no_session_after_the_decision_means_no_trade():
    """Not a trade that broke even - a trade that never happened."""
    sim = _run([("2026-09-17", 200, 201, 199, 200)])
    assert sim.trade is None
    assert "no session after" in sim.reason_not_taken


def test_no_history_means_no_trade():
    sim = simulate_trade(pd.DataFrame(), symbol="X.NS", decided_on=DECIDED,
                         planned_entry=1.0, stop=1.0, target=2.0, qty=1)
    assert sim.trade is None
    assert "no price history" in sim.reason_not_taken


def test_zero_quantity_is_not_a_trade():
    sim = _run([("2026-09-17", 200, 201, 199, 200),
                ("2026-09-18", 204, 210, 203, 209)], qty=0)
    assert sim.trade is None


# --- the horizon ---------------------------------------------------------

def test_an_unresolved_trade_exits_at_the_horizon_close():
    rows = [("2026-09-17", 200, 201, 199, 200)]
    for i, d in enumerate(["18", "21", "22", "23", "24"]):
        rows.append((f"2026-09-{d}", 205, 207, 203, 206))
    sim = _run(rows, horizon=5)
    assert sim.trade.exit_reason == ExitReason.TIME
    assert sim.trade.bars_held == 5
    assert sim.trade.exit_price == 206.0


def test_the_horizon_is_respected_even_with_more_bars_available():
    rows = [("2026-09-17", 200, 201, 199, 200)]
    for d in ["18", "21", "22", "23", "24", "25", "28"]:
        rows.append((f"2026-09-{d}", 205, 207, 203, 206))
    sim = _run(rows, horizon=3)
    assert sim.trade.bars_held == 3
    assert sim.trade.exit_date == date(2026, 9, 22)


def test_a_trade_with_no_target_can_only_stop_or_time_out():
    rows = [("2026-09-17", 200, 201, 199, 200)]
    for d in ["18", "21", "22"]:
        rows.append((f"2026-09-{d}", 300, 400, 299, 350))
    sim = _run(rows, target=None, horizon=3)
    assert sim.trade.exit_reason == ExitReason.TIME


# --- costs ---------------------------------------------------------------

def test_the_cost_stack_matches_the_documented_breakdown():
    c = CostModel()
    assert c.charges(value=100.0, side="buy") == pytest.approx(0.0186, abs=1e-3)
    assert c.charges(value=100.0, side="sell") == pytest.approx(0.1036, abs=1e-3)
    assert c.round_trip_pct() == pytest.approx(0.4222, abs=1e-3)


def test_stt_is_charged_on_the_sell_side_only():
    c = CostModel()
    assert c.charges(value=100.0, side="sell") > c.charges(value=100.0, side="buy")


def test_slippage_always_moves_against_us():
    c = CostModel(slippage_bps=100.0)
    assert c.fill_price(100.0, side="buy") == pytest.approx(101.0)
    assert c.fill_price(100.0, side="sell") == pytest.approx(99.0)


def test_a_full_service_broker_dominates_the_stack():
    """Worth being able to show: brokerage is the line to check against a
    real contract note."""
    assert CostModel(brokerage_pct=0.3).round_trip_pct() > 1.0


def test_the_zero_model_is_actually_zero():
    assert ZERO_COSTS.round_trip_pct() == 0.0
    assert ZERO_COSTS.round_trip(entry=100, exit_=200, qty=10).total == 0.0


def test_costs_reduce_the_net_result():
    t = SimulatedTrade(symbol="A.NS", entry_date=DECIDED, entry_price=200.0,
                       planned_entry=200.0, exit_date=DECIDED,
                       exit_price=250.0, exit_reason=ExitReason.TARGET,
                       qty=100, bars_held=1, stop=180.0, target=250.0)
    res = BacktestResult(start=DECIDED, end=DECIDED, trades=[t])
    assert res.gross_pnl == pytest.approx(5000.0)
    assert res.total_costs > 0
    assert res.net_pnl_total < res.gross_pnl


def test_an_unsized_trade_costs_nothing():
    assert CostModel().round_trip(entry=100, exit_=110, qty=0).total == 0.0


def test_an_unknown_side_is_refused():
    with pytest.raises(ValueError, match="buy.*sell"):
        CostModel().charges(value=100.0, side="hold")


# --- the result's denominators -------------------------------------------

def _res(trades=(), days=()):
    return BacktestResult(start=DECIDED, end=DECIDED, trades=list(trades),
                          days=list(days))


def _t(r_target, qty=10, entry=200.0, stop=180.0):
    """A trade whose R-multiple is r_target."""
    exit_price = entry + r_target * abs(entry - stop)
    return SimulatedTrade(
        symbol="X.NS", entry_date=DECIDED, entry_price=entry,
        planned_entry=entry, exit_date=DECIDED, exit_price=exit_price,
        exit_reason=ExitReason.TARGET if r_target > 0 else ExitReason.STOP,
        qty=qty, bars_held=1, stop=stop, target=None)


def test_expectancy_is_the_mean_r():
    res = _res([_t(2.5), _t(-1.0), _t(-1.0), _t(2.5)])
    assert res.expectancy_r == pytest.approx(0.75)
    assert res.win_rate == pytest.approx(50.0)
    assert res.total_r == pytest.approx(3.0)


def test_max_drawdown_is_measured_on_the_cumulative_r_curve():
    res = _res([_t(2.0), _t(-1.0), _t(-1.0), _t(-1.0), _t(1.0)])
    assert res.max_drawdown_r == pytest.approx(-3.0)


def test_a_backtest_with_no_trades_reports_that_plainly():
    res = _res()
    assert res.expectancy_r is None
    assert res.win_rate is None
    assert "no trades to measure" in res.report()


def test_days_that_could_not_run_are_separate_from_no_trade_days():
    """'We looked and declined' and 'we could not look' are different
    facts; averaging them understates both."""
    from desk.backtest import DayResult
    res = _res(days=[DayResult(as_of=DECIDED, proposed=0),
                     DayResult(as_of=DECIDED, error="no snapshot"),
                     DayResult(as_of=DECIDED, proposed=2)])
    assert res.sessions == 3
    assert res.scanned == 2
    assert res.failed_days == 1
    assert res.no_trade_days == 1


def test_proposals_that_could_not_be_simulated_are_reported():
    res = BacktestResult(start=DECIDED, end=DECIDED,
                         not_taken={"no session after the decision date": 4})
    assert "4 proposed but not taken" in res.report()


def test_the_engine_never_takes_an_llm_client():
    """Hard-coded, not a default. A model trained past the simulated date
    partly remembers the answer - look-ahead living in the weights, which
    no plumbing can audit."""
    import inspect

    from desk.backtest import engine
    src = inspect.getsource(engine.run_backtest)
    assert "client" not in src.split('"""')[2], \
        "run_backtest must not accept or forward an LLM client"


def test_the_funnel_accepts_a_simulated_today():
    """REGRESSION, and it silenced the entire backtest. The risk engine
    refuses data more than 5 days old, measured against "today". With the
    IST wall clock hard-coded into _run_funnel, every historical session
    was stale by definition: 29 sessions produced 0 trades and every day
    reported NO TRADE while the machinery looked like it was working.

    In a replay, today IS the simulated date."""
    import inspect

    from desk.api.main import _run_funnel
    sig = inspect.signature(_run_funnel)
    assert "today" in sig.parameters
    assert sig.parameters["today"].default is None, \
        "live callers must still get the IST clock by default"

    from desk.backtest import engine
    src = inspect.getsource(engine.run_backtest)
    assert "today=day" in src, "the backtest must pass the simulated date"


def test_the_engine_loads_forward_prices_once_not_per_trade():
    """REGRESSION. series() is on History, not BarStore. store.series()
    raised AttributeError, a bare except swallowed it, and all 60
    proposals were reported as "no price history for this symbol" - a real
    bug wearing a data problem's clothes."""
    import inspect

    from desk.backtest import engine
    # Code only - the comment above the fix names the old call on purpose.
    code = [ln.split("#")[0] for ln in
            inspect.getsource(engine.run_backtest).splitlines()]
    assert not any("store.series(" in ln for ln in code),         "series() is not a BarStore method"
    assert any("forward.series(" in ln for ln in code)

    # Both reads - the forward prices for exit simulation and the
    # backward lookback window the funnel needs - must happen BEFORE the
    # per-session loop. Counting calls is the wrong assertion (there are
    # legitimately two); what matters is that neither is inside the loop.
    loop_at = next(i for i, ln in enumerate(code)
                   if ln.strip().startswith("for i, day in enumerate("))
    assert not any("store.history(" in ln for ln in code[loop_at:]),         "a price read inside the session loop re-scans overlapping files"
    assert sum(1 for ln in code[:loop_at] if "store.history(" in ln) == 2,         "expected exactly two hoisted reads: forward prices and lookback"
