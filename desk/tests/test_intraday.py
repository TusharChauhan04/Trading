"""The intraday 1:2 lattice, and the arithmetic that answers the main goal.

The measured result lives in desk/backtest/intraday.py. These tests pin the
machinery that produced it, because the finding is only as good as the lattice
- and the two things most likely to be got wrong are the convention on a bar
that touches both levels, and letting a trade run past its own session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.backtest.intraday import (
    STATUTORY_ROUND_TRIP_PCT, breakeven_win_rate, session_outcomes,
)


def _bars(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    """(session, symbol, high, low, close) -> the long frame the lattice takes."""
    return pd.DataFrame(
        [{"session": s, "symbol": y, "high": h, "low": lo, "close": c}
         for s, y, h, lo, c in rows])


def test_a_clean_target_hit_is_counted_as_a_win() -> None:
    # Entry 100 at bar 0; stop 99, target 102 at a 1% stop and rr 2.
    out = session_outcomes(_bars([
        ("d1", "A", 100.0, 100.0, 100.0),
        ("d1", "A", 101.0, 100.0, 100.5),
        ("d1", "A", 102.5, 101.0, 102.2),
    ]), stop_pct=1.0, rr=2.0, max_bars=73)
    assert out.target == 1
    assert out.stopped == 0


def test_a_clean_stop_is_counted_as_a_loss() -> None:
    out = session_outcomes(_bars([
        ("d1", "A", 100.0, 100.0, 100.0),
        ("d1", "A", 100.2, 98.5, 98.8),
    ]), stop_pct=1.0, rr=2.0, max_bars=73)
    assert out.stopped == 1
    assert out.target == 0


def test_the_conservative_convention_takes_the_stop_on_a_bar_touching_both():
    """A bar spanning both levels is unknowable without ticks. Taking the
    target there is how a backtest flatters itself."""
    rows = _bars([
        ("d1", "A", 100.0, 100.0, 100.0),
        ("d1", "A", 103.0, 98.0, 100.0),      # touches stop 99 AND target 102
    ])
    pess = session_outcomes(rows, stop_pct=1.0, rr=2.0)
    opt = session_outcomes(rows, stop_pct=1.0, rr=2.0, optimistic=True)
    assert pess.stopped == 1 and pess.target == 0
    assert opt.target == 1 and opt.stopped == 0


def test_a_trade_never_runs_into_the_next_session() -> None:
    """THE look-ahead-shaped error here. Holding across the overnight gap is a
    different trade with different risk; if sessions leaked, a target reached
    on the following morning would be scored as an intraday win."""
    rows = _bars([
        ("d1", "A", 100.0, 100.0, 100.0),     # entry, unresolved in d1
        ("d1", "A", 100.3, 99.8, 100.1),
        ("d2", "A", 105.0, 104.0, 104.5),     # would hit the target
        ("d2", "A", 105.0, 104.0, 104.5),
    ])
    out = session_outcomes(rows, stop_pct=1.0, rr=2.0, max_bars=73)
    # d1's entry must time out, not win on d2's bars.
    assert out.target == 0, "a trade resolved using the next session's bars"
    assert out.timed_out >= 1


def test_max_bars_truncates_within_the_session() -> None:
    """The time limit is what truncates winners - it has to actually bite."""
    rows = [("d1", "A", 100.0, 99.9, 100.0)]
    rows += [("d1", "A", 100.2, 99.8, 100.0)] * 8
    rows += [("d1", "A", 103.0, 102.0, 102.5)]      # target, but late
    frame = _bars(rows)
    tight = session_outcomes(frame, stop_pct=1.0, rr=2.0, max_bars=3)
    loose = session_outcomes(frame, stop_pct=1.0, rr=2.0, max_bars=73)
    assert tight.target < loose.target


# -- the arithmetic ---------------------------------------------------------

def test_cost_r_is_round_trip_over_stop_width() -> None:
    """The identity the whole intraday answer turns on: quantity cancels and
    only stop WIDTH matters, so a tight stop is proportionally more
    expensive."""
    out = session_outcomes(_bars([("d1", "A", 100.0, 100.0, 100.0)]),
                           stop_pct=0.5, rr=2.0)
    assert out.cost_r(slippage_pct=0.300) == pytest.approx(
        (STATUTORY_ROUND_TRIP_PCT + 0.300) / 0.5)
    # Halving the stop doubles the cost in R.
    wide = session_outcomes(_bars([("d1", "A", 100.0, 100.0, 100.0)]),
                            stop_pct=1.0, rr=2.0)
    assert out.cost_r() == pytest.approx(2 * wide.cost_r())


def test_breakeven_win_rate_matches_the_published_table() -> None:
    """Pins the figures quoted in the module docstring, so the finding and the
    code cannot drift apart."""
    assert breakeven_win_rate(stop_pct=0.50) == pytest.approx(0.615, abs=0.002)
    assert breakeven_win_rate(stop_pct=1.00) == pytest.approx(0.474, abs=0.002)
    assert breakeven_win_rate(stop_pct=2.00) == pytest.approx(0.404, abs=0.002)
    # A 1:1 payoff needs a much higher rate at the same cost.
    assert breakeven_win_rate(stop_pct=1.0, rr=1.0) > breakeven_win_rate(
        stop_pct=1.0, rr=2.0)


def test_a_negative_required_round_trip_means_no_cost_structure_helps() -> None:
    """THE headline. When gross is negative the breakeven round-trip cost is
    negative - you would have to be paid to trade - so the result cannot be
    blamed on slippage or a broker."""
    # Construct a cell that loses gross: one win, four stops.
    rows = []
    for i in range(4):
        rows += [(f"d{i}", "A", 100.0, 100.0, 100.0),
                 (f"d{i}", "A", 100.1, 98.0, 98.5)]          # stop
    rows += [("d9", "A", 100.0, 100.0, 100.0),
             ("d9", "A", 103.0, 99.9, 102.5)]                # target
    out = session_outcomes(_bars(rows), stop_pct=1.0, rr=2.0)
    assert out.gross_r is not None and out.gross_r < 0
    assert out.required_round_trip_pct() < 0
    assert out.required_round_trip_pct() < STATUTORY_ROUND_TRIP_PCT


def test_a_timeout_is_scored_zero_not_a_loss() -> None:
    """Generous on purpose: the real exit is the session's last close, which
    after costs is usually slightly negative. Being generous means a negative
    verdict cannot be attributed to the convention."""
    out = session_outcomes(_bars([
        ("d1", "A", 100.0, 100.0, 100.0),
        ("d1", "A", 100.1, 99.9, 100.0),
    ]), stop_pct=1.0, rr=2.0)
    assert out.timed_out == 1
    assert out.gross_r == pytest.approx(0.0)


def test_empty_and_malformed_input() -> None:
    empty = session_outcomes(pd.DataFrame(), stop_pct=1.0)
    assert empty.trades == 0
    assert empty.win_rate is None and empty.gross_r is None
    with pytest.raises(ValueError, match="missing"):
        session_outcomes(pd.DataFrame({"symbol": ["A"], "session": ["d1"]}),
                         stop_pct=1.0)


def test_win_rate_and_counts_agree() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for s in range(12):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 40)))
        for v in px:
            rows.append((f"d{s}", "A", v * 1.001, v * 0.999, v))
    out = session_outcomes(_bars(rows), stop_pct=0.5, rr=2.0)
    assert out.trades == out.target + out.stopped + out.timed_out
    assert out.trades > 0
    assert 0.0 <= out.win_rate <= 1.0
