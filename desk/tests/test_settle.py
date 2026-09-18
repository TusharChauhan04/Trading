"""Closing the journal loop: settle (derived) and close (human).

Until these existed the journal recorded intentions and nothing could
record what happened - `record_outcomes` was reachable from no CLI and no
endpoint, so `unrecorded_outcomes` grew forever and nothing was ever
measurable against the market. A journal that only holds intentions is a
diary, not evidence.

The property tested hardest here: BOTH PATHS MEASURE R AGAINST THE
DECISION'S OWN LEVELS. The risk that was accepted is the one written down
that morning - not one measured against a stop moved afterwards, and not
one implied by where the trade actually filled.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from desk.journal import (
    Decision, ExitReason, JournalStore, Outcome, TradeRecord,
)
from desk.journal.models import IST
from desk.research.refresh import main as research_main

DAY = date(2026, 9, 11)


def _bars(tmp_path, rows):
    """rows: iso date -> (open, high, low, close), one parquet per day."""
    d = tmp_path / "bhavcopy"
    d.mkdir(parents=True, exist_ok=True)
    for iso, (o, h, l, c) in rows.items():
        pd.DataFrame({
            "symbol": ["AAA.NS"], "series": ["EQ"],
            "date": [date.fromisoformat(iso)],
            "open": [o], "high": [h], "low": [l], "close": [c],
            "volume": [100000],
        }).to_parquet(d / f"{iso}.parquet", index=False)
    return d


def _decision(tmp_path, *, entry=100.0, stop=90.0, target=120.0, qty=10):
    store = JournalStore(tmp_path / "journal")
    store.record(Decision(
        as_of=DAY, recorded_at=datetime(2026, 9, 11, 9, tzinfo=IST),
        regime="range", capital=100_000.0,
        trades=(TradeRecord(symbol="AAA.NS", stance="Buy", entry=entry,
                            stop=stop, target=target, qty=qty),)))
    return store


def _settle(tmp_path, *extra):
    return research_main([
        "settle", "--journal", str(tmp_path / "journal"),
        "--bhavcopy", str(tmp_path / "bhavcopy"), *extra])


# --- settle: derived from the bars ---------------------------------------

def test_a_target_is_settled_from_price_history(tmp_path):
    store = _decision(tmp_path)
    _bars(tmp_path, {
        "2026-09-11": (100, 101, 99, 100),
        "2026-09-14": (100, 105, 99, 104),
        "2026-09-15": (104, 125, 103, 124),      # target 120 reached
    })
    assert _settle(tmp_path) == 0

    out = store.outcomes(DAY)
    assert len(out) == 1
    assert out[0].exit_reason == ExitReason.TARGET
    assert out[0].r_multiple == pytest.approx(2.0)


def test_a_stop_is_settled_as_minus_one_r(tmp_path):
    store = _decision(tmp_path)
    _bars(tmp_path, {
        "2026-09-11": (100, 101, 99, 100),
        "2026-09-14": (100, 101, 85, 88),        # through the 90 stop
    })
    _settle(tmp_path)
    out = store.outcomes(DAY)
    assert out[0].exit_reason == ExitReason.STOP
    assert out[0].r_multiple == pytest.approx(-1.0)


def test_an_unresolved_trade_times_out_at_the_horizon(tmp_path):
    store = _decision(tmp_path)
    rows = {"2026-09-11": (100, 101, 99, 100)}
    for d in ("14", "15", "16", "17", "18"):
        rows[f"2026-09-{d}"] = (100, 102, 98, 101)
    _bars(tmp_path, rows)
    _settle(tmp_path, "--horizon", "3")

    out = store.outcomes(DAY)
    assert out[0].exit_reason == ExitReason.TIME
    assert out[0].status == "closed"


def test_a_trade_that_could_not_be_entered_is_recorded_not_dropped(tmp_path):
    """A plan whose trades are routinely skipped is a fact about the desk.
    Deleting those makes the journal describe a desk that does not exist."""
    store = _decision(tmp_path)
    _bars(tmp_path, {"2026-09-11": (100, 101, 99, 100)})   # no session after
    _settle(tmp_path)

    out = store.outcomes(DAY)
    assert len(out) == 1
    assert out[0].exit_reason == ExitReason.NOT_TAKEN
    assert out[0].r_multiple is None
    assert "no session after" in out[0].note


def test_settling_clears_the_unrecorded_list(tmp_path):
    """Which is the whole point - it is the journal's own to-do list."""
    store = _decision(tmp_path)
    _bars(tmp_path, {"2026-09-11": (100, 101, 99, 100),
                     "2026-09-14": (100, 101, 85, 88)})
    assert store.unrecorded() == [DAY]
    _settle(tmp_path)
    assert store.unrecorded() == []


def test_settling_twice_does_not_overwrite_without_force(tmp_path):
    store = _decision(tmp_path)
    _bars(tmp_path, {"2026-09-11": (100, 101, 99, 100),
                     "2026-09-14": (100, 101, 85, 88)})
    _settle(tmp_path)
    first = store.outcomes(DAY)[0].observed_at
    _settle(tmp_path)
    assert store.outcomes(DAY)[0].observed_at == first


def test_nothing_to_settle_is_success_not_an_error(tmp_path):
    (tmp_path / "journal").mkdir(parents=True)
    _bars(tmp_path, {"2026-09-11": (100, 101, 99, 100)})
    assert _settle(tmp_path) == 0


def test_a_no_trade_day_needs_no_settling(tmp_path):
    store = JournalStore(tmp_path / "journal")
    store.record(Decision(as_of=DAY, recorded_at=datetime.now(IST),
                          no_trade_reason="nothing cleared the gate"))
    _bars(tmp_path, {"2026-09-11": (100, 101, 99, 100)})
    assert _settle(tmp_path) == 0
    assert store.unrecorded() == []


def test_an_ambiguous_bar_is_noted_in_the_outcome(tmp_path):
    """The exit simulator resolves stop-and-target-on-one-bar as a stop.
    That assumption has to travel with the record, or a later reader
    cannot tell a measurement from a convention."""
    store = _decision(tmp_path)
    _bars(tmp_path, {
        "2026-09-11": (100, 101, 99, 100),
        "2026-09-14": (100, 125, 85, 110),       # both levels touched
    })
    _settle(tmp_path)
    out = store.outcomes(DAY)[0]
    assert out.exit_reason == ExitReason.STOP
    assert "AMBIGUOUS" in out.note


def test_settle_uses_the_same_simulator_as_the_backtest():
    """So a live decision and a replayed one are scored by identical code
    rather than by two implementations that can drift apart."""
    import inspect

    from desk.research import refresh as mod
    src = inspect.getsource(mod._cmd_settle)
    assert "simulate_trade" in src


# --- close: what a human actually did ------------------------------------

def _close(tmp_path, **kw):
    args = ["close", "--journal", str(tmp_path / "journal"),
            "--day", kw.pop("day", DAY.isoformat())]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return research_main(args)


def test_a_human_exit_is_recorded(tmp_path):
    store = _decision(tmp_path)
    assert _close(tmp_path, symbol="AAA.NS", price=110.0,
                  reason="discretion", on="2026-09-15") == 0

    out = store.outcomes(DAY)[0]
    assert out.exit_reason == ExitReason.DISCRETION
    assert out.exit_price == 110.0
    assert out.exit_date == date(2026, 9, 15)


def test_r_is_measured_against_the_decisions_own_levels(tmp_path):
    """Entry 100, stop 90 - so 10 of risk. Exiting at 110 is +1R whatever
    the stop was later moved to."""
    store = _decision(tmp_path, entry=100.0, stop=90.0)
    _close(tmp_path, symbol="AAA.NS", price=110.0, reason="discretion")
    assert store.outcomes(DAY)[0].r_multiple == pytest.approx(1.0)


def test_a_human_exit_replaces_a_derived_one(tmp_path):
    """What actually happened outranks what the bars imply."""
    store = _decision(tmp_path)
    _bars(tmp_path, {"2026-09-11": (100, 101, 99, 100),
                     "2026-09-14": (100, 101, 85, 88)})
    _settle(tmp_path)
    assert store.outcomes(DAY)[0].r_multiple == pytest.approx(-1.0)

    _close(tmp_path, symbol="AAA.NS", price=95.0, reason="discretion")
    out = store.outcomes(DAY)
    assert len(out) == 1, "it must replace, not duplicate"
    assert out[0].r_multiple == pytest.approx(-0.5)


def test_a_trade_never_taken_can_be_recorded_as_such(tmp_path):
    store = _decision(tmp_path)
    _close(tmp_path, symbol="AAA.NS", price=0, reason="not_taken",
           note="never filled")
    assert store.outcomes(DAY)[0].exit_reason == ExitReason.NOT_TAKEN


def test_an_unknown_reason_is_refused(tmp_path):
    _decision(tmp_path)
    assert _close(tmp_path, symbol="AAA.NS", price=100.0, reason="vibes") == 1


def test_a_symbol_not_in_the_decision_is_refused(tmp_path):
    """Recording an outcome against a trade the desk never proposed would
    put a result in the journal with no decision behind it."""
    _decision(tmp_path)
    assert _close(tmp_path, symbol="ZZZ.NS", price=100.0, reason="stop") == 1


def test_closing_an_unrecorded_day_is_refused(tmp_path):
    (tmp_path / "journal").mkdir(parents=True)
    assert _close(tmp_path, symbol="AAA.NS", price=100.0, reason="stop") == 1


def test_both_commands_are_reachable_from_the_cli():
    """The defect this file exists to close: record_outcomes was callable
    from nothing."""
    import inspect

    from desk.research import refresh as mod
    src = inspect.getsource(mod.main)
    assert '"settle"' in src and '"close"' in src
    assert "_cmd_settle" in src and "_cmd_close" in src
