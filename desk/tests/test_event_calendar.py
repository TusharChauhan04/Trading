"""Event-calendar persistence, and the Stage 3 -> Stage 4 narrowing.

The property this file exists for: A STALE CALENDAR MUST NOT LOOK LIKE A
CLEAN CHECK. Every other failure here is loud. This one is silent - a
calendar fetched two weeks ago answers "nothing scheduled" for every company
that has announced a board meeting since, and the gate reports that it ran
and found nothing. That is a false clear, and a position goes on through an
earnings print.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from desk.contracts.enums import Regime
from desk.research.events import (
    DEFAULT_MAX_AGE_DAYS, EventCalendar, load_calendar, save_calendar,
)
from desk.research.models import CorporateEvent
from desk.scanner.stage2 import DEFAULT_FACTORS, Stage2Result
from desk.scanner.stage3 import Stage3Result, run_stage3

IST = timezone(timedelta(hours=5, minutes=30))
DAY = date(2026, 9, 18)


def _events():
    return [
        CorporateEvent("AAA", date(2026, 9, 21), "Financial Results", "A Ltd"),
        CorporateEvent("BBB", date(2026, 11, 4), "Financial Results", "B Ltd"),
        CorporateEvent("CCC", None, "Undated thing", "C Ltd"),
    ]


def _fresh(tmp_path, when=None):
    p = tmp_path / "event_calendar.json"
    save_calendar(_events(), p, fetched_at=when or datetime(2026, 9, 18, 8,
                                                            tzinfo=IST))
    return p


# --- round trip ----------------------------------------------------------

def test_undated_events_are_dropped_on_save(tmp_path):
    """An undated event cannot answer 'is this inside my holding window'."""
    p = tmp_path / "cal.json"
    assert save_calendar(_events(), p) == 2
    assert len(json.loads(p.read_text())["events"]) == 2


def test_round_trip_preserves_the_gate_behaviour(tmp_path):
    stored = load_calendar(_fresh(tmp_path), as_of=DAY)
    assert stored is not None and not stored.stale
    win = stored.calendar.window("AAA.NS", as_of=DAY)
    assert win.days_until == 3
    assert win.blocks(5) is True
    assert stored.calendar.window("BBB", as_of=DAY).blocks(5) is False


def test_saving_is_atomic(tmp_path):
    """A truncated calendar still PARSES - as a shorter one. More names
    would report nothing scheduled."""
    p = tmp_path / "cal.json"
    save_calendar(_events(), p)
    save_calendar(_events(), p)
    assert not list(tmp_path.glob("*.tmp"))


# --- staleness, the reason this module exists ----------------------------

def test_a_fresh_calendar_is_usable(tmp_path):
    stored = load_calendar(_fresh(tmp_path), as_of=DAY)
    assert not stored.stale
    assert stored.age_days == 0
    assert stored.caveat is None


@pytest.mark.parametrize("age,stale", [
    (0, False), (DEFAULT_MAX_AGE_DAYS, False), (DEFAULT_MAX_AGE_DAYS + 1, True),
    (30, True),
])
def test_staleness_boundary(tmp_path, age, stale):
    when = datetime(2026, 9, 18, 8, tzinfo=IST) - timedelta(days=age)
    p = _fresh(tmp_path, when=when)
    stored = load_calendar(p, as_of=DAY)
    assert stored.age_days == age
    assert stored.stale is stale


def test_a_stale_calendar_says_the_gate_did_not_run(tmp_path):
    p = _fresh(tmp_path, when=datetime(2026, 9, 1, 8, tzinfo=IST))
    stored = load_calendar(p, as_of=DAY)
    assert stored.stale
    assert "NOT applied" in stored.caveat
    assert "17 day(s) ago" in stored.caveat


def test_a_calendar_from_the_future_is_stale(tmp_path):
    """Fetched after the scan date, so it holds announcements the market had
    not yet seen. Refused as look-ahead, not as age - see
    test_a_calendar_fetched_after_the_scan_date_is_named_as_look_ahead."""
    p = _fresh(tmp_path, when=datetime(2026, 10, 1, 8, tzinfo=IST))
    assert load_calendar(p, as_of=DAY).stale


# --- absence is absence, never a partial answer --------------------------

def test_a_missing_file_is_none(tmp_path):
    assert load_calendar(tmp_path / "nope.json", as_of=DAY) is None


def test_a_corrupt_file_is_none_not_a_partial_calendar(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"fetched_at": "2026-09-18T08:00', encoding="utf-8")
    assert load_calendar(p, as_of=DAY) is None


def test_a_file_with_no_timestamp_is_none(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"events": []}), encoding="utf-8")
    assert load_calendar(p, as_of=DAY) is None


def test_unreadable_rows_are_skipped_not_fatal(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({
        "fetched_at": datetime(2026, 9, 18, 8, tzinfo=IST).isoformat(),
        "events": [{"symbol": "AAA", "event_date": "2026-09-21",
                    "purpose": "Financial Results"},
                   {"symbol": "BAD", "event_date": "not-a-date"},
                   "not even a dict"],
    }), encoding="utf-8")
    stored = load_calendar(p, as_of=DAY)
    assert stored.calendar.symbols == ["AAA"]


def test_an_empty_calendar_loads_and_blocks_nothing(tmp_path):
    """It must still be distinguishable from a real one - EventCalendar
    already reports unknown symbols as known=False."""
    p = tmp_path / "cal.json"
    save_calendar([], p)
    stored = load_calendar(p, as_of=DAY)
    assert stored.calendar.symbols == []
    assert stored.calendar.window("AAA", as_of=DAY).known is False


# --- Stage 3 -> Stage 4 narrowing ----------------------------------------

def _stage2(symbols):
    n = len(symbols)
    data = {"score": [90.0 - i for i in range(n)], "factors_used": [2] * n,
            "penalty": [0.0] * n}
    for f in DEFAULT_FACTORS[:2]:
        data[f"rank_{f.name}"] = [95.0 - i for i in range(n)]
    ranked = pd.DataFrame(data, index=list(symbols))
    ranked.index.name = "symbol"
    return Stage2Result(as_of=DAY, regime=Regime.TRENDING_UP, ranked=ranked,
                        factors=DEFAULT_FACTORS[:2], universe_in=1598,
                        universe_out=n)


def test_narrow_keeps_only_survivors_in_ranking_order():
    s2 = _stage2(["A.NS", "B.NS", "C.NS", "D.NS"])
    s3 = Stage3Result(as_of=DAY, kept=["C.NS", "A.NS"])
    out = s3.narrow(s2)
    assert list(out.ranked.index) == ["A.NS", "C.NS"], "ranking order wins"
    assert out.universe_out == 2


def test_narrow_preserves_the_audit_trail():
    """Stage 2's explain() must still work on anything that survives, or the
    plan cannot say why a trade was taken."""
    s2 = _stage2(["A.NS", "B.NS"])
    out = Stage3Result(as_of=DAY, kept=["A.NS"]).narrow(s2)
    assert "A.NS" in out.explain("A.NS")
    assert list(out.ranked.columns) == list(s2.ranked.columns)


def test_narrow_cannot_introduce_a_name():
    s2 = _stage2(["A.NS"])
    out = Stage3Result(as_of=DAY, kept=["A.NS", "GHOST.NS"]).narrow(s2)
    assert list(out.ranked.index) == ["A.NS"]


def test_narrow_with_nothing_kept_gives_an_empty_ranking():
    s2 = _stage2(["A.NS", "B.NS"])
    out = Stage3Result(as_of=DAY, kept=[]).narrow(s2)
    assert out.ranked.empty and out.universe_out == 0


def test_a_skipped_stage3_narrows_to_exactly_what_it_was_given():
    """With no client, Stage 4 must see the identical shortlist it would
    have seen before Stage 3 existed."""
    s2 = _stage2(["A.NS", "B.NS", "C.NS"])
    out = run_stage3(s2, client=None).narrow(s2)
    assert list(out.ranked.index) == ["A.NS", "B.NS", "C.NS"]
    assert out.universe_out == s2.universe_out


# --- the funnel's own wiring ---------------------------------------------

def test_the_api_reports_a_missing_calendar_rather_than_ignoring_it(tmp_path,
                                                                    monkeypatch):
    from desk.api import main as api
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    cal, caveats = api._event_calendar(DAY)
    assert cal is None
    assert caveats and "earnings gate did NOT run" in caveats[0]
    assert "desk.research.refresh events" in caveats[0]


def test_the_api_treats_a_stale_calendar_as_absent(tmp_path, monkeypatch):
    from desk.api import main as api
    (tmp_path / "research").mkdir()
    save_calendar(_events(), tmp_path / "research" / "event_calendar.json",
                  fetched_at=datetime(2026, 8, 1, 8, tzinfo=IST))
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    cal, caveats = api._event_calendar(DAY)
    assert cal is None, "a stale calendar must not be used"
    assert caveats and "too old to trust" in caveats[0]


def test_the_api_uses_a_fresh_calendar(tmp_path, monkeypatch):
    """`today=DAY` is load-bearing, not tidiness.

    Without it `_event_calendar` measures staleness against the WALL CLOCK,
    so this test asserted "a calendar fetched on 2026-09-18 is fresh" and
    the answer changed as the real date advanced. It passed while the
    machine's clock was within the 3-day window of that fixture and began
    failing on the fourth day - a red suite caused by the calendar on the
    wall rather than by any code, which is the most expensive kind of
    failure to debug because nothing in the diff explains it.

    Staleness is measured against the DECISION date by design (see the
    function's own docstring), and the decision date here is DAY. Pinning
    it is also the more honest test: the rule under test is "a calendar
    fetched the morning of the decision is fresh", and that statement
    should not depend on when the suite happens to run.
    """
    from desk.api import main as api
    (tmp_path / "research").mkdir()
    save_calendar(_events(), tmp_path / "research" / "event_calendar.json",
                  fetched_at=datetime(2026, 9, 18, 8, tzinfo=IST))
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    cal, caveats = api._event_calendar(DAY, today=DAY)
    assert isinstance(cal, EventCalendar)
    assert caveats == []


def test_a_calendar_older_than_the_window_is_stale_on_any_clock(tmp_path,
                                                                monkeypatch):
    """The boundary the test above used to straddle, pinned on both sides.

    Measured: fetched 2026-09-18, the calendar reads fresh at a decision
    date of 2026-09-21 (3 days) and stale at 2026-09-22 (4 days). Notice
    is that short in Indian markets, which is why the window is tight.
    """
    from desk.api import main as api
    (tmp_path / "research").mkdir()
    save_calendar(_events(), tmp_path / "research" / "event_calendar.json",
                  fetched_at=datetime(2026, 9, 18, 8, tzinfo=IST))
    monkeypatch.setattr(api, "CONFIGS", tmp_path)

    fresh, _ = api._event_calendar(DAY, today=date(2026, 9, 21))
    assert isinstance(fresh, EventCalendar), "3 days old must still count"

    stale, caveats = api._event_calendar(DAY, today=date(2026, 9, 22))
    assert stale is None, "4 days old must be treated as absent, not used"
    assert caveats and "4 day(s) ago" in caveats[0]


def test_no_llm_key_means_no_client_and_no_spend(monkeypatch):
    """Set EMPTY rather than deleted, deliberately.

    Deleting it from os.environ is not enough once a real key exists in
    .env: Settings.from_env() re-reads the file and puts it back, which
    is exactly the behaviour production wants. An empty value in the
    process environment WINS over the file - load_env skips any key
    already present - so this asserts the real precedence rule rather
    than one that only holds on a machine with no key configured.
    """
    from desk.api import main as api
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert api._llm_client() is None


def test_a_calendar_fetched_after_the_scan_date_is_named_as_look_ahead():
    """Re-running a past day is normal, and a calendar fetched since then
    holds announcements the market had not yet seen. Refusing it is the
    point-in-time guarantee, not a staleness fault - and the caveat has to
    say which, or it reads like a clock bug."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cal.json"
        save_calendar(_events(), p,
                      fetched_at=datetime(2026, 9, 18, 8, tzinfo=IST))
        stored = load_calendar(p, as_of=date(2026, 9, 11))
        assert stored.stale and stored.from_the_future
        assert "7 day(s) AFTER" in stored.caveat
        assert "look-ahead protection" in stored.caveat
        assert "-7" not in stored.caveat


def test_an_ordinary_stale_calendar_is_not_called_look_ahead(tmp_path):
    p = _fresh(tmp_path, when=datetime(2026, 9, 1, 8, tzinfo=IST))
    stored = load_calendar(p, as_of=DAY)
    assert stored.stale and not stored.from_the_future
    assert "look-ahead" not in stored.caveat


# --- the bug that made this gate fire exactly never ----------------------

def test_a_live_plan_uses_todays_calendar(tmp_path, monkeypatch):
    """REGRESSION, and it silenced R4 completely in production.

    Staleness was measured against `as_of` - the date of the DATA being
    scanned - instead of against the DECISION date. Those are always
    different in live use: bhavcopy for day D publishes after D closes, so
    a plan built on D's bars is decided on D+1 at the earliest. Every
    calendar therefore looked like it came "from the future" and the
    look-ahead guard refused all of them.

    The calendar is FORWARD-LOOKING. For a live plan, today's calendar is
    the actual schedule of what is coming - that is the point of it.
    """
    from desk.api import main as api
    (tmp_path / "research").mkdir()
    save_calendar(_events(), tmp_path / "research" / "event_calendar.json",
                  fetched_at=datetime(2026, 9, 18, 8, tzinfo=IST))
    monkeypatch.setattr(api, "CONFIGS", tmp_path)

    # Scanning the 17th's bars, deciding on the 18th - the normal case.
    cal, caveats = api._event_calendar(date(2026, 9, 17),
                                       today=date(2026, 9, 18))
    assert cal is not None, "the live gate must use today's calendar"
    assert caveats == []


def test_a_replay_still_refuses_a_calendar_from_after_the_simulated_day(
        tmp_path, monkeypatch):
    """The protection that the fix must NOT remove. In a replay the
    simulated date is passed as `today`, and a calendar fetched later
    genuinely holds intimations the market had not seen."""
    from desk.api import main as api
    (tmp_path / "research").mkdir()
    save_calendar(_events(), tmp_path / "research" / "event_calendar.json",
                  fetched_at=datetime(2026, 9, 18, 8, tzinfo=IST))
    monkeypatch.setattr(api, "CONFIGS", tmp_path)

    cal, caveats = api._event_calendar(date(2026, 9, 1), today=date(2026, 9, 1))
    assert cal is None
    assert "look-ahead protection" in caveats[0]


def test_the_backtest_passes_its_simulated_date_as_today():
    """Which is what keeps the replay protected after the fix."""
    import inspect

    from desk.backtest import engine
    code = [ln.split("#")[0]
            for ln in inspect.getsource(engine.run_backtest).splitlines()]
    assert any("today=day" in ln for ln in code)


def test_a_genuinely_old_calendar_is_still_refused_live(tmp_path, monkeypatch):
    """The fix changes WHICH date staleness is measured against, not
    whether staleness matters. A calendar nobody has refreshed in weeks is
    still refused."""
    from desk.api import main as api
    (tmp_path / "research").mkdir()
    save_calendar(_events(), tmp_path / "research" / "event_calendar.json",
                  fetched_at=datetime(2026, 8, 20, 8, tzinfo=IST))
    monkeypatch.setattr(api, "CONFIGS", tmp_path)

    cal, caveats = api._event_calendar(date(2026, 9, 17),
                                       today=date(2026, 9, 18))
    assert cal is None
    assert "too old to trust" in caveats[0]
