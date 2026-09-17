"""Regression tests for the refresh CLI's calendar merge/atomic-write logic.

None of these touch the network. `_FakeSession` stands in for `NseSession` and
returns a canned holiday-master payload shaped exactly like the real API
response (segments keyed by name, CM = Capital Market / equities).

Added after a deployment audit found these code paths - genuinely fixed,
verified by hand that session - had zero automated coverage: a future
regression here would only be caught by someone re-running the same manual
check, which is exactly the failure mode this file exists to remove.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from desk.marketdata.calendar_in import CalendarError, TradingCalendar
from desk.marketdata.refresh import _merge_calendars, refresh_calendar

FIXTURES = Path(__file__).parent / "fixtures"


def _cm_payload(entries: list[tuple[str, str]]) -> dict:
    """entries: list of (tradingDate, description)."""
    return {"CM": [{"tradingDate": d, "weekDay": "", "description": n,
                    "morning_session": None, "evening_session": None}
                   for d, n in entries]}


class _FakeSession:
    """Stands in for NseSession. No network, no cookies, no HTTP."""
    def __init__(self, payload: dict):
        self._payload = payload

    def fetch_holiday_master(self) -> dict:
        return self._payload


# ===========================================================================
# _merge_calendars - pure function, no I/O
# ===========================================================================

def test_years_accumulate_rather_than_replace():
    existing = {"years": [2026], "holidays": [{"date": "2026-01-26", "name": "Republic Day"}],
               "special_sessions": []}
    fresh = {"years": [2027], "holidays": [{"date": "2027-01-26", "name": "Republic Day"}],
            "special_sessions": []}
    merged = _merge_calendars(existing, fresh)
    assert merged["years"] == [2026, 2027]
    assert {h["date"] for h in merged["holidays"]} == {"2026-01-26", "2027-01-26"}


def test_fresh_data_wins_on_the_same_date():
    """NSE amends its own list occasionally - a re-fetch of an already-loaded
    year should update the entry, not just leave the stale one."""
    existing = {"years": [2026], "holidays": [{"date": "2026-03-03", "name": "Holi (old)"}],
               "special_sessions": []}
    fresh = {"years": [2026], "holidays": [{"date": "2026-03-03", "name": "Holi (corrected)"}],
            "special_sessions": []}
    merged = _merge_calendars(existing, fresh)
    assert len(merged["holidays"]) == 1
    assert merged["holidays"][0]["name"] == "Holi (corrected)"


def test_exchange_mismatch_is_refused_not_silently_merged():
    existing = {"exchange": "NSE", "years": [2026], "holidays": [], "special_sessions": []}
    fresh = {"exchange": "BSE", "years": [2026], "holidays": [], "special_sessions": []}
    with pytest.raises(CalendarError, match="refusing to merge"):
        _merge_calendars(existing, fresh)


def test_malformed_years_raises_calendar_error_not_a_bare_value_error():
    """REGRESSION (deployment re-verification): int(y) on a malformed years
    field raised a bare ValueError, which main()'s handler - catching only
    CalendarError/SourceError - would not catch, producing an unhandled
    traceback instead of the module's normal 'ERROR: ...' message."""
    existing = {"years": [2026], "holidays": [], "special_sessions": []}
    fresh = {"years": ["not-a-year"], "holidays": [], "special_sessions": []}
    with pytest.raises(CalendarError, match="malformed 'years'"):
        _merge_calendars(existing, fresh)


def test_special_sessions_merge_the_same_way_as_holidays():
    existing = {"years": [2026], "holidays": [],
               "special_sessions": [{"date": "2026-11-08", "name": "Muhurat",
                                     "start": "18:15", "end": "19:15"}]}
    fresh = {"years": [2026], "holidays": [], "special_sessions": []}
    merged = _merge_calendars(existing, fresh)
    assert len(merged["special_sessions"]) == 1     # preserved, not dropped


# ===========================================================================
# refresh_calendar - merge, atomic write, --replace
# ===========================================================================

def test_refresh_merges_into_an_existing_file_without_erasing_prior_years(tmp_path):
    out = tmp_path / "holidays_nse.json"
    out.write_text(json.dumps({
        "exchange": "NSE", "years": [2026],
        "holidays": [{"date": "2026-01-26", "name": "Republic Day"}],
        "special_sessions": [],
    }), encoding="utf-8")

    session = _FakeSession(_cm_payload([("26-Jan-2027", "Republic Day")]))
    refresh_calendar(session, out)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["years"] == [2026, 2027]
    cal = TradingCalendar.from_file(out)
    assert cal.is_trading_day(date(2026, 1, 26)) is False    # 2026 survived
    assert cal.is_trading_day(date(2027, 1, 26)) is False    # 2027 added


def test_replace_flag_discards_prior_years_on_purpose(tmp_path):
    out = tmp_path / "holidays_nse.json"
    out.write_text(json.dumps({
        "exchange": "NSE", "years": [2026],
        "holidays": [{"date": "2026-01-26", "name": "Republic Day"}],
        "special_sessions": [],
    }), encoding="utf-8")

    session = _FakeSession(_cm_payload([("26-Jan-2027", "Republic Day")]))
    refresh_calendar(session, out, replace=True)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["years"] == [2027]                          # 2026 is GONE


def test_a_malformed_existing_file_fails_at_the_temp_stage_not_on_disk(tmp_path):
    """REGRESSION: the original wrote the file first and validated after, so
    a bad merge result destroyed the good file it was meant to update. Now it
    writes to a temp file, validates THAT, and only replaces on success.

    parse_holiday_master always hands TradingCalendar well-formed dates - the
    real way this fires is a pre-existing on-disk file with a typo (hand-
    edited, or from an older format) that _merge_calendars happily carries
    forward (it only checks a "date" KEY exists, not that the VALUE parses),
    and only TradingCalendar.from_file's own _parse_date catches.
    """
    out = tmp_path / "holidays_nse.json"
    bad_existing = json.dumps({
        "exchange": "NSE", "years": [2026],
        "holidays": [{"date": "not-a-real-date", "name": "Typo'd Holiday"}],
        "special_sessions": [],
    })
    out.write_text(bad_existing, encoding="utf-8")

    session = _FakeSession(_cm_payload([("26-Jan-2027", "Republic Day")]))
    with pytest.raises(CalendarError, match="bad date"):
        refresh_calendar(session, out)

    # The real file must be exactly what it was before the failed attempt -
    # still carrying the typo, but not further corrupted, and the operator
    # gets the real error instead of a silently swapped-in half-good file.
    assert out.read_text(encoding="utf-8") == bad_existing
    assert not out.with_suffix(".json.tmp").exists()   # temp file cleaned up


def test_exchange_mismatch_during_a_live_refresh_leaves_the_file_untouched(tmp_path):
    out = tmp_path / "holidays_nse.json"
    good = json.dumps({
        "exchange": "NSE", "years": [2026],
        "holidays": [{"date": "2026-01-26", "name": "Republic Day"}],
        "special_sessions": [],
    })
    out.write_text(good, encoding="utf-8")

    session = _FakeSession(_cm_payload([("26-Jan-2027", "Republic Day")]))
    # parse_holiday_master always tags "exchange": "NSE" itself, so simulate
    # a mismatch by refreshing into a file that claims to be BSE's.
    out.write_text(good.replace('"NSE"', '"BSE"'), encoding="utf-8")
    with pytest.raises(CalendarError, match="refusing to merge"):
        refresh_calendar(session, out)
    assert out.read_text(encoding="utf-8") == good.replace('"NSE"', '"BSE"')


# ===========================================================================
# refresh_bhavcopy - fetch + parse + atomic write, no network
# ===========================================================================

class _FakeBhavcopySession:
    def __init__(self, raw: bytes):
        self._raw = raw

    def fetch_bhavcopy(self, day):
        return self._raw


def test_refresh_bhavcopy_writes_a_readable_parquet_file(tmp_path):
    import pandas as pd
    from desk.marketdata.refresh import refresh_bhavcopy

    raw = (FIXTURES / "nse_bhavcopy_20260911.csv").read_bytes()
    session = _FakeBhavcopySession(raw)
    out_dir = tmp_path / "bhavcopy"
    rc = refresh_bhavcopy(session, date(2026, 9, 11), out_dir)
    assert rc == 0

    saved = out_dir / "2026-09-11.parquet"
    assert saved.exists()
    assert not saved.with_suffix(".parquet.tmp").exists()   # temp cleaned up

    df = pd.read_parquet(saved)
    assert len(df) == 3485
    assert (df["symbol"] == "RELIANCE.NS").any()


def test_refresh_bhavcopy_refuses_the_wrong_day_and_writes_nothing(tmp_path):
    from desk.marketdata.calendar_in import CalendarError
    from desk.marketdata.refresh import refresh_bhavcopy
    from desk.marketdata.sources.nse import SourceError

    raw = (FIXTURES / "nse_bhavcopy_20260911.csv").read_bytes()
    session = _FakeBhavcopySession(raw)
    out_dir = tmp_path / "bhavcopy"
    with pytest.raises(SourceError, match="expected bhavcopy for"):
        refresh_bhavcopy(session, date(2020, 1, 1), out_dir)
    assert not (out_dir / "2020-01-01.parquet").exists()


# ===========================================================================
# The bhavcopy range backfill
# ===========================================================================
#
# Added with the range option itself. The single-day command was the only
# way to fetch history, so a 60-day backfill meant a shell loop - and a
# shell loop starts a new NseSession per day, which means a fresh cookie
# handshake each time AND no throttle between days, because the throttle is
# per-session. That is the exact traffic shape that gets an IP blocked.

from datetime import timedelta

from desk.marketdata.refresh import backfill_bhavcopy
from desk.marketdata.sources.errors import RateLimited, SourceError


class _CountingSession:
    """Stands in for NseSession. Records which days were asked for."""

    def __init__(self, fail_on=None, rate_limit_on=None):
        self.asked: list[date] = []
        self.fail_on = fail_on or set()
        self.rate_limit_on = rate_limit_on

    def fetch_bhavcopy(self, day: date) -> bytes:
        self.asked.append(day)
        if self.rate_limit_on == day:
            raise RateLimited(f"429 on {day}")
        if day in self.fail_on:
            raise SourceError(f"no file for {day}")
        return b"stub"


def _patch_writer(monkeypatch, session_cls_out_dir=None):
    """Replace refresh_bhavcopy with a stub that writes a marker file.

    The parsing path is covered by test_nse_source.py; what matters here is
    which days are ATTEMPTED and how failures are classified.
    """
    from desk.marketdata import refresh as mod

    def _fake(session, day, out_dir):
        session.fetch_bhavcopy(day)           # so the session records it
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{day.isoformat()}.parquet").write_bytes(b"x")
        return 0

    monkeypatch.setattr(mod, "refresh_bhavcopy", _fake)


def test_backfill_skips_weekends(monkeypatch, tmp_path):
    _patch_writer(monkeypatch)
    s = _CountingSession()
    # 2026-09-05 is a Saturday, 09-06 a Sunday.
    rc = backfill_bhavcopy(s, date(2026, 9, 4), date(2026, 9, 7), tmp_path)
    assert rc == 0
    assert s.asked == [date(2026, 9, 4), date(2026, 9, 7)]


def test_backfill_skips_days_already_on_disk(monkeypatch, tmp_path):
    """So an interrupted run resumes instead of refetching from zero."""
    _patch_writer(monkeypatch)
    (tmp_path / "2026-09-07.parquet").write_bytes(b"already here")
    s = _CountingSession()
    backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 9), tmp_path)
    assert date(2026, 9, 7) not in s.asked
    assert s.asked == [date(2026, 9, 8), date(2026, 9, 9)]


def test_backfill_uses_one_session_for_the_whole_range(monkeypatch, tmp_path):
    """The reason this function exists. A shell loop would hand each day a
    fresh session: a new cookie handshake per day, and no throttle between
    them because the throttle is per-session."""
    _patch_writer(monkeypatch)
    s = _CountingSession()
    backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 11), tmp_path)
    assert len(s.asked) == 5, "every day must go through the same session"


def test_a_rate_limit_stops_the_run_and_exits_two(monkeypatch, tmp_path):
    """Exit 2 is distinct from exit 1: a scheduler should retry 'come back
    later' and should not retry 'something is broken'."""
    _patch_writer(monkeypatch)
    s = _CountingSession(rate_limit_on=date(2026, 9, 9))
    rc = backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 11), tmp_path)
    assert rc == 2
    assert date(2026, 9, 10) not in s.asked, "must stop, not grind on"


def test_work_done_before_a_rate_limit_survives(monkeypatch, tmp_path):
    _patch_writer(monkeypatch)
    s = _CountingSession(rate_limit_on=date(2026, 9, 9))
    backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 11), tmp_path)
    assert (tmp_path / "2026-09-07.parquet").exists()
    assert (tmp_path / "2026-09-08.parquet").exists()

    # And rerunning resumes from where it stopped.
    s2 = _CountingSession()
    backfill_bhavcopy(s2, date(2026, 9, 7), date(2026, 9, 11), tmp_path)
    assert date(2026, 9, 7) not in s2.asked
    assert date(2026, 9, 9) in s2.asked


def test_a_missing_file_without_a_calendar_is_reported_as_unresolved(
        monkeypatch, tmp_path, capsys):
    """NSE serves nothing for a non-trading day, so with no calendar a
    holiday and a broken fetch look identical. That ambiguity is REPORTED,
    never guessed - an earlier version sniffed the error text for "404" and
    called the remainder holidays, which would mark a real outage as a
    holiday and carry on.

    The run still succeeds, because a backfill spanning any holiday would
    otherwise always exit non-zero, and a command that always fails stops
    being read."""
    _patch_writer(monkeypatch)
    s = _CountingSession(fail_on={date(2026, 9, 9)})
    rc = backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 11), tmp_path,
                           calendar=None)
    assert rc == 0
    assert (tmp_path / "2026-09-10.parquet").exists(), "must keep going"

    out = capsys.readouterr().out
    assert "unresolved" in out
    assert "cannot be told apart" in out
    # And it must not ASSERT which one it was.
    assert "likely holiday" not in out
    assert "probably a holiday" not in out


def test_a_real_failure_with_a_calendar_exits_non_zero(monkeypatch, tmp_path):
    """With a calendar saying the day trades, no file IS a failure."""
    _patch_writer(monkeypatch)

    class _Cal:
        def is_trading_day(self, d):
            return d.weekday() < 5

    s = _CountingSession(fail_on={date(2026, 9, 9)})
    rc = backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 11), tmp_path,
                           calendar=_Cal())
    assert rc == 1


def test_a_calendar_removes_holidays_from_the_range(monkeypatch, tmp_path):
    _patch_writer(monkeypatch)

    class _Cal:
        def is_trading_day(self, d):
            return d.weekday() < 5 and d != date(2026, 9, 9)

    s = _CountingSession()
    backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 11), tmp_path,
                      calendar=_Cal())
    assert date(2026, 9, 9) not in s.asked


def test_a_reversed_range_is_refused(tmp_path):
    s = _CountingSession()
    assert backfill_bhavcopy(s, date(2026, 9, 11), date(2026, 9, 1),
                             tmp_path) == 1
    assert s.asked == []


def test_a_fully_cached_range_asks_for_nothing(monkeypatch, tmp_path):
    _patch_writer(monkeypatch)
    for d in ("2026-09-07", "2026-09-08"):
        (tmp_path / f"{d}.parquet").write_bytes(b"x")
    s = _CountingSession()
    assert backfill_bhavcopy(s, date(2026, 9, 7), date(2026, 9, 8),
                             tmp_path) == 0
    assert s.asked == []
