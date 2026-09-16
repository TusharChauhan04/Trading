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
