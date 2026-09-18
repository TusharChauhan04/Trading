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


# ===========================================================================
# refresh_crosscheck - the command that PRODUCES the report
# ===========================================================================
#
# Found by auditing for public functions nothing references: the endpoint
# that SERVES the cross-check was tested, and the command that produces it
# had zero tests. It had been written, wired and run live against NSE and
# BSE - which is three of the four conditions for done, and the missing
# one is the one that catches a regression.

import json as _json

from desk.marketdata.isin import IsinMap
from desk.marketdata.refresh import refresh_crosscheck


class _StubNse:
    """Stands in for NseSession. Named distinctly from this file's own
    _FakeSession, which serves the calendar tests and takes a payload."""

    def fetch_with_retry(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def fetch_equity_master(self):
        return b"\n".join([
            b"SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER",
            b"AAA,A Ltd,EQ,INE000A01001",
            b"",
        ])


class _StubBse:
    """Stands in for BseSession, which refresh_crosscheck constructs
    internally - so it is patched at the module it is imported from."""

    def __init__(self, raw=b"", fail=None):
        self.raw = raw
        self.fail = fail
        self.calls = 0

    def fetch_with_retry(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def fetch_bhavcopy(self, day):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.raw


def _bse_csv(rows):
    """rows: (isin, ticker, close, turnover)."""
    head = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,"
            "SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,"
            "OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,"
            "UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
            "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,"
            "Rsvd3,Rsvd4\n")
    body = "".join(
        f"2026-09-17,2026-09-17,CM,BSE,STK,1,{isin},{tkr},A,,,,,{tkr} LTD,"
        f"{close},{close},{close},{close},{close},{close},,{close},,,1000,"
        f"{turn},10,F1,1,,,,,\n"
        for isin, tkr, close, turn in rows)
    return (head + body).encode()


def _nse_snapshot(tmp_path, rows):
    """rows: (symbol, close)."""
    import pandas as pd
    d = tmp_path / "bhavcopy"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "symbol": [s for s, _ in rows], "series": ["EQ"] * len(rows),
        "date": [date(2026, 9, 17)] * len(rows),
        "open": [c for _, c in rows], "high": [c for _, c in rows],
        "low": [c for _, c in rows], "close": [c for _, c in rows],
        "volume": [100000] * len(rows), "turnover_lacs": [500.0] * len(rows),
    }).to_parquet(d / "2026-09-17.parquet", index=False)
    return d


def _isin_file(tmp_path, mapping):
    p = tmp_path / "isin.json"
    IsinMap(by_symbol=mapping,
            by_isin={v: k for k, v in mapping.items()}).save(p)
    return p


def _run(tmp_path, monkeypatch, *, bse_rows, nse_rows, mapping, fail=None):
    from desk.marketdata.sources import bse as bse_mod

    stub = _StubBse(_bse_csv(bse_rows), fail=fail)
    monkeypatch.setattr(bse_mod, "BseSession", lambda *a, **k: stub)
    out = tmp_path / "crosscheck"
    rc = refresh_crosscheck(
        _StubNse(), date(2026, 9, 17), out,
        bhavcopy_dir=_nse_snapshot(tmp_path, nse_rows),
        isin_path=_isin_file(tmp_path, mapping))
    return rc, out / "2026-09-17.json", stub


def test_a_clean_crosscheck_writes_a_report(tmp_path, monkeypatch):
    rc, report, _ = _run(
        tmp_path, monkeypatch,
        nse_rows=[("AAA.NS", 100.0)],
        bse_rows=[("INE000A01001", "AAA", 100.0, 5e7)],
        mapping={"AAA": "INE000A01001"})

    assert rc == 0
    assert report.is_file()
    body = _json.loads(report.read_text(encoding="utf-8"))
    assert body["as_of"] == "2026-09-17"
    assert body["coverage"]["checkable"] == 1
    assert body["disagreements"] == []


def test_a_disagreement_is_recorded(tmp_path, monkeypatch):
    rc, report, _ = _run(
        tmp_path, monkeypatch,
        nse_rows=[("AAA.NS", 100.0)],
        bse_rows=[("INE000A01001", "AAA", 130.0, 5e7)],
        mapping={"AAA": "INE000A01001"})

    body = _json.loads(report.read_text(encoding="utf-8"))
    assert len(body["disagreements"]) == 1
    assert body["disagreements"][0]["symbol"] == "AAA.NS"
    assert body["disagreements"][0]["diff_pct"] > 20
    assert rc == 0, ("a price disagreement is a WARNING for the plan to "
                     "surface, not a failed refresh - both exchanges are "
                     "real and a scheduled run must not look broken")


def test_a_thin_bse_name_is_not_checked_rather_than_cleared(tmp_path,
                                                            monkeypatch):
    """Below the liquidity gate the BSE close is one trade, not an
    independent measurement. Agreeing with it is not corroboration."""
    rc, report, _ = _run(
        tmp_path, monkeypatch,
        nse_rows=[("AAA.NS", 100.0)],
        bse_rows=[("INE000A01001", "AAA", 140.0, 1000.0)],   # ~nothing traded
        mapping={"AAA": "INE000A01001"})

    body = _json.loads(report.read_text(encoding="utf-8"))
    assert body["coverage"]["thin_on_bse"] == 1
    assert body["coverage"]["checkable"] == 0
    assert body["disagreements"] == [], "a thin print must not raise a flag"


def test_the_coverage_buckets_add_up(tmp_path, monkeypatch):
    """A coverage report that does not balance is hiding a case."""
    rc, report, _ = _run(
        tmp_path, monkeypatch,
        nse_rows=[("AAA.NS", 100.0), ("BBB.NS", 50.0), ("NOISIN.NS", 10.0)],
        bse_rows=[("INE000A01001", "AAA", 100.0, 5e7)],
        mapping={"AAA": "INE000A01001", "BBB": "INE111A01011"})

    cov = _json.loads(report.read_text(encoding="utf-8"))["coverage"]
    assert cov["balanced"] is True
    assert (cov["checkable"] + cov["no_isin"] + cov["not_on_bse"]
            + cov["thin_on_bse"] + cov["no_usable_close"]
            + cov["ambiguous_isin"]) == cov["nse_symbols"]


def test_a_missing_nse_snapshot_names_the_command(tmp_path, monkeypatch):
    from desk.marketdata.sources import bse as bse_mod
    monkeypatch.setattr(bse_mod, "BseSession", lambda *a, **k: _StubBse(b""))
    (tmp_path / "bhavcopy").mkdir()
    rc = refresh_crosscheck(_StubNse(), date(2026, 9, 17),
                            tmp_path / "out",
                            bhavcopy_dir=tmp_path / "bhavcopy",
                            isin_path=_isin_file(tmp_path, {"A": "INE000A01001"}))
    assert rc == 1


def test_a_bse_failure_does_not_write_a_half_report(tmp_path, monkeypatch):
    """Better no report than one the plan would read as a clean check."""
    from desk.marketdata.sources.errors import SourceError

    rc, report, _ = _run(
        tmp_path, monkeypatch,
        nse_rows=[("AAA.NS", 100.0)], bse_rows=[],
        mapping={"AAA": "INE000A01001"},
        fail=SourceError("BSE returned HTTP 503"))

    assert rc == 1
    assert not report.exists()


def test_the_isin_map_is_reused_not_refetched(tmp_path, monkeypatch):
    """It changes on listings, not daily - refetching NSE's equity master
    on every cross-check would be a request for nothing."""
    from desk.marketdata.sources import bse as bse_mod

    calls = []

    class _CountingNse(_StubNse):
        def fetch_with_retry(self, fn, *a, **k):
            calls.append(1)
            return fn(*a, **k)

    stub = _StubBse(_bse_csv([("INE000A01001", "AAA", 100.0, 5e7)]))
    monkeypatch.setattr(bse_mod, "BseSession", lambda *a, **k: stub)
    refresh_crosscheck(_CountingNse(), date(2026, 9, 17), tmp_path / "out",
                       bhavcopy_dir=_nse_snapshot(tmp_path, [("AAA.NS", 100.0)]),
                       isin_path=_isin_file(tmp_path, {"AAA": "INE000A01001"}))
    assert calls == [], "an existing ISIN map must not trigger a fetch"


def test_the_report_is_written_atomically(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch,
         nse_rows=[("AAA.NS", 100.0)],
         bse_rows=[("INE000A01001", "AAA", 100.0, 5e7)],
         mapping={"AAA": "INE000A01001"})
    assert not list((tmp_path / "crosscheck").glob("*.tmp"))
