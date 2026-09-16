"""R2 - the `refresh research` CLI.

No network: a stub session serves the captured fixtures. What is under test is
the OPERATIONAL behaviour - resume, progress, failure accounting, and whether
a record NSE published but we could not date reaches somewhere a human will
see it.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from desk.marketdata.refresh import RESEARCH_KINDS, refresh_research
from desk.marketdata.sources.nse import RateLimited, SourceError

FIXTURES = Path(__file__).parent / "fixtures"


def _rows(name: str):
    d = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d.get("data", [])


class StubSession:
    """Serves the captured fixtures. Records every call so a test can assert
    on how many requests a run would really have made."""

    def __init__(self, *, fail: set[str] = frozenset(),
                 rate_limit_on: str | None = None):
        self.calls: list[tuple[str, str]] = []
        self.fail = fail
        self.rate_limit_on = rate_limit_on

    def _guard(self, kind: str, symbol: str):
        self.calls.append((kind, symbol))
        if symbol == self.rate_limit_on:
            raise RateLimited(f"{symbol}: slow down")
        if symbol in self.fail:
            raise SourceError(f"{symbol}: endpoint returned HTTP 500")

    def fetch_results(self, s, period="Quarterly"):
        self._guard("results", s); return _rows(f"nse_results_{s}.json")

    def fetch_announcements(self, s):
        self._guard("announcements", s); return _rows(f"nse_announcements_{s}.json")

    def fetch_board_meetings(self, s):
        self._guard("boardmeetings", s); return _rows(f"nse_boardmeetings_{s}.json")

    def fetch_shareholding(self, s):
        self._guard("shareholding", s); return _rows(f"nse_shareholding_{s}.json")

    def fetch_insider_deals(self, s):
        self._guard("insider", s); return _rows(f"nse_insider_{s}.json")

    def fetch_xbrl(self, url):
        self.calls.append(("xbrl", url))
        return (FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes()


# ===========================================================================
# the happy path
# ===========================================================================

def test_a_run_writes_every_kind_and_reports_counts(tmp_path, capsys):
    rc = refresh_research(StubSession(), ["IRCTC"], tmp_path, with_xbrl=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[1/1] IRCTC" in out
    for kind in RESEARCH_KINDS:
        assert kind in out
    for kind in ("announcements", "boardmeetings", "shareholding", "insider"):
        assert (tmp_path / kind / "IRCTC.json").is_file()
    assert list((tmp_path / "filings" / "IRCTC").iterdir())


def test_progress_is_printed_because_a_multi_hour_run_must_not_look_hung(
        tmp_path, capsys):
    """A full-universe pass is ~1,598 symbols at a 1s throttle - over two
    hours. Silence is indistinguishable from a hang."""
    refresh_research(StubSession(), ["IRCTC", "RELIANCE"], tmp_path,
                     kinds=("insider",), with_xbrl=False)
    out = capsys.readouterr().out
    assert "[1/2]" in out and "[2/2]" in out


def test_xbrl_is_fetched_and_the_numbers_land_in_the_store(tmp_path):
    from desk.research.store import FilingStore, at_open

    s = StubSession()
    refresh_research(s, ["RELIANCE"], tmp_path, kinds=("filings",))
    assert any(k == "xbrl" for k, _ in s.calls)

    store = FilingStore(tmp_path / "filings")
    rec = store.latest("RELIANCE", as_of=at_open(date(2026, 1, 1)), months=3)
    assert rec is not None
    assert rec.period(3).revenue / 1e7 == pytest.approx(128_260, abs=1)


def test_no_xbrl_skips_the_documents_entirely(tmp_path):
    s = StubSession()
    refresh_research(s, ["RELIANCE"], tmp_path, kinds=("filings",),
                     with_xbrl=False)
    assert not any(k == "xbrl" for k, _ in s.calls)
    assert list((tmp_path / "filings" / "RELIANCE").iterdir())   # still stored


# ===========================================================================
# resume - the point of the whole design
# ===========================================================================

def test_a_second_run_skips_what_is_already_fetched(tmp_path, capsys):
    """A full pass will not reliably complete in one sitting, so an
    interrupted run must restart where it stopped rather than from zero."""
    first = StubSession()
    refresh_research(first, ["IRCTC"], tmp_path, kinds=("insider",),
                     with_xbrl=False)
    n = len(first.calls)

    second = StubSession()
    rc = refresh_research(second, ["IRCTC"], tmp_path, kinds=("insider",),
                          with_xbrl=False)
    assert rc == 0
    assert second.calls == [], "a resumed run refetched an already-done symbol"
    assert "already fetched and skipped" in capsys.readouterr().out
    assert n > 0


def test_force_refetches(tmp_path):
    refresh_research(StubSession(), ["IRCTC"], tmp_path, kinds=("insider",),
                     with_xbrl=False)
    s = StubSession()
    refresh_research(s, ["IRCTC"], tmp_path, kinds=("insider",),
                     with_xbrl=False, force=True)
    assert s.calls, "--force did not refetch"


def test_a_rate_limit_stops_the_run_rather_than_grinding_on(tmp_path, capsys):
    """Continuing through 1,500 more symbols against a host already refusing
    us makes the block worse and achieves nothing. The marker files mean the
    next run resumes from here."""
    s = StubSession(rate_limit_on="RELIANCE")
    rc = refresh_research(s, ["IRCTC", "RELIANCE", "IRCTC"], tmp_path,
                          kinds=("insider",), with_xbrl=False)
    assert rc == 2
    out = capsys.readouterr().out
    assert "RATE LIMITED" in out
    assert "rerun the same command to resume" in out
    assert [sym for _, sym in s.calls].count("RELIANCE") == 1


# ===========================================================================
# failure is visible and counted
# ===========================================================================

def test_a_failed_symbol_does_not_stop_the_others_but_does_fail_the_run(
        tmp_path, capsys):
    s = StubSession(fail={"IRCTC"})
    rc = refresh_research(s, ["IRCTC", "RELIANCE"], tmp_path,
                          kinds=("insider",), with_xbrl=False)
    assert rc == 1, "a failed symbol must not report success"
    out = capsys.readouterr().out
    assert "FAILED" in out and "1 symbol(s) failed" in out
    assert (tmp_path / "insider" / "RELIANCE.json").is_file()


def test_a_failed_symbol_is_not_marked_done_so_a_rerun_retries_it(tmp_path):
    refresh_research(StubSession(fail={"IRCTC"}), ["IRCTC"], tmp_path,
                     kinds=("insider",), with_xbrl=False)
    assert not (tmp_path / "_done" / "IRCTC.json").exists()
    s = StubSession()
    refresh_research(s, ["IRCTC"], tmp_path, kinds=("insider",),
                     with_xbrl=False)
    assert s.calls, "a previously failed symbol was treated as done"


def test_an_unusable_symbol_is_skipped_and_counted(tmp_path, capsys):
    rc = refresh_research(StubSession(), ["!!bad!!"], tmp_path,
                          kinds=("insider",), with_xbrl=False)
    assert rc == 1
    assert "SKIPPED" in capsys.readouterr().out


# ===========================================================================
# undated records must reach a human
# ===========================================================================

def test_undated_records_are_printed_as_they_happen(tmp_path, capsys):
    """RELIANCE has 8 filings NSE published with no usable timestamp. They are
    the early warning that a field has been renamed, so an operator watching a
    run should see them then - not months later when a filing is
    inexplicably missing from a backtest."""
    refresh_research(StubSession(), ["RELIANCE"], tmp_path, kinds=("filings",),
                     with_xbrl=False)
    out = capsys.readouterr().out
    assert "UNDATED" in out
    assert "8 record(s) had no usable disclosure timestamp" in out
    assert "a field has probably been renamed" in out


def test_undated_records_are_written_to_the_file_not_only_printed(tmp_path):
    """Printed-only means lost the moment the terminal scrolls."""
    refresh_research(StubSession(), ["RELIANCE"], tmp_path,
                     kinds=("announcements",), with_xbrl=False)
    blob = json.loads((tmp_path / "announcements" / "RELIANCE.json")
                      .read_text(encoding="utf-8"))
    assert "undated" in blob
    assert isinstance(blob["records"], list) and blob["records"]
    assert "fetched_at" in blob


def test_the_marker_records_what_was_fetched_and_when(tmp_path):
    refresh_research(StubSession(), ["IRCTC"], tmp_path, kinds=("insider",),
                     with_xbrl=False)
    marker = json.loads((tmp_path / "_done" / "IRCTC.json")
                        .read_text(encoding="utf-8"))
    assert marker["symbol"] == "IRCTC"
    assert marker["counts"]["insider"] == 5
    datetime.fromisoformat(marker["fetched_at"])       # parses


def test_kinds_limits_what_is_fetched(tmp_path):
    s = StubSession()
    refresh_research(s, ["IRCTC"], tmp_path, kinds=("insider", "shareholding"),
                     with_xbrl=False)
    assert {k for k, _ in s.calls} == {"insider", "shareholding"}
    assert not (tmp_path / "announcements").exists()
