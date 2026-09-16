"""R1 - the NSE research source: filings, announcements, events.

Every parser here is checked against a payload actually captured from NSE
(desk/tests/fixtures/nse_*.json), not against an invented shape. That matters
more here than anywhere else in the project: these endpoints are undocumented,
their fields are inconsistently populated, and the failure mode is a filing
that silently does not arrive.

The single most important behaviour under test is that a record NSE published
without a usable timestamp comes back in the `undated` list rather than being
dropped or given a plausible date.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from desk.marketdata.sources.nse import (
    SourceError,
    parse_announcements,
    parse_board_meetings,
    parse_event_calendar,
    parse_insider_deals,
    parse_nse_datetime,
    parse_results,
    parse_shareholding,
)
from desk.research import Announcement, Filing, ResultPeriod, Undated

FIXTURES = Path(__file__).parent / "fixtures"


def rows(name: str) -> list[dict]:
    d = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d.get("data", [])


# Both fixtures below are parsed once per module and reused - several
# tests check different facets of the same RELIANCE payload, and re-reading
# + re-parsing the same JSON file per test added nothing but repetition.
@pytest.fixture(scope="module")
def reliance_results_raw() -> list[dict]:
    return rows("nse_results_RELIANCE.json")


@pytest.fixture(scope="module")
def reliance_filings(reliance_results_raw):
    return parse_results(reliance_results_raw, "RELIANCE")


@pytest.fixture(scope="module")
def reliance_announcements():
    return parse_announcements(rows("nse_announcements_RELIANCE.json"), "RELIANCE")


# ===========================================================================
# the timestamp parser - four real formats, mixed case
# ===========================================================================

@pytest.mark.parametrize("raw,expected", [
    ("16-Jan-2025 20:20:21", datetime(2025, 1, 16, 20, 20, 21)),
    ("2026-09-16 17:45:33", datetime(2026, 9, 16, 17, 45, 33)),
    ("26-Apr-2007 18:00", datetime(2007, 4, 26, 18, 0)),
    ("17-Jul-2026", datetime(2026, 7, 17)),
    ("16-JUL-2026 19:24:44", datetime(2026, 7, 16, 19, 24, 44)),   # upper
])
def test_every_date_format_nse_actually_uses(raw, expected):
    """All five were measured across the captured fixtures, not guessed. NSE
    uses different spellings on different endpoints and mixes case."""
    assert parse_nse_datetime(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "-", "NA", "null", None,
                                 "not a date", "31-Feb-2025"])
def test_an_unreadable_timestamp_is_none_never_a_guess(raw):
    assert parse_nse_datetime(raw) is None


# ===========================================================================
# undated records must surface, not vanish
# ===========================================================================

def test_filings_nse_cannot_date_are_reported_not_dropped(
        reliance_filings, reliance_results_raw):
    """REGRESSION target for the whole design: NSE publishes results with no
    broadcast time at all - RELIANCE has several, all 2005-2007, from before
    electronic dissemination timestamps. Dropping them loses history silently;
    dating them invents information. They come back in `undated`."""
    filings, undated = reliance_filings
    assert undated, "expected NSE's undated 2005-2007 filings to be reported"
    assert all(isinstance(u, Undated) for u in undated)
    assert all(u.kind == "result" for u in undated)
    # Nothing is lost: every input row is accounted for in one list or other.
    assert len(filings) + len(undated) == len(reliance_results_raw)
    # The reason names the period, so a human can find the filing.
    assert any("Quarter" in u.reason for u in undated)
    # And the raw row is kept for inspection.
    assert all(isinstance(u.raw, dict) and u.raw for u in undated)


def test_filing_date_rescues_a_row_with_no_broadcast_time():
    """The fallback chain is broadCastDate -> exchdisstime -> filingDate, and
    it is load-bearing: one RELIANCE row has no broadcast time but does carry
    filingDate '26-Apr-2007 18:00'. Without the fallback it would be reported
    as undated, which would be true but needlessly lossy."""
    raw = [{"fromDate": "01-Jan-2007", "toDate": "31-Mar-2007",
            "relatingTo": "Fourth Quarter", "filingDate": "26-Apr-2007 18:00",
            "exchdisstime": None, "period": "Quarterly", "xbrl": ""}]
    filings, undated = parse_results(raw, "RELIANCE")
    assert not undated
    assert filings[0].disclosed_at == datetime(2007, 4, 26, 18, 0)


def test_broadcast_time_wins_over_filing_time():
    """Order is not arbitrary. filingDate is when the company submitted;
    broadCastDate is when the exchange disseminated it and therefore when the
    market could act. The later, tradeable one must win."""
    raw = [{"broadCastDate": "16-Jan-2025 20:20:21",
            "filingDate": "16-Jan-2025 20:20", "period": "Quarterly"}]
    filings, _ = parse_results(raw, "X")
    assert filings[0].disclosed_at == datetime(2025, 1, 16, 20, 20, 21)


# ===========================================================================
# filings
# ===========================================================================

def test_reliance_filings_parse_against_the_real_payload(reliance_filings):
    filings, undated = reliance_filings
    assert len(filings) == 122
    assert len(undated) == 8
    latest = max(filings, key=lambda f: f.disclosed_at)
    assert latest.disclosed_at == datetime(2025, 1, 16, 20, 20, 21)
    assert latest.relating_to == "Third Quarter"
    assert latest.period_start == date(2024, 10, 1)
    assert latest.period_end == date(2024, 12, 31)
    assert latest.period is ResultPeriod.QUARTERLY


def test_the_period_described_is_never_confused_with_when_it_was_disclosed(
        reliance_filings):
    """The leakage trap this whole module exists to avoid: a Q3 result covers
    Oct-Dec but is disclosed in mid-January. A backtest simulating 1 January
    must not see it."""
    filings, _ = reliance_filings
    latest = max(filings, key=lambda f: f.disclosed_at)
    assert latest.period_end < latest.disclosed_at.date()
    assert (latest.disclosed_at.date() - latest.period_end).days > 10


def test_a_missing_xbrl_document_is_none_not_a_broken_url(reliance_filings):
    """NSE emits '.../corporate/xbrl/-' rather than omitting the field. Left
    as-is that is a URL that 404s on fetch; `has_numbers` would lie."""
    filings, _ = reliance_filings
    assert any(f.xbrl_url is None for f in filings)
    assert all(not (f.xbrl_url or "").endswith("/-") for f in filings)
    assert all(f.has_numbers is (f.xbrl_url is not None) for f in filings)


def test_audited_and_consolidated_are_three_state():
    """None means NSE did not say, which is not the same as saying no - the
    same discipline as Sizing.checks_skipped."""
    filings, _ = parse_results([
        {"broadCastDate": "16-Jan-2025 20:20:21", "audited": "Audited",
         "consolidated": "Consolidated", "period": "Quarterly"},
        {"broadCastDate": "16-Jan-2025 20:20:21", "audited": "Un-Audited",
         "consolidated": "Non-Consolidated", "period": "Quarterly"},
        {"broadCastDate": "16-Jan-2025 20:20:21", "period": "Quarterly"},
    ], "X")
    assert [f.audited for f in filings] == [True, False, None]
    assert [f.consolidated for f in filings] == [True, False, None]


def test_a_smaller_company_parses_too():
    """RELIANCE is well-behaved. The bugs live in the awkward cases, which is
    why a second, smaller symbol was captured."""
    filings, undated = parse_results(rows("nse_results_IRCTC.json"), "IRCTC")
    assert len(filings) == 25 and not undated
    assert sum(1 for f in filings if f.has_numbers) == 22


# ===========================================================================
# announcements - tier 1 news
# ===========================================================================

def test_announcements_parse_and_every_one_is_dated(reliance_announcements):
    ann, undated = reliance_announcements
    assert len(ann) == 3345
    assert not undated
    assert all(isinstance(a.disclosed_at, datetime) for a in ann)


@pytest.mark.parametrize("clock,phase", [
    ("09:14:59", "before_market"),
    ("09:15:00", "during_market"),     # the bell
    ("12:00:00", "during_market"),
    ("15:30:00", "during_market"),     # the close is still the session
    ("15:30:01", "after_hours"),
    ("17:45:33", "after_hours"),
])
def test_session_phase_boundaries(clock, phase):
    """Master prompt section 11 wants before / during / after-market, and it
    is only answerable because an_dt carries a clock time. The boundaries are
    NSE's real session, 09:15-15:30 IST - and both edges are inclusive of the
    session, because a trade at the bell is a trade."""
    a = Announcement(symbol="X",
                     disclosed_at=datetime.fromisoformat(f"2026-09-16T{clock}"),
                     category="", text="")
    assert a.session_phase == phase


def test_most_announcements_land_outside_market_hours(reliance_announcements):
    """A sanity check on reality rather than on our code: companies file after
    the close. If this ever inverts, the timestamp field has changed meaning."""
    ann, _ = reliance_announcements
    after = sum(1 for a in ann if a.session_phase == "after_hours")
    assert after > len(ann) / 2


def test_a_missing_attachment_is_none(reliance_announcements):
    """456 of RELIANCE's 3,345 announcements have no attachment."""
    ann, _ = reliance_announcements
    assert any(a.attachment_url is None for a in ann)
    assert all(a.attachment_url is None or a.attachment_url.startswith("http")
               for a in ann)


# ===========================================================================
# shareholding, insiders, board meetings, events
# ===========================================================================

def test_shareholding_promoter_and_public_account_for_the_company():
    """Hand-checkable against reality: RELIANCE's promoter holding is ~50.5%
    and the two must sum to 100."""
    snaps, undated = parse_shareholding(
        rows("nse_shareholding_RELIANCE.json"), "RELIANCE")
    assert snaps and not undated
    latest = max(snaps, key=lambda s: s.disclosed_at)
    assert latest.promoter_pct == pytest.approx(50.48, abs=0.01)
    assert latest.promoter_pct + latest.public_pct == pytest.approx(100.0, abs=0.05)
    assert latest.as_at == date(2026, 6, 30)


def test_a_government_owned_company_shows_its_real_promoter():
    """IRCTC is ~62.4% government-held. A parser reading the wrong column
    would still return a plausible-looking number, so this is pinned to a
    value that can be checked against the public record."""
    snaps, _ = parse_shareholding(rows("nse_shareholding_IRCTC.json"), "IRCTC")
    latest = max(snaps, key=lambda s: s.disclosed_at)
    assert latest.promoter_pct == pytest.approx(62.4, abs=0.1)


def test_insider_deals_parse():
    deals, undated = parse_insider_deals(
        rows("nse_insider_RELIANCE.json"), "RELIANCE")
    assert len(deals) == 20 and not undated
    assert all(d.acquirer for d in deals)
    assert all(d.disclosed_at for d in deals)


def test_board_meetings_carry_the_forward_looking_date():
    """`meeting_date` is the field the event gate reads - it is what keeps a
    trade away from a results announcement."""
    meetings, undated = parse_board_meetings(
        rows("nse_boardmeetings_RELIANCE.json"), "RELIANCE")
    assert len(meetings) == 20 and not undated
    assert any(m.meeting_date for m in meetings)
    assert all(m.purpose for m in meetings)


def test_event_calendar_is_market_wide_and_dated():
    events = parse_event_calendar(rows("nse_eventcalendar.json"))
    assert len(events) == 44
    assert all(e.event_date for e in events)
    assert len({e.symbol for e in events}) > 1      # market-wide, not one name


def test_event_calendar_drops_a_row_with_no_symbol_rather_than_guessing():
    events = parse_event_calendar([
        {"symbol": "AAA", "date": "17-Sep-2026", "purpose": "Results"},
        {"symbol": "", "date": "17-Sep-2026", "purpose": "Results"},
        {"date": "17-Sep-2026", "purpose": "Results"},
    ])
    assert [e.symbol for e in events] == ["AAA"]


# ===========================================================================
# the fetch side: URLs are built, never accepted
# ===========================================================================

def test_fetch_xbrl_refuses_a_url_that_is_not_an_nse_archive():
    """Same reasoning as get_json refusing absolute URLs. The XBRL link comes
    from a filing row, so it must be checked rather than trusted - otherwise
    a poisoned payload steers an authenticated session at any host."""
    from desk.marketdata.sources.nse import NseSession
    s = NseSession()
    for bad in ("https://evil.example/x.xml", "http://nsearchives.nseindia.com/x",
                "https://www.nseindia.com.evil.example/x"):
        with pytest.raises(SourceError, match="not an NSE archive"):
            s.fetch_xbrl(bad)


def test_research_endpoints_refuse_a_symbol_that_is_not_alphanumeric():
    from desk.marketdata.sources.nse import NseSession
    s = NseSession()
    for bad in ("RELIANCE&index=x", "../../etc/passwd", "A B", "A;rm -rf"):
        with pytest.raises(SourceError, match="refusing to build a URL"):
            s.fetch_announcements(bad)


def test_results_period_argument_is_constrained():
    from desk.marketdata.sources.nse import NseSession
    s = NseSession()
    with pytest.raises(SourceError, match="unsupported results period"):
        s.fetch_results("RELIANCE", period="Quarterly&evil=1")


# ===========================================================================
# the shape contract
# ===========================================================================

def test_disclosed_at_is_required_on_every_disclosure_record():
    """The design rule, enforced by the type rather than by discipline. If
    this ever becomes optional, point-in-time correctness becomes a thing you
    have to remember instead of a thing you cannot get wrong."""
    with pytest.raises(TypeError):
        Filing(symbol="X")                                    # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Announcement(symbol="X", category="c", text="t")      # type: ignore[call-arg]


def test_records_are_immutable():
    """A filing is a historical fact. Nothing downstream should be able to
    edit when the market learned something."""
    filings, _ = parse_results(
        [{"broadCastDate": "16-Jan-2025 20:20:21", "period": "Quarterly"}], "X")
    with pytest.raises(Exception):
        filings[0].disclosed_at = datetime(2020, 1, 1)        # type: ignore[misc]


def test_a_non_dict_row_is_skipped_rather_than_crashing_the_parse():
    """NSE occasionally returns a stray value in a list. One bad row must not
    cost the other 3,344."""
    for parse in (parse_results, parse_announcements, parse_board_meetings,
                  parse_shareholding, parse_insider_deals):
        recs, undated = parse(
            ["junk", None, 42, {"broadCastDate": "16-Jan-2025 20:20:21",
                                "an_dt": "16-Jan-2025 20:20:21",
                                "bm_timestamp": "16-Jan-2025 20:20:21",
                                "broadcastDate": "16-Jan-2025 20:20:21",
                                "date": "16-Jan-2025 20:20:21",
                                "period": "Quarterly"}], "X")
        assert len(recs) == 1, parse.__name__


def test_nse_sentinel_dash_is_treated_as_absent_everywhere(
        reliance_announcements):
    """REGRESSION, found by the fixture: 456 of RELIANCE's 3,345
    announcements carry "-" in place of an attachment URL. Passed through
    verbatim that is a link a downstream fetch will dutifully try to open,
    and `attachment_url is not None` becomes a lie."""
    ann, _ = parse_announcements([
        {"an_dt": "16-Jan-2025 20:20:21", "desc": "d", "attchmntFile": "-"},
        {"an_dt": "16-Jan-2025 20:20:21", "desc": "d", "attchmntFile": ""},
        {"an_dt": "16-Jan-2025 20:20:21", "desc": "d", "attchmntFile": "NA"},
        {"an_dt": "16-Jan-2025 20:20:21", "desc": "d",
         "attchmntFile": "https://nsearchives.nseindia.com/x.pdf"},
    ], "X")
    assert [a.attachment_url for a in ann] == [
        None, None, None, "https://nsearchives.nseindia.com/x.pdf"]

    # And against the real payload: exactly the 456 measured, no more.
    real, _ = reliance_announcements
    assert sum(1 for a in real if a.attachment_url is None) == 456
    assert all(a.attachment_url is None or a.attachment_url.startswith("http")
               for a in real)
