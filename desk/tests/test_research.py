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

from desk.marketdata.sources.nse import SourceError
from desk.research.sources.nse import (
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
    for bad in (
        "https://evil.example/x.xml",
        "http://nsearchives.nseindia.com/x",            # not https
        "https://www.nseindia.com.evil.example/x",
        # REGRESSION: these three all pass a startswith(ARCHIVES) check. The
        # first is the real bypass - the prefix matches exactly and the host
        # is attacker-controlled, so a cookie-bearing session would have been
        # sent straight to it. The XBRL URL comes from an NSE payload, i.e.
        # remote data, so it must be PARSED rather than pattern-matched.
        "https://nsearchives.nseindia.com.evil.com/x.xml",
        "https://nsearchives.nseindia.com@evil.example/x.xml",
        "https://nsearchives.nseindia.com.evil.example:443/x.xml",
    ):
        with pytest.raises(SourceError, match="not an NSE archive"):
            s.fetch_xbrl(bad)


def test_fetch_xbrl_accepts_the_genuine_archive_host(monkeypatch):
    """The guard must not be so tight it rejects the real links NSE gives us."""
    from desk.marketdata.sources.nse import NseSession
    s = NseSession()
    monkeypatch.setattr(NseSession, "_fetch_raw", lambda self, url: b"<xbrl/>")
    real = ("https://nsearchives.nseindia.com/corporate/xbrl/"
            "INDAS_117298_1348254_16012025082021.xml")
    assert s.fetch_xbrl(real) == b"<xbrl/>"


def test_research_endpoints_refuse_an_injection_shaped_symbol():
    from desk.marketdata.sources.nse import NseSession
    s = NseSession()
    for bad in ("RELIANCE&index=x", "../../etc/passwd", "A B", "A;rm -rf",
                "A?x=1", "A/B", "A=1", ""):
        with pytest.raises(SourceError, match="refusing to build a URL"):
            s.fetch_announcements(bad)


@pytest.mark.parametrize("symbol", [
    "M&M", "BAJAJ-AUTO", "J&KBANK", "M&MFIN", "NAM-INDIA",
    "IL&FSENGG", "GVT&D", "ARE&M", "BOSCH-HCIL", "RELIANCE",
])
def test_real_nse_symbols_containing_ampersand_or_hyphen_are_accepted(symbol,
                                                                      monkeypatch):
    """REGRESSION: the validator used `.isalnum()`, which rejects `&` and `-`.
    Measured against one real day's bhavcopy: 22 of 3,485 symbols failed,
    including M&M - a Nifty 50 constituent - and BAJAJ-AUTO. A check that
    looks conservative and silently refuses a fifth of the index is not
    conservative, it is broken."""
    from desk.marketdata.sources.nse import NseSession
    s = NseSession()
    seen = {}
    monkeypatch.setattr(NseSession, "get_json",
                        lambda self, path: seen.setdefault("path", path) and [] or [])
    s.fetch_announcements(symbol)
    assert f"symbol={symbol}" in seen["path"]


def test_every_symbol_in_a_real_bhavcopy_passes_the_validator():
    """The validator is checked against an entire real trading day rather
    than a handful of hand-picked names, because the failure mode is 'a name
    I did not think of is silently unfetchable'."""
    from desk.marketdata.sources.nse import _SYMBOL_RE
    import csv, io as _io
    raw = (FIXTURES / "nse_bhavcopy_20260911.csv").read_bytes().decode("utf-8")
    symbols = {r[0].strip() for r in csv.reader(_io.StringIO(raw))
               if r and r[0].strip() and r[0].strip() != "SYMBOL"}
    rejected = [s for s in symbols if not _SYMBOL_RE.fullmatch(s.upper())]
    assert not rejected, f"validator rejects real NSE symbols: {sorted(rejected)[:10]}"


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


def test_an_insider_transaction_date_is_never_used_as_a_disclosure_date():
    """REGRESSION: the fallback chain ended in `acqtoDt`, which is when the
    TRANSACTION completed - measured 1-5 days before the actual broadcast in
    every sampled RELIANCE row. Using it as `disclosed_at` tells a backtest
    the market knew about an insider trade days before it was disclosed,
    which is exactly the leak this module exists to prevent."""
    row = {"acqName": "SOMEONE", "acqMode": "Off Market",
           "acqfromDt": "05-Jan-2025", "acqtoDt": "05-Jan-2025"}
    deals, undated = parse_insider_deals([row], "X")
    assert not deals, "a row with only a transaction date must not be dated"
    assert undated and undated[0].kind == "insider_deal"

    # intimDt IS a disclosure event, so it is an acceptable last resort.
    with_intim = dict(row, intimDt="09-Jan-2025 19:06:00")
    deals, undated = parse_insider_deals([with_intim], "X")
    assert not undated
    assert deals[0].disclosed_at == datetime(2025, 1, 9, 19, 6)
    # and the transaction dates are still preserved, just not as disclosure
    assert deals[0].to_date == date(2025, 1, 5)


def test_an_undated_event_is_dropped_as_the_docstring_promises():
    """REGRESSION: parse_event_calendar's docstring said it drops undated
    events; the code appended them with event_date=None. The real fixture has
    no bad rows, so the test passed straight through the discrepancy. A None
    here either crashes date arithmetic in the event gate or silently
    miscounts how many events fall inside a holding window."""
    events = parse_event_calendar([
        {"symbol": "AAA", "date": "17-Sep-2026", "purpose": "Results"},
        {"symbol": "BBB", "date": "not-a-date", "purpose": "Results"},
        {"symbol": "CCC", "purpose": "Results"},
        {"symbol": "DDD", "date": "-", "purpose": "Results"},
    ])
    assert [e.symbol for e in events] == ["AAA"]
    assert all(e.event_date is not None for e in events)


def test_session_phase_converts_a_timezone_aware_datetime_rather_than_dropping_it():
    """REGRESSION: `.time()` discards tzinfo without complaint, so a UTC
    09:00 - which is 14:30 IST, mid-session - read as 'before_market'. Wrong
    answer, no error. Nothing builds an aware datetime today; this is so that
    whoever wires a UTC-normalised store cannot be caught silently."""
    from datetime import timedelta, timezone

    utc_mid_session = datetime(2026, 9, 16, 9, 0, tzinfo=timezone.utc)
    a = Announcement(symbol="X", disclosed_at=utc_mid_session,
                     category="c", text="t")
    assert a.session_phase == "during_market"

    ist = timezone(timedelta(hours=5, minutes=30))
    before = Announcement(symbol="X", category="c", text="t",
                          disclosed_at=datetime(2026, 9, 16, 9, 0, tzinfo=ist))
    assert before.session_phase == "before_market"

    # A naive datetime is still read as IST wall-clock, unchanged.
    naive = Announcement(symbol="X", category="c", text="t",
                         disclosed_at=datetime(2026, 9, 16, 14, 30))
    assert naive.session_phase == "during_market"


def test_mixed_case_months_parse_without_the_removed_retry():
    """The title-casing retry was removed as dead code - strptime's %b is
    already case-insensitive. Pinned so nobody re-adds it 'to be safe'."""
    assert parse_nse_datetime("16-JUL-2026 19:24:44") == datetime(2026, 7, 16, 19, 24, 44)
    assert parse_nse_datetime("16-jul-2026") == datetime(2026, 7, 16)
    assert parse_nse_datetime("16-JuL-2026") == datetime(2026, 7, 16)


# ===========================================================================
# REGRESSION: security-review findings
# ===========================================================================

@pytest.mark.parametrize("bad_field,value", [
    ("broadCastDate", 12345), ("xbrl", ["a"]), ("companyName", {"x": 1}),
    ("relatingTo", True), ("isin", 3.14), ("audited", 7),
])
def test_a_non_string_field_does_not_destroy_the_whole_batch(bad_field, value):
    """REGRESSION: `(v or "").strip()` substitutes "" only for FALSY values, so
    a truthy non-string - an int, a list, a dict - reached .strip() and raised
    AttributeError. There is no per-row try/except, so ONE such field anywhere
    in a 3,345-row payload discarded the entire batch including the `undated`
    list this module's whole contract rests on. These endpoints are
    undocumented; a field changing type between revisions is when, not if."""
    good = {"broadCastDate": "16-Jan-2025 20:20:21", "period": "Quarterly"}
    filings, undated = parse_results([dict(good, **{bad_field: value}), good], "X")
    assert len(filings) + len(undated) == 2


def test_every_parser_survives_a_hostile_row():
    ugly = {k: [{"nested": 1}] for k in
            ("an_dt", "desc", "attchmntText", "attchmntFile", "bm_purpose",
             "bm_desc", "acqName", "acqMode", "anex", "symbol", "purpose",
             "company", "companyName", "xbrl", "isin")}
    ok = {"an_dt": "16-Jan-2025 20:20:21", "broadCastDate": "16-Jan-2025 20:20:21",
          "bm_timestamp": "16-Jan-2025 20:20:21",
          "broadcastDate": "16-Jan-2025 20:20:21", "date": "16-Jan-2025 20:20:21",
          "period": "Quarterly"}
    for parse in (parse_results, parse_announcements, parse_board_meetings,
                  parse_shareholding, parse_insider_deals):
        recs, und = parse([dict(ok, **ugly), ok], "X")      # must not raise
        assert len(recs) + len(und) == 2, parse.__name__
    parse_event_calendar([dict(ugly, date="17-Sep-2026"), {"symbol": "A",
                          "date": "17-Sep-2026"}])          # must not raise


def test_a_nan_never_becomes_a_shareholding_percentage():
    """json.loads accepts bare NaN/Infinity even though neither is valid JSON.
    A NaN promoter holding compares false against every threshold downstream
    while looking like a real number on a dashboard."""
    from desk.marketdata.sources.nse import _num
    for bad in (float("nan"), float("inf"), float("-inf"), "NaN", "Infinity"):
        assert _num(bad) is None, bad
    assert _num("0") == 0.0 and _num(0) == 0.0        # a real 0% must survive
    assert _num("62.4") == 62.4 and _num("1,234.5") == 1234.5


def test_get_json_refuses_non_finite_constants_rather_than_passing_them_on():
    from desk.marketdata.sources.nse import _reject_constant
    with pytest.raises(SourceError, match="non-finite JSON constant"):
        json.loads('{"promoter_pct": NaN}', parse_constant=_reject_constant)


def test_only_a_real_nse_host_passes_the_host_check():
    """Suffix matching is not a hostname check. Every string below starts with
    the archive prefix or contains the domain, and none of them IS the domain."""
    from desk.marketdata.sources.nse import _is_nse_host
    assert _is_nse_host("nseindia.com")
    assert _is_nse_host("nsearchives.nseindia.com")
    assert _is_nse_host("www.nseindia.com")
    assert _is_nse_host("NSEArchives.NSEIndia.Com")          # case
    assert not _is_nse_host("nsearchives.nseindia.com.evil.example")
    assert not _is_nse_host("evil-nseindia.com")
    assert not _is_nse_host("nseindia.com.evil.example")
    assert not _is_nse_host("evil.example")


def test_a_redirect_off_nse_is_refused_even_though_the_url_we_asked_for_was_fine():
    """REGRESSION: urllib follows redirects transparently and cross-host, so
    validating the URL we REQUEST constrains nothing about where we end up.
    Checking the URL is necessary and not sufficient; the destination is
    re-checked after the fact."""
    from desk.marketdata.sources.nse import NseSession

    class FakeResp:
        headers = {}
        def __init__(self, final): self._final = final
        def geturl(self): return self._final
        def read(self, n=-1): return b"payload"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    # min_interval=0: this test is about the redirect check, not the
    # throttle, and a real sleep here buys nothing.
    s = NseSession(min_interval=0)
    s._warmed = True
    s._opener = type("O", (), {"open": lambda self, req, timeout=None:
                               FakeResp("https://evil.example/stolen")})()
    with pytest.raises(SourceError, match="redirected off NSE"):
        s._fetch_raw("https://nsearchives.nseindia.com/corporate/xbrl/x.xml")

    # A redirect that stays inside NSE is fine - they do this legitimately.
    s._opener = type("O", (), {"open": lambda self, req, timeout=None:
                               FakeResp("https://www.nseindia.com/ok")})()
    assert s._fetch_raw("https://nsearchives.nseindia.com/x") == b"payload"


def test_a_gzip_bomb_is_refused_rather_than_decompressed_into_memory():
    """Measured amplification is ~1029x: 200MB of compressible data gzips to
    ~204KB. gzip.decompress() does it in one unbounded allocation, so a
    few-MB response that passes the wire-size cap still detonates."""
    import gzip as _gzip
    from desk.marketdata.sources.nse import MAX_DECOMPRESSED_BYTES, _gunzip_bounded

    bomb = _gzip.compress(b"\0" * (MAX_DECOMPRESSED_BYTES + 1024))
    assert len(bomb) < 1_000_000, "the bomb should be small on the wire"
    with pytest.raises(SourceError, match="decompressed past"):
        _gunzip_bounded(bomb, "https://nsearchives.nseindia.com/x")

    # A normal body still round-trips.
    assert _gunzip_bounded(_gzip.compress(b"hello"), "u") == b"hello"


# ===========================================================================
# the throttle - R2's precondition for fetching anything at scale
# ===========================================================================

def _stub_session(monkeypatch, min_interval, **kw):
    """A session whose socket is replaced but whose throttle is real."""
    from desk.marketdata.sources.nse import NseSession

    class Resp:
        headers = {}
        def geturl(self): return "https://www.nseindia.com/x"
        def read(self, n=-1): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    s = NseSession(min_interval=min_interval, **kw)
    s._warmed = True
    s._opener = type("O", (), {"open": lambda self, req, timeout=None: Resp()})()
    return s


def test_requests_are_spaced_by_at_least_min_interval(monkeypatch):
    """REGRESSION target for the whole of R2's first step: providers.py has
    said "Throttle and cache" since the registry was written and NseSession
    did neither. R3 wants ~8,000 sequential requests; unthrottled, that is how
    an IP stops being able to reach NSE."""
    import time as _t
    s = _stub_session(monkeypatch, min_interval=0.05)
    start = _t.monotonic()
    for _ in range(4):
        s._fetch_raw("https://www.nseindia.com/x")
    elapsed = _t.monotonic() - start
    # 4 requests, 3 gaps. The first is free.
    assert elapsed >= 0.05 * 3 * 0.9, f"throttle did not hold: {elapsed:.3f}s"
    assert s.requests_made == 4


def test_the_throttle_uses_a_monotonic_clock():
    """time.time() steps backwards over an NTP correction or a DST change,
    which would silently disable the throttle at exactly the wrong moment."""
    import ast
    import inspect
    from desk.marketdata.sources.nse import NseSession

    # Parse rather than grep: the docstring deliberately MENTIONS time.time()
    # to explain why it is not used, and a naive substring check flags that.
    tree = ast.parse(inspect.getsource(NseSession._wait_turn).strip())
    calls = {ast.unparse(n.func) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "time.monotonic" in calls
    assert "time.time" not in calls


def test_a_request_budget_stops_a_loop_becoming_a_ban(monkeypatch):
    """A bug that loops is otherwise an all-night hammering session, and the
    first anyone knows of it is the block."""
    s = _stub_session(monkeypatch, min_interval=0, max_requests=3)
    for _ in range(3):
        s._fetch_raw("https://www.nseindia.com/x")
    with pytest.raises(SourceError, match="request budget"):
        s._fetch_raw("https://www.nseindia.com/x")


def test_rate_limiting_is_distinguishable_from_a_broken_endpoint(monkeypatch):
    """"Wait and try again" and "the endpoint moved, stop and tell a human"
    need different responses, and before this they were the same exception."""
    import urllib.error
    from desk.marketdata.sources.nse import NseSession, RateLimited

    def make(code):
        s = NseSession(min_interval=0)
        s._warmed = True
        def boom(req, timeout=None):
            raise urllib.error.HTTPError("u", code, "no", {}, None)
        s._opener = type("O", (), {"open": lambda self, r, timeout=None: boom(r)})()
        return s

    with pytest.raises(RateLimited):
        make(429)._fetch_raw("https://www.nseindia.com/x")
    # 403 after a successful handshake is how NSE usually says "slow down"
    with pytest.raises(RateLimited):
        make(403)._fetch_raw("https://www.nseindia.com/x")
    # 404 is not transient - retrying it just arrives at the same wrong answer
    with pytest.raises(SourceError) as exc:
        make(404)._fetch_raw("https://www.nseindia.com/x")
    assert not isinstance(exc.value, RateLimited)


def test_backoff_retries_only_rate_limits(monkeypatch):
    from desk.marketdata.sources.nse import NseSession, RateLimited
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = NseSession(min_interval=0)

    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimited("slow down")
        return "ok"
    assert s.fetch_with_retry(flaky) == "ok"
    assert calls["n"] == 3

    # A non-transient failure must NOT be retried.
    tries = {"n": 0}
    def broken():
        tries["n"] += 1
        raise SourceError("endpoint renamed")
    with pytest.raises(SourceError, match="endpoint renamed"):
        s.fetch_with_retry(broken)
    assert tries["n"] == 1


def test_backoff_gives_up_and_reraises_rather_than_looping_forever(monkeypatch):
    from desk.marketdata.sources.nse import NseSession, RateLimited
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = NseSession(min_interval=0)
    n = {"c": 0}
    def always():
        n["c"] += 1
        raise RateLimited("still limited")
    with pytest.raises(RateLimited):
        s.fetch_with_retry(always, attempts=3)
    assert n["c"] == 3


def test_the_research_and_transport_modules_do_not_form_an_import_cycle():
    """REGRESSION: the split initially kept a back-compat re-export in the
    transport module, which made `desk.marketdata.sources.nse` import
    `desk.research.sources.nse` and vice versa. It passed the whole suite,
    because the tests happened to import transport first - the cycle only
    appeared when something imported research first. A shim that creates a
    cycle is worse than moving two import lines."""
    import subprocess
    import sys

    # Run in a SUBPROCESS. Clearing desk.* out of sys.modules in-process
    # rebinds every class this suite has already imported, so a later test
    # asserting isinstance() against the pre-reload class fails for reasons
    # that have nothing to do with it - which is exactly what happened when
    # this test was first written.
    for first in ("desk.research.sources.nse", "desk.marketdata.sources.nse"):
        r = subprocess.run([sys.executable, "-c", f"import {first}"],
                           capture_output=True, text=True)
        assert r.returncode == 0, (
            f"importing {first} first failed:\n{r.stderr}")


def test_the_transport_module_does_not_depend_on_the_research_domain():
    """Direction of the dependency is the point of the split: research knows
    about transport, transport knows nothing about research. If this inverts,
    R6's BSE source has nowhere symmetric to live."""
    import ast
    import pathlib

    src = pathlib.Path("desk/marketdata/sources/nse.py").read_text(encoding="utf-8")
    imported = {n.module for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.startswith("desk.research") for m in imported), imported


def test_the_marketdata_package_does_not_depend_on_the_research_domain():
    """Direction of the dependency. research imports marketdata for the HTTP
    session, which is genuinely cross-cutting; marketdata must not import
    research back, or neither package sits above the other and R5/R6 have
    nowhere unambiguous to live.

    The one permitted exception is the CLI dispatcher in
    desk/marketdata/refresh.py, which imports the research entry point purely
    to route a subcommand to it."""
    import ast
    import pathlib

    offenders = []
    for f in pathlib.Path("desk/marketdata").rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.ImportFrom) and n.module
                    and n.module.startswith("desk.research")):
                offenders.append((f.name, n.module))
    assert offenders == [("refresh.py", "desk.research.refresh")], offenders


def test_the_shared_payload_helpers_are_public_not_borrowed_privates():
    """REGRESSION: research/sources/nse.py imported _num, _s and _text_or_none
    from marketdata/sources/nse.py - three private names across a package
    boundary. R6's BSE parser needs the same helpers, and the pattern on offer
    was 'reach into NSE's internals or copy them'."""
    from desk.marketdata.text import ABSENT, as_float, as_text, text_or_none

    assert as_text(None) == "" and as_text(123) == "123" and as_text(" x ") == "x"
    assert text_or_none("-") is None and text_or_none("NA") is None
    assert text_or_none("https://x") == "https://x"
    assert as_float("1,234.5") == 1234.5 and as_float(0) == 0.0
    assert as_float(float("nan")) is None and as_float("Infinity") is None
    assert "-" in ABSENT


def test_the_two_stores_no_longer_share_an_exception_name():
    """Two unrelated StoreError classes meant `except StoreError` written
    against one import silently would not catch the other - and Stage 2 will
    want to catch a bars problem and a filings problem in one place."""
    from desk.research.store import FilingStoreError
    from desk.store import StoreError as BarsStoreError

    assert FilingStoreError is not BarsStoreError
    assert not issubclass(FilingStoreError, BarsStoreError)
