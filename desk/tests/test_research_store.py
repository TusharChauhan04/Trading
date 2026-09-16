"""R2 - the point-in-time filings store.

The price store gets its guarantee from its layout: the date is in the
filename, so a file after `as_of` is never opened. This transposes that to
filings, and the one thing that does NOT transpose - date vs datetime
granularity - is where most of these tests point.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from desk.research.models import Filing, ResultPeriod
from desk.research.store import FilingStore, StoreError, at_open
from desk.research.xbrl import parse_xbrl

FIXTURES = Path(__file__).parent / "fixtures"


def _filing(disclosed: datetime, *, symbol="AAA", end=date(2024, 12, 31),
            **kw) -> Filing:
    base = dict(
        symbol=symbol, disclosed_at=disclosed,
        period_start=date(2024, 10, 1), period_end=end,
        period=ResultPeriod.QUARTERLY, audited=False, consolidated=False,
        company_name="A Ltd", relating_to="Third Quarter",
        xbrl_url="https://nsearchives.nseindia.com/x.xml", isin="INE000A01001",
    )
    base.update(kw)
    return Filing(**base)


@pytest.fixture
def store(tmp_path) -> FilingStore:
    return FilingStore(tmp_path / "filings")


# ===========================================================================
# the datetime cutoff - the thing that does not transpose from BarStore
# ===========================================================================

def test_an_after_close_filing_is_not_visible_on_its_own_disclosure_day(store):
    """The leak this store exists to stop, on real numbers. RELIANCE's Q3 FY25
    results were disclosed 2025-01-16 at 20:20:21 - after the close. A
    backtest simulating the 16th must not see them; they were tradeable on the
    17th. A date-granular cutoff treats 20:20 and 11:00 identically, which is
    precisely the distinction Announcement.session_phase exists to make."""
    f = _filing(datetime(2025, 1, 16, 20, 20, 21))
    store.write(f)

    assert store.latest("AAA", as_of=at_open(date(2025, 1, 16))) is None
    got = store.latest("AAA", as_of=at_open(date(2025, 1, 17)))
    assert got is not None and got.disclosed_at == f.disclosed_at


def test_an_intraday_filing_becomes_visible_within_the_same_day(store):
    """The other half: a filing at 11:00 DID move that day's price, so a
    simulation standing at 15:00 must see it."""
    store.write(_filing(datetime(2025, 1, 16, 11, 0, 0)))
    assert store.latest("AAA", as_of=datetime(2025, 1, 16, 10, 0)) is None
    assert store.latest("AAA", as_of=datetime(2025, 1, 16, 15, 0)) is not None


def test_a_bare_date_is_refused_rather_than_silently_coerced(store):
    """Accepting a date here would quietly reintroduce the bug above. The
    error names the fix rather than just complaining."""
    store.write(_filing(datetime(2025, 1, 16, 20, 20, 21)))
    with pytest.raises(StoreError, match="must be a datetime"):
        store.for_symbol("AAA", as_of=date(2025, 1, 17))
    with pytest.raises(StoreError, match="at_open"):
        store.for_symbol("AAA", as_of=date(2025, 1, 17))


def test_at_open_is_the_market_open_not_midnight(store):
    """Midnight would wrongly exclude everything filed the previous evening -
    which is most of them."""
    assert at_open(date(2025, 1, 17)) == datetime(2025, 1, 17, 9, 15)


def test_a_file_after_as_of_is_never_opened(monkeypatch, store):
    """The structural guarantee, not merely the visible result. A filter is a
    line someone can delete; a file that was never listed cannot leak."""
    store.write(_filing(datetime(2025, 1, 10, 9, 0), end=date(2024, 9, 30)))
    store.write(_filing(datetime(2025, 1, 16, 20, 20, 21)))
    store.write(_filing(datetime(2025, 2, 20, 18, 0), end=date(2025, 3, 31)))

    opened = []
    original = FilingStore._read
    monkeypatch.setattr(FilingStore, "_read",
                        lambda self, p: (opened.append(p.name), original(self, p))[1])

    store.for_symbol("AAA", as_of=at_open(date(2025, 1, 17)))
    assert len(opened) == 2, opened
    assert not any(n.startswith("202502") for n in opened)


# ===========================================================================
# the period trap, carried through the store
# ===========================================================================

def test_period_defaults_to_the_quarter_not_the_year_to_date():
    """The stored filing must not make it easy to pick up the 9-month figure
    by accident - it is 3x larger and looks entirely plausible. Uses the real
    RELIANCE XBRL, which carries both."""
    from desk.research.store import StoredFiling

    periods, _ = parse_xbrl((FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes(),
                            symbol="RELIANCE")
    rec = StoredFiling(filing=_filing(datetime(2025, 1, 16, 20, 20, 21),
                                      symbol="RELIANCE"),
                       periods=tuple(periods))
    assert rec.period().months == 3                      # the default
    assert rec.period(3).revenue / 1e7 == pytest.approx(128_260, abs=1)
    assert rec.period(9).revenue / 1e7 == pytest.approx(396_645, abs=1)
    assert rec.period(12) is None                        # absent, not guessed


def test_filtering_by_months_excludes_filings_without_that_period(store):
    store.write(_filing(datetime(2025, 1, 16, 20, 0)))    # written with no periods
    # months=3 asks for the numbers, and there are none on file yet.
    assert store.latest("AAA", as_of=at_open(date(2025, 1, 17)), months=3) is None
    # But "the latest filing" must still find it - a filing whose XBRL has not
    # been fetched has not stopped existing.
    assert store.latest("AAA", as_of=at_open(date(2025, 1, 17))) is not None


# ===========================================================================
# round trip
# ===========================================================================

def test_a_round_trip_preserves_types_not_just_values(store):
    """A generic asdict/**dict round trip silently turns every date, datetime
    and enum back into a string, and everything downstream then compares a
    str against a date and quietly finds nothing."""
    periods, _ = parse_xbrl((FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes(),
                            symbol="RELIANCE")
    original = _filing(datetime(2025, 1, 16, 20, 20, 21), symbol="RELIANCE")
    store.write(original, periods)

    got = store.latest("RELIANCE", as_of=at_open(date(2025, 1, 17)))
    f = got.filing
    assert isinstance(f.disclosed_at, datetime)
    assert isinstance(f.period_end, date)
    assert isinstance(f.period, ResultPeriod)
    assert f == original                                  # frozen dataclass equality

    q = got.period(3)
    assert isinstance(q.period_start, date)
    assert q.revenue / 1e7 == pytest.approx(128_260, abs=1)
    assert q.eps_basic == pytest.approx(6.44)
    assert q.net_margin_pct == pytest.approx(6.80, abs=0.01)


def test_named_fields_are_rebuilt_from_facts_rather_than_stored_twice(store):
    """One copy cannot drift from another if there is only one copy."""
    periods, _ = parse_xbrl((FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes(),
                            symbol="RELIANCE")
    store.write(_filing(datetime(2025, 1, 16, 20, 20, 21), symbol="RELIANCE"),
                periods)
    path = next((store.root / "RELIANCE").iterdir())
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert "revenue" not in blob["periods"][0]            # not stored separately
    assert "RevenueFromOperations" in blob["periods"][0]["facts"]
    assert store.latest("RELIANCE", as_of=at_open(date(2025, 1, 17))).period(3).revenue


# ===========================================================================
# writing
# ===========================================================================

def test_rewriting_the_same_filing_overwrites_rather_than_duplicating(store):
    """A refresh run twice must not double-count a quarter."""
    f = _filing(datetime(2025, 1, 16, 20, 20, 21))
    store.write(f); store.write(f); store.write(f)
    assert len(list((store.root / "AAA").iterdir())) == 1
    assert len(store.for_symbol("AAA", as_of=at_open(date(2025, 1, 17)))) == 1


def test_two_filings_disclosed_in_the_same_second_both_survive(store):
    """A company files standalone and consolidated together. Keying on the
    timestamp alone silently overwrites one with the other."""
    when = datetime(2025, 1, 16, 20, 20, 21)
    store.write(_filing(when, end=date(2024, 12, 31), consolidated=False))
    store.write(_filing(when, end=date(2024, 9, 30), consolidated=True))
    assert len(store.for_symbol("AAA", as_of=at_open(date(2025, 1, 17)))) == 2


def test_writes_are_atomic_and_leave_no_temp_file(store):
    store.write(_filing(datetime(2025, 1, 16, 20, 20, 21)))
    assert not list((store.root / "AAA").glob("*.tmp"))


def test_a_symbol_cannot_escape_the_store_directory(store):
    """The symbol becomes a directory name, so it is validated rather than
    trusted."""
    for bad in ("../../etc", "A/B", "A\\B", "..", "A B", ""):
        with pytest.raises(StoreError, match="refusing to use"):
            store.write(_filing(datetime(2025, 1, 16, 20, 0), symbol=bad))


def test_real_nse_symbols_with_ampersands_still_work(store):
    """M&M is a Nifty 50 constituent - the validator must not repeat the
    isalnum mistake."""
    for sym in ("M&M", "BAJAJ-AUTO", "J&KBANK"):
        store.write(_filing(datetime(2025, 1, 16, 20, 0), symbol=sym))
    assert set(store.symbols()) == {"M&M", "BAJAJ-AUTO", "J&KBANK"}


# ===========================================================================
# reading degradations
# ===========================================================================

def test_a_stray_file_is_ignored_rather_than_breaking_the_read(store):
    store.write(_filing(datetime(2025, 1, 16, 20, 20, 21)))
    (store.root / "AAA" / "notes.txt").write_text("hi", encoding="utf-8")
    (store.root / "AAA" / "20250116T202021_x.json.tmp").write_text("{}",
                                                                  encoding="utf-8")
    assert len(store.for_symbol("AAA", as_of=at_open(date(2025, 1, 17)))) == 1


def test_a_corrupt_filing_file_is_named_rather_than_silently_skipped(store):
    store.write(_filing(datetime(2025, 1, 16, 20, 20, 21)))
    path = next((store.root / "AAA").iterdir())
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(StoreError, match="cannot read"):
        store.for_symbol("AAA", as_of=at_open(date(2025, 1, 17)))


def test_an_unknown_symbol_is_empty_not_an_error(store):
    assert store.for_symbol("NOPE", as_of=at_open(date(2025, 1, 17))) == []
    assert store.latest("NOPE", as_of=at_open(date(2025, 1, 17))) is None
    assert store.symbols() == []


def test_latest_many_is_what_stage_2_will_call(store):
    for sym in ("AAA", "BBB", "CCC"):
        store.write(_filing(datetime(2025, 1, 16, 20, 20, 21), symbol=sym))
    store.write(_filing(datetime(2025, 2, 20, 18, 0), symbol="AAA",
                        end=date(2025, 3, 31)))

    got = store.latest_many(["AAA", "BBB", "NOPE", "../bad"],
                            as_of=at_open(date(2025, 1, 17)))
    assert set(got) == {"AAA", "BBB"}            # missing and bad both skipped
    assert got["AAA"].disclosed_at == datetime(2025, 1, 16, 20, 20, 21)

    later = store.latest_many(["AAA"], as_of=at_open(date(2025, 2, 21)))
    assert later["AAA"].disclosed_at == datetime(2025, 2, 20, 18, 0)


def test_filings_come_back_oldest_first(store):
    for d, end in ((datetime(2025, 3, 1, 18, 0), date(2024, 12, 31)),
                   (datetime(2025, 1, 16, 20, 0), date(2024, 9, 30)),
                   (datetime(2025, 2, 10, 9, 0), date(2024, 6, 30))):
        store.write(_filing(d, end=end))
    got = store.for_symbol("AAA", as_of=at_open(date(2025, 4, 1)))
    assert [r.disclosed_at for r in got] == sorted(r.disclosed_at for r in got)


def test_latest_opens_one_file_not_every_file_up_to_as_of(monkeypatch, store):
    """REGRESSION, measured: latest() first read EVERY filing up to as_of and
    discarded all but the last. Across 1,598 symbols with 8 filings each that
    was 12,784 JSON parses and 101.6 seconds - unusable for a pre-open batch.
    Walking filenames newest-first and stopping at the first match is 11.8s.

    The filename scan is also what preserves the point-in-time guarantee, so
    this is the same mechanism doing both jobs."""
    for q, end in enumerate((date(2024, 3, 31), date(2024, 6, 30),
                             date(2024, 9, 30), date(2024, 12, 31))):
        store.write(_filing(datetime(2024 + q // 4, 1 + 3 * (q % 4), 16, 20, 0),
                            end=end))

    opened = []
    original = FilingStore._read
    monkeypatch.setattr(FilingStore, "_read",
                        lambda self, p: (opened.append(p.name), original(self, p))[1])

    got = store.latest("AAA", as_of=at_open(date(2026, 1, 1)))
    assert got is not None
    assert len(opened) == 1, f"opened {len(opened)} files to find the latest"


def test_latest_keeps_looking_when_the_newest_lacks_the_period_asked_for(store):
    """Stopping at the first file would be wrong when months= is supplied:
    the newest filing may have no XBRL on file yet while an older one does."""
    from desk.research.xbrl import parse_xbrl

    periods, _ = parse_xbrl((FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes(),
                            symbol="AAA")
    store.write(_filing(datetime(2025, 1, 16, 20, 0), end=date(2024, 12, 31)),
                periods)                                   # older, has numbers
    store.write(_filing(datetime(2025, 4, 20, 20, 0), end=date(2025, 3, 31)))
    # newest, no numbers

    assert store.latest("AAA", as_of=at_open(date(2025, 5, 1))).disclosed_at \
        == datetime(2025, 4, 20, 20, 0)
    with_numbers = store.latest("AAA", as_of=at_open(date(2025, 5, 1)), months=3)
    assert with_numbers.disclosed_at == datetime(2025, 1, 16, 20, 0)


def test_standalone_and_consolidated_filed_in_the_same_second_both_survive(store):
    """REGRESSION, and it cost 5 of RELIANCE's 122 filings before it was
    caught. A company files its STANDALONE and CONSOLIDATED results in the
    same second for the same period - RELIANCE did it on 2022-10-21,
    2018-01-24, 2017-10-17, 2015-10-20 and 2014-10-13. Keyed on timestamp and
    period alone, one silently overwrote the other, and WHICH one survived
    depended on write order. Consolidated revenue is roughly twice standalone,
    so the survivor was not merely arbitrary - it was arbitrarily one of two
    very different numbers."""
    when = datetime(2022, 10, 21, 19, 58, 47)
    store.write(_filing(when, end=date(2022, 9, 30), consolidated=True))
    store.write(_filing(when, end=date(2022, 9, 30), consolidated=False))
    store.write(_filing(when, end=date(2022, 9, 30), consolidated=None))

    got = store.for_symbol("AAA", as_of=at_open(date(2022, 10, 22)))
    assert len(got) == 3
    assert {r.filing.consolidated for r in got} == {True, False, None}


def test_the_whole_reliance_filing_history_round_trips_without_loss(store):
    """Against the real payload rather than a constructed pair - 122 filings
    in, 122 files out. This is the test that would have caught the collision
    at the time, and it is the shape of check worth having wherever a
    filename is a primary key."""
    import json as _json

    rows = _json.loads((FIXTURES / "nse_results_RELIANCE.json")
                       .read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    from desk.research.sources.nse import parse_results

    filings, _ = parse_results(rows, "RELIANCE")
    for f in filings:
        store.write(f)
    assert len(list((store.root / "RELIANCE").iterdir())) == len(filings) == 122
