"""R6: BSE as a second source, and the NSE<->BSE reconciliation.

The property that matters most is NOT that disagreements are found - it is
that a symbol BSE barely traded is reported as NOT CHECKED rather than as
agreeing. Two thirds of matched names fall in that bucket, so a check that
quietly counted them as passes would be claiming corroboration it does not
have across most of the universe.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from desk.marketdata import quality as q
from desk.marketdata.crosscheck import (
    DEFAULT_TOLERANCE_PCT, MIN_BSE_TURNOVER, bse_reference, reconcile,
)
from desk.marketdata.isin import IsinMap, parse_equity_master
from desk.marketdata.sources.bse import (
    BseSession, bhavcopy_url, parse_bhavcopy,
)
from desk.marketdata.sources.errors import RateLimited, SourceError

FIXTURES = Path(__file__).parent / "fixtures"
DAY = date(2026, 9, 11)


@pytest.fixture(scope="module")
def bse_raw() -> bytes:
    return (FIXTURES / "bse_bhavcopy_20260911.csv").read_bytes()


@pytest.fixture(scope="module")
def isin_map() -> IsinMap:
    return parse_equity_master((FIXTURES / "nse_equity_master.csv").read_bytes(),
                               as_of=date(2026, 9, 18))


@pytest.fixture(scope="module")
def bse(bse_raw) -> pd.DataFrame:
    return parse_bhavcopy(bse_raw, day=DAY)


@pytest.fixture(scope="module")
def nse() -> pd.DataFrame:
    return pd.read_parquet(Path("configs/bhavcopy/2026-09-11.parquet"))


def _bse_rows(rows) -> pd.DataFrame:
    """A minimal BSE frame in the shape parse_bhavcopy produces."""
    return pd.DataFrame(rows, columns=["isin", "date", "close", "turnover"])


def _nse_rows(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["symbol", "date", "close"])


# --- the BSE source ------------------------------------------------------

def test_real_bse_bhavcopy_parses(bse):
    assert len(bse) == 5007
    assert set(bse["date"]) == {DAY}
    assert bse["symbol"].str.endswith(".BO").all()
    assert bse["isin"].str.match(r"^IN[A-Z0-9]{10}$").all()
    assert (bse["close"] > 0).all()


def test_bse_wrong_day_is_refused(bse_raw):
    with pytest.raises(SourceError, match="different TradDt"):
        parse_bhavcopy(bse_raw, day=date(2026, 9, 10))


def test_bse_missing_columns_refused():
    with pytest.raises(SourceError, match="missing expected columns"):
        parse_bhavcopy(b"TradDt,ISIN\n2026-09-11,INE002A01018\n")


def test_bse_empty_file_refused():
    hdr = (b"TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,"
           b"SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal\n")
    with pytest.raises(SourceError, match="no rows"):
        parse_bhavcopy(hdr)


def test_bse_non_equity_segment_refused():
    hdr = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,"
           "SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal\n")
    row = ("2026-09-11,2026-09-11,FO,BSE,FUT,1,INE002A01018,RELIANCE,A,"
           "1,1,1,1,1,1\n")
    with pytest.raises(SourceError, match="no STK rows"):
        parse_bhavcopy((hdr + row).encode())


def test_bhavcopy_url_is_dated():
    assert bhavcopy_url(DAY).endswith("BhavCopy_BSE_CM_0_0_0_20260911_F_0000.CSV")


# --- the HTTP-200-with-HTML trap -----------------------------------------

def test_bse_answers_a_missing_file_with_html_under_http_200():
    """Captured live. BSE does NOT 404 a file it does not have - it serves
    the site's single-page-app shell with status 200 and text/html. Anything
    trusting the status code would hand this to a CSV parser."""
    html = (FIXTURES / "bse_missing_file_response.html").read_bytes()
    assert html.lstrip().startswith(b"<!DOCTYPE html")

    sess = BseSession(min_interval=0.0)
    sess._fetch_raw = lambda url: html          # noqa: SLF001 - the point
    with pytest.raises(SourceError, match="not a bhavcopy"):
        sess.fetch_bhavcopy(DAY)


def test_the_html_rejection_says_what_actually_happened():
    html = (FIXTURES / "bse_missing_file_response.html").read_bytes()
    sess = BseSession(min_interval=0.0)
    sess._fetch_raw = lambda url: html          # noqa: SLF001
    with pytest.raises(SourceError) as exc:
        sess.fetch_bhavcopy(DAY)
    assert "status code was 200" in str(exc.value)


def test_non_html_garbage_is_also_refused():
    sess = BseSession(min_interval=0.0)
    sess._fetch_raw = lambda url: b"\x00\x01binary nonsense"   # noqa: SLF001
    with pytest.raises(SourceError, match="not a bhavcopy"):
        sess.fetch_bhavcopy(DAY)


def test_valid_csv_passes_the_content_check(bse_raw):
    sess = BseSession(min_interval=0.0)
    sess._fetch_raw = lambda url: bse_raw       # noqa: SLF001
    assert sess.fetch_bhavcopy(DAY) is bse_raw


# --- session discipline --------------------------------------------------

def test_session_refuses_a_non_bse_url():
    sess = BseSession(min_interval=0.0)
    for bad in ("https://bseindia.com.evil.example/x.csv",
                "http://www.bseindia.com/x.csv",
                "https://evil.example/x.csv"):
        with pytest.raises(SourceError, match="non-BSE URL"):
            sess._fetch_raw(bad)                # noqa: SLF001


def test_request_budget_is_enforced():
    sess = BseSession(min_interval=0.0, max_requests=2)
    sess._wait_turn(); sess._wait_turn()        # noqa: SLF001
    with pytest.raises(SourceError, match="request budget"):
        sess._wait_turn()                       # noqa: SLF001


def test_throttle_uses_a_monotonic_clock():
    """time.time() can step backwards over an NTP correction, which would
    disable the throttle exactly when a long batch is running."""
    import inspect
    src = inspect.getsource(BseSession._wait_turn)
    body = src.split('"""')[2]                  # past the docstring
    assert "time.monotonic()" in body
    assert "time.time()" not in body


def test_rate_limit_is_retried_but_a_source_error_is_not():
    sess = BseSession(min_interval=0.0)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RateLimited("429")
        return "ok"

    assert sess.fetch_with_retry(flaky, base_delay=0.0) == "ok"
    assert len(calls) == 3

    def broken():
        calls.append(1)
        raise SourceError("endpoint moved")

    before = len(calls)
    with pytest.raises(SourceError):
        sess.fetch_with_retry(broken, base_delay=0.0)
    assert len(calls) == before + 1, "a non-transient failure must not retry"


# --- the ISIN map --------------------------------------------------------

def test_real_equity_master_parses(isin_map):
    assert len(isin_map) == 2577
    assert isin_map.ambiguous == ()
    assert isin_map.rejected == ()
    assert isin_map.isin_for("RELIANCE") == "INE002A01018"
    assert isin_map.isin_for("RELIANCE.NS") == "INE002A01018"
    assert isin_map.symbol_for("INE002A01018") == "RELIANCE"


def test_malformed_isins_are_rejected_not_kept():
    csv = ("SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\n"
           "GOOD,Good Ltd,EQ,INE002A01018\n"
           "BAD,Bad Ltd,EQ,-\n"
           "ALSOBAD,Also Ltd,EQ,US0378331005\n").encode()
    m = parse_equity_master(csv)
    assert len(m) == 1
    assert m.isin_for("BAD") is None
    assert len(m.rejected) == 2


def test_an_isin_claimed_by_two_symbols_is_ambiguous_and_unusable():
    """Picking one is how a join silently attaches the wrong security."""
    csv = ("SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\n"
           "AAA,A Ltd,EQ,INE000A01001\n"
           "BBB,B Ltd,EQ,INE000A01001\n"
           "CCC,C Ltd,EQ,INE111A01011\n").encode()
    m = parse_equity_master(csv)
    assert m.ambiguous == ("INE000A01001",)
    assert m.symbol_for("INE000A01001") is None, "must refuse, not guess"
    assert m.symbol_for("INE111A01011") == "CCC"


def test_equity_master_with_no_usable_rows_refused():
    with pytest.raises(SourceError, match="zero usable rows"):
        parse_equity_master(b"SYMBOL,ISIN NUMBER\nX,-\n")


def test_equity_master_missing_columns_refused():
    with pytest.raises(SourceError, match="missing expected columns"):
        parse_equity_master(b"SYMBOL,NAME\nX,Y\n")


def test_isin_map_round_trips(tmp_path, isin_map):
    p = tmp_path / "isin.json"
    isin_map.save(p)
    back = IsinMap.load(p)
    assert back.by_symbol == isin_map.by_symbol
    assert back.as_of == isin_map.as_of
    assert back.symbol_for("INE002A01018") == "RELIANCE"


def test_saving_leaves_no_partial_file(tmp_path, isin_map):
    p = tmp_path / "isin.json"
    isin_map.save(p)
    isin_map.save(p)
    assert not list(tmp_path.glob("*.tmp"))


# --- coverage, which is the honest part ----------------------------------

def test_coverage_buckets_account_for_every_symbol(nse, bse, isin_map):
    _ref, cov = bse_reference(nse, bse, isin_map)
    assert cov.nse_symbols == len(nse)
    assert cov.balanced, "a coverage report that does not add up hides a case"


def test_real_coverage_is_about_a_third(nse, bse, isin_map):
    """Measured 2026-09-11. If this moves sharply something changed upstream
    and the daily plan's coverage claim would be wrong."""
    _ref, cov = bse_reference(nse, bse, isin_map)
    assert cov.checkable == 759
    assert cov.thin_on_bse > cov.checkable, "most matched names are thin"
    assert "759" in cov.report()


def test_thin_symbols_are_excluded_from_the_reference_entirely():
    """A caller must not be able to reconcile against a thin print by
    forgetting to filter."""
    m = IsinMap(by_symbol={"THIN": "INE000A01001", "DEEP": "INE111A01011"},
                by_isin={"INE000A01001": "THIN", "INE111A01011": "DEEP"})
    n = _nse_rows([("THIN.NS", DAY, 100.0), ("DEEP.NS", DAY, 200.0)])
    b = _bse_rows([("INE000A01001", DAY, 140.0, MIN_BSE_TURNOVER / 100),
                   ("INE111A01011", DAY, 200.0, MIN_BSE_TURNOVER * 10)])
    ref, cov = bse_reference(n, b, m)
    assert list(ref["symbol"]) == ["DEEP.NS"]
    assert cov.thin_on_bse == 1
    assert cov.checkable == 1


def test_symbol_absent_from_bse_is_counted_separately_from_thin():
    m = IsinMap(by_symbol={"ONLYNSE": "INE000A01001"},
                by_isin={"INE000A01001": "ONLYNSE"})
    n = _nse_rows([("ONLYNSE.NS", DAY, 100.0)])
    b = _bse_rows([])
    _ref, cov = bse_reference(n, b, m)
    assert (cov.not_on_bse, cov.thin_on_bse, cov.checkable) == (1, 0, 0)


def test_symbol_with_no_isin_is_counted_separately():
    m = IsinMap(by_symbol={}, by_isin={})
    n = _nse_rows([("SMESTOCK.NS", DAY, 100.0)])
    _ref, cov = bse_reference(n, _bse_rows([]), m)
    assert cov.no_isin == 1 and cov.balanced


def test_ambiguous_isin_is_never_joined():
    m = IsinMap(by_symbol={"AAA": "INE000A01001"},
                by_isin={}, ambiguous=("INE000A01001",))
    n = _nse_rows([("AAA.NS", DAY, 100.0)])
    b = _bse_rows([("INE000A01001", DAY, 999.0, MIN_BSE_TURNOVER * 10)])
    ref, cov = bse_reference(n, b, m)
    assert ref.empty
    assert cov.ambiguous_isin == 1 and cov.balanced


def test_duplicate_isin_on_one_date_is_refused_not_silently_fanned_out():
    m = IsinMap(by_symbol={"AAA": "INE000A01001"},
                by_isin={"INE000A01001": "AAA"})
    n = _nse_rows([("AAA.NS", DAY, 100.0)])
    b = _bse_rows([("INE000A01001", DAY, 100.0, MIN_BSE_TURNOVER * 10),
                   ("INE000A01001", DAY, 101.0, MIN_BSE_TURNOVER * 10)])
    with pytest.raises(ValueError, match="more than once on the same date"):
        bse_reference(n, b, m)


def test_missing_columns_are_named():
    m = IsinMap()
    with pytest.raises(KeyError, match="close"):
        bse_reference(pd.DataFrame({"symbol": [], "date": []}),
                      _bse_rows([]), m)


# --- reconciliation ------------------------------------------------------

def test_reconcile_finds_a_planted_disagreement():
    m = IsinMap(by_symbol={"AAA": "INE000A01001"},
                by_isin={"INE000A01001": "AAA"})
    n = _nse_rows([("AAA.NS", DAY, 200.0)])
    b = _bse_rows([("INE000A01001", DAY, 100.0, MIN_BSE_TURNOVER * 10)])
    out, cov = reconcile(n, b, m)
    assert len(out) == 1
    assert out.iloc[0]["diff_pct"] == pytest.approx(100.0)
    assert cov.checkable == 1


def test_reconcile_is_quiet_when_the_exchanges_agree():
    m = IsinMap(by_symbol={"AAA": "INE000A01001"},
                by_isin={"INE000A01001": "AAA"})
    n = _nse_rows([("AAA.NS", DAY, 100.0)])
    b = _bse_rows([("INE000A01001", DAY, 100.2, MIN_BSE_TURNOVER * 10)])
    out, _cov = reconcile(n, b, m)
    assert out.empty


def test_reconcile_on_real_data_is_within_the_measured_envelope(nse, bse, isin_map):
    """Six sessions of measurement put the pooled max at 4.393% above the
    liquidity gate. One day should sit well inside that."""
    out, cov = reconcile(nse, bse, isin_map)
    assert cov.checkable == 759
    assert len(out) == 0, f"unexpected disagreements: {out.to_dict('records')}"


def test_tolerance_default_matches_the_measurement():
    assert DEFAULT_TOLERANCE_PCT == 2.0


# --- the quality.py wiring, including the bug this found -----------------

def _panel(sym_closes: dict[str, list[float]], days) -> pd.DataFrame:
    rows = []
    for sym, closes in sym_closes.items():
        for d, c in zip(days, closes):
            rows.append({"symbol": sym, "date": pd.Timestamp(d), "open": c,
                         "high": c, "low": c, "close": c, "volume": 1000})
    return pd.DataFrame(rows).set_index("date").sort_index()


def test_each_symbol_is_compared_against_its_own_reference_only():
    """REGRESSION. check_panel forwarded ONE reference frame to every symbol
    while its own comment said the reference is per-symbol - the fix had been
    applied to corporate actions and not to this. With a multi-symbol
    reference, CHEAP's 10.0 was joined against PRICEY's 1000.0 on the same
    date and reported as a 99% disagreement between exchanges.
    """
    days = [date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11)]
    panel = _panel({"CHEAP.NS": [10.0, 10.0, 10.0],
                    "PRICEY.NS": [1000.0, 1000.0, 1000.0]}, days)
    ref = pd.DataFrame({
        "symbol": ["CHEAP.NS"] * 3 + ["PRICEY.NS"] * 3,
        "close": [10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0],
    }, index=pd.DatetimeIndex(list(days) * 2, name="date")).sort_index()

    reps = q.check_panel(panel, by="symbol", reference=ref,
                         reference_name="BSE", price_tolerance_pct=2.0)

    for sym, rep in reps.items():
        assert q.Check.CROSS_SOURCE in rep.checks_run, sym
        assert not [i for i in rep.warnings
                    if i.check is q.Check.CROSS_SOURCE], (
            f"{sym} was compared against another symbol's prices")


def test_a_real_disagreement_still_surfaces_through_check_panel():
    days = [date(2026, 9, 9), date(2026, 9, 10)]
    panel = _panel({"AAA.NS": [100.0, 100.0], "BBB.NS": [50.0, 50.0]}, days)
    ref = pd.DataFrame({
        "symbol": ["AAA.NS", "AAA.NS", "BBB.NS", "BBB.NS"],
        "close": [100.0, 130.0, 50.0, 50.0],
    }, index=pd.DatetimeIndex(list(days) * 2, name="date")).sort_index()

    reps = q.check_panel(panel, by="symbol", reference=ref,
                         reference_name="BSE", price_tolerance_pct=2.0)
    assert [i for i in reps["AAA.NS"].warnings
            if i.check is q.Check.CROSS_SOURCE]
    assert not [i for i in reps["BBB.NS"].warnings
                if i.check is q.Check.CROSS_SOURCE]


def test_a_symbol_missing_from_the_reference_is_skipped_not_passed():
    """Two thirds of the universe lands here. Counting it as a pass would
    claim corroboration across most of the market that does not exist."""
    days = [date(2026, 9, 9), date(2026, 9, 10)]
    panel = _panel({"COVERED.NS": [100.0, 100.0], "THIN.NS": [7.0, 7.0]}, days)
    ref = pd.DataFrame({"symbol": ["COVERED.NS"] * 2, "close": [100.0, 100.0]},
                       index=pd.DatetimeIndex(days, name="date"))

    reps = q.check_panel(panel, by="symbol", reference=ref,
                         reference_name="BSE")
    assert q.Check.CROSS_SOURCE in reps["COVERED.NS"].checks_run
    assert q.Check.CROSS_SOURCE not in reps["THIN.NS"].checks_run
    assert q.Check.CROSS_SOURCE in reps["THIN.NS"].checks_skipped


def test_single_symbol_reference_without_a_symbol_column_still_works():
    """The original contract. A frame with no symbol column is one
    instrument and is forwarded whole."""
    days = [date(2026, 9, 9), date(2026, 9, 10)]
    panel = _panel({"AAA.NS": [100.0, 100.0]}, days)
    ref = pd.DataFrame({"close": [100.0, 140.0]},
                       index=pd.DatetimeIndex(days, name="date"))
    reps = q.check_panel(panel, by="symbol", reference=ref,
                         price_tolerance_pct=2.0)
    assert [i for i in reps["AAA.NS"].warnings
            if i.check is q.Check.CROSS_SOURCE]


def test_the_reference_produced_by_crosscheck_plugs_straight_into_quality(
        nse, bse, isin_map):
    """End to end on real data: build the reference, run the panel gate."""
    ref, cov = bse_reference(nse, bse, isin_map)
    picked = ["RELIANCE.NS", "TCS.NS", "SANWARIA.NS"]
    sub = nse[nse["symbol"].isin(picked)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.set_index("date").sort_index()

    reps = q.check_panel(sub, by="symbol", reference=ref,
                         reference_name="BSE",
                         price_tolerance_pct=DEFAULT_TOLERANCE_PCT)

    assert q.Check.CROSS_SOURCE in reps["RELIANCE.NS"].checks_run
    assert q.Check.CROSS_SOURCE in reps["TCS.NS"].checks_run
    # Traded 6,504 shares on BSE against 549,342 on NSE - below the gate.
    assert q.Check.CROSS_SOURCE in reps["SANWARIA.NS"].checks_skipped
    assert cov.checkable == 759
