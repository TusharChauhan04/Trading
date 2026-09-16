"""Calendar and data-quality tests.

The calendar's most important behaviour is what it does with data it does not
have, so that gets tested first and hardest.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time

import pandas as pd
import pytest

# desk is an installed package (pip install -e .) - no sys.path hack needed.

from desk.marketdata import quality as q
from desk.marketdata.calendar_in import (
    CalendarError,
    CalendarNotLoaded,
    TradingCalendar,
)
from desk.marketdata.corporate_actions import CorporateAction


# ===========================================================================
# Calendar
# ===========================================================================

def _cal() -> TradingCalendar:
    """A deliberately small, INVENTED calendar. These are not real NSE dates -
    they exist to exercise the machinery, which is exactly why the module
    refuses to ship a hardcoded list of its own."""
    cal = TradingCalendar(exchange="NSE")
    cal.load({
        "exchange": "NSE",
        "years": [2026],
        "holidays": [
            {"date": "2026-01-26", "name": "Republic Day"},
            {"date": "2026-08-15", "name": "Independence Day"},
        ],
        "special_sessions": [
            {"date": "2026-11-09", "name": "Muhurat Trading",
             "start": "18:15", "end": "19:15"},
        ],
    })
    return cal


def test_unloaded_year_refuses_rather_than_assuming_weekends():
    """The single most important calendar behaviour.

    Falling back to weekends-only would be right about 95% of days and wrong
    about precisely the ones that matter, and it would fail silently.
    """
    cal = _cal()
    with pytest.raises(CalendarNotLoaded):
        cal.is_trading_day(date(2027, 3, 2))     # a plain Tuesday, never loaded


def test_a_loaded_year_with_no_holidays_is_not_the_same_as_unloaded():
    cal = TradingCalendar()
    cal.load({"years": [2026], "holidays": []})
    assert cal.is_trading_day(date(2026, 3, 3)) is True     # loaded, no holiday
    with pytest.raises(CalendarNotLoaded):
        cal.is_trading_day(date(2025, 3, 3))                # never loaded


def test_payload_must_declare_its_years():
    cal = TradingCalendar()
    with pytest.raises(CalendarError, match="declare which years"):
        cal.load({"holidays": [{"date": "2026-01-26", "name": "x"}]})


def test_weekends_and_holidays_are_both_closed():
    cal = _cal()
    assert not cal.is_trading_day(date(2026, 1, 26))    # Republic Day, a Monday
    assert cal.holiday_name(date(2026, 1, 26)) == "Republic Day"
    assert not cal.is_trading_day(date(2026, 3, 7))     # Saturday
    assert not cal.is_trading_day(date(2026, 3, 8))     # Sunday
    assert cal.is_trading_day(date(2026, 3, 6))         # Friday


def test_muhurat_day_is_not_a_trading_day():
    """The regular session is shut; a separate evening window opens. Treating
    it as a normal session misaligns every bar that day."""
    cal = _cal()
    d = date(2026, 11, 9)
    special = cal.special_session(d)
    assert special is not None and special.name == "Muhurat Trading"
    # It is a Monday and not in the holiday list, so it reads as a trading day -
    # which is why callers must ask special_session() before assuming hours.
    assert cal.is_open_at(datetime.combine(d, time(11, 0))) is False
    assert cal.is_open_at(datetime.combine(d, time(18, 30))) is True


def test_navigation_skips_holidays_and_weekends():
    cal = _cal()
    # Fri 2026-08-14 -> Sat, Sun, then Independence Day falls on Sat 15th.
    assert cal.next_trading_day(date(2026, 1, 23)) == date(2026, 1, 27)
    assert cal.previous_trading_day(date(2026, 1, 27)) == date(2026, 1, 23)


def test_sessions_between_counts_only_open_days():
    cal = _cal()
    sessions = cal.sessions_between(date(2026, 1, 23), date(2026, 1, 27))
    assert sessions == [date(2026, 1, 23), date(2026, 1, 27)]
    assert cal.session_count(date(2026, 1, 23), date(2026, 1, 27)) == 2


def test_reversed_range_is_an_error_not_an_empty_list():
    cal = _cal()
    with pytest.raises(CalendarError, match="before start"):
        cal.sessions_between(date(2026, 3, 10), date(2026, 3, 1))


def test_settlement_is_t_plus_one_trading_day():
    cal = _cal()
    # Friday trade settles Monday, not Saturday.
    assert cal.settlement_date(date(2026, 3, 6)) == date(2026, 3, 9)
    with pytest.raises(CalendarError, match="not a trading day"):
        cal.settlement_date(date(2026, 1, 26))


def test_session_hours():
    cal = _cal()
    d = date(2026, 3, 6)                                  # a Friday
    assert not cal.is_open_at(datetime.combine(d, time(9, 5)))    # pre-open
    assert cal.is_open_at(datetime.combine(d, time(9, 15)))
    assert cal.is_open_at(datetime.combine(d, time(15, 30)))
    assert not cal.is_open_at(datetime.combine(d, time(15, 31)))


def test_from_file_round_trip(tmp_path):
    payload = {"exchange": "NSE", "years": [2026],
               "holidays": [{"date": "2026-01-26", "name": "Republic Day"}]}
    p = tmp_path / "hol.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    cal = TradingCalendar.from_file(p)
    assert cal.loaded_years == [2026]
    assert not cal.is_trading_day(date(2026, 1, 26))


def test_missing_holiday_file_says_what_to_do():
    with pytest.raises(CalendarError, match="will not guess"):
        TradingCalendar.from_file("does/not/exist.json")


# ===========================================================================
# Data quality
# ===========================================================================

def _bars(closes, start="2026-01-05", volume=1000):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {"open": closes, "high": [c * 1.01 for c in closes],
         "low": [c * 0.99 for c in closes], "close": closes,
         "volume": [volume] * len(closes)}, index=idx)


def test_clean_series_passes_everything():
    df = _bars([100, 101, 102, 103, 104])
    rep = q.check(df, symbol="X.NS")
    assert rep.usable
    assert rep.fatal == []
    assert q.Check.OHLC_COHERENCE in rep.checks_run


def test_missing_columns_stop_the_run_immediately():
    """Every later check reads a price column, so continuing would raise."""
    df = pd.DataFrame({"close": [1, 2]}, index=pd.bdate_range("2026-01-05", periods=2))
    rep = q.check(df, symbol="X.NS")
    assert not rep.usable
    assert rep.fatal[0].check is q.Check.SCHEMA
    assert len(rep.checks_run) == 1              # bailed out, did not crash


def test_empty_and_wrong_index_are_fatal_not_crashes():
    empty = pd.DataFrame(columns=["open", "high", "low", "close"],
                         index=pd.DatetimeIndex([]))
    assert not q.check(empty).usable

    wrong = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]})
    rep = q.check(wrong)
    assert not rep.usable
    assert rep.fatal[0].check is q.Check.SCHEMA


def test_duplicate_bars_are_fatal():
    df = _bars([100, 101, 102])
    df = pd.concat([df, df.iloc[[1]]]).sort_index()
    rep = q.check(df, symbol="X.NS")
    assert not rep.usable
    assert any(i.check is q.Check.DUPLICATE_BARS for i in rep.fatal)


def test_unsorted_index_is_fatal():
    df = _bars([100, 101, 102]).iloc[::-1]
    rep = q.check(df)
    assert any(i.check is q.Check.INDEX_ORDER for i in rep.fatal)


def test_incoherent_ohlc_is_caught():
    df = _bars([100, 101, 102])
    df.loc[df.index[1], "high"] = 50.0            # high below low and close
    rep = q.check(df)
    assert any(i.check is q.Check.OHLC_COHERENCE for i in rep.fatal)


def test_non_positive_and_null_prices_are_fatal():
    df = _bars([100, 0, 102])
    assert any(i.check is q.Check.NON_POSITIVE_PRICE for i in q.check(df).fatal)

    df2 = _bars([100, 101, 102])
    df2.loc[df2.index[1], "close"] = None
    assert any(i.check is q.Check.NON_POSITIVE_PRICE for i in q.check(df2).fatal)


def test_stale_quote_run_is_flagged():
    df = _bars([100, 100, 100, 100, 100, 100, 105])
    rep = q.check(df)
    stale = [i for i in rep.issues if i.check is q.Check.STALE_QUOTE]
    assert stale and stale[0].severity is q.Severity.WARN


def test_zero_volume_escalates_with_prevalence():
    mild = _bars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    mild.loc[mild.index[0], "volume"] = 0                 # 10%
    assert q.check(mild).usable                           # warn only

    bad = _bars([100, 101, 102, 103], volume=0)           # 100%
    assert not q.check(bad).usable


def test_extreme_move_without_actions_warns_and_says_why():
    df = _bars([1000, 1000, 200, 200])
    rep = q.check(df, symbol="X.NS")
    assert rep.usable                                     # warn, not fatal
    assert q.Check.CORPORATE_ACTION in rep.checks_skipped
    assert any(i.check is q.Check.EXTREME_RETURN for i in rep.warnings)


def test_extreme_move_with_no_matching_action_is_fatal():
    df = _bars([1000, 1000, 200, 200])
    rep = q.check(df, symbol="X.NS", actions=[])
    assert not rep.usable
    assert any(i.check is q.Check.CORPORATE_ACTION for i in rep.fatal)


def test_extreme_move_explained_by_a_split_is_only_informational():
    df = _bars([1000, 1000, 200, 200])
    split = CorporateAction.split("X.NS", df.index[2].date(), 1, 5)
    rep = q.check(df, symbol="X.NS", actions=[split])
    assert rep.usable
    assert any(i.severity is q.Severity.INFO for i in rep.issues)


def test_missing_and_unexpected_sessions():
    from datetime import timedelta
    df = _bars([100, 101, 102])                            # Mon Tue Wed
    # Ask for one more session than exists -> missing bar (a warning).
    expected = sorted({*[d.date() for d in df.index],
                       df.index[-1].date() + timedelta(days=1)})
    rep = q.check(df, expected_sessions=expected)
    assert any(i.check is q.Check.MISSING_BARS for i in rep.warnings)

    # A bar on a day the exchange was shut is fatal - padded data.
    rep2 = q.check(df, expected_sessions=[d.date() for d in df.index[:2]])
    assert any(i.check is q.Check.UNEXPECTED_BARS for i in rep2.fatal)


def test_cross_source_disagreement_is_reported():
    a = _bars([100, 101, 102])
    b = a.copy()
    b.loc[b.index[1], "close"] = 120.0
    rep = q.check(a, reference=b, reference_name="kite")
    assert any(i.check is q.Check.CROSS_SOURCE for i in rep.warnings)


def test_skipped_checks_are_recorded_not_silently_passed():
    df = _bars([100, 101, 102]).drop(columns=["volume"])
    rep = q.check(df)
    assert q.Check.ZERO_VOLUME in rep.checks_skipped
    assert q.Check.CROSS_SOURCE in rep.checks_skipped
    assert q.Check.MISSING_BARS in rep.checks_skipped
    # Keyed by the enum the module already defines, not ad-hoc strings - so a
    # skip can be filtered and waived the same way a finding can.
    assert all(isinstance(k, q.Check) for k in rep.checks_skipped)


def test_assert_clean_raises_on_fatal_and_returns_on_pass():
    good = _bars([100, 101, 102])
    assert q.assert_clean(good, symbol="X.NS").usable

    bad = _bars([100, 101, 102])
    bad.loc[bad.index[1], "high"] = 1.0
    with pytest.raises(ValueError, match="failed the data-quality gate"):
        q.assert_clean(bad, symbol="X.NS")


# ===========================================================================
# Regressions from the 2026-09-11 reviews
# ===========================================================================

def test_multi_symbol_panel_is_refused_not_silently_certified():
    """REGRESSION, and the worst of the set.

    Handed a stacked panel - the natural shape of a Parquet read - the index
    looked monotonic and unduplicated, so the seam between two different shares
    read as a legitimate price move. `check()` returned usable=True on a frame
    containing a fabricated several-thousand-percent return.
    """
    a = _bars([100, 101, 102], start="2026-01-05")
    a["symbol"] = "A.NS"
    b = _bars([4800, 4810, 4820], start="2026-01-08")
    b["symbol"] = "B.NS"
    panel = pd.concat([a, b])

    with pytest.raises(q.PanelError, match="ONE instrument"):
        q.check(panel)


def test_check_panel_splits_and_reports_per_symbol():
    a = _bars([100, 101, 102], start="2026-01-05"); a["symbol"] = "A.NS"
    b = _bars([4800, 4810, 4820], start="2026-01-08"); b["symbol"] = "B.NS"
    reports = q.check_panel(pd.concat([a, b]))
    assert set(reports) == {"A.NS", "B.NS"}
    assert all(r.usable for r in reports.values())
    assert reports["A.NS"].bars == 3


def test_single_symbol_column_still_passes():
    """A symbol column is fine as long as it names one instrument."""
    df = _bars([100, 101, 102])
    df["symbol"] = "A.NS"
    assert q.check(df, symbol="A.NS").usable


def test_overnight_gap_is_caught_even_when_close_to_close_looks_calm():
    """REGRESSION: the corporate-action check was gated behind a close-to-close
    screen, but it measures OPEN against the previous close. A 30% gap down
    that recovered intraday never reached the check, and assert_clean passed a
    series with a missing split in it."""
    df = _bars([1000, 1000, 1000])
    # Gap the open down 30% on the last bar but recover the close to -15%.
    df.loc[df.index[2], "open"] = 700.0
    df.loc[df.index[2], "low"] = 690.0
    df.loc[df.index[2], "close"] = 850.0

    close_to_close = abs(850 / 1000 - 1)
    assert close_to_close < q.CIRCUIT_LIMIT_PCT / 100    # invisible to screen 2

    rep = q.check(df, symbol="X.NS", actions=[])
    assert not rep.usable
    assert any(i.check is q.Check.CORPORATE_ACTION for i in rep.fatal)


def test_extreme_moves_are_only_called_explained_when_a_matching_action_exists():
    """REGRESSION: `explained` subtracted one screen's count from the other's -
    unrelated sets - so an EMPTY action list produced the report line
    'all explained by corporate actions on file'."""
    df = _bars([100, 130, 169, 220])          # +30% a day, no gaps
    rep = q.check(df, symbol="X.NS", actions=[])
    assert not any(
        i.severity is q.Severity.INFO and "ex-date" in i.message
        for i in rep.issues
    )
    assert any(i.check is q.Check.EXTREME_RETURN and i.severity is q.Severity.WARN
               for i in rep.issues)


def test_non_numeric_columns_are_reported_not_raised():
    """REGRESSION: a feed returning "N/A" crashed the gate with a TypeError
    from inside a comparison, instead of reporting a schema failure."""
    df = _bars([100, 101, 102])
    df["close"] = ["100", "N/A", "102"]
    rep = q.check(df, symbol="X.NS")
    assert not rep.usable
    assert any("non-numeric" in i.message for i in rep.fatal)


def test_duplicate_column_labels_are_reported_not_raised():
    df = _bars([100, 101, 102])
    df = pd.concat([df, df[["close"]]], axis=1)      # two 'close' columns
    rep = q.check(df, symbol="X.NS")
    assert not rep.usable
    assert any("duplicate column" in i.message for i in rep.fatal)


def test_null_volume_is_not_reported_as_zero_volume():
    df = _bars([100, 101, 102, 103])
    # float NaN, so the column stays numeric and reaches the volume check.
    # An all-None object column is a different failure and is caught earlier
    # by the schema gate, correctly.
    df["volume"] = float("nan")
    rep = q.check(df)
    assert any("null volume" in i.message for i in rep.issues)
    assert not any(i.check is q.Check.ZERO_VOLUME and i.severity is q.Severity.FATAL
                   for i in rep.issues)


def test_stale_run_counts_bars_not_gaps():
    """Six identical closes are six bars, not five."""
    df = _bars([100, 100, 100, 100, 100, 100])
    stale = [i for i in q.check(df).issues if i.check is q.Check.STALE_QUOTE]
    assert stale and stale[0].rows == 6


def test_special_session_day_is_not_a_trading_day():
    """REGRESSION: is_trading_day consulted only the holiday map, so a Muhurat
    day counted as a full session while is_open_at said the market was shut."""
    cal = _cal()
    d = date(2026, 11, 9)
    assert cal.special_session(d) is not None
    assert cal.is_trading_day(d) is False
    assert d not in cal.sessions_between(date(2026, 11, 2), date(2026, 11, 13))


def test_exchange_mismatch_between_file_and_caller_is_refused(tmp_path):
    p = tmp_path / "nse.json"
    p.write_text(json.dumps({"exchange": "NSE", "years": [2026], "holidays": []}),
                 encoding="utf-8")
    with pytest.raises(CalendarError, match="declares exchange"):
        TradingCalendar.from_file(p, exchange="BSE")


def test_a_failed_load_leaves_no_partial_state():
    """REGRESSION: holidays parsed before the bad entry were already applied."""
    cal = TradingCalendar()
    with pytest.raises(CalendarError):
        cal.load({"years": [2026], "holidays": [
            {"date": "2026-01-26", "name": "good"},
            {"date": "not-a-date", "name": "bad"},
        ]})
    assert cal.loaded_years == []
    cal.load({"years": [2026], "holidays": [{"date": "2026-01-26", "name": "good"}]})
    assert cal.is_holiday(date(2026, 1, 26))


def test_holidays_outside_the_declared_years_are_refused():
    """They would be stored and then permanently unreachable."""
    cal = TradingCalendar()
    with pytest.raises(CalendarError, match="outside the declared years"):
        cal.load({"years": [2026], "holidays": [{"date": "2027-01-26", "name": "x"}]})


def test_translate_refuses_a_non_dialect_attribute():
    """REGRESSION: unbounded getattr returned bound methods and classes, typed
    as str, which then flowed downstream."""
    from desk.marketdata.symbols import SymbolError, translate
    for bad in ("parse", "__class__", "nope"):
        with pytest.raises(SymbolError, match="unknown dialect"):
            translate("RELIANCE", bad)


def test_a_bad_ticker_base_is_not_blamed_on_the_exchange():
    """REGRESSION: SymbolError subclasses ValueError, so one try around both
    resolutions reported a bad base as 'unknown exchange'."""
    from desk.marketdata.symbols import Symbol, SymbolError
    with pytest.raises(SymbolError, match="implausible ticker base"):
        Symbol.parse("NSE:RELIANCE INDUSTRIES")
    with pytest.raises(SymbolError, match="unknown exchange"):
        Symbol.parse("XYZ:RELIANCE")


def test_report_is_human_readable():
    df = _bars([100, 100, 100, 100, 100, 100])
    text = q.check(df, symbol="X.NS").report()
    assert "X.NS" in text and "stale_quote" in text


def test_check_panel_reports_rows_with_a_missing_symbol_rather_than_dropping():
    """REGRESSION: groupby(col) defaults to dropna=True, so rows whose symbol
    is NaN vanished with no report, no issue, and nothing in checks_skipped -
    in the module whose stated job is that nothing passes unexamined."""
    a = _bars([100, 101, 102], start="2026-01-05"); a["symbol"] = "A.NS"
    b = _bars([200, 201, 202], start="2026-01-05"); b["symbol"] = float("nan")
    reps = q.check_panel(pd.concat([a, b]))
    assert "A.NS" in reps and reps["A.NS"].usable
    assert "<missing symbol>" in reps
    assert not reps["<missing symbol>"].usable
    assert reps["<missing symbol>"].bars == 3


# ===========================================================================
# Regressions from the 2026-09-13 second-pass re-verification
# ===========================================================================

def test_load_converts_malformed_shapes_to_calendar_error_not_a_raw_exception():
    """REGRESSION: five distinct malformed-input shapes each raised a raw
    KeyError/TypeError/ValueError/AttributeError instead of CalendarError,
    which meant /health and /calendar/{day} crashed with an unhandled 500
    instead of the module's own designed 503 fail-closed behaviour."""
    cases = [
        {"years": [2026], "holidays": [{"name": "no date key"}]},   # KeyError
        {"years": ["abc"], "holidays": []},                          # ValueError
        {"years": [2026], "holidays": {"a": 1}},                     # not a list
        {"years": [2026], "holidays": [],
         "special_sessions": [{"date": "2026-11-08", "name": "x", "end": "19:15"}]},
    ]
    for payload in cases:
        cal = TradingCalendar()
        with pytest.raises(CalendarError):
            cal.load(payload)


def test_load_rejects_a_non_dict_payload():
    cal = TradingCalendar()
    with pytest.raises(CalendarError, match="must be a JSON object"):
        cal.load(["not", "a", "dict"])


def test_from_file_converts_invalid_json_to_calendar_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json...", encoding="utf-8")
    with pytest.raises(CalendarError, match="not valid JSON"):
        TradingCalendar.from_file(p)


def test_from_file_rejects_a_non_dict_top_level(tmp_path):
    p = tmp_path / "list.json"
    p.write_text('["a", "b"]', encoding="utf-8")
    with pytest.raises(CalendarError, match="must contain a JSON object"):
        TradingCalendar.from_file(p)


def test_ambiguous_session_time_with_no_meridiem_is_refused():
    """REGRESSION: the docstring claimed an hour with no AM/PM in a context
    that clearly needs one is refused rather than defaulting to a 24-hour
    reading nobody wrote - but 1-12 with no meridiem was accepted literally,
    silently misreading an evening Muhurat session as a morning one."""
    from desk.marketdata.sources.nse import _session_window
    assert _session_window({"evening_session": "6:15 to 7:15"}) is None
    # Still correct: an explicit meridiem, or an hour that is unambiguous
    # even without one (0, or 13-23), continues to work.
    assert _session_window({"evening_session": "6:15 PM to 7:15 PM"}) == ("18:15", "19:15")
    assert _session_window({"evening_session": "18:15 to 19:15"}) == ("18:15", "19:15")
