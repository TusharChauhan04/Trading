"""NSE source tests.

Every one of these runs against payloads **actually captured from NSE** and
committed under `fixtures/`. No network, no hand-written sample that agrees
with my idea of the format - the free-text subject lines here are the real ones
NSE has published over the last decade, in all their inconsistency.

The parser's contract is the thing under test: it must never silently drop an
action. A dropped bonus becomes an unexplained 50% gap, which a backtest reads
as a crash and a strategy reads as a signal.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

# desk is an installed package (pip install -e .) - no sys.path hack needed.

from desk.marketdata.calendar_in import TradingCalendar
from desk.marketdata.corporate_actions import ActionType
from desk.marketdata.sources.nse import (
    SourceError,
    Unparsed,
    parse_corporate_actions,
    parse_holiday_master,
    parse_nse_date,
    parse_subject,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fx(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _actions(symbol: str):
    return parse_corporate_actions(_fx(f"nse_ca_{symbol}.json"), symbol)


# ===========================================================================
# Dates
# ===========================================================================

def test_nse_date_formats_including_the_empty_marker():
    assert parse_nse_date("04-Sep-2018") == date(2018, 9, 4)
    assert parse_nse_date("  28-Oct-2024  ") == date(2024, 10, 28)
    assert parse_nse_date("-") is None          # NSE's "no date" marker
    assert parse_nse_date("") is None
    assert parse_nse_date("garbage") is None


# ===========================================================================
# The subject parser - the part that actually matters
# ===========================================================================

def test_both_real_split_spellings_give_the_same_factor():
    """NSE has used two formats over the years and the punctuation differs.

    Rs 10 -> Rs 2 means each share becomes five, so the factor is 5 either way.
    """
    modern = parse_subject(
        "HDFCBANK.NS", date(2011, 7, 14),
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share")
    legacy = parse_subject(
        "HDFCBANK.NS", date(2011, 7, 14), "Face Value Split From Rs.10/- To Rs.2/-")
    for a in (modern, legacy):
        assert a.type is ActionType.SPLIT
        assert a.factor == pytest.approx(5.0)


def test_bonus_uses_the_indian_convention():
    """1:1 means one free share per share held, so a holder ends with two."""
    one_for_one = parse_subject("X.NS", date(2024, 10, 28), "Bonus 1:1")
    assert one_for_one.type is ActionType.BONUS
    assert one_for_one.factor == pytest.approx(2.0)

    # ITC's real 2016 action: one free share for every TWO held -> 1.5, not 2.
    one_for_two = parse_subject("ITC.NS", date(2016, 7, 1), "Bonus 1:2")
    assert one_for_two.factor == pytest.approx(1.5)


def test_a_split_wins_over_a_dividend_in_a_combined_subject():
    """Order matters: the structural event dwarfs the payout, and adjusting for
    the dividend instead would leave the split entirely unapplied."""
    a = parse_subject(
        "X.NS", date(2020, 1, 1),
        "Face Value Split From Rs 10 To Rs 1 / Dividend - Rs 5 Per Share")
    assert a.type is ActionType.SPLIT
    assert a.factor == pytest.approx(10.0)


def test_dividend_spellings_seen_in_the_archive():
    for text, amount in [
        ("Dividend - Rs 6 Per Share", 6.0),
        ("Dividend -  Rs 11/- Per Share", 11.0),
        ("Dividend- Rs 5.15 Per Share", 5.15),
        ("Dividend-Rs.8.50 Per Share (Including Special Dividend Of Rs.2/- Per Share).", 8.50),
        ("Interim Dividend - Re 1 Per Share", 1.0),
        ("Annual General Meeting/Dividend - Rs 15 Per Share", 15.0),
        ("Dividend Rs 15 Per Sh", 15.0),          # truncated, and still real
    ]:
        a = parse_subject("X.NS", date(2024, 1, 1), text)
        assert a.type is ActionType.DIVIDEND, text
        assert a.amount == pytest.approx(amount), text


def test_price_moving_events_we_cannot_factor_are_flagged_not_dropped():
    """Rights, demergers and mergers all move the price and none reduces to a
    single multiplier. Silence here would leave an unexplained gap that reads
    as a crash."""
    for text in ["Rights 1:15 @ Premium Rs 1247", "Demerger",
                 "Scheme of Arrangement", "Capital Reduction"]:
        u = parse_subject("X.NS", date(2024, 1, 1), text)
        assert isinstance(u, Unparsed), text
        assert "cannot be reduced" in u.reason


def test_non_price_events_are_labelled_rather_than_ignored():
    for text in ["Annual General Meeting", "Buyback", "Buy Back", "Board Meeting"]:
        u = parse_subject("X.NS", date(2024, 1, 1), text)
        assert isinstance(u, Unparsed)
        assert u.reason == "not price-affecting"


def test_a_consolidation_is_refused_rather_than_guessed():
    """A face value that RISES is a reverse split. Real, rare, and not worth
    guessing the direction of."""
    u = parse_subject("X.NS", date(2024, 1, 1),
                      "Face Value Split From Rs 1 To Rs 10")
    assert isinstance(u, Unparsed)
    assert "consolidation" in u.reason


def test_parse_subject_never_returns_none():
    """The contract. An absence is something a caller can fail to notice; a
    value carrying its own reason is not."""
    for text in ["", "   ", "Something Nobody Has Ever Written Before", "???"]:
        assert parse_subject("X.NS", date(2024, 1, 1), text) is not None
    assert isinstance(parse_subject("X.NS", None, "Bonus 1:1"), Unparsed)


# ===========================================================================
# Whole payloads, from the real archive
# ===========================================================================

@pytest.mark.parametrize("symbol", ["RELIANCE", "INFY", "TCS", "ITC",
                                    "WIPRO", "HDFCBANK", "IRCTC"])
def test_every_record_is_either_parsed_or_explained(symbol):
    payload = _fx(f"nse_ca_{symbol}.json")
    actions, unparsed = parse_corporate_actions(payload, symbol)
    assert len(actions) + len(unparsed) == len(payload)
    assert all(u.reason for u in unparsed)
    # Nothing in the real archive should be wholly unreadable.
    assert not [u for u in unparsed if u.reason == "unrecognised subject"]


def test_known_real_corporate_actions_are_recovered_exactly():
    """Spot-checks against events that actually happened."""
    acts, _ = _actions("RELIANCE")
    bonuses = [a for a in acts if a.type is ActionType.BONUS]
    assert {a.ex_date for a in bonuses} >= {date(2017, 9, 7), date(2024, 10, 28)}
    assert all(a.factor == pytest.approx(2.0) for a in bonuses)

    acts, _ = _actions("IRCTC")
    splits = [a for a in acts if a.type is ActionType.SPLIT]
    assert [(s.ex_date, round(s.factor, 4)) for s in splits] == [(date(2021, 10, 28), 5.0)]

    acts, _ = _actions("ITC")
    bonus = [a for a in acts if a.type is ActionType.BONUS]
    assert bonus[0].factor == pytest.approx(1.5)     # 1:2, not 1:1


def test_actions_come_back_in_date_order():
    acts, _ = _actions("HDFCBANK")
    assert acts == sorted(acts, key=lambda a: a.ex_date)


def test_symbols_are_canonicalised():
    acts, _ = _actions("INFY")
    assert all(a.symbol.endswith(".NS") for a in acts)


def test_an_empty_archive_is_not_an_error():
    """TATAMOTORS genuinely returns no rows - an empty result, not a failure."""
    actions, unparsed = parse_corporate_actions(_fx("nse_ca_TATAMOTORS.json"),
                                                "TATAMOTORS")
    assert actions == [] and unparsed == []


# ===========================================================================
# Holiday master
# ===========================================================================

def test_holiday_master_parses_into_something_the_calendar_accepts():
    payload = parse_holiday_master(_fx("nse_holiday_master.json"))
    assert payload["exchange"] == "NSE"
    assert payload["years"] == [2026]
    assert len(payload["holidays"]) == 20

    cal = TradingCalendar()
    cal.load(payload)
    assert cal.is_trading_day(date(2026, 1, 26)) is False
    assert cal.holiday_name(date(2026, 1, 26)) == "Republic Day"
    assert cal.is_trading_day(date(2026, 9, 11)) is True


def test_muhurat_day_without_published_timings_is_surfaced():
    """NSE marks it with a trailing '*' and publishes hours by separate
    circular. Treating it as a plain holiday is safe; losing the fact that an
    evening session exists is not."""
    payload = parse_holiday_master(_fx("nse_holiday_master.json"))
    assert payload["_needs_session_timings"] == ["2026-11-08"]
    assert "Muhurat" in payload["_note"] or "timings" in payload["_note"]


def test_weekend_holidays_do_not_break_anything():
    """The master legitimately lists holidays that fall on weekends."""
    cal = TradingCalendar()
    cal.load(parse_holiday_master(_fx("nse_holiday_master.json")))
    assert cal.is_trading_day(date(2026, 2, 15)) is False     # a Sunday
    assert cal.session_count(date(2026, 1, 1), date(2026, 12, 31)) > 200


def test_a_missing_segment_is_an_error_naming_what_was_present():
    with pytest.raises(SourceError, match="present:"):
        parse_holiday_master({"FO": []}, segment="CM")


def test_the_shipped_config_matches_what_the_parser_produces():
    """configs/holidays_nse.json is generated, not hand-edited. If this fails,
    someone edited it by hand or the source format changed."""
    shipped = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "holidays_nse.json")
        .read_text(encoding="utf-8"))
    fresh = parse_holiday_master(_fx("nse_holiday_master.json"))
    assert shipped["holidays"] == fresh["holidays"]
    assert shipped["years"] == fresh["years"]


# ===========================================================================
# Provider registry
# ===========================================================================

def test_every_provider_declares_all_twelve_criteria():
    from desk.marketdata.providers import PROVIDERS
    required = ("india_coverage", "realtime", "historical_depth", "rate_limits",
                "cost", "reliability", "licence", "redistribution",
                "api_quality", "python_support", "websocket", "corporate_actions")
    for p in PROVIDERS:
        for field_name in required:
            assert getattr(p, field_name) is not None, f"{p.key}.{field_name}"


def test_unassessed_criteria_are_named_not_hidden_behind_a_verdict():
    """UNKNOWN must not read as passing. If a provider carries unknowns, the
    registry has to say which ones."""
    from desk.marketdata.providers import PROVIDERS, unassessed_criteria
    gaps = unassessed_criteria()
    for p in PROVIDERS:
        if p.unassessed:
            assert p.key in gaps
            assert set(gaps[p.key]) == set(p.unassessed)


def test_redistribution_is_assessed_for_every_provider():
    """The criterion that invalidates an architecture after it is built."""
    from desk.marketdata.providers import PROVIDERS, Redistribution
    for p in PROVIDERS:
        assert isinstance(p.redistribution, Redistribution)
    # Nothing currently in the registry may be redistributed.
    assert not [p for p in PROVIDERS if p.safe_to_redistribute]


def test_only_sources_actually_exercised_are_marked_verified():
    from desk.marketdata.providers import BY_KEY
    assert BY_KEY["nse_official"].verified is True      # fetched and parsed here
    assert BY_KEY["zerodha_kite"].verified is False     # no credentials
    assert "NOT SUBSCRIBED" in " ".join(BY_KEY["zerodha_kite"].blockers)


def test_provider_status_is_json_safe():
    import json
    from desk.marketdata.providers import provider_status
    json.dumps(provider_status())


# ===========================================================================
# Regressions from the 2026-09-12 correctness review
# ===========================================================================

def test_combined_split_and_bonus_are_multiplied_not_one_dropped():
    """REGRESSION: parse_subject returned on the first regex hit, so a subject
    naming both a split AND a bonus silently lost the bonus - 5x applied where
    10x was required, and the residual read as a genuine -50% day."""
    from desk.marketdata.sources.nse import parse_subject
    a = parse_subject(
        "X.NS", date(2024, 3, 5),
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- "
        "Per Share And Bonus 1:1")
    assert a.factor == pytest.approx(10.0)      # 5 (split) x 2 (bonus)


def test_multi_dividend_lines_are_summed_not_truncated_to_the_first():
    """REGRESSION: re.search took the first Rs-amount and discarded the rest.
    Measured against the real archive this understated seven real TCS/INFY/ITC
    dividends by up to 89%."""
    from desk.marketdata.sources.nse import parse_subject
    a = parse_subject("X.NS", date(2024, 1, 1),
                      "Interim Dividend - Rs 8 Per Share Special Dividend - "
                      "Rs 67 Per Share")
    assert a.amount == pytest.approx(75.0)


def test_an_already_inclusive_dividend_total_is_not_summed_again():
    """The counterpart to the fix above: a figure already stated as inclusive
    of its own breakdown must not be double-counted."""
    from desk.marketdata.sources.nse import parse_subject
    a = parse_subject(
        "X.NS", date(2016, 5, 30),
        "Dividend-Rs.8.50 Per Share (Including Special Dividend Of "
        "Rs.2/- Per Share).")
    assert a.amount == pytest.approx(8.50)      # NOT 10.50


def test_split_regex_cannot_borrow_an_unrelated_from_to_clause():
    """REGRESSION: unanchored 'split' + DOTALL '.*?' let the regex reach past
    the split clause into an unrelated sentence - a record-date change on its
    own line was read as a 5:1 split."""
    from desk.marketdata.sources.nse import Unparsed, parse_subject
    a = parse_subject(
        "X.NS", date(2024, 1, 1),
        "Face Value Split (Ratio Awaited)\nRecord Date Moved From 10 To 2 March")
    assert isinstance(a, Unparsed)

    b = parse_subject(
        "X.NS", date(2024, 1, 1),
        "Annual General Meeting / Splitting Of Share Certificates From Rs 10 "
        "To Rs 2")
    assert isinstance(b, Unparsed)


def test_bonus_regex_cannot_borrow_an_unrelated_ratio():
    """REGRESSION: the unbounded gap let 'Bonus' (describing bonus debentures)
    reach across a slash into a RIGHTS issue's ratio, turning a warn-worthy
    rights issue into a silently-applied bonus factor."""
    from desk.marketdata.sources.nse import Unparsed, parse_subject
    for text in ["Bonus Debentures / Rights 1:15 @ Premium Rs 1247",
                 "Scheme Of Amalgamation / Bonus Issue Pending / Rights 3:7"]:
        a = parse_subject("X.NS", date(2024, 1, 1), text)
        assert isinstance(a, Unparsed), text
        assert "cannot be reduced" in a.reason


def test_dividend_amount_accepts_thousands_separators():
    from desk.marketdata.sources.nse import parse_subject
    a = parse_subject("X.NS", date(2024, 1, 1), "Dividend - Rs 1,250 Per Share")
    assert a.amount == pytest.approx(1250.0)


@pytest.mark.parametrize("symbol,ex_date,expected", [
    ("TCS", date(2023, 1, 16), 75.0), ("TCS", date(2025, 1, 17), 76.0),
    ("TCS", date(2026, 1, 16), 57.0), ("TCS", date(2024, 1, 19), 27.0),
    ("INFY", date(2024, 5, 31), 28.0), ("INFY", date(2018, 6, 14), 30.5),
    ("ITC", date(2023, 5, 30), 9.5),
])
def test_real_archive_dividends_recovered_in_full(symbol, ex_date, expected):
    """Each of these was truncated before the fix - measured, not estimated,
    against the actual NSE payload."""
    acts, _ = _actions(symbol)
    match = [a for a in acts if a.ex_date == ex_date]
    assert match, f"{symbol} {ex_date} missing from parsed actions"
    assert match[0].amount == pytest.approx(expected)


def test_real_archive_total_unparsed_is_unchanged_by_the_regex_tightening():
    """The bounding/anchoring fixes must not turn genuine splits or bonuses
    into false negatives - re-assert the same totals as the original capture."""
    acts, un = _actions("RELIANCE")
    assert len([a for a in acts if a.type is ActionType.BONUS]) == 2
    acts, un = _actions("IRCTC")
    assert len([a for a in acts if a.type is ActionType.SPLIT]) == 1
    acts, un = _actions("ITC")
    assert len([a for a in acts if a.type is ActionType.BONUS]) == 1


def test_session_window_respects_am_pm():
    """REGRESSION: dropping the meridiem read '6:15 PM' as 06:15 - exactly
    opposite the real Muhurat window."""
    from desk.marketdata.sources.nse import _session_window
    row = {"evening_session": "6:15 PM to 7:15 PM", "morning_session": None}
    assert _session_window(row) == ("18:15", "19:15")

    row2 = {"evening_session": "6.15 - 7.15 PM"}      # meridiem stated once
    assert _session_window(row2) == ("18:15", "19:15")


def test_session_window_ignores_open_closed_placeholders():
    """Non-CM segments carry the literal strings 'Open'/'Closed' instead of a
    time - these must not be misread as a session window."""
    from desk.marketdata.sources.nse import _session_window
    assert _session_window({"evening_session": "Open", "morning_session": "Closed"}) is None


def test_parse_holiday_master_rejects_non_list_segment():
    from desk.marketdata.sources.nse import SourceError, parse_holiday_master
    with pytest.raises(SourceError, match="expected a list"):
        parse_holiday_master({"CM": {"a": 1}})


def test_parse_holiday_master_rejects_non_dict_payload():
    from desk.marketdata.sources.nse import SourceError, parse_holiday_master
    with pytest.raises(SourceError, match="expected a dict"):
        parse_holiday_master(["not", "a", "dict"])


def test_symbol_is_not_double_suffixed():
    """REGRESSION: symbol="ITC.NS" with a row lacking its own 'symbol' field
    produced 'ITC.NS.NS', matching nothing downstream."""
    from desk.marketdata.sources.nse import parse_corporate_actions
    actions, _ = parse_corporate_actions(
        [{"exDate": "05-Mar-2024", "subject": "Bonus 1:1"}], "ITC.NS")
    assert actions[0].symbol == "ITC.NS"


def test_review_worthy_filter_catches_unparseable_ex_dates():
    """REGRESSION: _REVIEW_WORTHY omitted 'no usable ex-date', the reason
    parse_subject returns whenever NSE's exDate was unparseable - checked
    BEFORE any split/bonus/dividend classification, so a genuine bonus whose
    date didn't parse was silently absent from every REVIEW line."""
    from desk.marketdata.refresh import _REVIEW_WORTHY
    from desk.marketdata.sources.nse import parse_subject
    u = parse_subject("X.NS", None, "Bonus 1:1")
    assert any(kw in u.reason for kw in _REVIEW_WORTHY)


# ===========================================================================
# Bhavcopy - the whole exchange, one file, one request per day
# ===========================================================================

def _bhavcopy_raw():
    return (FIXTURES / "nse_bhavcopy_20260911.csv").read_bytes()


def test_bhavcopy_parses_the_real_captured_archive():
    from desk.marketdata.sources.nse import parse_bhavcopy
    df = parse_bhavcopy(_bhavcopy_raw(), day=date(2026, 9, 11))
    assert len(df) == 3485
    assert set(df.columns) == {
        "symbol", "series", "date", "prev_close", "open", "high", "low",
        "close", "last", "volume", "turnover_lacs", "trades",
        "delivery_qty", "delivery_pct",
    }
    assert (df["date"] == date(2026, 9, 11)).all()


def test_bhavcopy_symbols_are_canonicalised_and_series_preserved():
    from desk.marketdata.sources.nse import parse_bhavcopy
    df = parse_bhavcopy(_bhavcopy_raw())
    reliance = df[df["symbol"] == "RELIANCE.NS"]
    assert len(reliance) == 1
    assert reliance.iloc[0]["series"] == "EQ"
    assert reliance.iloc[0]["close"] == pytest.approx(1257.50)
    assert reliance.iloc[0]["delivery_pct"] == pytest.approx(54.50)


def test_bhavcopy_missing_value_marker_becomes_nan_not_a_parse_failure():
    """REGRESSION: NSE writes '-' for delivery figures it has nothing to
    report (measured: 297 of 3485 rows, concentrated in BE/BZ/SM/ST/GB where
    delivery isn't tracked the same way as EQ) - the same convention
    parse_nse_date already handles for exDate elsewhere in this module."""
    from desk.marketdata.sources.nse import parse_bhavcopy
    df = parse_bhavcopy(_bhavcopy_raw())
    assert df["delivery_qty"].isna().sum() == 297
    assert df["delivery_pct"].isna().sum() == 297
    # And a normal EQ row must NOT be NaN.
    assert not df[df["symbol"] == "RELIANCE.NS"]["delivery_pct"].isna().any()


def test_bhavcopy_equity_only_filters_to_the_mainstream_universe():
    from desk.marketdata.sources.nse import bhavcopy_equity_only, parse_bhavcopy
    df = parse_bhavcopy(_bhavcopy_raw())
    eq = bhavcopy_equity_only(df)
    assert len(eq) == 2637
    assert set(eq["series"].unique()) == {"EQ"}
    assert eq["symbol"].nunique() == 2637          # one row per symbol in EQ


def test_bhavcopy_series_distribution_matches_the_real_archive():
    """Pinned to the actual measured distribution so a future regex/parsing
    change that silently drops or reclassifies a series is caught."""
    from desk.marketdata.sources.nse import parse_bhavcopy
    df = parse_bhavcopy(_bhavcopy_raw())
    counts = df["series"].value_counts().to_dict()
    assert counts["EQ"] == 2637
    assert counts["SM"] == 373
    assert counts["BE"] == 248
    assert sum(counts.values()) == 3485


def test_bhavcopy_refuses_a_file_for_the_wrong_day():
    """REGRESSION guard: the wrong day's file accepted silently would mean a
    scanner reads yesterday's prices as today's, with no way to notice."""
    from desk.marketdata.sources.nse import SourceError, parse_bhavcopy
    with pytest.raises(SourceError, match="expected bhavcopy for"):
        parse_bhavcopy(_bhavcopy_raw(), day=date(2020, 1, 1))


def test_bhavcopy_missing_columns_are_reported_not_a_raw_keyerror():
    from desk.marketdata.sources.nse import SourceError, parse_bhavcopy
    truncated = b"SYMBOL,SERIES,DATE1\nRELIANCE,EQ,11-Sep-2026\n"
    with pytest.raises(SourceError, match="missing expected columns"):
        parse_bhavcopy(truncated)


def test_bhavcopy_empty_file_is_reported():
    from desk.marketdata.sources.nse import SourceError, parse_bhavcopy
    header_only = (b"SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, "
                  b"LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, "
                  b"TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n")
    with pytest.raises(SourceError, match="no rows"):
        parse_bhavcopy(header_only)
