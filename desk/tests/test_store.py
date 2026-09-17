"""desk/store/ - the accumulation layer. Every test here targets a way the
obvious implementation would silently return wrong history rather than fail.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from desk.marketdata.calendar_in import TradingCalendar
from desk.store import BarStore, StoreError

SESSIONS = [date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9),
            date(2026, 9, 10), date(2026, 9, 11)]      # Mon-Fri, no holiday


def _snapshot(day: date, symbols=("AAA.NS", "BBB.NS"), close=100.0):
    return pd.DataFrame({
        "symbol": list(symbols),
        "date": [day] * len(symbols),
        "open": [close] * len(symbols),
        "high": [close * 1.02] * len(symbols),
        "low": [close * 0.98] * len(symbols),
        "close": [close + i for i in range(len(symbols))],
        "volume": [1000 + i for i in range(len(symbols))],
    })


@pytest.fixture
def store(tmp_path) -> BarStore:
    s = BarStore(tmp_path / "bhavcopy")
    for i, day in enumerate(SESSIONS):
        s.write_day(day, _snapshot(day, close=100.0 + i))
    return s


@pytest.fixture
def calendar() -> TradingCalendar:
    cal = TradingCalendar(exchange="NSE")
    cal.load({"exchange": "NSE", "years": [2026], "holidays": [],
              "special_sessions": []})
    return cal


# ===========================================================================
# point-in-time
# ===========================================================================

def test_history_never_returns_a_bar_dated_after_as_of(store):
    """The whole reason this layer exists. A backtest simulating 2026-09-09
    must not be able to see Thursday's or Friday's close."""
    h = store.history(as_of=date(2026, 9, 9), lookback=50)
    assert h.frame["date"].max() == date(2026, 9, 9)
    assert date(2026, 9, 10) not in set(h.frame["date"])


def test_files_after_as_of_are_never_opened(store, monkeypatch):
    """Stronger than the assertion above: a future day must not be READ and
    then filtered, because a filter is one deleted line away from a leak.

    Asserted against the EXACT day list handed to the batched read, and
    separately against every file whose footer is opened for the
    name-versus-contents check - so neither path can start touching a future
    file without failing here.
    """
    batched: list[list[date]] = []
    footers: list[date] = []
    original_many = BarStore._read_many
    original_verify = BarStore._day_verified_from_footer

    def spy_many(self, days, columns, symbols):
        batched.append(list(days))
        return original_many(self, days, columns, symbols)

    def spy_verify(self, path, day):
        footers.append(day)
        return original_verify(self, path, day)

    monkeypatch.setattr(BarStore, "_read_many", spy_many)
    monkeypatch.setattr(BarStore, "_day_verified_from_footer", spy_verify)
    store.history(as_of=date(2026, 9, 9), lookback=50)

    assert batched and max(batched[0]) == date(2026, 9, 9)
    assert date(2026, 9, 10) not in batched[0]
    assert footers and max(footers) == date(2026, 9, 9)


def test_as_of_before_every_snapshot_gives_an_empty_history_not_an_error(store):
    """No data is an ANSWER, with a coverage that says so - not an exception.
    A scanner run on a symbol's first listed day takes this path."""
    h = store.history(as_of=date(2020, 1, 1), lookback=10)
    assert h.frame.empty
    assert h.coverage.days_loaded == 0
    assert h.coverage.complete is False
    assert list(h.frame.columns)[:2] == ["symbol", "date"]


# ===========================================================================
# coverage - a gap must never read as clean data
# ===========================================================================

def test_a_missing_session_is_reported_not_silently_skipped(tmp_path, calendar):
    """The failure this guards: fetch Mon, Tue, Thu, Fri; ask for a week;
    get four bars that LOOK contiguous. A 5-day average over them is wrong
    and nothing says so."""
    s = BarStore(tmp_path, calendar=calendar)
    for day in SESSIONS:
        if day == date(2026, 9, 9):
            continue                      # Wednesday never fetched
        s.write_day(day, _snapshot(day))

    h = s.history(as_of=date(2026, 9, 11), start=date(2026, 9, 7))
    assert h.coverage.missing == (date(2026, 9, 9),)
    assert h.coverage.sessions_expected == 5
    assert h.coverage.days_loaded == 4
    assert h.coverage.complete is False
    assert "MISSING" in h.coverage.describe()


def test_complete_is_false_when_the_gap_check_could_not_run(store):
    """No calendar supplied -> gaps are UNCHECKED. 'I could not check' must
    never be reported as 'it is fine' - the same third state as
    Sizing.checks_skipped."""
    h = store.history(as_of=date(2026, 9, 11), start=date(2026, 9, 7))
    assert h.coverage.calendar_checked is False
    assert h.coverage.sessions_expected is None
    assert h.coverage.missing == ()
    assert h.coverage.complete is False           # NOT True
    assert "UNCHECKED" in h.coverage.describe()


def test_an_unloaded_calendar_year_degrades_to_unchecked_not_to_weekends_only(
        tmp_path, calendar):
    """calendar_in refuses to guess holidays for a year it has no data for.
    The store must inherit that refusal rather than falling back to a
    weekends-only assumption, which would invent five clean sessions a week
    for a year full of Indian holidays."""
    s = BarStore(tmp_path, calendar=calendar)        # only 2026 loaded
    day = date(2025, 6, 16)
    s.write_day(day, _snapshot(day))
    h = s.history(as_of=day, lookback=1)
    assert h.coverage.calendar_checked is False
    assert h.coverage.sessions_expected is None


def test_truncated_marks_a_window_shorter_than_requested(store, calendar):
    """Asking for 60 bars and getting 5 must be visible, or a 50-period
    indicator quietly returns all-NaN and the caller blames the indicator."""
    store.calendar = calendar
    h = store.history(as_of=date(2026, 9, 11), lookback=60)
    assert h.coverage.days_loaded == 5
    assert h.coverage.truncated is True
    assert h.coverage.complete is False


def test_a_complete_window_reports_complete(store, calendar):
    store.calendar = calendar
    h = store.history(as_of=date(2026, 9, 11), lookback=5)
    assert h.coverage.complete is True
    assert h.coverage.describe().endswith("complete")


# ===========================================================================
# reading
# ===========================================================================

def test_lookback_counts_snapshots_ending_at_as_of(store):
    h = store.history(as_of=date(2026, 9, 10), lookback=2)
    assert sorted(set(h.frame["date"])) == [date(2026, 9, 9), date(2026, 9, 10)]


def test_symbol_and_column_projection(store):
    h = store.history(as_of=date(2026, 9, 11), lookback=3,
                      symbols=["AAA.NS"], columns=["close"])
    assert h.symbols == ["AAA.NS"]
    assert list(h.frame.columns) == ["symbol", "date", "close"]


def test_series_is_date_indexed_and_sorted_for_the_indicator_layer(store):
    """desk.indicators functions take a date-indexed OHLCV frame. This is the
    handoff, so it is pinned."""
    from desk.indicators.trend import sma

    h = store.history(as_of=date(2026, 9, 11), lookback=5)
    s = h.series("AAA.NS")
    assert s.index.is_monotonic_increasing
    assert "symbol" not in s.columns
    assert sma(s["close"], 3).iloc[-1] == pytest.approx(103.0)   # 102,103,104


def test_wide_builds_a_date_by_symbol_matrix(store):
    h = store.history(as_of=date(2026, 9, 11), lookback=5)
    w = h.wide("close")
    assert list(w.columns) == ["AAA.NS", "BBB.NS"]
    assert len(w) == 5
    assert w.index.is_monotonic_increasing


def test_wide_names_the_available_fields_when_asked_for_a_missing_one(store):
    h = store.history(as_of=date(2026, 9, 11), lookback=2, columns=["close"])
    with pytest.raises(StoreError, match="close"):
        h.wide("turnover_lacs")


def test_series_of_an_unknown_symbol_is_empty_not_an_error(store):
    """A name that was not listed during the window is a normal scanner
    outcome, not a bug."""
    assert store.history(as_of=date(2026, 9, 11),
                         lookback=3).series("NOPE.NS").empty


# ===========================================================================
# integrity
# ===========================================================================

def test_a_file_whose_contents_disagree_with_its_name_is_refused(tmp_path):
    """A snapshot named 2026-09-11 holding Thursday's rows would shift every
    bar by one session and look completely plausible. parse_bhavcopy guards
    the fetch; this guards the file that is actually read a year later."""
    s = BarStore(tmp_path)
    mislabelled = _snapshot(date(2026, 9, 10))
    mislabelled.to_parquet(tmp_path / "2026-09-11.parquet", index=False)
    with pytest.raises(StoreError, match="contains rows dated"):
        s.load_day(date(2026, 9, 11))


def test_writing_an_incomplete_snapshot_is_refused(tmp_path):
    s = BarStore(tmp_path)
    bad = _snapshot(date(2026, 9, 11)).drop(columns=["volume"])
    with pytest.raises(StoreError, match="volume"):
        s.write_day(date(2026, 9, 11), bad)


def test_writes_are_atomic_and_leave_no_temp_file(tmp_path):
    s = BarStore(tmp_path)
    s.write_day(date(2026, 9, 11), _snapshot(date(2026, 9, 11)))
    assert list(tmp_path.glob("*.tmp")) == []
    assert (tmp_path / "2026-09-11.parquet").is_file()


def test_a_stray_non_date_file_does_not_break_the_listing(store):
    """An interrupted fetch leaves .parquet.tmp; an editor may leave a backup.
    Neither may take down a read of otherwise-good data."""
    (store.root / "notes.parquet").write_bytes(b"not parquet")
    assert store.available_days() == SESSIONS


def test_load_day_names_the_refresh_command_for_a_day_never_fetched(store):
    with pytest.raises(StoreError, match="refresh bhavcopy"):
        store.load_day(date(2026, 9, 14))


def test_lookback_and_start_together_are_refused(store):
    with pytest.raises(StoreError, match="not both"):
        store.history(as_of=date(2026, 9, 11), lookback=5,
                      start=date(2026, 9, 7))


def test_a_missing_store_directory_is_empty_not_an_error(tmp_path):
    """The very first run, before any bhavcopy has been fetched."""
    s = BarStore(tmp_path / "does-not-exist")
    assert s.available_days() == []
    assert s.history(as_of=date(2026, 9, 11), lookback=5).frame.empty


def test_string_and_datetime_date_columns_both_normalise(tmp_path):
    """Parquet round-trips python dates as object dtype, but a frame built
    elsewhere may hold strings or datetime64. All three must compare equal to
    a datetime.date or the file-name guard fires on good data."""
    s = BarStore(tmp_path)
    as_strings = _snapshot(date(2026, 9, 11))
    as_strings["date"] = "2026-09-11"
    as_strings.to_parquet(tmp_path / "2026-09-11.parquet", index=False)
    assert s.load_day(date(2026, 9, 11))["date"].iloc[0] == date(2026, 9, 11)


def test_a_mislabelled_file_is_caught_from_the_footer_without_reading_it(store):
    """The name-versus-contents guard moved to parquet footer statistics when
    the read was batched. It must still fire - and must fire from metadata
    alone, since the batched read no longer looks at rows per file."""
    import pandas as _pd

    wrong = _snapshot(date(2026, 9, 10))
    wrong.to_parquet(store.root / "2026-09-11.parquet", index=False)
    with pytest.raises(StoreError, match="contains rows dated"):
        store.history(as_of=date(2026, 9, 11), lookback=1)


def test_the_batched_read_returns_the_same_frame_as_a_per_file_read(store):
    """PARITY. The batched pyarrow path replaced a read-one-file-per-day loop
    for speed (4.95s -> ~1s at full scale). Speed is only worth having if the
    answer is identical, so the two are compared directly."""
    import pandas as _pd

    h = store.history(as_of=SESSIONS[-1], lookback=5)
    per_file = _pd.concat(
        [store._read(store._path(d), d, None) for d in SESSIONS],
        ignore_index=True,
    )
    a = h.frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    b = per_file.sort_values(["symbol", "date"]).reset_index(drop=True)
    _pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False)


# ===========================================================================
# One instrument per (date, symbol)
# ===========================================================================
#
# REGRESSION, and one that only appeared once 93 sessions were on disk. A
# base symbol can trade in more than one series on the same day: AARTISURF
# trades EQ and P1 (partly paid) together, M&MFIN trades EQ and N3. The
# bhavcopy parser keeps both, correctly - they are different instruments.
#
# wide_many unstacks on (date, symbol), so the duplicated index killed the
# whole scan with "Index contains duplicate entries, cannot reshape". That
# is the GOOD failure. The dangerous fix is drop_duplicates(), which keeps
# whichever row happens to come first - for a partly-paid line, a completely
# different price - and puts an artificial step in the series with nothing
# reported. Same corruption the unadjusted-corporate-action guard exists to
# prevent.

def _multi_series_day(tmp_path, day: date, rows):
    """rows: list of (symbol, series, close)."""
    import pandas as pd
    frame = pd.DataFrame([
        {"symbol": sym, "series": ser, "date": day, "open": close,
         "high": close, "low": close, "close": close, "volume": 1000}
        for sym, ser, close in rows
    ])
    frame.to_parquet(tmp_path / f"{day.isoformat()}.parquet", index=False)


def test_the_equity_series_wins_a_same_day_collision(tmp_path):
    from desk.store import BarStore
    day = date(2026, 9, 11)
    _multi_series_day(tmp_path, day, [
        ("AARTISURF.NS", "P1", 11.0),      # partly paid, listed FIRST
        ("AARTISURF.NS", "EQ", 550.0),     # the real cash line
        ("RELIANCE.NS", "EQ", 1257.5),
    ])
    h = BarStore(tmp_path).history(as_of=day, lookback=1,
                                   columns=["open", "high", "low", "close",
                                            "volume"])
    got = h.frame.set_index("symbol")["close"]
    assert got["AARTISURF.NS"] == 550.0, "took the partly-paid price"
    assert got["RELIANCE.NS"] == 1257.5


def test_a_collision_is_reported_never_silent(tmp_path):
    from desk.store import BarStore
    day = date(2026, 9, 11)
    _multi_series_day(tmp_path, day, [
        ("AARTISURF.NS", "EQ", 550.0),
        ("AARTISURF.NS", "P1", 11.0),
        ("RELIANCE.NS", "EQ", 1257.5),
    ])
    h = BarStore(tmp_path).history(as_of=day, lookback=1,
                                   columns=["close"])
    assert h.coverage.collapsed_series == {"AARTISURF.NS": 1}
    assert "collapsed" in h.coverage.describe()
    assert "AARTISURF.NS" in h.coverage.describe()


def test_a_collision_no_longer_breaks_the_reshape(tmp_path):
    """The symptom that exposed it."""
    from desk.store import BarStore
    day = date(2026, 9, 11)
    _multi_series_day(tmp_path, day, [
        ("AARTISURF.NS", "EQ", 550.0),
        ("AARTISURF.NS", "P1", 11.0),
    ])
    h = BarStore(tmp_path).history(as_of=day, lookback=1, columns=["close"])
    wide = h.wide_many(["close"])
    assert wide["close"].shape == (1, 1)
    assert wide["close"].iloc[0, 0] == 550.0


def test_a_symbol_with_no_equity_series_is_kept(tmp_path):
    """An SME name that trades only in SM has no EQ row. Dropping it would
    remove a tradeable instrument on a technicality."""
    from desk.store import BarStore
    day = date(2026, 9, 11)
    _multi_series_day(tmp_path, day, [("SMECO.NS", "SM", 42.0)])
    h = BarStore(tmp_path).history(as_of=day, lookback=1, columns=["close"])
    assert list(h.frame["symbol"]) == ["SMECO.NS"]
    assert h.coverage.collapsed_series == {}


def test_series_is_not_returned_unless_it_was_asked_for(tmp_path):
    from desk.store import BarStore
    day = date(2026, 9, 11)
    _multi_series_day(tmp_path, day, [("RELIANCE.NS", "EQ", 1257.5)])
    h = BarStore(tmp_path).history(as_of=day, lookback=1, columns=["close"])
    assert "series" not in h.frame.columns

    h2 = BarStore(tmp_path).history(as_of=day, lookback=1,
                                    columns=["close", "series"])
    assert "series" in h2.frame.columns


def test_no_collision_reports_nothing(tmp_path):
    from desk.store import BarStore
    day = date(2026, 9, 11)
    _multi_series_day(tmp_path, day, [("RELIANCE.NS", "EQ", 1257.5),
                                      ("TCS.NS", "EQ", 3000.0)])
    h = BarStore(tmp_path).history(as_of=day, lookback=1, columns=["close"])
    assert h.coverage.collapsed_series == {}
    assert "collapsed" not in h.coverage.describe()
