"""The historical accumulation layer: many one-day snapshots, read as history.

`desk/marketdata/refresh.py bhavcopy` writes ONE FILE PER TRADING DAY into
configs/bhavcopy/<date>.parquet, each holding the whole exchange for that day.
That layout is deliberate and stays: a bad fetch on one day cannot corrupt any
other day, and re-fetching a single day is a single file replace. But it is
the wrong SHAPE for everything downstream - Stage 1 wants one symbol's last
60 bars, not one day's 3,485 rows. This module is the transposition, and the
only place in the project that reads across days.

THREE RULES IT ENFORCES, each of which exists because the obvious shortcut is
wrong:

1. POINT-IN-TIME. `history()` takes a mandatory `as_of` and never OPENS a file
   dated after it. Not "reads and filters" - never opens. A filter is a line
   of code someone can delete; a file that was never listed cannot leak into a
   backtest by accident.

2. A MISSING DAY IS NOT AN ABSENT DAY. If the calendar says 2026-09-10 was a
   session and no snapshot exists, that is a HOLE in the data, and a 20-day
   average computed over it is quietly wrong. `Coverage.missing` names those
   days. If no calendar is supplied the check cannot run, and `Coverage`
   says so via `calendar_checked=False` rather than implying a clean bill -
   the same third-state discipline as `Sizing.checks_skipped`.

3. PRICES COME BACK RAW. Bhavcopy is unadjusted. Across a 1:2 split the raw
   close halves, and every return, moving average and volatility measure over
   that window is garbage. Adjustment is `desk.marketdata.corporate_actions`'
   job, applied by the caller WITH ITS OWN `as_of` so a split announced after
   the simulated date cannot reach back. This module will not do it silently,
   because doing it silently means the caller never learns whether it happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from desk.marketdata.calendar_in import CalendarError, TradingCalendar

__all__ = ["BarStore", "Coverage", "History", "StoreError"]

SUFFIX = ".parquet"

#: Columns every snapshot is expected to carry. `history()` refuses a file
#: missing any of them rather than returning a frame with a silently absent
#: column, because `df["close"]` failing loudly beats `df.get("close")`
#: returning None three layers downstream.
REQUIRED_COLUMNS = frozenset({"symbol", "date", "open", "high", "low",
                              "close", "volume"})


class StoreError(Exception):
    """The store cannot answer honestly. Never raised for 'no data' - that is
    an empty History with an explicit Coverage, which is an answer."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """What the returned history actually covers, as opposed to what was asked
    for. Read this before trusting any window-based calculation over the
    frame: a 50-bar average over 31 bars is not a 50-bar average."""

    start: date | None
    end: date | None
    days_loaded: int
    sessions_expected: int | None
    """Trading sessions the calendar says exist in [start, end]. None when no
    calendar was supplied - meaning UNKNOWN, not zero."""
    missing: tuple[date, ...] = ()
    """Trading sessions in range with no snapshot on disk. Always empty when
    `calendar_checked` is False, which is not the same as 'none missing'."""
    calendar_checked: bool = False
    truncated: bool = False
    """True when fewer days were available than were asked for."""

    @property
    def complete(self) -> bool:
        """Every expected session present. False when unverifiable, because
        'I could not check' must never read as 'it is fine'."""
        return self.calendar_checked and not self.missing and not self.truncated

    def describe(self) -> str:
        if self.days_loaded == 0:
            return "no snapshots in range"
        span = f"{self.days_loaded} sessions, {self.start} to {self.end}"
        if not self.calendar_checked:
            return f"{span} (gaps UNCHECKED - no calendar supplied)"
        if self.missing:
            shown = ", ".join(d.isoformat() for d in self.missing[:5])
            more = f" +{len(self.missing) - 5} more" if len(self.missing) > 5 else ""
            return f"{span}, {len(self.missing)} MISSING: {shown}{more}"
        if self.truncated:
            return f"{span} (all available, fewer than requested)"
        return f"{span}, complete"


@dataclass(slots=True)
class History:
    """Long-format bars plus an honest account of what is in them."""

    frame: pd.DataFrame
    """Columns: symbol, date, and whatever price/volume fields were kept.
    `date` holds datetime.date objects. Row order is NOT part of the
    contract - `wide`/`wide_many` reindex and `series` sorts, so nothing
    downstream depends on it and sorting 567k rows here would be pure cost."""
    coverage: Coverage

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def symbols(self) -> list[str]:
        if self.frame.empty:
            return []
        return sorted(self.frame["symbol"].unique().tolist())

    def series(self, symbol: str) -> pd.DataFrame:
        """One symbol's bars, indexed by date - the shape desk.indicators
        expects. Returns an EMPTY frame with the right columns for a symbol
        that never traded in range, rather than raising: 'this name has no
        bars' is a normal scanner outcome, not an error."""
        if self.frame.empty:
            return self.frame.copy()
        one = self.frame[self.frame["symbol"] == symbol]
        return one.drop(columns=["symbol"]).set_index("date").sort_index()

    def wide(self, field_name: str) -> pd.DataFrame:
        """A date x symbol matrix of one field, for cross-sectional work
        (relative strength, breadth, correlation). Missing cells stay NaN -
        a symbol listed halfway through the window genuinely has no earlier
        price, and forward-filling one would invent history."""
        if self.frame.empty:
            return pd.DataFrame()
        if field_name not in self.frame.columns:
            raise StoreError(
                f"{field_name!r} is not in this history (have: "
                f"{sorted(c for c in self.frame.columns if c != 'symbol')})"
            )
        return self.wide_many([field_name])[field_name]

    def wide_many(self, field_names: list[str]) -> dict[str, pd.DataFrame]:
        """Several fields' matrices, built from ONE index factorisation.

        `pivot` rebuilds the (date, symbol) MultiIndex from scratch on every
        call, and Stage 1 wants five fields. Measured on 2,637 symbols x 215
        sessions: five separate `pivot` calls cost 2.7s, one `set_index` +
        `unstack` costs 0.85s for byte-identical output.
        """
        if self.frame.empty:
            return {f: pd.DataFrame() for f in field_names}
        missing = [f for f in field_names if f not in self.frame.columns]
        if missing:
            raise StoreError(
                f"{missing[0]!r} is not in this history (have: "
                f"{sorted(c for c in self.frame.columns if c != 'symbol')})"
            )
        wide = (self.frame.set_index(["date", "symbol"])[list(field_names)]
                .unstack("symbol").sort_index())
        return {f: wide[f] for f in field_names}


class BarStore:
    """Reads and writes the one-file-per-day snapshot directory.

    >>> store = BarStore(Path("configs/bhavcopy"))
    >>> h = store.history(as_of=date(2026, 9, 11), lookback=60)
    >>> h.coverage.describe()
    """

    def __init__(self, root: str | Path,
                 calendar: TradingCalendar | None = None) -> None:
        self.root = Path(root)
        self.calendar = calendar

    # ------------------------------------------------------------- layout ---

    def _path(self, day: date) -> Path:
        return self.root / f"{day.isoformat()}{SUFFIX}"

    def available_days(self) -> list[date]:
        """Every day with a snapshot on disk, ascending.

        A file whose name is not a date is SKIPPED silently rather than
        raising - a stray .parquet.tmp from an interrupted fetch, or an
        editor backup, must not take down a read of otherwise-good data.
        """
        if not self.root.is_dir():
            return []
        days = []
        for p in self.root.glob(f"*{SUFFIX}"):
            try:
                days.append(date.fromisoformat(p.stem))
            except ValueError:
                continue
        return sorted(days)

    def has(self, day: date) -> bool:
        return self._path(day).is_file()

    def write_day(self, day: date, frame: pd.DataFrame) -> Path:
        """Write one day's snapshot atomically (temp file then replace), so an
        interrupted write cannot leave a half-file that a later read treats as
        a real session. Mirrors what refresh.py does."""
        if frame.empty:
            raise StoreError(f"refusing to write an empty snapshot for {day}")
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise StoreError(
                f"snapshot for {day} is missing required columns: "
                f"{sorted(missing)}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        out = self._path(day)
        tmp = out.with_suffix(f"{SUFFIX}.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(out)
        return out

    def load_day(self, day: date, *,
                 columns: list[str] | None = None) -> pd.DataFrame:
        """One day's snapshot. Raises StoreError if it was never fetched -
        asking for a specific day you do not have is a caller bug, unlike
        `history()` where a gap is data to be reported."""
        p = self._path(day)
        if not p.is_file():
            raise StoreError(
                f"no snapshot for {day} in {self.root} - fetch it with "
                f"`python -m desk.marketdata.refresh bhavcopy --date {day}`"
            )
        return self._read(p, day, columns)

    def _read(self, path: Path, day: date,
              columns: list[str] | None) -> pd.DataFrame:
        cols = None
        if columns is not None:
            cols = sorted(set(columns) | {"symbol", "date"})
        try:
            df = pd.read_parquet(path, columns=cols)
        except Exception as exc:                       # corrupt / truncated
            raise StoreError(f"cannot read {path}: {exc}") from exc

        missing = REQUIRED_COLUMNS - set(df.columns)
        if columns is None and missing:
            raise StoreError(
                f"{path} is missing required columns {sorted(missing)} - it "
                f"was not written by parse_bhavcopy, or the format changed"
            )
        # Column projection is pushed into the parquet read as a sorted set,
        # so put the identity columns back in front - an empty result and a
        # populated one must not differ in column order.
        lead = [c for c in ("symbol", "date") if c in df.columns]
        df = df[lead + [c for c in df.columns if c not in lead]]
        df["date"] = _as_dates(df["date"])

        # A file named 2026-09-11 holding 2026-09-10's rows would silently
        # shift every bar by a session. parse_bhavcopy already guards the
        # fetch side; this guards the file on disk, which is what actually
        # gets read a year later.
        wrong = df["date"] != day
        if wrong.any():
            offenders = sorted({d.isoformat() for d in df.loc[wrong, "date"]})[:3]
            raise StoreError(
                f"{path} is named {day} but contains rows dated "
                f"{', '.join(offenders)} - refusing to use it"
            )
        return df

    # ------------------------------------------------------------ history ---

    def history(
        self,
        *,
        as_of: date,
        lookback: int | None = None,
        start: date | None = None,
        symbols: list[str] | set[str] | None = None,
        columns: list[str] | None = None,
    ) -> History:
        """Bars up to and including `as_of`, never after it.

        Give EITHER `lookback` (that many of the most recent available
        sessions, ending at as_of) OR `start` (an explicit earliest date).
        `lookback` counts SNAPSHOTS ON DISK, not calendar sessions - so a
        missing fetch quietly widens the real window, which is exactly why
        `Coverage.missing` exists and why the calendar check matters.

        `symbols` filters after load. `columns` is a projection pushed down
        into the parquet read, so asking for just close+volume across 250
        days reads roughly a fifth of the bytes.
        """
        if lookback is not None and start is not None:
            raise StoreError("pass lookback or start, not both")
        if lookback is not None and lookback < 1:
            raise StoreError(f"lookback must be >= 1, got {lookback}")

        # The point-in-time guard. Everything after as_of is dropped HERE,
        # before any file is opened.
        days = [d for d in self.available_days() if d <= as_of]

        truncated = False
        if start is not None:
            days = [d for d in days if d >= start]
        elif lookback is not None:
            truncated = len(days) < lookback
            days = days[-lookback:]

        if not days:
            return History(
                frame=_empty_frame(columns),
                coverage=Coverage(start=None, end=None, days_loaded=0,
                                  sessions_expected=None, truncated=truncated,
                                  calendar_checked=False),
            )

        frame = self._read_many(days, columns, symbols)
        if not frame.empty:
            # Deliberately NOT sorted: measured at 0.45-0.63s for 567k rows,
            # and no consumer needs it - `wide_many` unstacks (which sorts its
            # own index) and `series` calls sort_index itself.
            frame = frame.reset_index(drop=True)

        cov = self._coverage(days[0], days[-1], days, truncated=truncated)
        return History(frame=frame, coverage=cov)

    def _read_many(self, days: list[date], columns: list[str] | None,
                   symbols: list[str] | set[str] | None) -> pd.DataFrame:
        """One batched read across many day files.

        A read-one-file-per-day loop is the obvious implementation and it is
        four times slower: measured on 2,637 symbols x 215 sessions, the loop
        took 4.95s of which only 1.87s was actually parquet - the rest was
        per-file Python overhead repeated 215 times. A single pyarrow dataset
        read of the same data, with the column projection and the symbol
        predicate both pushed down, takes about 1s.

        Nothing about the safety guarantees changes. The file LIST is already
        point-in-time filtered by the caller, so no file dated after `as_of`
        is in `days` and none is opened. The filename-versus-contents check
        still runs on every file, from the parquet footer's own min/max
        statistics - metadata only, no column data read.
        """
        import pyarrow.compute as pc
        import pyarrow.dataset as pads

        paths = [self._path(d) for d in days]
        needs_scan = [(d, p) for d, p in zip(days, paths)
                      if not self._day_verified_from_footer(p, d)]
        for d, path in needs_scan:
            # No usable footer statistics (an older or unusual writer), so
            # this one file is read the slow way rather than trusted.
            self._read(path, d, columns)

        cols = None
        if columns is not None:
            cols = sorted(set(columns) | {"symbol", "date"})

        try:
            dataset = pads.dataset([str(p) for p in paths], format="parquet")
            filt = (pc.field("symbol").isin(list(symbols))
                    if symbols is not None else None)
            table = dataset.to_table(columns=cols, filter=filt)
        except Exception as exc:
            raise StoreError(f"cannot read the snapshot set: {exc}") from exc

        frame = table.to_pandas()
        if frame.empty:
            return _empty_frame(columns)

        missing = REQUIRED_COLUMNS - set(frame.columns)
        if columns is None and missing:
            raise StoreError(
                f"snapshots are missing required columns {sorted(missing)} - "
                f"they were not written by parse_bhavcopy, or the format "
                f"changed"
            )
        lead = [c for c in ("symbol", "date") if c in frame.columns]
        frame = frame[lead + [c for c in frame.columns if c not in lead]]
        frame["date"] = _as_dates(frame["date"])
        return frame

    def _day_verified_from_footer(self, path: Path, day: date) -> bool:
        """True when the footer PROVES every row in this file is dated `day`.

        A file named 2026-09-11 holding 2026-09-10's rows would shift every
        bar by a session and look entirely plausible. parse_bhavcopy guards
        the fetch side; this guards the file on disk, which is what actually
        gets read a year later.

        Returns False (meaning "could not prove it, check the slow way")
        rather than raising when statistics are absent - but RAISES when the
        statistics are present and disagree, because that is proof of a bad
        file, not an inability to check.
        """
        import pyarrow.parquet as pq

        if not path.is_file():
            raise StoreError(
                f"no snapshot for {day} in {self.root} - fetch it with "
                f"`python -m desk.marketdata.refresh bhavcopy --date {day}`"
            )
        try:
            meta = pq.ParquetFile(path).metadata
        except Exception as exc:
            raise StoreError(f"cannot read {path}: {exc}") from exc
        if meta.num_rows == 0:
            return True

        try:
            idx = [meta.schema.column(i).name
                   for i in range(meta.num_columns)].index("date")
        except ValueError:
            return False

        for g in range(meta.num_row_groups):
            stats = meta.row_group(g).column(idx).statistics
            if stats is None or not stats.has_min_max:
                return False
            lo, hi = _one_date(stats.min), _one_date(stats.max)
            if lo is None or hi is None:
                return False
            if lo != day or hi != day:
                bad = lo if lo != day else hi
                raise StoreError(
                    f"{path} is named {day} but contains rows dated "
                    f"{bad.isoformat()} - refusing to use it"
                )
        return True

    def _coverage(self, start: date, end: date, loaded: list[date], *,
                  truncated: bool) -> Coverage:
        """Compare what is on disk against what the calendar says should be."""
        if self.calendar is None:
            return Coverage(start=start, end=end, days_loaded=len(loaded),
                            sessions_expected=None, calendar_checked=False,
                            truncated=truncated)
        try:
            sessions = self.calendar.sessions_between(start, end)
        except CalendarError:
            # The year isn't loaded. The calendar refuses to guess, and so do
            # we: report UNCHECKED rather than downgrading to weekends-only.
            return Coverage(start=start, end=end, days_loaded=len(loaded),
                            sessions_expected=None, calendar_checked=False,
                            truncated=truncated)
        have = set(loaded)
        missing = tuple(d for d in sessions if d not in have)
        return Coverage(start=start, end=end, days_loaded=len(loaded),
                        sessions_expected=len(sessions), missing=missing,
                        calendar_checked=True, truncated=truncated)


def _one_date(value) -> date | None:
    """Coerce a parquet statistic to a plain `date`, or None if it is not a
    date at all (a string-typed date column, say). None means "cannot
    verify", never "verified fine"."""
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _as_dates(col: pd.Series) -> pd.Series:
    """Normalise a date column to datetime.date objects.

    Parquet round-trips a python `date` column as object dtype, but a frame
    built elsewhere may hold strings or datetime64. All three must compare
    equal to a `datetime.date` or the file-name guard below fires on good
    data.
    """
    if col.empty:
        return col
    first = col.iloc[0]
    if isinstance(first, date) and not isinstance(first, pd.Timestamp):
        return col
    return pd.to_datetime(col).dt.date


#: The order parse_bhavcopy emits, so an empty result has the same column
#: order as a populated one - code that does `frame.columns[2]` or writes the
#: frame straight to CSV must not behave differently on an empty day.
_DEFAULT_COLUMNS = ("symbol", "date", "open", "high", "low", "close", "volume")


def _empty_frame(columns: list[str] | None) -> pd.DataFrame:
    if columns is None:
        cols = list(_DEFAULT_COLUMNS)
    else:
        extra = [c for c in columns if c not in ("symbol", "date")]
        cols = ["symbol", "date", *extra]
    return pd.DataFrame({
        c: pd.Series(dtype="object" if c in ("symbol", "date") else "float64")
        for c in cols
    })
