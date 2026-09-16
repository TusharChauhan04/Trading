"""The data-quality gate. Nothing reaches a backtest without passing here.

Bad Indian market data produces extremely convincing false results. An
unadjusted split looks like a crash; a stale quote looks like a consolidation;
a duplicated bar quietly doubles a day's weight in every average. All three
produce a backtest that is wrong in a way that reads as insight.

So this module reports rather than repairs. `check()` returns a report; only
`assert_clean()` raises. The distinction matters - repairing data silently is
how a pipeline develops opinions nobody reviewed.

Severity has a precise meaning here:
  FATAL - the series cannot be used. Results computed on it are meaningless.
  WARN  - usable, but a human should know. Often legitimate (a genuine limit-up).
  INFO  - observed and worth recording, not a problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from desk.marketdata.corporate_actions import CorporateAction, find_discontinuities

# Indian equities carry a 20% circuit band at the widest; index scrips are
# narrower. A move beyond this with no corporate action on file is not a market
# event, it is a data error.
CIRCUIT_LIMIT_PCT = 20.0

# A price that has not moved at all across this many sessions, on zero volume,
# is a feed that stopped updating rather than a share nobody traded.
STALE_RUN_LENGTH = 5


class Severity(str, Enum):
    FATAL = "fatal"
    WARN = "warn"
    INFO = "info"


class Check(str, Enum):
    """Named so a report can be filtered and so waivers are explicit."""
    SCHEMA = "schema"
    INDEX_ORDER = "index_order"
    DUPLICATE_BARS = "duplicate_bars"
    MISSING_BARS = "missing_bars"
    UNEXPECTED_BARS = "unexpected_bars"
    OHLC_COHERENCE = "ohlc_coherence"
    NON_POSITIVE_PRICE = "non_positive_price"
    NEGATIVE_VOLUME = "negative_volume"
    STALE_QUOTE = "stale_quote"
    ZERO_VOLUME = "zero_volume"
    EXTREME_RETURN = "extreme_return"
    CORPORATE_ACTION = "corporate_action"
    SYMBOL_MAPPING = "symbol_mapping"
    CROSS_SOURCE = "cross_source"


@dataclass(frozen=True, slots=True)
class Issue:
    check: Check
    severity: Severity
    message: str
    rows: int = 0
    sample: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        n = f" [{self.rows} rows]" if self.rows else ""
        return f"{self.severity.value.upper():5} {self.check.value}: {self.message}{n}"


@dataclass(slots=True)
class QualityReport:
    symbol: str
    bars: int
    issues: list[Issue] = field(default_factory=list)
    checks_run: list[Check] = field(default_factory=list)
    checks_skipped: dict[Check, str] = field(default_factory=dict)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    @property
    def fatal(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.FATAL]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARN]

    @property
    def usable(self) -> bool:
        return not self.fatal

    def summary(self) -> str:
        head = (f"{self.symbol}: {self.bars} bars, {len(self.fatal)} fatal, "
                f"{len(self.warnings)} warnings, "
                f"{len(self.checks_run)} checks run")
        if self.checks_skipped:
            head += f", {len(self.checks_skipped)} skipped"
        return head

    def report(self) -> str:
        lines = [self.summary()]
        lines += [f"  {i}" for i in self.issues]
        lines += [f"  SKIP  {k.value}: {v}" for k, v in self.checks_skipped.items()]
        return "\n".join(lines)


# ===========================================================================
# The checks
# ===========================================================================

REQUIRED_COLUMNS = ("open", "high", "low", "close")
SYMBOL_COLUMNS = ("symbol", "ticker", "instrument", "scrip")


class PanelError(ValueError):
    """A multi-symbol frame reached a single-symbol function."""


def _symbol_column(df: pd.DataFrame) -> str | None:
    for c in SYMBOL_COLUMNS:
        if c in df.columns:
            return c
    return None


def _reject_panel(df: pd.DataFrame, fn: str) -> None:
    col = _symbol_column(df)
    if col is None:
        return
    n = df[col].nunique(dropna=False)
    if n > 1:
        raise PanelError(
            f"{fn}() works on ONE instrument; got {n} in column {col!r}. "
            f"The seam between two shares reads as a legitimate price move, "
            f"so this would pass a frame containing fabricated returns. "
            f"Use check_panel(), or group by {col!r} first."
        )


def check(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    actions: list[CorporateAction] | None = None,
    expected_sessions: list[date] | None = None,
    reference: pd.DataFrame | None = None,
    reference_name: str = "reference",
    price_tolerance_pct: float = 1.0,
) -> QualityReport:
    """Run every applicable check. Never raises on bad data - that is
    `assert_clean`'s job. Raises only if `df` is not a usable frame at all.

    Optional inputs gate optional checks, and a check that could not run is
    recorded in `checks_skipped`. "Not checked" and "checked and passed" are
    different states.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")

    # Every check below assumes ONE instrument on a date index. Handed a
    # multi-symbol panel - the natural shape of a Parquet/DuckDB read - the
    # index looks monotonic and unduplicated, and the seam between two shares
    # reads as a legitimate price move. The gate would return usable=True on a
    # frame containing a fabricated several-thousand-percent return.
    #
    # Refusing loudly is the only safe answer, and it matches how the rest of
    # this package behaves: CalendarNotLoaded, SymbolError, assert_clean.
    _reject_panel(df, "check")

    rep = QualityReport(symbol=symbol or "<unnamed>", bars=len(df))

    # --- schema, before anything tries to read a column --------------------
    rep.checks_run.append(Check.SCHEMA)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rep.add(Issue(Check.SCHEMA, Severity.FATAL,
                      f"missing required columns: {', '.join(missing)}"))
        return rep                      # every later check would raise

    if df.empty:
        rep.add(Issue(Check.SCHEMA, Severity.FATAL, "series is empty"))
        return rep

    if not isinstance(df.index, pd.DatetimeIndex):
        rep.add(Issue(Check.SCHEMA, Severity.FATAL,
                      f"index is {type(df.index).__name__}, expected DatetimeIndex"))
        return rep

    # A duplicated column label makes df["close"] return a DataFrame, and every
    # comparison below then fails with "Operands are not aligned".
    dupe_cols = df.columns[df.columns.duplicated()].tolist()
    if dupe_cols:
        rep.add(Issue(Check.SCHEMA, Severity.FATAL,
                      f"duplicate column labels: {', '.join(map(str, dupe_cols))}"))
        return rep

    # Non-numeric price columns are the common real-world case, not an exotic
    # one: a feed that emits "N/A", or a CSV read without dtype coercion. This
    # gate must REPORT that, not raise a TypeError from inside a comparison.
    non_numeric = [c for c in (*REQUIRED_COLUMNS, "volume")
                   if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        rep.add(Issue(Check.SCHEMA, Severity.FATAL,
                      f"non-numeric columns: {', '.join(non_numeric)} - the "
                      f"feed is returning strings where prices belong"))
        return rep

    _check_index(df, rep)
    _check_prices(df, rep)
    _check_volume(df, rep)
    _check_stale(df, rep)
    _check_returns(df, rep, actions, symbol or None)
    _check_sessions(df, rep, expected_sessions)
    _check_reference(df, rep, reference, reference_name, price_tolerance_pct)
    return rep


def _check_index(df: pd.DataFrame, rep: QualityReport) -> None:
    rep.checks_run.append(Check.INDEX_ORDER)
    if not df.index.is_monotonic_increasing:
        rep.add(Issue(Check.INDEX_ORDER, Severity.FATAL,
                      "timestamps are not in increasing order - every rolling "
                      "window and every lag is computed over the wrong bars"))

    rep.checks_run.append(Check.DUPLICATE_BARS)
    dupes = df.index.duplicated(keep=False)
    if dupes.any():
        sample = [str(t.date()) for t in df.index[dupes][:5]]
        rep.add(Issue(Check.DUPLICATE_BARS, Severity.FATAL,
                      "duplicate timestamps double a day's weight in every "
                      "average and every sum", int(dupes.sum()), sample))


def _check_prices(df: pd.DataFrame, rep: QualityReport) -> None:
    rep.checks_run.append(Check.NON_POSITIVE_PRICE)
    price_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    nulls = df[price_cols].isna().any(axis=1)
    if nulls.any():
        rep.add(Issue(Check.NON_POSITIVE_PRICE, Severity.FATAL,
                      "null prices", int(nulls.sum()),
                      [str(t.date()) for t in df.index[nulls][:5]]))

    nonpos = (df[price_cols] <= 0).any(axis=1) & ~nulls
    if nonpos.any():
        rep.add(Issue(Check.NON_POSITIVE_PRICE, Severity.FATAL,
                      "zero or negative prices", int(nonpos.sum()),
                      [str(t.date()) for t in df.index[nonpos][:5]]))

    # OHLC coherence. A high below the open is not a market event.
    rep.checks_run.append(Check.OHLC_COHERENCE)
    hi, lo = df["high"], df["low"]
    op, cl = df["open"], df["close"]
    bad = (
        (hi < lo)
        | (hi < op) | (hi < cl)
        | (lo > op) | (lo > cl)
    ) & ~nulls
    if bad.any():
        rep.add(Issue(Check.OHLC_COHERENCE, Severity.FATAL,
                      "high/low do not bracket open/close - the bars are "
                      "malformed or the columns are mislabelled",
                      int(bad.sum()),
                      [str(t.date()) for t in df.index[bad][:5]]))


def _check_volume(df: pd.DataFrame, rep: QualityReport) -> None:
    if "volume" not in df.columns:
        rep.checks_skipped[Check.ZERO_VOLUME] = "no volume column"
        return

    rep.checks_run.append(Check.NEGATIVE_VOLUME)
    neg = df["volume"] < 0
    if neg.any():
        rep.add(Issue(Check.NEGATIVE_VOLUME, Severity.FATAL,
                      "negative volume", int(neg.sum())))

    # Missing volume and zero volume are different facts. Conflating them
    # reports a broken feed as an illiquid share.
    nan_vol = df["volume"].isna()
    if nan_vol.any():
        rep.add(Issue(Check.NEGATIVE_VOLUME, Severity.WARN,
                      "null volume - missing data, not zero trading",
                      int(nan_vol.sum())))

    rep.checks_run.append(Check.ZERO_VOLUME)
    zero = (df["volume"] == 0) & ~nan_vol
    if zero.any():
        pct = 100.0 * zero.sum() / len(df)
        sev = Severity.WARN if pct < 20 else Severity.FATAL
        rep.add(Issue(Check.ZERO_VOLUME, sev,
                      f"{pct:.1f}% of bars have zero volume - illiquid, "
                      f"suspended, or a padded series", int(zero.sum())))


def _check_stale(df: pd.DataFrame, rep: QualityReport) -> None:
    """A price that never moves is a feed that stopped, not a quiet market."""
    rep.checks_run.append(Check.STALE_QUOTE)
    unchanged = df["close"].diff() == 0

    # Longest run of consecutive unchanged closes.
    run = best = 0
    best_end = -1
    for i, flat in enumerate(unchanged.to_numpy()):
        run = run + 1 if flat else 0
        if run > best:
            best, best_end = run, i

    if best >= STALE_RUN_LENGTH:
        start = df.index[max(0, best_end - best)].date()
        end = df.index[best_end].date()
        # `best` counts unchanged DIFFS; n diffs span n+1 identical bars, so
        # the bar count is one more than the run length.
        rep.add(Issue(Check.STALE_QUOTE, Severity.WARN,
                      f"close unchanged across {best + 1} consecutive bars "
                      f"({start} to {end})", best + 1))


def _check_returns(df: pd.DataFrame, rep: QualityReport,
                   actions: list[CorporateAction] | None,
                   symbol: str | None = None) -> None:
    """Moves beyond the circuit band that no corporate action explains.

    This reuses `find_discontinuities` rather than reimplementing the ratio
    inference, so the two can never drift apart.
    """
    # These are TWO independent screens measuring different things, and they
    # must stay independent. `find_discontinuities` looks at the OPEN against
    # the previous close - the overnight gap, which is where a missing split
    # shows up. The screen below looks CLOSE to close. A 30% gap down that
    # recovers intraday is invisible to the second and obvious to the first.
    #
    # Gating one behind the other (as this function originally did) meant a
    # missing corporate action could pass the gate entirely, which defeats the
    # module's whole purpose.

    # --- screen 1: unexplained overnight gaps ------------------------------
    if actions is None:
        rep.checks_skipped[Check.CORPORATE_ACTION] = (
            "no action list supplied - overnight gaps could not be explained")
    else:
        rep.checks_run.append(Check.CORPORATE_ACTION)
        unexplained = find_discontinuities(df, known=actions)
        if unexplained:
            rep.add(Issue(Check.CORPORATE_ACTION, Severity.FATAL,
                          "price discontinuity that no corporate action "
                          "explains - almost certainly a missing split or bonus",
                          len(unexplained),
                          [f"{d.date} ~{d.nearest_ratio}"
                           for d in unexplained[:5]]))

    # --- screen 2: close-to-close moves beyond the circuit band ------------
    rep.checks_run.append(Check.EXTREME_RETURN)
    # fill_method=None is deliberate: padding a NaN close would hide a gap
    # behind a fabricated flat return, in the one module whose job is to find
    # exactly that. It is also the pandas 3 default.
    ret = df["close"].pct_change(fill_method=None)
    extreme = ret.abs() > (CIRCUIT_LIMIT_PCT / 100.0)
    if not extreme.any():
        return

    extreme_days = {t.date() for t in df.index[extreme]}
    if actions is None:
        rep.add(Issue(Check.EXTREME_RETURN, Severity.WARN,
                      f"{len(extreme_days)} moves beyond {CIRCUIT_LIMIT_PCT}%, "
                      f"unexplained because no corporate actions were supplied",
                      len(extreme_days),
                      sorted(str(d) for d in extreme_days)[:5]))
        return

    # A move counts as explained only if an action falls on THAT day. The
    # original subtracted one screen's count from the other's, which is
    # comparing unrelated sets - it reported "all explained by corporate
    # actions on file" for an empty action list.
    # Scoped to this instrument for the same reason as the screen above.
    action_days = {a.ex_date for a in actions
                   if symbol is None or a.symbol == symbol}
    unmatched = sorted(extreme_days - action_days)
    matched = len(extreme_days) - len(unmatched)

    if unmatched:
        rep.add(Issue(Check.EXTREME_RETURN, Severity.WARN,
                      f"{len(unmatched)} close-to-close moves beyond "
                      f"{CIRCUIT_LIMIT_PCT}% with no corporate action that day "
                      f"- a limit move, or a bad print",
                      len(unmatched), [str(d) for d in unmatched[:5]]))
    if matched:
        rep.add(Issue(Check.EXTREME_RETURN, Severity.INFO,
                      f"{matched} extreme move(s) fall on a known ex-date",
                      matched))


def _check_sessions(df: pd.DataFrame, rep: QualityReport,
                    expected: list[date] | None) -> None:
    """Compare the bars present against the calendar's trading days."""
    if expected is None:
        rep.checks_skipped[Check.MISSING_BARS] = "no expected session list supplied"
        return

    rep.checks_run.append(Check.MISSING_BARS)
    have = {t.date() for t in df.index}
    want = set(expected)

    missing = sorted(want - have)
    if missing:
        rep.add(Issue(Check.MISSING_BARS, Severity.WARN,
                      "trading sessions with no bar - the feed has gaps",
                      len(missing), [str(d) for d in missing[:5]]))

    rep.checks_run.append(Check.UNEXPECTED_BARS)
    extra = sorted(have - want)
    if extra:
        rep.add(Issue(Check.UNEXPECTED_BARS, Severity.FATAL,
                      "bars on days the exchange was shut - the series is "
                      "padded, or the calendar and the feed disagree",
                      len(extra), [str(d) for d in extra[:5]]))


def _check_reference(df: pd.DataFrame, rep: QualityReport,
                     reference: pd.DataFrame | None, name: str,
                     tolerance_pct: float) -> None:
    """Cross-source reconciliation. Two feeds that disagree on a close mean at
    least one is wrong, and you cannot tell which from one of them alone."""
    if reference is None:
        rep.checks_skipped[Check.CROSS_SOURCE] = "no second source supplied"
        return
    if "close" not in reference.columns:
        rep.checks_skipped[Check.CROSS_SOURCE] = f"{name} has no close column"
        return

    rep.checks_run.append(Check.CROSS_SOURCE)
    joined = df[["close"]].join(reference[["close"]], how="inner",
                                lsuffix="_a", rsuffix="_b")
    if joined.empty:
        rep.add(Issue(Check.CROSS_SOURCE, Severity.WARN,
                      f"no overlapping dates with {name} - cannot reconcile"))
        return

    diff_pct = (joined["close_a"] - joined["close_b"]).abs() / joined["close_b"] * 100
    off = diff_pct > tolerance_pct
    if off.any():
        worst = diff_pct.max()
        rep.add(Issue(Check.CROSS_SOURCE, Severity.WARN,
                      f"disagrees with {name} on {int(off.sum())} of "
                      f"{len(joined)} shared bars, worst {worst:.2f}%",
                      int(off.sum()),
                      [str(t.date()) for t in joined.index[off][:5]]))


# ===========================================================================
# The gate
# ===========================================================================

def check_panel(df: pd.DataFrame, *, by: str | None = None,
                **kwargs) -> dict[str, QualityReport]:
    """Check a multi-symbol frame, one instrument at a time.

    This is what the scanner calls. Splitting here rather than inside `check`
    keeps the single-symbol contract honest and makes the grouping explicit at
    the call site instead of implied.
    """
    col = by or _symbol_column(df)
    if col is None:
        raise PanelError(
            f"no symbol column found (looked for {', '.join(SYMBOL_COLUMNS)}); "
            f"pass by=<column>, or call check() for a single instrument"
        )
    kwargs.pop("symbol", None)
    actions = kwargs.pop("actions", None)
    reference = kwargs.pop("reference", None)

    out: dict[str, QualityReport] = {}
    # dropna=False: the default SILENTLY DISCARDS every row whose symbol is
    # NaN - no report, no issue, nothing in checks_skipped - inside the one
    # module whose stated job is that nothing passes unexamined.
    for sym, group in df.groupby(col, sort=True, dropna=False):
        named = not (sym is None or (isinstance(sym, float) and pd.isna(sym)))
        name = str(sym) if named else "<missing symbol>"

        # Corporate actions and the cross-source reference are PER SYMBOL.
        # Forwarding one shared list to every group is how symbol A's split
        # came to mark symbol B's identical unexplained gap as "explained by a
        # known ex-date" - the original F1 failure reappearing inside its own
        # fix. Match on the action's own symbol, not just its ex-date.
        per_symbol = dict(kwargs)
        if actions is not None:
            per_symbol["actions"] = [
                a for a in actions if getattr(a, "symbol", None) in (None, name)
            ] if named else []
        if reference is not None and named:
            per_symbol["reference"] = reference

        rep = check(group.drop(columns=[col]), symbol=name, **per_symbol)
        if not named:
            # A bar that cannot say which instrument it belongs to is not a
            # minor annotation problem - it cannot be joined, adjusted or
            # priced, so it is fatal rather than a warning.
            rep.add(Issue(
                Check.SCHEMA, Severity.FATAL,
                f"{len(group)} row(s) have no symbol - they cannot be "
                f"attributed to an instrument, so no corporate action, "
                f"reference price or index membership can be applied to them",
            ))
        out[name] = rep
    return out


def assert_clean(df: pd.DataFrame, **kwargs) -> QualityReport:
    """Run every check and raise on anything FATAL.

    This is what a backtest harness calls. Reject rather than guess.
    """
    rep = check(df, **kwargs)
    if rep.fatal:
        detail = "\n".join(f"  - {i}" for i in rep.fatal)
        raise ValueError(
            f"{rep.symbol} failed the data-quality gate with "
            f"{len(rep.fatal)} fatal issue(s):\n{detail}"
        )
    return rep
