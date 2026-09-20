"""Refresh the data that cannot be derived and must be fetched.

    python -m desk.marketdata.refresh calendar
    python -m desk.marketdata.refresh actions RELIANCE INFY TCS

Writes into `configs/`. Deliberately a separate entry point rather than
something the API does on startup: a fetch that happens implicitly is a fetch
nobody notices failing, and this data changes a few times a year, not per
request.

Every run prints what it could NOT parse. That list is the point - an action
this tool cannot read is an action your backtest will see as an unexplained
gap, and you want to find out here rather than six months into a result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from desk.marketdata.calendar_in import CalendarError, TradingCalendar
from desk.marketdata.sources.nse import (
    NseSession,
    RateLimited,
    SourceError,
    bhavcopy_equity_only,
    parse_bhavcopy,
    parse_corporate_actions,
    parse_holiday_master,
)
from desk.marketdata.symbols import Symbol, SymbolError
from desk.research.refresh import RESEARCH_KINDS, refresh_research

# NSE trades in IST. date.today() reads server-local time, which is silently
# 5:30h out on a UTC-clock VPS - the same reasoning as desk/api/main.py.
IST = ZoneInfo("Asia/Kolkata")


def _merge_calendars(existing: dict, fresh: dict) -> dict:
    """Fold a freshly fetched year into whatever is already on file.

    Fresh data wins on any date present in both, because NSE amends its own
    list. Years only ever accumulate.
    """
    if existing.get("exchange") and fresh.get("exchange") and \
            existing["exchange"] != fresh["exchange"]:
        raise CalendarError(
            f"refusing to merge {fresh['exchange']} data into an "
            f"{existing['exchange']} file - the holiday lists differ"
        )

    def by_date(rows):
        return {r["date"]: r for r in rows if isinstance(r, dict) and "date" in r}

    holidays = by_date(existing.get("holidays", []))
    holidays.update(by_date(fresh.get("holidays", [])))
    specials = by_date(existing.get("special_sessions", []))
    specials.update(by_date(fresh.get("special_sessions", [])))

    merged = dict(fresh)
    merged["holidays"] = sorted(holidays.values(), key=lambda r: r["date"])
    merged["special_sessions"] = sorted(specials.values(), key=lambda r: r["date"])
    # int() on a junk entry raises a BARE ValueError, which main()'s handler -
    # catching only CalendarError and SourceError - would not catch, turning a
    # malformed file on disk into an unhandled traceback instead of the
    # "existing calendar left untouched" message the CLI promises.
    try:
        merged["years"] = sorted(
            {int(y) for y in existing.get("years", [])}
            | {int(y) for y in fresh.get("years", [])}
        )
    except (TypeError, ValueError) as exc:
        raise CalendarError(
            f"malformed 'years' in the holiday data: {exc}"
        ) from exc
    return merged

#: Must agree with desk.api.main.CONFIGS - the refresh CLI writes what the
#: API reads, so a deployment that moves one and not the other silently
#: refreshes into a directory nobody serves from.
CONFIGS = Path(os.environ.get("DESK_CONFIG_DIR")
               or Path(__file__).resolve().parents[2] / "configs")


def refresh_calendar(session: NseSession, out: Path, *, replace: bool = False) -> int:
    raw = session.fetch_holiday_master()
    payload = parse_holiday_master(raw)
    out.parent.mkdir(parents=True, exist_ok=True)

    # MERGE, do not overwrite. NSE's holiday master carries the current year
    # only. A straight write in January 2027 would produce years [2027] and
    # silently erase 2026 - after which every backtest spanning 2026 raises
    # CalendarNotLoaded, correctly but for entirely the wrong reason.
    if out.exists() and not replace:
        payload = _merge_calendars(json.loads(out.read_text(encoding="utf-8")),
                                   payload)

    # Write to a temp file, validate THAT, and only then replace the real one.
    # Writing first and validating after means a bad fetch destroys the good
    # file it was meant to update.
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        TradingCalendar.from_file(tmp)
    except CalendarError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)

    cal = TradingCalendar.from_file(out)
    print(f"calendar -> {out}")
    print(f"  years {cal.loaded_years}, {len(payload['holidays'])} holidays, "
          f"{len(payload['special_sessions'])} special sessions")

    if payload.get("_needs_session_timings"):
        print(f"  NOTE: {len(payload['_needs_session_timings'])} special-session "
              f"date(s) have no published timings: "
              f"{', '.join(payload['_needs_session_timings'])}")
        print("        Add them from the Muhurat circular before trading intraday.")
    return 0


#: Reasons that mean a HUMAN must look, not that a retry might help. An
#: action with no usable ex-date cannot be applied point-in-time at all, so it
#: belongs here beside the ones that cannot be reduced to a single factor.
_REVIEW_WORTHY = ("cannot be reduced", "no usable ex-date")


def refresh_actions(session: NseSession, symbols: list[str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_unparsed = 0
    failed = 0

    for raw_symbol in symbols:
        # Validate through the parser that already exists. Without it,
        # `C:/Windows/Temp/pwn` escapes out_dir entirely - pathlib discards
        # the left operand when the right is absolute.
        try:
            base = Symbol.parse(raw_symbol).base
        except SymbolError as exc:
            print(f"{raw_symbol[:30]:14} SKIPPED  {exc}")
            failed += 1
            continue
        try:
            payload = session.fetch_corporate_actions(base)
        except SourceError as exc:
            print(f"{base:14} FAILED  {exc}")
            failed += 1
            continue

        actions, unparsed = parse_corporate_actions(payload, base)
        dest = out_dir / f"{base}.json"
        # Atomic, like every other write in this file. A direct write_text
        # here could leave a truncated corporate-action file if the process
        # died mid-write - and unlike a truncated bhavcopy, a restart would
        # NOT repair it: nothing re-fetches actions automatically, and
        # load_actions would then either raise or, worse, hand the scanner a
        # short list that reads as "this symbol has no splits".
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "symbol": f"{base}.NS",
            "actions": [
                {"ex_date": a.ex_date.isoformat(), "type": a.type.value,
                 "factor": a.factor, "amount": a.amount, "note": a.note}
                for a in actions
            ],
            "unparsed": [
                {"ex_date": u.ex_date.isoformat() if u.ex_date else None,
                 "subject": u.subject, "reason": u.reason}
                for u in unparsed
            ],
        }, indent=2), encoding="utf-8")
        tmp.replace(dest)

        structural = [a for a in actions if a.factor != 1.0]
        print(f"{base:14} {len(actions):3} parsed "
              f"({len(structural)} price-affecting), {len(unparsed):2} unparsed")

        for u in unparsed:
            if any(k in u.reason for k in _REVIEW_WORTHY):
                print(f"               REVIEW {u.ex_date} {u.subject[:60]!r}")
                total_unparsed += 1

    if total_unparsed:
        print(f"\n{total_unparsed} action(s) move the price and need a human. "
              f"Until they are handled, any series spanning those dates will "
              f"show a gap the quality gate reports as a missing action.")
    if failed:
        print(f"\n{failed} symbol(s) failed. Exiting non-zero so a scheduled "
              f"run does not report success.")
    return 1 if failed else 0


def refresh_bhavcopy(session: NseSession, day: date, out_dir: Path) -> int:
    """Fetch and save ONE day's full-market snapshot as Parquet.

    Parquet, not JSON: this is 2,600+ numeric rows that will accumulate day
    over day into a genuine time series, exactly the shape the project's own
    stated stack (pandas + DuckDB over Parquet) was chosen for - unlike the
    small, hand-editable JSON configs elsewhere in this module.

    One file per day (never merged into a single growing file the way the
    calendar is), so a bad fetch on one day cannot corrupt any other day's
    data - the failure mode a merged file would have is structurally absent
    here, not merely guarded against.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = session.fetch_bhavcopy(day)
    df = parse_bhavcopy(raw, day=day)          # raises if the file is for the wrong day

    out = out_dir / f"{day.isoformat()}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)                            # atomic: no half-written file survives a crash

    eq = bhavcopy_equity_only(df)
    print(f"bhavcopy -> {out}")
    print(f"  {len(df)} rows total, {len(eq)} EQ-series "
          f"({eq['symbol'].nunique()} unique symbols)")
    zero_vol = int((eq["volume"] == 0).sum())
    if zero_vol:
        print(f"  NOTE: {zero_vol} EQ symbol(s) traded zero volume today "
              f"(suspended, or genuinely no trades)")
    return 0


def refresh_crosscheck(session: NseSession, day: date, out_dir: Path, *,
                       bhavcopy_dir: Path, isin_path: Path) -> int:
    """Reconcile one day's NSE closes against BSE, and store the verdict.

    AT INGEST, NOT AT QUERY TIME. "Do the two exchanges agree on the 17th"
    is answered once for that day and never changes; running it inside
    /plan/today would put a BSE fetch and a 5,000-row join on every page
    load to recompute a constant.

    The report is what the plan reads. It is deliberately a SUMMARY plus
    the offending rows rather than the full join: 2,400 matched names a day
    accumulates, and nobody needs the 2,397 that agreed.
    """
    from desk.marketdata.crosscheck import DEFAULT_TOLERANCE_PCT, reconcile
    from desk.marketdata.isin import IsinMap, parse_equity_master
    from desk.marketdata.sources.bse import BseSession
    from desk.marketdata.sources.bse import parse_bhavcopy as parse_bse

    snapshot = bhavcopy_dir / f"{day.isoformat()}.parquet"
    if not snapshot.is_file():
        print(f"no NSE snapshot for {day} at {snapshot} - fetch it first with "
              f"'python -m desk.marketdata.refresh bhavcopy --date {day}'")
        return 1

    # The ISIN map is the join key and NSE's bhavcopy has no ISIN column.
    # Cached because it changes on listings, not daily.
    isin_map = IsinMap.load(isin_path) if isin_path.is_file() else None
    if isin_map is None:
        print("fetching NSE's equity master for the ISIN map")
        isin_map = parse_equity_master(
            session.fetch_with_retry(session.fetch_equity_master),
            as_of=day)
        isin_map.save(isin_path)
    print(f"ISIN map: {len(isin_map)} symbols"
          + (f", {len(isin_map.ambiguous)} ambiguous" if isin_map.ambiguous else ""))

    bse_session = BseSession()
    try:
        bse = parse_bse(bse_session.fetch_with_retry(
            bse_session.fetch_bhavcopy, day), day=day)
    except (SourceError, RateLimited) as exc:
        print(f"could not fetch BSE for {day}: {exc}")
        return 1

    nse = pd.read_parquet(snapshot)
    disagreements, coverage = reconcile(nse, bse, isin_map)

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{day.isoformat()}.json"
    payload = {
        "as_of": day.isoformat(),
        "checked_at": datetime.now(IST).isoformat(timespec="seconds"),
        "tolerance_pct": DEFAULT_TOLERANCE_PCT,
        "coverage": {
            "nse_symbols": coverage.nse_symbols,
            "checkable": coverage.checkable,
            "no_isin": coverage.no_isin,
            "ambiguous_isin": coverage.ambiguous_isin,
            "not_on_bse": coverage.not_on_bse,
            "thin_on_bse": coverage.thin_on_bse,
            "no_usable_close": coverage.no_usable_close,
            "balanced": coverage.balanced,
        },
        "disagreements": [
            {"symbol": str(r.symbol),
             "nse_close": float(r.nse_close),
             "bse_close": float(r.bse_close),
             "diff_pct": round(float(r.diff_pct), 4),
             "bse_turnover": float(r.bse_turnover)}
            for r in disagreements.itertuples()
        ],
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(dest)

    print(coverage.report())
    n = len(disagreements)
    if n:
        print(f"\n{n} symbol(s) disagree by more than "
              f"{DEFAULT_TOLERANCE_PCT}%:")
        for r in disagreements.head(10).itertuples():
            print(f"  {r.symbol:<16} NSE {r.nse_close:>10.2f}  "
                  f"BSE {r.bse_close:>10.2f}  {r.diff_pct:>6.2f}%")
    else:
        print(f"\nno symbol disagrees by more than {DEFAULT_TOLERANCE_PCT}%")
    print(f"wrote {dest}")

    # A disagreement is a WARNING, not a failed run. The exchanges are both
    # real and a thin print is not an error in our pipeline; the plan
    # surfaces it and a human decides. Exiting non-zero here would make a
    # scheduled refresh look broken on a perfectly ordinary day.
    return 0


def backfill_bhavcopy(session: NseSession, start: date, end: date,
                      out_dir: Path, *, calendar=None) -> int:
    """Fetch every trading day in [start, end] that is not already on disk.

    ONE SESSION for the whole range. The single-day command was the only way
    to do this, and looping it from a shell means a fresh cookie handshake
    per day - 60 extra requests to NSE's homepage for a 60-day backfill,
    which is both wasteful and exactly the traffic pattern that gets an IP
    blocked. The throttle is per-session too, so a shell loop bypasses it
    entirely.

    Days already on disk are SKIPPED, so an interrupted run resumes instead
    of refetching. Weekends and holidays are skipped when a calendar is
    available; without one, every weekday is attempted and NSE's 404 for a
    non-trading day is treated as "not a trading day" rather than a failure.

    Returns 0 if every attempted day succeeded, 1 if any genuinely failed,
    2 if NSE rate-limited (distinct, because a scheduler should retry that
    one and not the other).
    """
    if start > end:
        print(f"--from {start} is after --to {end}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:          # Mon-Fri; NSE cash does not trade weekends
            days.append(cursor)
        cursor += timedelta(days=1)

    if calendar is not None:
        try:
            days = [d for d in days if calendar.is_trading_day(d)]
        except Exception:                 # noqa: BLE001 - calendar is optional
            print("  (calendar could not classify these dates; attempting "
                  "every weekday and letting NSE decide)")

    todo = [d for d in days if not (out_dir / f"{d.isoformat()}.parquet").exists()]
    have = len(days) - len(todo)
    print(f"{len(days)} trading day(s) in range; {have} already on disk, "
          f"{len(todo)} to fetch")
    if not todo:
        return 0

    fetched = failed = unresolved = 0
    for i, day in enumerate(todo, 1):
        try:
            print(f"[{i}/{len(todo)}] {day}", end="  ")
            refresh_bhavcopy(session, day, out_dir)
            fetched += 1
        except RateLimited as exc:
            # Stop the whole run. Days already written stay written, and
            # rerunning the same command resumes from here.
            print(f"\nRATE LIMITED at {day}: {exc}")
            print(f"{fetched} day(s) fetched before stopping. Rerun the same "
                  f"command to resume - days already on disk are skipped.")
            return 2
        except SourceError as exc:
            if calendar is None:
                # NSE serves no file for a non-trading day, so with no
                # calendar this is genuinely AMBIGUOUS - a holiday and a
                # broken fetch look identical from here.
                #
                # It is deliberately not guessed. An earlier version
                # sniffed the error text for "404" and called the rest
                # holidays, which is a coin flip dressed as a diagnosis: it
                # would mark a real outage as a holiday and move on. These
                # days are counted separately and named at the end, and the
                # run does not fail on them - a backfill spanning any
                # holiday would otherwise always exit non-zero, and a
                # command that always fails stops being read.
                print(f"  no file - cannot tell holiday from failure: {exc}")
                unresolved += 1
                continue
            # With a calendar saying this day trades, no file IS a failure.
            print(f"  FAILED  {exc}")
            failed += 1

    print(f"\nfetched {fetched} day(s)"
          + (f", {unresolved} unresolved" if unresolved else "")
          + (f", {failed} FAILED" if failed else ""))
    if unresolved:
        print(f"{unresolved} day(s) returned no file, and with no holiday "
              f"calendar loaded a market holiday cannot be told apart from a "
              f"failed fetch. Run 'python -m desk.marketdata.refresh "
              f"calendar' to remove the ambiguity, then rerun - days already "
              f"on disk are skipped.")
    if failed:
        print("Exiting non-zero so a scheduled run does not report success.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="desk.marketdata.refresh",
                                description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="what", required=True)

    c = sub.add_parser("calendar", help="NSE trading holidays")
    c.add_argument("--out", type=Path, default=CONFIGS / "holidays_nse.json")
    c.add_argument("--replace", action="store_true",
                   help="overwrite instead of merging (DESTROYS earlier years)")

    a = sub.add_parser("actions", help="corporate actions for one or more symbols")
    a.add_argument("symbols", nargs="+")
    a.add_argument("--out-dir", type=Path, default=CONFIGS / "corporate_actions")

    r = sub.add_parser("research",
                       help="DEPRECATED alias - use "
                            "'python -m desk.research.refresh research'")
    r.add_argument("symbols", nargs="+")
    r.add_argument("--out-dir", type=Path, default=CONFIGS / "research")
    r.add_argument("--kinds", default=",".join(RESEARCH_KINDS),
                   help=f"comma-separated subset of {','.join(RESEARCH_KINDS)}")
    r.add_argument("--no-xbrl", action="store_true",
                   help="skip the financial documents - much faster, but the "
                        "filings are then records that a result was announced "
                        "rather than the numbers themselves")
    r.add_argument("--since", type=date.fromisoformat, metavar="YYYY-MM-DD",
                   help="ignore records disclosed before this date. Saves no "
                        "requests (these endpoints return whole history) but "
                        "bounds the XBRL documents fetched and the disk "
                        "written - both dominated by old records")
    r.add_argument("--force", action="store_true",
                   help="refetch symbols already on file (default is to skip "
                        "them, so an interrupted run resumes)")
    r.add_argument("--min-interval", type=float, default=1.0,
                   help="seconds between requests (default 1.0). NSE "
                        "rate-limits; lower this deliberately or not at all")

    b = sub.add_parser("bhavcopy",
                       help="one day's full bhavcopy - the WHOLE exchange in "
                            "a single request, not per symbol")
    b.add_argument("--date", type=date.fromisoformat, metavar="YYYY-MM-DD")
    b.add_argument("--from", dest="start", type=date.fromisoformat,
                   metavar="YYYY-MM-DD",
                   help="backfill a RANGE using one session. Days already "
                        "on disk are skipped, so an interrupted run resumes.")
    b.add_argument("--to", dest="end", type=date.fromisoformat,
                   metavar="YYYY-MM-DD", help="defaults to today (IST)")
    b.add_argument("--out-dir", type=Path, default=CONFIGS / "bhavcopy")

    x = sub.add_parser(
        "crosscheck",
        help="reconcile one day's NSE closes against BSE (the second source)")
    x.add_argument("--date", type=date.fromisoformat, metavar="YYYY-MM-DD",
                   help="defaults to the newest NSE snapshot on disk")
    x.add_argument("--bhavcopy-dir", type=Path, default=CONFIGS / "bhavcopy")
    x.add_argument("--out-dir", type=Path, default=CONFIGS / "crosscheck")
    x.add_argument("--isin", type=Path, default=CONFIGS / "isin_map.json")

    args = p.parse_args(argv)
    interval = getattr(args, "min_interval", 1.0)
    session = NseSession(min_interval=interval)

    try:
        if args.what == "calendar":
            return refresh_calendar(session, args.out, replace=args.replace)
        if args.what == "bhavcopy":
            if args.start:
                end = args.end or datetime.now(IST).date()
                cal = None
                try:
                    cal = TradingCalendar.from_file(CONFIGS / "holidays_nse.json")
                except Exception:              # noqa: BLE001 - optional
                    print("no holiday calendar loaded; every weekday will be "
                          "attempted and NSE will decide")
                return backfill_bhavcopy(session, args.start, end,
                                         args.out_dir, calendar=cal)
            if not args.date:
                print("bhavcopy needs either --date or --from/--to")
                return 1
            return refresh_bhavcopy(session, args.date, args.out_dir)
        if args.what == "crosscheck":
            day = args.date
            if day is None:
                snaps = sorted(args.bhavcopy_dir.glob("*.parquet"))
                if not snaps:
                    print(f"no NSE snapshots in {args.bhavcopy_dir}")
                    return 1
                day = date.fromisoformat(snaps[-1].stem)
            return refresh_crosscheck(session, day, args.out_dir,
                                      bhavcopy_dir=args.bhavcopy_dir,
                                      isin_path=args.isin)
        if args.what == "research":
            # KEPT AS AN ALIAS, NOT REMOVED. The same subcommand, with the
            # same arguments, calling the same refresh_research(), exists
            # in desk.research.refresh - which is where the implementation
            # and RESEARCH_KINDS actually live, and where the four
            # sibling commands (events, fundamentals, news, settle, close)
            # exist with no equivalent here.
            #
            # Two documented ways to run one job is how they silently
            # diverge when someone edits one and forgets the other. It
            # still works because a script may depend on it; the pointer
            # says where the canonical one is.
            print("NOTE: 'desk.marketdata.refresh research' is an alias. "
                  "The canonical command is 'python -m desk.research.refresh "
                  "research', which also has events / fundamentals / news / "
                  "settle / close.", file=sys.stderr)
            kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
            unknown = [k for k in kinds if k not in RESEARCH_KINDS]
            if unknown:
                print(f"ERROR: unknown kind(s) {unknown}. Choose from "
                      f"{list(RESEARCH_KINDS)}.", file=sys.stderr)
                return 1
            return refresh_research(session, args.symbols, args.out_dir,
                                    kinds=kinds, with_xbrl=not args.no_xbrl,
                                    since=args.since, force=args.force)
        return refresh_actions(session, args.symbols, args.out_dir)
    except CalendarError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("The existing calendar was left untouched.", file=sys.stderr)
        return 1
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("NSE blocks bare clients; this needs a browser-like session and "
              "occasionally rate-limits. Retry in a minute.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
