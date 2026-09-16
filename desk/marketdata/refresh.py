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
from datetime import date
from pathlib import Path

from desk.marketdata.calendar_in import CalendarError, TradingCalendar
from desk.marketdata.sources.nse import (
    NseSession,
    SourceError,
    bhavcopy_equity_only,
    parse_bhavcopy,
    parse_corporate_actions,
    parse_holiday_master,
)
from desk.marketdata.symbols import Symbol, SymbolError
from desk.research.refresh import RESEARCH_KINDS, refresh_research


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
                       help="filings, announcements, board meetings, "
                            "shareholding and insider deals for one or more "
                            "symbols")
    r.add_argument("symbols", nargs="+")
    r.add_argument("--out-dir", type=Path, default=CONFIGS / "research")
    r.add_argument("--kinds", default=",".join(RESEARCH_KINDS),
                   help=f"comma-separated subset of {','.join(RESEARCH_KINDS)}")
    r.add_argument("--no-xbrl", action="store_true",
                   help="skip the financial documents - much faster, but the "
                        "filings are then records that a result was announced "
                        "rather than the numbers themselves")
    r.add_argument("--force", action="store_true",
                   help="refetch symbols already on file (default is to skip "
                        "them, so an interrupted run resumes)")
    r.add_argument("--min-interval", type=float, default=1.0,
                   help="seconds between requests (default 1.0). NSE "
                        "rate-limits; lower this deliberately or not at all")

    b = sub.add_parser("bhavcopy",
                       help="one day's full bhavcopy - the WHOLE exchange in "
                            "a single request, not per symbol")
    b.add_argument("--date", type=date.fromisoformat, required=True,
                   metavar="YYYY-MM-DD")
    b.add_argument("--out-dir", type=Path, default=CONFIGS / "bhavcopy")

    args = p.parse_args(argv)
    interval = getattr(args, "min_interval", 1.0)
    session = NseSession(min_interval=interval)

    try:
        if args.what == "calendar":
            return refresh_calendar(session, args.out, replace=args.replace)
        if args.what == "bhavcopy":
            return refresh_bhavcopy(session, args.date, args.out_dir)
        if args.what == "research":
            kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
            unknown = [k for k in kinds if k not in RESEARCH_KINDS]
            if unknown:
                print(f"ERROR: unknown kind(s) {unknown}. Choose from "
                      f"{list(RESEARCH_KINDS)}.", file=sys.stderr)
                return 1
            return refresh_research(session, args.symbols, args.out_dir,
                                    kinds=kinds, with_xbrl=not args.no_xbrl,
                                    force=args.force)
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
