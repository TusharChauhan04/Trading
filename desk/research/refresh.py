"""Fetching exchange-disclosed research, symbol by symbol.

Separate from desk/marketdata/refresh.py because this drives the RESEARCH
domain - it writes filings, parses XBRL and accounts for undated records,
none of which is marketdata's concern. Keeping it there made
desk.marketdata import three of desk.research's four modules, which inverts
the layering: research depends on marketdata for its HTTP session, so
marketdata cannot also sit above it. R5 (news) and R6 (BSE) need a home to
slot into, and this is it.

The HTTP session still comes from desk.marketdata.sources.nse - a throttled,
cookie-bearing client is genuinely cross-cutting and both domains need it.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from desk.marketdata.sources.nse import NseSession, RateLimited, SourceError
from desk.marketdata.symbols import Symbol, SymbolError
from desk.research.sources.nse import (
    parse_announcements,
    parse_board_meetings,
    parse_insider_deals,
    parse_results,
    parse_shareholding,
)
from desk.research.store import FilingStore
from desk.research.xbrl import XbrlError, parse_xbrl

__all__ = ["RESEARCH_KINDS", "refresh_research"]

#: The five per-symbol research endpoints. `filings` is listed first because
#: it is the only one that also fetches a second document per record (the
#: XBRL), so a run bounded by --kinds filings is the expensive one.
RESEARCH_KINDS = ("filings", "announcements", "boardmeetings",
                  "shareholding", "insider")


def refresh_research(session: NseSession, symbols: list[str], out_dir: Path, *,
                     kinds: tuple[str, ...] = RESEARCH_KINDS,
                     with_xbrl: bool = True,
                     since: date | None = None,
                     force: bool = False) -> int:
    """Fetch exchange-disclosed research for one or more symbols.

    RESUMABLE BY DESIGN, not as an afterthought. A full-universe pass is
    ~1,598 symbols x 5 endpoints, and at the session's 1s throttle that is
    over two hours - it will not reliably complete in one sitting. Work
    already on file is skipped unless --force, so an interrupted or
    rate-limited run restarts from where it stopped rather than from zero.

    RESUME IS PER KIND, NOT PER SYMBOL, and that distinction is load-bearing.
    Keyed on the symbol alone, `--kinds insider` marked the symbol done and a
    later FULL run skipped it entirely - fetching no filings and no
    announcements, silently, while the operator believed they had complete
    research data. The marker therefore records WHICH kinds were fetched, and
    a symbol is skipped only when every kind being asked for is already
    there.

    `since` drops records disclosed before that date. It does NOT save
    requests - these endpoints return a symbol's whole history and take no
    date parameter - but it bounds the two things that actually hurt: the
    XBRL documents fetched (53 per symbol for RELIANCE, one request each,
    and the dominant cost of a full backfill) and the disk written (3,345
    announcements for RELIANCE, most of them a decade old).

    Prints a progress counter because a multi-hour run with no output is
    indistinguishable from a hung one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filings_store = FilingStore(out_dir / "filings")
    failed = 0
    skipped = 0
    total_undated = 0
    total = len(symbols)

    for i, raw_symbol in enumerate(symbols, 1):
        # Same validation as refresh_actions, and for the same reason: this
        # value becomes a path component, and pathlib discards the left
        # operand when the right is absolute.
        try:
            base = Symbol.parse(raw_symbol).base
        except SymbolError as exc:
            print(f"[{i}/{total}] {raw_symbol[:24]:14} SKIPPED  {exc}")
            failed += 1
            continue

        marker = out_dir / "_done" / f"{base}.json"
        done: dict[str, int] = {}
        if marker.exists() and not force:
            try:
                done = json.loads(marker.read_text(encoding="utf-8")).get(
                    "counts", {}) or {}
            except (OSError, json.JSONDecodeError):
                done = {}          # unreadable marker: refetch rather than skip

        todo = tuple(k for k in kinds if k not in done)
        if not todo:
            skipped += 1
            continue

        counts: dict[str, int] = dict(done)
        undated_here: list = []
        try:
            for kind in todo:
                n, und = _fetch_one_kind(session, base, kind, out_dir,
                                         filings_store, with_xbrl=with_xbrl,
                                         since=since)
                counts[kind] = n
                undated_here.extend(und)
        except RateLimited as exc:
            # Stop the whole run rather than grinding through 1,500 more
            # symbols against a host that is already refusing us. The marker
            # files mean the next run picks up here.
            print(f"[{i}/{total}] {base:14} RATE LIMITED  {exc}")
            print(f"\nStopped at {base}. {i - 1} symbol(s) completed; rerun the "
                  f"same command to resume from here.")
            return 2
        except SourceError as exc:
            print(f"[{i}/{total}] {base:14} FAILED  {exc}")
            failed += 1
            continue

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(
            {"symbol": base, "counts": counts,
             "undated": len(undated_here),
             "fetched_at": datetime.now().isoformat(timespec="seconds")},
            indent=1), encoding="utf-8")

        # Kinds carried over from a previous run are marked, so a resumed run
        # does not read as though it refetched everything.
        summary = " ".join(f"{k}={counts[k]}" + ("*" if k in done else "")
                           for k in kinds if k in counts)
        print(f"[{i}/{total}] {base:14} {summary}")

        # Surfaced on the console AS IT HAPPENS, not only in a file. A record
        # NSE published without a usable timestamp is the early warning that
        # a field has been renamed, and an operator watching a run should see
        # it then rather than discover it months later when a filing is
        # inexplicably missing from a backtest.
        for u in undated_here[:3]:
            print(f"               UNDATED {u}")
        if len(undated_here) > 3:
            print(f"               UNDATED +{len(undated_here) - 3} more")
        total_undated += len(undated_here)

    if skipped:
        print(f"\n{skipped} symbol(s) already had every requested kind and "
              f"were skipped. Pass --force to refetch. A '*' above marks a "
              f"kind carried over from an earlier run rather than refetched.")
    if total_undated:
        print(f"{total_undated} record(s) had no usable disclosure timestamp "
              f"and were NOT stored. They are listed above; if this number is "
              f"large, a field has probably been renamed.")
    if failed:
        print(f"{failed} symbol(s) failed. Exiting non-zero so a scheduled "
              f"run does not report success.")
    return 1 if failed else 0


def _fetch_one_kind(session: NseSession, base: str, kind: str, out_dir: Path,
                    filings_store: "FilingStore", *, with_xbrl: bool,
                    since: date | None = None):
    """One endpoint for one symbol. Returns (records written, undated).

    `undated` is NEVER filtered by `since`. A record with no timestamp cannot
    be placed in time, so excluding it by date would be inventing the very
    fact it is missing - and these are the records that warn a field has been
    renamed.
    """
    if kind == "filings":
        filings, undated = parse_results(session.fetch_results(base), base)
        if since is not None:
            filings = [f for f in filings if f.disclosed_at.date() >= since]
        for f in filings:
            periods = ()
            if with_xbrl and f.xbrl_url:
                # A historical filing's XBRL never changes, and RELIANCE alone
                # has 53 of them. Refetching every one on every run is what
                # turns a full-universe backfill into ~92,000 requests and 25
                # hours; skipping the ones already parsed makes an incremental
                # run cost one document per genuinely new filing.
                if filings_store.has_numbers_for(f):
                    continue
                try:
                    periods, _warn = parse_xbrl(session.fetch_xbrl(f.xbrl_url),
                                                symbol=base)
                except (SourceError, XbrlError) as exc:
                    # A filing whose document cannot be read is still a filing.
                    # Store the record without numbers rather than losing the
                    # disclosure entirely.
                    print(f"               XBRL unreadable for "
                          f"{f.relating_to or f.period_end}: {exc}")
            filings_store.write(f, periods)
        return len(filings), undated

    fetch, parse = {
        "announcements": (session.fetch_announcements, parse_announcements),
        "boardmeetings": (session.fetch_board_meetings, parse_board_meetings),
        "shareholding": (session.fetch_shareholding, parse_shareholding),
        "insider": (session.fetch_insider_deals, parse_insider_deals),
    }[kind]

    records, undated = parse(fetch(base), base)
    if since is not None:
        records = [r for r in records if r.disclosed_at.date() >= since]
    dest = out_dir / kind / f"{base}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "symbol": f"{base}.NS",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "since": since.isoformat() if since else None,
        "records": [_research_to_json(r) for r in records],
        # Written into the file as well as printed. Printed-only means lost
        # the moment the terminal scrolls, and this is the list that says
        # what the exchange published and we could not use.
        "undated": [{"kind": u.kind, "reason": u.reason, "raw": u.raw}
                    for u in undated],
    }, indent=1), encoding="utf-8")
    tmp.replace(dest)
    return len(records), undated


def _research_to_json(rec) -> dict:
    out = {}
    for field_name in rec.__slots__:
        v = getattr(rec, field_name)
        out[field_name] = v.isoformat() if hasattr(v, "isoformat") else v
    return out



# ===========================================================================
# The command line
# ===========================================================================
#
# refresh_research() existed with no way to invoke it - the same defect the
# 2026-09-16 recovery found in desk/marketdata/refresh.py, where the whole
# `bhavcopy` subcommand was missing from argparse while the function it
# called was fully written and fully tested. A function with no entry point
# looks finished from inside the test suite and is unreachable in practice.
#
# `events` is here rather than in RESEARCH_KINDS on purpose: the other five
# are PER-SYMBOL and cost ~1,598 requests each, while the event calendar is
# ONE market-wide request covering every listed company. Putting it in the
# per-symbol loop would refetch the same market-wide file once per symbol.

def _cmd_fundamentals(args) -> int:
    """Build the pre-open fundamentals table. Touches no network.

    Separate from `research` because it is a different kind of work: that
    one is ~1,598 throttled HTTP requests over hours, this is a pure
    recomputation from what is already on disk and takes seconds. Running
    them together would mean you could not rebuild the table after fixing a
    parser without refetching the internet.
    """
    import pandas as pd

    from desk.research.fundamentals import FundamentalsCache, build_table
    from desk.research.store import FilingStore

    universe = Path(args.universe)
    snapshots = sorted(universe.glob("*.parquet")) if universe.is_dir() else []
    if not snapshots:
        print(f"no bhavcopy snapshots in {universe} - the symbol universe "
              f"comes from there. Run 'python -m desk.marketdata.refresh "
              f"bhavcopy' first.")
        return 1

    if args.as_of:
        try:
            as_of = date.fromisoformat(args.as_of)
        except ValueError:
            print(f"--as-of must be YYYY-MM-DD, got {args.as_of!r}")
            return 1
    else:
        as_of = date.fromisoformat(snapshots[-1].stem)

    # The universe must come from a snapshot dated AT OR BEFORE as_of. Taking
    # today's symbol list for a historical scan would quietly introduce
    # survivorship: names listed since as_of would appear, and names delisted
    # since would not. The filings store already refuses to read a filing
    # disclosed after as_of; this closes the same hole one level up.
    eligible = [p for p in snapshots if p.stem <= as_of.isoformat()]
    if not eligible:
        print(f"no bhavcopy snapshot on or before {as_of} - cannot build a "
              f"point-in-time universe for that date.")
        return 1
    snap = eligible[-1]
    symbols = sorted(pd.read_parquet(snap)["symbol"].astype(str).unique())

    store = FilingStore(args.filings)
    frame, coverage = build_table(store, symbols, as_of=as_of)

    dest = FundamentalsCache(args.out).save(frame, as_of=as_of,
                                            coverage=coverage)
    total = sum(coverage.values()) or 1
    print(f"built {len(frame)} row(s) for {as_of} from {len(symbols)} "
          f"symbols (universe: {snap.name})")
    for reason, n in sorted(coverage.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6}  {reason}  ({100.0 * n / total:.1f}%)")
    print(f"wrote {dest}")

    if not len(frame):
        # An empty table saves, loads, and then reports every company as
        # "no filing on file" - which is TRUE but reads like a working
        # pipeline producing a quiet answer. Say it plainly instead.
        print("WARNING: the table is EMPTY. Every fundamental check will "
              "report NOT CHECKED. Fetch filings first with 'python -m "
              "desk.research.refresh research <SYMBOLS> --kinds filings'.")
        return 1

    # The age of the newest filing decides whether an age filter is usable at
    # all, and it is not obvious from the row count. Print it so nobody sets
    # max_filing_age_days without seeing what it would exclude.
    if "days_since_filing" in frame and len(frame):
        ages = frame["days_since_filing"].dropna()
        if len(ages):
            print(f"  filing age (days): min {int(ages.min())}, "
                  f"median {int(ages.median())}, max {int(ages.max())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="desk.research.refresh",
        description="Fetch exchange-disclosed research from NSE.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser(
        "events",
        help="the market-wide corporate event calendar (ONE request)")
    ev.add_argument("--out", default="configs/research/event_calendar.json")

    fu = sub.add_parser(
        "fundamentals",
        help="build the pre-open fundamentals table (no network)")
    fu.add_argument("--as-of", default=None,
                    help="YYYY-MM-DD, the point-in-time date to build FOR. "
                         "Defaults to the newest bhavcopy on file, because "
                         "that is the day the scanner can actually run.")
    fu.add_argument("--filings", default="configs/research/filings")
    fu.add_argument("--out", default="configs/research/fundamentals")
    fu.add_argument("--universe", default="configs/bhavcopy",
                    help="bhavcopy directory; its newest day supplies the "
                         "symbol list")

    rs = sub.add_parser("research", help="per-symbol research (SLOW)")
    rs.add_argument("symbols", nargs="+")
    rs.add_argument("--out", default="configs/research")
    rs.add_argument("--kinds", default=",".join(RESEARCH_KINDS))
    rs.add_argument("--no-xbrl", action="store_true",
                    help="skip the results documents - measured 26h vs 2.2h "
                         "for a full-universe pass")
    rs.add_argument("--since", default=None, help="YYYY-MM-DD")
    rs.add_argument("--force", action="store_true")

    args = p.parse_args(argv)

    # Handled before any session exists: this command reads only local files,
    # and building a throttled HTTP client for it would imply otherwise.
    if args.cmd == "fundamentals":
        return _cmd_fundamentals(args)

    session = NseSession()

    if args.cmd == "events":
        from desk.research.events import save_calendar
        from desk.research.sources.nse import parse_event_calendar
        try:
            events = parse_event_calendar(
                session.fetch_with_retry(session.fetch_event_calendar))
        except RateLimited as exc:
            print(f"NSE is rate-limiting: {exc}")
            return 2
        except SourceError as exc:
            print(f"could not fetch the event calendar: {exc}")
            return 1
        n = save_calendar(events, Path(args.out))
        print(f"wrote {n} dated event(s) to {args.out}")
        if not n:
            # An empty calendar parses, saves, and then answers "nothing
            # scheduled" for the entire market. It must not look like success.
            print("WARNING: the calendar is EMPTY. The earnings gate will "
                  "clear every name. Check the endpoint before relying on "
                  "today's plan.")
            return 1
        return 0

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    unknown = [k for k in kinds if k not in RESEARCH_KINDS]
    if unknown:
        print(f"unknown kind(s): {unknown}. Known: {list(RESEARCH_KINDS)}")
        return 1
    try:
        since = date.fromisoformat(args.since) if args.since else None
    except ValueError:
        print(f"--since must be YYYY-MM-DD, got {args.since!r}")
        return 1

    try:
        # This is an EXIT CODE, not a record count. refresh_research already
        # prints its own per-symbol summary, and it returns 1 when any symbol
        # failed and 2 when NSE rate-limited. An earlier version of this
        # function read it as a count, printed "refreshed 0 record(s)" after
        # storing 35 filings, and - the part that mattered - returned 0
        # regardless, so a scheduled run reported success on a run where
        # symbols had failed. That is exactly what refresh_research's own
        # docstring says the non-zero exit exists to prevent.
        return refresh_research(session, list(args.symbols), Path(args.out),
                                kinds=kinds, with_xbrl=not args.no_xbrl,
                                since=since, force=args.force)
    except RateLimited as exc:
        # Exit 2, distinct from a real failure: "come back later" and
        # "something is broken" need different responses from a scheduler.
        print(f"NSE is rate-limiting, stopping early: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
