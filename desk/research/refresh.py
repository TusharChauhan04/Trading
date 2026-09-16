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
from datetime import datetime
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
                                         filings_store, with_xbrl=with_xbrl)
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
                    filings_store: "FilingStore", *, with_xbrl: bool):
    """One endpoint for one symbol. Returns (records written, undated)."""
    if kind == "filings":
        filings, undated = parse_results(session.fetch_results(base), base)
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
    dest = out_dir / kind / f"{base}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "symbol": f"{base}.NS",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
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

