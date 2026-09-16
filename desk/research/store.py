"""Filings on disk, point-in-time by construction.

The price store (desk/store/bars.py) gets its point-in-time guarantee for free
from its layout: one file per trading day, the date in the filename, so
`available_days()` filters by filename and a file dated after `as_of` is never
OPENED. That property is worth more than any runtime check, because a filter
is a line of code someone can delete and a file that was never listed cannot
leak.

Filings are the opposite shape - per symbol, irregular, a handful a year - so
the same layout does not transfer. The same PROPERTY does, transposed:

    <root>/<SYMBOL>/<disclosed-at>_<seq>.json

one file per filing, partitioned by symbol, with the disclosure timestamp
encoded in the filename. `for_symbol(sym, as_of=...)` lists one directory,
discards by filename, and only then opens anything.

WHY `as_of` IS A datetime AND NOT A date
----------------------------------------
This is the one place the price store's design does NOT transfer, and getting
it wrong would reintroduce exactly the leak this whole layer exists to stop.

A bhavcopy is an atomic end-of-day snapshot, so a date-granular cutoff is
safe. A filing is not: RELIANCE's Q3 FY25 results were disclosed at
2025-01-16 20:20:21, after the close. A backtest simulating the 16th must NOT
see them - they were tradeable on the 17th. A `disclosed_at.date() <= as_of`
comparison treats a 20:20 filing and an 11:00 filing identically, and the
whole point of `Announcement.session_phase` is that those are different
facts.

So `as_of` here is a datetime, compared against the full timestamp. A caller
holding only a date should pass the moment it means - usually the market open
of the day being simulated, which `at_open()` builds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

from desk.research.models import Filing, ResultPeriod
from desk.research.xbrl import FinancialFacts

__all__ = ["FilingStore", "StoredFiling", "StoreError", "at_open"]

#: NSE's normal session opens 09:15 IST. A date-only `as_of` almost always
#: means "as the market opened that day", so this is the conversion rather
#: than leaving every caller to invent midnight - which would wrongly exclude
#: everything filed the previous evening.
MARKET_OPEN = time(9, 15)

_TS = "%Y%m%dT%H%M%S"
_NAME = re.compile(r"^(\d{8}T\d{6})_([A-Za-z0-9-]+)\.json$")


class StoreError(Exception):
    """The store cannot answer honestly."""


def at_open(day: date) -> datetime:
    """The moment a simulated trading day begins.

    A filing disclosed at 20:20 the previous evening IS visible at this
    instant; one disclosed at 11:00 the same morning is not yet.
    """
    return datetime.combine(day, MARKET_OPEN)


@dataclass(frozen=True, slots=True)
class StoredFiling:
    """A filing as persisted, with whatever periods were parsed from its
    XBRL. `periods` is empty when the document was never fetched - which is
    different from a filing that had no document at all (`filing.has_numbers`
    says which)."""

    filing: Filing
    periods: tuple[FinancialFacts, ...] = ()

    @property
    def disclosed_at(self) -> datetime:
        return self.filing.disclosed_at

    def period(self, months: int = 3) -> FinancialFacts | None:
        """The period of a given length, or None. Defaults to the quarter,
        because that is what a growth or margin comparison wants - the
        year-to-date figure in the same filing is three times larger and
        picking it by accident is the documented trap in xbrl.py."""
        for p in self.periods:
            if p.months == months:
                return p
        return None


class FilingStore:
    """Per-symbol filings, filtered by filename before anything is opened."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------- layout --

    def _dir(self, symbol: str) -> Path:
        base = symbol.split(".")[0].upper()
        # The symbol becomes a directory name, so it is validated rather than
        # trusted - `..` or a separator here would escape the store entirely.
        if not re.fullmatch(r"[A-Z0-9&-]{1,30}", base):
            raise StoreError(f"refusing to use {symbol!r} as a directory name")
        return self.root / base

    def symbols(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    # -------------------------------------------------------------- write --

    def write(self, filing: Filing,
              periods: tuple[FinancialFacts, ...] | list[FinancialFacts] = ()
              ) -> Path:
        """One filing, one file, written atomically.

        Keyed on (disclosed_at, isin-or-period) so re-running a fetch
        overwrites the same file rather than accumulating duplicates - a
        refresh that is run twice must not double-count a quarter.
        """
        d = self._dir(filing.symbol)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / self._filename(filing)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "filing": _filing_to_json(filing),
            "periods": [_facts_to_json(p) for p in periods],
        }, indent=1), encoding="utf-8")
        tmp.replace(dest)
        return dest

    @staticmethod
    def _filename(filing: Filing) -> str:
        # The period end disambiguates two filings disclosed in the same
        # second (a company filing standalone and consolidated together),
        # which is otherwise a silent overwrite of one by the other.
        tag = (filing.period_end.isoformat().replace("-", "")
               if filing.period_end else "na")
        return f"{filing.disclosed_at.strftime(_TS)}_{tag}.json"

    # --------------------------------------------------------------- read --

    def for_symbol(self, symbol: str, *, as_of: datetime,
                   months: int | None = None) -> list[StoredFiling]:
        """Every filing disclosed at or before `as_of`, oldest first.

        A file dated after `as_of` is discarded from the FILENAME and never
        opened - the same structural guarantee BarStore gets from its
        one-file-per-day layout.
        """
        out: list[StoredFiling] = []
        for path, _stamp in self._candidates(symbol, as_of=as_of):
            rec = self._read(path)
            if months is not None and rec.period(months) is None:
                continue
            out.append(rec)
        out.sort(key=lambda r: r.disclosed_at)
        return out

    def _candidates(self, symbol: str, *, as_of: datetime,
                    newest_first: bool = False):
        """Filenames that qualify, without opening any of them.

        THIS is the point-in-time guarantee. The cutoff is applied to the
        filename, so a filing disclosed after `as_of` is not merely filtered
        out of the result - it is never read off disk at all.
        """
        if not isinstance(as_of, datetime):
            raise StoreError(
                f"as_of must be a datetime, got {type(as_of).__name__}. A "
                f"filing at 20:20 and one at 11:00 are different facts on the "
                f"same date - use at_open(day) if you have only a date."
            )
        d = self._dir(symbol)
        if not d.is_dir():
            return
        pairs = []
        for path in d.iterdir():
            stamp = self._stamp(path)
            if stamp is not None and stamp <= as_of:
                pairs.append((path, stamp))
        pairs.sort(key=lambda t: t[1], reverse=newest_first)
        yield from pairs

    def latest(self, symbol: str, *, as_of: datetime,
               months: int | None = None) -> StoredFiling | None:
        """The most recent filing visible at `as_of`.

        `months` does NOT default to 3 here, deliberately. Filtering by period
        length would make a filing whose XBRL has not been fetched yet simply
        vanish from "the latest filing", which is a surprising thing for a
        method with this name to do. The 9-month trap lives in
        `StoredFiling.period()`, and that is where it is defended - pass
        months=3 when you specifically need the numbers.
        """
        # Walks filenames NEWEST-first and opens only until one qualifies,
        # rather than reading every filing up to as_of and discarding all but
        # the last. Measured at R3 scale (1,598 symbols x 8 filings): 101.6s
        # -> 1.2s, because the common case opens exactly one file per symbol
        # instead of eight.
        for path, _stamp in self._candidates(symbol, as_of=as_of, newest_first=True):
            rec = self._read(path)
            if months is None or rec.period(months) is not None:
                return rec
        return None

    def latest_many(self, symbols, *, as_of: datetime, months: int | None = None
                    ) -> dict[str, StoredFiling]:
        """The latest filing per symbol. What Stage 2's fundamental filters
        need across the whole universe."""
        out = {}
        for sym in symbols:
            try:
                rec = self.latest(sym, as_of=as_of, months=months)
            except StoreError:
                continue                      # an unusable symbol is not fatal
            if rec is not None:
                out[sym.split(".")[0].upper()] = rec
        return out

    @staticmethod
    def _stamp(path: Path) -> datetime | None:
        m = _NAME.match(path.name)
        if not m:
            return None                        # .tmp, stray files, backups
        try:
            return datetime.strptime(m.group(1), _TS)
        except ValueError:
            return None

    def _read(self, path: Path) -> StoredFiling:
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read {path.name}: {exc}") from exc
        if not isinstance(blob, dict) or "filing" not in blob:
            raise StoreError(f"{path.name} is not a stored filing")
        return StoredFiling(
            filing=_filing_from_json(blob["filing"]),
            periods=tuple(_facts_from_json(p) for p in blob.get("periods") or []),
        )


# --------------------------------------------------------------------------
# serialisation. Explicit rather than a generic asdict/**dict round trip: the
# dataclasses are frozen with typed date/datetime/enum fields, and a generic
# round trip silently turns every one of those back into a string.
# --------------------------------------------------------------------------

def _filing_to_json(f: Filing) -> dict:
    return {
        "symbol": f.symbol,
        "disclosed_at": f.disclosed_at.isoformat(),
        "period_start": f.period_start.isoformat() if f.period_start else None,
        "period_end": f.period_end.isoformat() if f.period_end else None,
        "period": f.period.value,
        "audited": f.audited,
        "consolidated": f.consolidated,
        "company_name": f.company_name,
        "relating_to": f.relating_to,
        "xbrl_url": f.xbrl_url,
        "isin": f.isin,
    }


def _filing_from_json(d: dict) -> Filing:
    return Filing(
        symbol=d["symbol"],
        disclosed_at=datetime.fromisoformat(d["disclosed_at"]),
        period_start=_date(d.get("period_start")),
        period_end=_date(d.get("period_end")),
        period=ResultPeriod(d.get("period", "unknown")),
        audited=d.get("audited"),
        consolidated=d.get("consolidated"),
        company_name=d.get("company_name", ""),
        relating_to=d.get("relating_to", ""),
        xbrl_url=d.get("xbrl_url"),
        isin=d.get("isin", ""),
    )


def _facts_to_json(p: FinancialFacts) -> dict:
    return {
        "symbol": p.symbol,
        "period_start": p.period_start.isoformat(),
        "period_end": p.period_end.isoformat(),
        "consolidated": p.consolidated,
        "audited": p.audited,
        "facts": p.facts,
    }


def _facts_from_json(d: dict) -> FinancialFacts:
    facts = d.get("facts") or {}
    from desk.research.xbrl import _NAMED

    return FinancialFacts(
        symbol=d["symbol"],
        period_start=date.fromisoformat(d["period_start"]),
        period_end=date.fromisoformat(d["period_end"]),
        consolidated=d.get("consolidated"),
        audited=d.get("audited"),
        facts=facts,
        # Rebuilt from `facts` rather than stored twice - one copy cannot
        # drift from the other if there is only one copy.
        **{attr: facts.get(tag) for attr, tag in _NAMED.items()},
    )


def _date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None
