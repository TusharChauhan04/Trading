"""Which industry a symbol belongs to, and which index it sits in.

WHY THIS WAS MISSING FOR SO LONG
--------------------------------
Three separate caveats have been shipping on every plan since the scanner
was built:

    "Sector strength not computed - no sector classification in bhavcopy
     and no sector map in the project."
    "Sector caps are not enforced per name - bhavcopy carries no sector
     classification, so every setup is sized as sector UNKNOWN."
    "Point-in-time index membership not checked."

All three are the same missing fact, and it turns out NSE publishes it for
free in a place nobody had looked: the index constituent CSVs at
nsearchives.nseindia.com/content/indices/ind_nifty200list.csv carry
`Company Name, Industry, Symbol, Series, ISIN Code`. 200 names, 18
industries, one request. The same path serves nifty50, nifty100, nifty500
and the sectoral indices.

WHAT THIS IS NOT: POINT-IN-TIME
--------------------------------
NSE publishes the CURRENT constituent list. There is no archive of "who was
in the Nifty 200 in March", so a historical scan using today's list has
SURVIVORSHIP BIAS - a name added after a big run appears in the backtest as
though it had always been eligible, and a name dropped for collapsing
disappears.

That is a real limitation and it is NOT fixed by this module. What is done
instead is that every snapshot records `fetched_at`, and `SectorMap.as_of`
carries it, so a caller can say which vintage it used and a backtest can
report the bias rather than inherit it silently. The honest use is: sector
CAPS and sector STRENGTH on a live plan, where today's membership is the
right membership. Treat index membership in a historical replay as
approximate and say so.

Industry strings are NSE's own - "Financial Services", "Information
Technology", "Oil Gas & Consumable Fuels" - and are kept verbatim rather
than mapped to a tidier taxonomy. A translation layer is one more thing to
get wrong, and the exchange's own label is what a user will recognise.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

__all__ = ["INDEX_FILES", "SectorMap", "parse_constituents"]

#: The constituent files worth having. Each is ONE request and covers its
#: whole index; nifty500 alone classifies most of the tradeable universe.
INDEX_FILES: dict[str, str] = {
    "NIFTY 50": "ind_nifty50list.csv",
    "NIFTY 100": "ind_nifty100list.csv",
    "NIFTY 200": "ind_nifty200list.csv",
    "NIFTY 500": "ind_nifty500list.csv",
    "NIFTY MIDCAP 150": "ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "ind_niftysmallcap250list.csv",
}

_REQUIRED = ("Symbol", "Industry")


@dataclass(frozen=True, slots=True)
class SectorMap:
    """symbol -> industry, plus index membership, plus a vintage."""

    industry: dict[str, str] = field(default_factory=dict)
    indices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """symbol -> the indices it appears in, e.g. ("NIFTY 50", "NIFTY 100")."""
    as_of: date | None = None
    """When the underlying lists were FETCHED. Not a point-in-time
    membership date - see the module docstring. Present so a caller can
    state the vintage instead of implying there is none."""

    def sector_for(self, symbol: str) -> str | None:
        return self.industry.get(_base(symbol))

    def indices_for(self, symbol: str) -> tuple[str, ...]:
        return self.indices.get(_base(symbol), ())

    def in_index(self, symbol: str, index: str) -> bool:
        return index in self.indices_for(symbol)

    def symbols_in(self, index: str) -> list[str]:
        return sorted(s for s, ix in self.indices.items() if index in ix)

    def sectors(self) -> list[str]:
        return sorted(set(self.industry.values()))

    def coverage(self, symbols) -> tuple[int, int]:
        """(classified, asked). The honest counterpart to sector_for
        returning None: a sector cap that could only see a third of the
        book is not a sector cap, and the caller has to be able to say so.
        """
        asked = [_base(s) for s in symbols]
        return sum(1 for s in asked if s in self.industry), len(asked)

    def __len__(self) -> int:
        return len(self.industry)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "industry": self.industry,
            "indices": {k: list(v) for k, v in self.indices.items()},
        }, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "SectorMap | None":
        path = Path(path)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return cls(
            industry={str(k).upper(): str(v)
                      for k, v in raw.get("industry", {}).items()},
            indices={str(k).upper(): tuple(v)
                     for k, v in raw.get("indices", {}).items()},
            as_of=(date.fromisoformat(raw["as_of"])
                   if raw.get("as_of") else None),
        )

    def merged_with(self, other: "SectorMap") -> "SectorMap":
        """Combine two index files into one map.

        On a conflicting industry the EXISTING value wins, because the maps
        are merged narrowest-first: NIFTY 50's label for RELIANCE and NIFTY
        500's should agree, and if they ever disagree the smaller, more
        curated list is the better source.
        """
        industry = dict(self.industry)
        for sym, ind in other.industry.items():
            industry.setdefault(sym, ind)
        indices: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in self.indices.items()}
        for sym, ix in other.indices.items():
            indices[sym] = tuple(dict.fromkeys(indices.get(sym, ()) + ix))
        return SectorMap(industry=industry, indices=indices,
                         as_of=self.as_of or other.as_of)


def parse_constituents(raw: bytes, *, index: str,
                       as_of: date | None = None) -> SectorMap:
    """One NSE index constituent CSV -> a SectorMap.

    utf-8-sig because NSE ships these with a BOM, which turns the first
    header into "\\ufeffCompany Name" and makes a plain DictReader silently
    miss the column it is looking for.
    """
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{index}: constituent file has no rows")

    headers = {h.strip() for h in (rows[0].keys() or ()) if h}
    missing = [c for c in _REQUIRED if c not in headers]
    if missing:
        raise ValueError(
            f"{index}: constituent file missing {missing}; got "
            f"{sorted(headers)}")

    industry: dict[str, str] = {}
    for row in rows:
        sym = (row.get("Symbol") or "").strip().upper()
        ind = (row.get("Industry") or "").strip()
        if not sym or not ind:
            continue
        industry[sym] = ind

    if not industry:
        raise ValueError(f"{index}: no usable rows - format has changed?")

    return SectorMap(industry=industry,
                     indices={s: (index,) for s in industry},
                     as_of=as_of or datetime.now().date())


def _base(symbol: str) -> str:
    return str(symbol).split(".")[0].strip().upper()
