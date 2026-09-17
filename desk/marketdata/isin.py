"""ISIN: the only sound key for joining one exchange to another.

WHY A WHOLE MODULE FOR AN IDENTIFIER
------------------------------------
NSE's daily bhavcopy (sec_bhavdata_full) has NO ISIN column - SYMBOL, SERIES,
prices, volumes, nothing else. BSE's file HAS one. So a cross-exchange join
cannot be done from the two price files alone; the mapping has to come from
NSE's equity master, and that is a separate fetch with its own lifetime.

WHY NOT JUST MATCH ON THE TICKER
--------------------------------
It looks like it would work. On 2026-09-11, all 2,397 names that joined by
ISIN also had identical NSE and BSE tickers - 100% agreement.

That number is misleading and should not be used to justify dropping ISIN.
It is 100% agreement AMONG THE NAMES THAT ALREADY JOINED BY ISIN: any company
whose tickers disagree is absent from that population by construction, so the
measurement cannot see the cases it would need to see. Ticker agreement is
also a convention, not a guarantee - the two exchanges assign their own codes
and nothing obliges them to match, whereas an ISIN identifies the SECURITY.
Matching the wrong company's close against ours would not raise anything; it
would just quietly report a disagreement, or quietly fail to report a real
one.

MEASURED, on NSE's EQUITY_L.csv (2,577 rows, fetched 2026-09-18): no
duplicate symbols, no duplicate ISINs, every ISIN matching ^IN[A-Z0-9]{10}$.
The duplicate handling below is therefore defensive rather than responding to
current data - historically, DVR (differential voting rights) lines did share
economics with their ordinary counterpart, and an ambiguous key is exactly
the kind of thing that silently attaches the wrong price.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from desk.marketdata.sources.errors import SourceError

__all__ = ["IsinMap", "ISIN_RE", "parse_equity_master"]

#: Two-letter country prefix plus a nine-character NSIN and a check digit.
#: Indian securities are all IN-prefixed.
ISIN_RE = re.compile(r"^IN[A-Z0-9]{10}$")

_REQUIRED = ("SYMBOL", "ISIN NUMBER")


@dataclass(frozen=True, slots=True)
class IsinMap:
    """A symbol <-> ISIN mapping that refuses to answer when it is unsure."""

    by_symbol: dict[str, str] = field(default_factory=dict)
    by_isin: dict[str, str] = field(default_factory=dict)
    ambiguous: tuple[str, ...] = ()
    """ISINs claimed by more than one NSE symbol. Excluded from `by_isin`
    entirely rather than resolved by picking one, because picking one is how
    a join silently attaches another security's price. They stay listed so a
    caller can report them instead of wondering where the rows went."""
    rejected: tuple[str, ...] = ()
    """Rows dropped because the ISIN was not well-formed."""
    as_of: date | None = None

    def isin_for(self, symbol: str) -> str | None:
        """ISIN for an NSE symbol. Accepts 'RELIANCE' or 'RELIANCE.NS'."""
        return self.by_symbol.get(symbol.split(".")[0].strip().upper())

    def symbol_for(self, isin: str) -> str | None:
        """NSE symbol for an ISIN, or None if unknown OR ambiguous."""
        return self.by_isin.get(isin.strip().upper())

    def __len__(self) -> int:
        return len(self.by_symbol)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write atomically. A half-written map read by tomorrow's pre-open
        run would produce a partial join reported as a real coverage figure.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "by_symbol": self.by_symbol,
            "ambiguous": list(self.ambiguous),
            "rejected": list(self.rejected),
        }
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "IsinMap":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        by_symbol = {str(k).upper(): str(v).upper()
                     for k, v in raw.get("by_symbol", {}).items()}
        ambiguous = tuple(raw.get("ambiguous", ()))
        return cls(
            by_symbol=by_symbol,
            by_isin=_invert(by_symbol, set(ambiguous)),
            ambiguous=ambiguous,
            rejected=tuple(raw.get("rejected", ())),
            as_of=date.fromisoformat(raw["as_of"]) if raw.get("as_of") else None,
        )


def parse_equity_master(raw: bytes, *, as_of: date | None = None) -> IsinMap:
    """NSE's EQUITY_L.csv -> an IsinMap.

    Every series in the file is kept, not just EQ. The cross-check is about
    whether two exchanges agree on a price, and a BE- or BZ-series name has
    exactly the same claim on being checked - Stage 0 decides tradability,
    and that is a different question answered elsewhere.
    """
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise SourceError(f"equity master missing expected columns: {missing}")
    if df.empty:
        raise SourceError("equity master contained no rows")

    symbols = df["SYMBOL"].astype(str).str.strip().str.upper()
    isins = df["ISIN NUMBER"].astype(str).str.strip().str.upper()

    by_symbol: dict[str, str] = {}
    rejected: list[str] = []
    for sym, isin in zip(symbols, isins):
        if not sym or not ISIN_RE.match(isin):
            rejected.append(f"{sym}={isin}")
            continue
        by_symbol[sym] = isin

    if not by_symbol:
        raise SourceError(
            "equity master parsed to zero usable rows - the format has "
            "changed, or the response was not the master file")

    counts: dict[str, int] = {}
    for isin in by_symbol.values():
        counts[isin] = counts.get(isin, 0) + 1
    ambiguous = tuple(sorted(i for i, n in counts.items() if n > 1))

    return IsinMap(by_symbol=by_symbol,
                   by_isin=_invert(by_symbol, set(ambiguous)),
                   ambiguous=ambiguous, rejected=tuple(rejected), as_of=as_of)


def _invert(by_symbol: dict[str, str], ambiguous: set[str]) -> dict[str, str]:
    return {isin: sym for sym, isin in by_symbol.items()
            if isin not in ambiguous}
