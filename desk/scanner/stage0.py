"""Stage 0 of the daily scanner funnel: universe hygiene.

Deterministic. No AI. Runs on ONE day's bhavcopy snapshot and narrows the
full listed universe down to names worth spending Stage 1's per-symbol
indicator computation on. In the master plan's worked example this is the
500 -> 380 step.

Stage 1 (relative volume, ATR percentile, distance from moving averages,
relative-strength trend) needs multi-day HISTORY, which this project does
not accumulate yet - desk/store/ is still an empty package, and deciding its
Parquet layout is its own design task. Stage 0 deliberately does not need
that: every filter here is computable from a single day's snapshot alone,
which is why it is buildable and useful today while Stage 1 is not.

Two filters the master plan names for this stage are NOT implemented, and
said so explicitly rather than silently skipped - see Stage0Result.caveats:
  - F&O ban list exclusion (no endpoint found for it yet)
  - Point-in-time index membership (equity-stockIndices 404s under the name
    tried; NSE has evidently renamed or moved it)
A name that should have been excluded by either is not caught here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from desk.marketdata.sources.nse import bhavcopy_equity_only

CAVEATS = (
    "F&O ban list not checked - no endpoint found yet for it.",
    "Point-in-time index membership not checked - equity-stockIndices 404s "
    "under the name tried; a name that should be excluded on this basis is "
    "not caught here.",
)


@dataclass(slots=True)
class Stage0Result:
    as_of: date
    universe_in: int
    universe_out: int
    survivors: pd.DataFrame
    excluded: dict[str, int] = field(default_factory=dict)
    """Reason -> count. A name can be counted under more than one reason if
    it fails more than one filter - this is a diagnostic breakdown, not a
    partition, and the reasons deliberately overlap rather than picking one
    arbitrary "first" reason to report."""
    caveats: list[str] = field(default_factory=lambda: list(CAVEATS))

    def summary(self) -> str:
        lines = [f"Stage 0: {self.universe_in} -> {self.universe_out} "
                 f"({self.as_of})"]
        for reason, n in sorted(self.excluded.items(), key=lambda kv: -kv[1]):
            lines.append(f"  -{n:5} {reason}")
        for c in self.caveats:
            lines.append(f"  CAVEAT: {c}")
        return "\n".join(lines)


def run_stage0(
    bhavcopy: pd.DataFrame,
    *,
    min_price: float = 20.0,
    min_turnover_lacs: float = 100.0,
    exclude_circuit_locked: bool = False,
) -> Stage0Result:
    """Narrow a bhavcopy snapshot to a hygienic tradeable universe.

    `min_price` and `min_turnover_lacs` are YOUR numbers, same philosophy as
    RiskConfig - defaults are a starting point, not a recommendation.
    100 lacs (1 crore) of daily turnover is a conservative retail-liquidity
    floor; 20 INR excludes the thinnest penny-stock tier without being
    aggressive about it.

    `exclude_circuit_locked` defaults to False deliberately: a stock frozen
    at its circuit (open == high == low == close, or high == low with real
    volume) is not tradeable IN THE CIRCUIT'S DIRECTION today, but it is
    exactly the kind of name a momentum scanner wants to know exists - "NO
    TRADE today" and "not interesting" are different facts, and this project
    treats NO TRADE as a first-class output rather than something to hide.
    Circuit-locked names are always flagged in the `circuit_locked` column
    of `survivors`; this flag only controls whether they are also dropped.
    """
    if bhavcopy.empty:
        raise ValueError("bhavcopy is empty - nothing to scan")

    required = {"symbol", "series", "date", "close", "high", "low", "volume",
               "turnover_lacs"}
    missing = required - set(bhavcopy.columns)
    if missing:
        raise ValueError(f"bhavcopy missing required columns: {sorted(missing)}")

    as_of = bhavcopy["date"].iloc[0]
    universe_in = len(bhavcopy)

    df = bhavcopy_equity_only(bhavcopy).copy()
    excluded = {"non-EQ series": universe_in - len(df)}

    # A stock frozen at one price all day - either circuit-locked or
    # genuinely untraded. Flagged always, dropped only if asked.
    df["circuit_locked"] = (df["high"] == df["low"]) & (df["volume"] > 0)

    zero_volume = df["volume"] <= 0
    excluded["zero volume (suspended or untraded)"] = int(zero_volume.sum())
    df = df[~zero_volume]

    below_price = df["close"] < min_price
    excluded["below price floor"] = int(below_price.sum())
    df = df[~below_price]

    below_turnover = df["turnover_lacs"] < min_turnover_lacs
    excluded["below liquidity floor"] = int(below_turnover.sum())
    df = df[~below_turnover]

    if exclude_circuit_locked:
        locked = df["circuit_locked"]
        excluded["circuit-locked"] = int(locked.sum())
        df = df[~locked]

    survivors = df.reset_index(drop=True)
    return Stage0Result(
        as_of=as_of, universe_in=universe_in, universe_out=len(survivors),
        survivors=survivors, excluded=excluded,
    )
