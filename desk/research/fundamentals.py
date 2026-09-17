"""Per-symbol fundamental metrics, point-in-time, for the whole universe.

This is what Stage 2's fundamental filters read. It turns the filings store
into one row per symbol: the latest known quarter, its year-ago comparison,
and the derived growth and margin numbers.

THE TRAP THIS MODULE EXISTS TO AVOID
------------------------------------
A company files STANDALONE and CONSOLIDATED results for the same quarter, and
for a holding company they are wildly different numbers. Measured, RELIANCE
Q3 FY25:

    standalone     Rs 128,260 crore
    consolidated   Rs 243,865 crore        1.9x

Compare consolidated-this-year against standalone-last-year and you get
+87% revenue growth. The real standalone growth was -1.8%. Nothing about the
output looks wrong; it is simply a different company's worth of revenue.

It is not a hypothetical mistake either. IRCTC filed standalone only until
2024-05 and both after, so a naive "prefer consolidated, fall back to
standalone" rule does EXACTLY this comparison on real data.

So: one nature is chosen per symbol per comparison, and it must be present in
BOTH the current and the year-ago period or no growth figure is produced.
`nature` is reported on every row so a caller can see which was used.

POINT-IN-TIME
-------------
Every lookup goes through FilingStore with a datetime `as_of`, so a filing
disclosed after the simulated moment is never read. See store.py.

WHY THIS IS A BATCH, NOT A PER-REQUEST CALL
-------------------------------------------
`FilingStore.latest_many` measured 11.8s for 1,598 symbols - it is one
directory listing plus a file read per symbol, inherently O(universe)
filesystem work. The API layer is stateless and recomputes the funnel on
every call, so calling this inline would add ~12s to every /plan/today.
`build_table` is therefore meant to run once pre-open and be cached to
parquet, the same way the bhavcopy snapshot is fetched once rather than per
request.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from desk.research.store import FilingStore, StoredFiling, at_open

__all__ = ["COLUMNS", "build_table", "latest_comparable"]

#: One row per symbol. Named explicitly so Stage 2 can assert on the shape and
#: a downstream factor cannot silently depend on a column a refactor renames.
COLUMNS = (
    "nature", "period_end", "disclosed_at", "days_since_filing",
    "revenue", "profit_after_tax", "eps_basic", "net_margin_pct",
    "revenue_growth_yoy_pct", "profit_growth_yoy_pct", "margin_change_pp",
)

#: How far either side of "one year ago" a filing may sit and still count as
#: the year-ago comparison. Quarters drift by days, and NSE's period_end is
#: the real accounting date - 365 +/- 45 catches the right quarter without
#: reaching the adjacent one, which is 91 days away.
_YEAR_AGO_TOLERANCE = timedelta(days=45)


def latest_comparable(recs: list[StoredFiling]
                      ) -> tuple[object | None, object | None, str]:
    """(current quarter, year-ago quarter, nature) from one symbol's filings.

    Returns the most recent quarter for which a SAME-NATURE year-ago quarter
    also exists, preferring consolidated when both natures qualify. When no
    matched pair exists, the current quarter is still returned with a None
    comparison - a company's latest margin is usable even when its growth is
    not.
    """
    quarters = [(r, r.period(3)) for r in recs]
    quarters = [(r, q) for r, q in quarters if q is not None and q.period_end]
    if not quarters:
        return None, None, "none"

    by_nature: dict[str, list] = {"consolidated": [], "standalone": []}
    for r, q in quarters:
        if q.consolidated is True:
            by_nature["consolidated"].append((r, q))
        elif q.consolidated is False:
            by_nature["standalone"].append((r, q))

    best_current = None
    # Consolidated first: for a holding company it is the real business.
    for nature in ("consolidated", "standalone"):
        rows = sorted(by_nature[nature], key=lambda t: t[1].period_end)
        if not rows:
            continue
        current = rows[-1]
        if best_current is None:
            best_current = (current[1], None, nature)
        target = current[1].period_end - timedelta(days=365)
        for _r, q in rows[:-1]:
            if abs(q.period_end - target) <= _YEAR_AGO_TOLERANCE:
                return current[1], q, nature
    # No same-nature pair anywhere. Return the latest quarter uncompared
    # rather than nothing - and never pair across natures.
    if best_current:
        return best_current
    return None, None, "none"


def build_table(store: FilingStore, symbols, *, as_of: datetime | date
                ) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per symbol with fundamentals known at `as_of`.

    Returns (frame, coverage). `coverage` counts why symbols are missing,
    because "no filing on file" and "filed but no comparable year-ago
    quarter" are different facts and a filter must be able to tell them
    apart - Stage 2's whole reporting convention depends on it.
    """
    if not isinstance(as_of, datetime):
        as_of = at_open(as_of)

    rows: dict[str, dict] = {}
    coverage = {"no filings on file": 0, "no quarterly numbers": 0,
                "no year-ago comparison": 0, "complete": 0}

    for raw in symbols:
        base = raw.split(".")[0].upper()
        try:
            recs = store.for_symbol(base, as_of=as_of, months=3)
        except Exception:
            coverage["no filings on file"] += 1
            continue
        if not recs:
            coverage["no filings on file"] += 1
            continue

        current, prior, nature = latest_comparable(recs)
        if current is None:
            coverage["no quarterly numbers"] += 1
            continue

        row = {
            "nature": nature,
            "period_end": current.period_end,
            "disclosed_at": max(r.disclosed_at for r in recs),
            "revenue": current.revenue,
            "profit_after_tax": current.profit_after_tax,
            "eps_basic": current.eps_basic,
            "net_margin_pct": current.net_margin_pct,
            "revenue_growth_yoy_pct": _growth(current.revenue,
                                              prior.revenue if prior else None),
            "profit_growth_yoy_pct": _growth(
                current.profit_after_tax,
                prior.profit_after_tax if prior else None),
            "margin_change_pp": _delta(
                current.net_margin_pct,
                prior.net_margin_pct if prior else None),
        }
        row["days_since_filing"] = (as_of.date() - row["disclosed_at"].date()).days
        rows[base] = row
        coverage["complete" if prior is not None
                 else "no year-ago comparison"] += 1

    frame = pd.DataFrame.from_dict(rows, orient="index",
                                   columns=list(COLUMNS))
    frame.index.name = "symbol"
    return frame, coverage


def _growth(now, before) -> float | None:
    """Percent change, or None.

    A negative or zero base is refused rather than divided by: a company that
    swung from a loss to a profit has no meaningful "percent growth", and the
    number you get from the arithmetic is a sign-flipped artefact that reads
    as a spectacular result.
    """
    if now is None or before is None or before <= 0:
        return None
    return 100.0 * (now - before) / before


def _delta(now, before) -> float | None:
    if now is None or before is None:
        return None
    return now - before
