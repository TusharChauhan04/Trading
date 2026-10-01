"""Market cap, P/E and earnings yield, from a trailing-twelve-month base.

WHY THIS COMPLETES SOMETHING THAT WAS LEFT HALF DONE
----------------------------------------------------
`desk/research/derived.py` found `shares_outstanding` for 186 of 186 symbols -
paid-up capital over face value - and its docstring says that field "unlocks
everything price-based: market cap, P/E, earnings yield, because the desk
already has close prices for the whole universe". That was true and nothing
acted on it. This is the acting on it.

It also makes OpenTerminalUI's `value` screen one field closer: that screen
wants `pe`, `roe_pct` and `debt_to_market_cap`, and `pe` is now available. The
other two still need a balance sheet.

TRAILING TWELVE MONTHS, NOT A QUARTER TIMES FOUR
-----------------------------------------------
Everything in `derived.py` is a single quarter, correctly labelled as such.
A P/E built on one quarter annualised is wrong for any business with
seasonality, which in India is most of them - a cement company's monsoon
quarter and a jeweller's festive quarter are not a quarter of their year.

So TTM here means FOUR ACTUAL CONSECUTIVE QUARTERS summed. Measured: 180 of 186
symbols have four or more quarters of the same nature on disk, so this is
available for almost the whole covered universe rather than a corner of it.

THREE TRAPS, EACH ONE MEASURED OR INHERITED
-------------------------------------------
1. NEVER MIX STANDALONE AND CONSOLIDATED. `fundamentals.py` documents this with
   real numbers: RELIANCE Q3 FY25 was Rs 128,260 crore standalone against
   Rs 243,865 crore consolidated. Summing across natures would produce a
   revenue figure belonging to no company. Every TTM here uses one nature and
   reports which.

2. A NEGATIVE P/E MUST BE None, NOT A NUMBER. A loss-making company has
   negative trailing earnings, so market cap over earnings is negative - and a
   negative number sorts BELOW a genuinely cheap profitable stock in an
   ascending screen. "P/E of -4" is not cheaper than "P/E of 8"; it is not a
   P/E at all. Earnings yield is still reported for those names, because
   negative there means exactly what it looks like.

3. SHARES OUTSTANDING COMES FROM THE LATEST FILING, never an average over the
   window. RELIANCE's count doubled on the October 2024 bonus; averaging across
   that would invent a company with 10 billion shares that never existed.

4. A STALE SHARE COUNT AGAINST A CURRENT PRICE IS THE DANGEROUS ONE, and the
   first version of this module shipped it. Trap 3 guards against averaging a
   count across a corporate action; this is the opposite direction and it is
   worse. `shares_outstanding` is point-in-time as of a filing 637 days old. If
   a split or bonus happened since, the count is pre-action and the price is
   post-action, and market cap is wrong by the action's whole ratio.

   MEASURED: BAJFINANCE came out at a TTM P/E of 3.75, which reads as a
   screaming bargain for a lender that normally trades near 25x. It split and
   issued a bonus in 2025, so a ~62 crore share count was being multiplied by a
   post-split price - understating market cap roughly tenfold and dividing the
   P/E by the same.

   The desk cannot correct this: `configs/corporate_actions/` holds three
   symbols (IRCTC, ITC, RELIANCE) of 186. So instead of correcting, it REFUSES:
   `find_discontinuities` flags any overnight gap beyond the 20% circuit band
   between the filing date and today, which is what an unrecorded split looks
   like, and every price-based field for that symbol becomes None with the
   reason recorded. A missing number is recoverable; a fake bargain at the top
   of a value screen is not.

WHAT STAYS STALE EVEN WHEN EVERYTHING ELSE IS RIGHT. Every filing on disk is
555-616 days old, because NSE's results endpoint returns nothing newer than Jan
2025. The PRICE is current and the EARNINGS are not, so a P/E computed here is
today's price over year-before-last's profit. That is a limitation of the data
rather than of the arithmetic, and it is reported on every row as
`earnings_age_days` so a caller cannot overlook it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from desk.research.store import FilingStore, StoreError, at_open

__all__ = ["TTM", "TTM_COLUMNS", "build_valuation_table", "trailing_twelve"]

#: How far from a perfect 91-day step two quarter-ends may sit and still count
#: as consecutive. Quarters drift by days and NSE reports the real accounting
#: date; 91 +/- 35 catches the neighbour without reaching the one beyond it,
#: which is 182 days away.
_STEP_TOLERANCE = timedelta(days=35)

TTM_COLUMNS = (
    "nature", "quarters", "period_end", "earnings_age_days",
    "revenue_ttm", "pat_ttm", "eps_ttm",
    "shares_outstanding", "close", "market_cap",
    "pe_ttm", "earnings_yield_pct", "ps_ttm",
    "share_count_suspect",
)

#: Widest common Indian circuit band. A cash-equity scrip cannot legitimately
#: gap further than this overnight, so anything beyond it with no corporate
#: action on file is an unrecorded split, bonus, or bad data.
_CIRCUIT_PCT = 20.0


@dataclass(frozen=True, slots=True)
class TTM:
    """Four consecutive same-nature quarters, summed.

    `quarters` is always exactly 4 when this object exists - a partial TTM is
    not returned, because summing three quarters and calling it a year
    understates earnings by a quarter and makes everything look expensive.
    """

    nature: str
    period_end: date
    quarters: int
    revenue_ttm: float | None
    pat_ttm: float | None
    eps_ttm: float | None
    shares_outstanding: float | None


def _consecutive(quarters: list, n: int = 4) -> list | None:
    """The `n` most recent quarters that actually step by about one quarter.

    Returns None rather than a best effort. A gap means a filing is missing
    from the store, and bridging it would sum a year that has a hole in it
    while reporting a full-year figure.
    """
    if len(quarters) < n:
        return None
    run = sorted(quarters, key=lambda q: q.period_end)[-n:]
    for earlier, later in zip(run, run[1:]):
        step = later.period_end - earlier.period_end
        if abs(step - timedelta(days=91)) > _STEP_TOLERANCE:
            return None
    return run


def trailing_twelve(recs: list, *, prefer: str = "consolidated"
                    ) -> TTM | None:
    """Four consecutive same-nature quarters from one symbol's filings.

    Consolidated first when both natures qualify: for a holding company that is
    the real business. Falls back to standalone, and NEVER mixes - see the
    module docstring for the RELIANCE numbers that make this non-optional.
    """
    by_nature: dict[str, list] = {"consolidated": [], "standalone": []}
    for r in recs:
        q = r.period(3)
        if q is None or not q.period_end:
            continue
        if q.consolidated is True:
            by_nature["consolidated"].append(q)
        elif q.consolidated is False:
            by_nature["standalone"].append(q)

    order = ([prefer] + [n for n in by_nature if n != prefer]
             if prefer in by_nature else list(by_nature))
    for nature in order:
        run = _consecutive(by_nature[nature])
        if run is None:
            continue

        def _sum(attr: str) -> float | None:
            vals = [getattr(q, attr, None) for q in run]
            if any(v is None or not math.isfinite(float(v)) for v in vals):
                return None
            return float(sum(float(v) for v in vals))

        latest = run[-1]
        # From the LATEST filing only. A bonus or split inside the window makes
        # an averaged count describe a company that never existed.
        fv = getattr(latest, "face_value", None)
        pu = getattr(latest, "paid_up_capital", None)
        shares = None
        if fv and pu and math.isfinite(float(fv)) and float(fv) > 0:
            shares = float(pu) / float(fv)

        return TTM(nature=nature, period_end=latest.period_end, quarters=4,
                   revenue_ttm=_sum("revenue"), pat_ttm=_sum("profit_after_tax"),
                   eps_ttm=_sum("eps_basic"), shares_outstanding=shares)
    return None


def suspect_share_counts(history: pd.DataFrame, *,
                         threshold_pct: float = _CIRCUIT_PCT) -> set[str]:
    """Symbols whose price gapped past the circuit band inside `history`.

    Pass a history window that STARTS at the filing date - the caller decides
    the window, because "since when" is a question about which filing the share
    count came from and this function cannot know that.

    That is what an unrecorded split or bonus looks like, and it means the share
    count from a filing older than the gap is pre-action while the price is
    post-action. Returns the symbols to REFUSE a market cap for.

    `history` is a long frame with symbol, date, open and close - the shape the
    bar store yields. Works on whatever actions are on file, which is currently
    three symbols of 186, so this detection is doing nearly all the work.
    """
    from desk.marketdata.corporate_actions import find_discontinuities

    need = {"symbol", "open", "close"}
    if not need <= set(history.columns):
        raise ValueError(f"history needs {sorted(need)}, has "
                         f"{sorted(history.columns)}")
    out: set[str] = set()
    for sym, grp in history.groupby("symbol", sort=False):
        frame = grp.sort_index() if grp.index.name == "date" else grp
        try:
            hits = find_discontinuities(frame, threshold_pct=threshold_pct,
                                        symbol=str(sym))
        except ValueError:
            continue
        if hits:
            out.add(str(sym).split(".")[0].upper())
    return out


def build_valuation_table(store: FilingStore, symbols, closes: pd.Series, *,
                          as_of: datetime | date,
                          suspect: set[str] | None = None
                          ) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per symbol: TTM fundamentals joined to a current price.

    `closes` is a Series of close prices indexed by symbol. KEY NORMALISATION:
    Stage 1 indexes by the full NSE symbol ("RELIANCE.NS") and the filing store
    by the base ("RELIANCE"). That mismatch once made the fundamentals join
    match ZERO rows in production while reporting an honest-looking "438 of 438
    have no filing on file", so both sides are normalised here and the overlap
    is returned in `coverage` to be checked rather than assumed.
    """
    if not isinstance(as_of, datetime):
        as_of = at_open(as_of)

    price = {str(k).split(".")[0].upper(): float(v)
             for k, v in closes.items()
             if v is not None and math.isfinite(float(v)) and float(v) > 0}

    rows: dict[str, dict] = {}
    coverage = {"no filings": 0, "no four consecutive quarters": 0,
                "no price": 0, "complete": 0, "unreadable": 0}

    for raw in symbols:
        base = str(raw).split(".")[0].upper()
        try:
            recs = store.for_symbol(base, as_of=as_of, months=3)
        except StoreError:
            coverage["no filings"] += 1
            continue
        except Exception:                            # noqa: BLE001
            # Counted separately on purpose. Folding an unexpected error into
            # "no filings" is the exact disguise that hid a broken join here
            # once before.
            coverage["unreadable"] += 1
            continue
        if not recs:
            coverage["no filings"] += 1
            continue

        ttm = trailing_twelve(recs)
        if ttm is None:
            coverage["no four consecutive quarters"] += 1
            continue
        px = price.get(base)
        if px is None:
            coverage["no price"] += 1
            continue

        # REFUSE rather than mislead. A price gap past the circuit band since
        # the filing means the share count is pre-action and the price is
        # post-action, so market cap is wrong by the action's whole ratio -
        # and wrong in the direction that makes a stock look cheap. BAJFINANCE
        # read as a P/E of 3.75 this way.
        flagged = bool(suspect and base in suspect)
        mcap = (None if flagged or not ttm.shares_outstanding
                else ttm.shares_outstanding * px)

        # A NEGATIVE P/E IS NOT A P/E. market cap over a loss is negative, and
        # a negative sorts below a cheap profitable name in an ascending
        # screen - so it is None, while earnings yield keeps the sign because
        # there a negative means what it looks like.
        pe = (mcap / ttm.pat_ttm
              if mcap and ttm.pat_ttm and ttm.pat_ttm > 0 else None)
        ey = (100.0 * ttm.pat_ttm / mcap
              if mcap and ttm.pat_ttm is not None else None)
        ps = (mcap / ttm.revenue_ttm
              if mcap and ttm.revenue_ttm and ttm.revenue_ttm > 0 else None)

        rows[base] = {
            "nature": ttm.nature, "quarters": ttm.quarters,
            "period_end": ttm.period_end,
            "earnings_age_days": (as_of.date() - ttm.period_end).days,
            "revenue_ttm": ttm.revenue_ttm, "pat_ttm": ttm.pat_ttm,
            "eps_ttm": ttm.eps_ttm,
            "shares_outstanding": ttm.shares_outstanding,
            "close": px, "market_cap": mcap, "pe_ttm": pe,
            "earnings_yield_pct": ey, "ps_ttm": ps,
            "share_count_suspect": flagged,
        }
        coverage["share count suspect" if flagged else "complete"] = (
            coverage.get("share count suspect" if flagged else "complete", 0)
            + 1)

    frame = pd.DataFrame.from_dict(rows, orient="index",
                                  columns=list(TTM_COLUMNS))
    frame.index.name = "symbol"
    return frame, coverage
