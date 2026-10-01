"""Valuation inputs derived from the XBRL the desk already has.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
Track A wants long-term investing, and the salvage map marked OpenTerminalUI's
`core/valuation.py` - a multi-stage FCFF DCF plus reverse DCF - as "take". The
module imports and runs. The DATA does not exist, and this is where the real
answer lives.

A FCFF DCF needs

    FCFF = EBIT x (1 - t) + D&A - capex - change in net working capital

and then net debt and a share count to reach a per-share value. Measured
against what is on disk for all 186 symbols with filings:

    EBIT                 DERIVABLE   profit_before_tax + finance_costs   170/186
    tax rate             DERIVABLE   tax_expense / profit_before_tax     170/186
    D&A                  PRESENT     already parsed                      170/186
    shares outstanding   DERIVABLE   paid_up_capital / face_value        186/186
    ------------------------------------------------------------------------
    capex                ABSENT      no PP&E purchase line
    change in NWC        ABSENT      needs a balance sheet
    net debt             ABSENT      no borrowings, no cash balance

NSE's quarterly results XBRL is a PROFIT AND LOSS statement. Capex, working
capital and debt live in the balance sheet and cash-flow statement, which
quarterly filings do not carry. So three of the seven inputs are missing, and
they are precisely the three that make FCFF a CASH flow rather than an accrual
one. Computing "FCFF" from what we have would produce EBITDA wearing a cash
flow's name - and for a capital-intensive business, capex is most of the
difference. That is not a DCF, and shipping it as one would be the worst kind
of plausible number.

SO THE DCF IS BLOCKED ON DATA, NOT ON CODE, and this module delivers what the
data does support instead: the accrual metrics, honestly labelled. Everything
here is one arithmetic step from facts already in the filing store, which is
why none of it needed a new fetch.

WHAT IS STILL WORTH KNOWING ABOUT THE COVERAGE. 186 symbols of roughly 3,500,
and every filing on disk is 555-616 days old because NSE's results endpoint
returns nothing newer than Jan 2025. These numbers are real and they are
stale, and any screen built on them inherits both facts.

ONE QUARTER, NOT A YEAR. Every value here comes from a single quarterly
filing, so `ebit`, `ebitda` and `nopat` are QUARTERLY figures. Multiplying by
four to annualise would assume no seasonality, which is wrong for most Indian
businesses and very wrong for some. They are left as reported; a caller that
wants a trailing-twelve-month figure has to sum four filings and should say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BLOCKED_SCORES", "DCF_MISSING_INPUTS", "DerivedFundamentals",
           "MAX_PLAUSIBLE_GROWTH_PCT",
           "derive", "peg_ratio"]

#: What a FCFF DCF still needs that NSE quarterly XBRL does not carry. Kept as
#: data rather than prose so the API and the dashboard can report the blocker
#: precisely instead of saying "valuation unavailable".
DCF_MISSING_INPUTS = (
    "capex - no purchase-of-PP&E line in quarterly results XBRL",
    "change in net working capital - needs a balance sheet",
    "net debt - no borrowings and no cash balance in quarterly results XBRL",
)


def _pos(x: float | None) -> float | None:
    """A finite, strictly positive number, or None.

    Zero is rejected alongside None because every use of these values below is
    a denominator or a ratio base, and a zero there yields inf - the one bad
    value that survives isna() and still sorts to the top of a ranking.
    """
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) and x > 0 else None


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


@dataclass(frozen=True, slots=True)
class DerivedFundamentals:
    """Accrual valuation inputs for one quarterly filing.

    Every field is None when its inputs are absent. None means "not derivable
    from this filing", never zero - a company with no reported finance cost and
    a company whose filing omitted the line are different facts, and a screen
    must be able to tell them apart.
    """

    shares_outstanding: float | None = None
    """paid_up_capital / face_value. Verified on RELIANCE: 67,660,000,000 /
    10 = 6.766 bn shares, against ~6.77 bn actual.

    THE ONE FIELD AVAILABLE FOR ALL 186 SYMBOLS, and the one that unlocks
    everything price-based - market cap, P/E, earnings yield - because the
    desk already has close prices for the whole universe. It is a
    point-in-time share count as of that filing, so it misses any subsequent
    issue or buyback."""

    ebit: float | None = None
    """profit_before_tax + finance_costs, QUARTERLY.

    The standard derivation, and preferred here over the taxonomy's own
    `SegmentProfitLossBeforeTaxAndFinanceCosts` because that fact appears in
    only 112 of 186 filings while these two appear in 170."""

    ebitda: float | None = None
    """ebit + depreciation, QUARTERLY. Not a cash flow: no capex, no working
    capital. See the module docstring."""

    effective_tax_rate_pct: float | None = None
    """tax_expense / profit_before_tax x 100.

    Only computed when profit_before_tax is positive. A loss-making quarter
    produces a meaningless or negative rate, and a negative tax rate fed into
    NOPAT would silently turn a loss into a gain."""

    nopat: float | None = None
    """ebit x (1 - effective tax rate), QUARTERLY. The closest thing to a cash
    return this data supports, and still an accrual figure."""

    op_margin_pct: float | None = None
    """100 x ebit / revenue. Operating margin before interest and tax.

    Added because it is the one field OpenTerminalUI's `quality` screen wants
    that our data can actually supply - the other two, ROE and
    debt-to-market-cap, need a balance sheet. Distinct from net_margin_pct,
    which is after both: a company can carry a healthy operating margin and a
    poor net one purely through leverage, and a quality screen wants to see the
    difference."""

    interest_cover: float | None = None
    """ebit / finance_costs. How many times over operating profit covers the
    interest bill - the most useful solvency read available here, and the one
    a long-term screen should not skip. None when finance_costs is zero, which
    is genuinely debt-free rather than infinitely covered.

    MEANINGLESS FOR LENDERS, AND THIS IS A TRAP. For a bank or an NBFC,
    interest paid is the COST OF GOODS, not a financing charge, so a low
    reading is the business model rather than distress. Measured on the real
    table, 2026-09-29:

        PFC      1.6x        a power-sector NBFC, entirely healthy
        RECLTD   1.6x        likewise
        IDEA    -0.11x       genuinely distressed
        TCS     72.2x        debt-free

    PFC and RECLTD sort next to Vodafone Idea on this column while being
    nothing like it. A screen that filters on interest_cover MUST exclude
    financials first.

    CORRECTION: an earlier version of this docstring said "the desk has no
    sector classification wired yet". That was WRONG, and a documented
    non-limitation is worse than none - it tells a reader to avoid a column
    that is usable with one extra step. `configs/sectors.json` carries an
    `industry` map and `desk.marketdata.sectors.SectorMap.sector_for()` reads
    it; coverage on the 186 filing symbols is 186/186 across 20 industries, of
    which 40 are Financial Services.

    Measured: excluding them changes the bottom of the ranking from
    TATAINVEST (-93.67x) and PAYTM (-46.28x), both financials, to SWIGGY,
    IDEA, GMRAIRPORT, PRESTIGE and JSWENERGY - genuinely leveraged or
    loss-making operating businesses. The `industry` column on the
    fundamentals table exists so that filter is one comparison away."""

    @property
    def derivable(self) -> tuple[str, ...]:
        """Which fields actually came out. For reporting coverage honestly."""
        return tuple(f for f in ("shares_outstanding", "ebit", "ebitda",
                                 "effective_tax_rate_pct", "nopat",
                                 "op_margin_pct", "interest_cover")
                     if getattr(self, f) is not None)


#: OpenTerminalUI's core/fundamental_scores.py holds six scores. Five are
#: blocked on the same balance-sheet gap as the DCF, and this records exactly
#: what each one needs - so if a source ever appears, the cost of each is known
#: rather than re-derived.
BLOCKED_SCORES = {
    "piotroski_f_score": ("cash flow from operations, ROA and its prior year, "
                          "long-term debt, total assets, current ratio"),
    "altman_z_score": ("working capital, retained earnings, total "
                       "liabilities, total assets - 4 of its 7 inputs. We do "
                       "have EBIT, market value of equity and sales"),
    "graham_number": "book value per share",
    "magic_formula_rank": "ROIC, which needs invested capital",
    "dupont_analysis": "total assets and shareholders' equity",
}


#: Growth above this is a BASE EFFECT, not a growth rate, and PEG's premise -
#: price relative to SUSTAINABLE growth - does not survive it. A judgement, not
#: a measurement, and it is here as a named constant so it can be argued with.
#:
#: MEASURED, and the reason this exists: PFC's trailing profit growth comes out
#: at 12,227.6% because its year-ago quarter was near zero. PEG then collapses
#: to 0.0003, and PEG SORTS ASCENDING - so the single most extreme base-effect
#: artefact in the table would rank FIRST in a value screen. That is the same
#: structural failure as upstream's missing-data 0.0, arrived at from the other
#: direction, and it would have shipped.
MAX_PLAUSIBLE_GROWTH_PCT = 100.0


def peg_ratio(pe: float | None, earnings_growth_pct: float | None, *,
              max_growth_pct: float = MAX_PLAUSIBLE_GROWTH_PCT
              ) -> float | None:
    """P/E divided by earnings growth. The one score of six our data supports.

    `earnings_growth_pct` is a PERCENTAGE, so a P/E of 20 against 15% growth
    gives 1.33 - the conventional reading, where under 1 is cheap for the
    growth.

    REIMPLEMENTED RATHER THAN IMPORTED, because upstream's version returns 0.0
    for missing or zero input and PEG IS SORTED ASCENDING. A company with no
    growth figure would score 0.0 and rank FIRST - the most attractive position
    in the screen - purely for having no data. That is the same trap as a
    negative P/E sorting below a cheap one, and all six of upstream's scores
    share it: a missing-data Altman Z of 0.0 reads as severe distress, a
    missing-data Graham number of 0.0 reads as no intrinsic value.

    None for every case where the ratio is not meaningful:
      - either input missing
      - P/E non-positive (a loss-making company has no P/E; see
        desk/research/valuation.py)
      - growth non-positive. A SHRINKING company's PEG is negative, and a
        negative sorts first in an ascending screen, which would put the
        fastest-declining businesses at the top of a value list.
      - growth above `max_growth_pct`. See MAX_PLAUSIBLE_GROWTH_PCT: a
        near-zero prior-year base produces a growth figure in the thousands of
        percent, PEG collapses toward zero, and the artefact ranks first.
    """
    p, g = _pos(pe), _pos(earnings_growth_pct)
    if p is None or g is None:
        return None
    if g > max_growth_pct:
        # Refused, not clamped. Clamping to the ceiling would still produce a
        # very low PEG and still rank the artefact near the top; the honest
        # answer is that this company's PEG is not computable from a base
        # effect.
        return None
    return _finite(p / g)


def derive(*, paid_up_capital: float | None = None,
           face_value: float | None = None,
           profit_before_tax: float | None = None,
           finance_costs: float | None = None,
           depreciation: float | None = None,
           tax_expense: float | None = None,
           revenue: float | None = None) -> DerivedFundamentals:
    """Derive what the P&L supports. Keyword-only, because six positional
    floats in a row is how a caller silently swaps two of them.

    Takes the promoted fields off `FinancialFacts` rather than a raw facts
    dict, so the taxonomy's spelling stays in one place - xbrl.py's `_NAMED`.
    """
    shares = None
    fv = _pos(face_value)
    pu = _pos(paid_up_capital)
    if fv is not None and pu is not None:
        shares = pu / fv

    pbt = _finite(profit_before_tax)
    fc = _finite(finance_costs)

    # EBIT needs BOTH terms. Treating an absent finance cost as zero would
    # report PBT as EBIT and quietly flatter every leveraged company's
    # operating profit.
    ebit = pbt + fc if (pbt is not None and fc is not None) else None

    dep = _finite(depreciation)
    ebitda = ebit + dep if (ebit is not None and dep is not None) else None

    tax = _finite(tax_expense)
    rate = None
    # Only meaningful on a profitable quarter. On a loss, tax/PBT is negative
    # or nonsensical, and feeding that into (1 - t) below would turn a loss
    # into a gain.
    if tax is not None and pbt is not None and pbt > 0:
        rate = 100.0 * tax / pbt

    nopat = None
    if ebit is not None and rate is not None:
        nopat = ebit * (1.0 - rate / 100.0)

    cover = None
    if ebit is not None and fc is not None and fc > 0:
        cover = ebit / fc

    # Revenue must be strictly positive: a zero-revenue quarter makes the
    # margin undefined, not zero, and inf would rank first in any screen.
    rev = _pos(revenue)
    op_margin = 100.0 * ebit / rev if (ebit is not None and rev) else None

    return DerivedFundamentals(
        shares_outstanding=_finite(shares), ebit=ebit, ebitda=ebitda,
        effective_tax_rate_pct=_finite(rate), nopat=_finite(nopat),
        op_margin_pct=_finite(op_margin), interest_cover=_finite(cover),
    )
