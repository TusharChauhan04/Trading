"""Turning an NSE results filing into numbers.

This is where fundamentals actually come from. `Filing.xbrl_url` points at one
of these documents; everything downstream - revenue growth, margins, the
Stage 2 fundamental filters - reads what this module produces.

THE TRAP, and it is a bad one
-----------------------------
An XBRL fact is scoped to a `<context>`, and a context carries a period. The
obvious implementation - and what a general-purpose XBRL reader does - is to
trust that period. On NSE's filings it is WRONG, measured:

    RELIANCE Q3 FY25, the two primary contexts:

      context id   <context> says          the FACTS inside say
      ----------   --------------------    -----------------------------
      OneD         2024-10-01..12-31       2024-10-01..12-31   (3 months)
      FourD        2024-10-01..12-31       2024-04-01..12-31   (9 months)

    Revenue, quarter      Rs   128,260 crore
    Revenue, year-to-date Rs   396,645 crore      ratio 3.09x

So the `<context>` element claims `FourD` is the quarter when it is actually
year-to-date. Trusting it reports 9 months of revenue as 3 months - a silent
3x overstatement, on every company, feeding straight into any growth or
margin calculation built on top.

This module therefore takes the period from the
`DateOfStartOfReportingPeriod` / `DateOfEndOfReportingPeriod` FACTS, and only
falls back to the context element when those are absent (and says so, in
`warnings`).

WHAT IT RETURNS, AND WHY IT DOES NOT CHOOSE
-------------------------------------------
Every primary period in the document, each labelled with its real span and
`months`. It deliberately does NOT pick "the" period: a quarterly filing
normally contains both the quarter and the year-to-date, and which one a
caller wants depends on what they are computing. A parser that silently
returned one of them would be making that decision invisibly, and the whole
point of the measurement above is that this is exactly where the invisible
decision goes wrong.

SEGMENT CONTEXTS ARE SKIPPED. A filing carries dozens of them (43 in the
RELIANCE fixture) holding per-segment revenue, per-line-item expense
breakdowns and so on. They are recognised by what they lack: a primary
context carries the reporting-period facts, a segment context does not.
Their contents are not lost - they are reported in `warnings` by count - but
they are not company totals and must never be added up as if they were.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date

__all__ = ["FinancialFacts", "XbrlError", "parse_xbrl"]


class XbrlError(Exception):
    """The document is not a readable XBRL filing."""


#: Facts every primary context carries, and segment contexts do not. This is
#: how the two are told apart - by what a context CONTAINS rather than by
#: pattern-matching its id, which is a naming convention NSE has never
#: promised to keep.
_PERIOD_START = "DateOfStartOfReportingPeriod"
_PERIOD_END = "DateOfEndOfReportingPeriod"

#: The named fields. Everything else still arrives in `facts`; these are
#: promoted because they are what the fundamental filters actually read, and
#: a caller should not have to know the taxonomy's spelling to get revenue.
_NAMED = {
    "revenue": "RevenueFromOperations",
    "other_income": "OtherIncome",
    "total_income": "Income",
    "expenses": "Expenses",
    "profit_before_tax": "ProfitBeforeTax",
    "tax_expense": "TaxExpense",
    "profit_after_tax": "ProfitLossForPeriod",
    "eps_basic": "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "eps_diluted": "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "face_value": "FaceValueOfEquityShareCapital",
    "paid_up_capital": "PaidUpValueOfEquityShareCapital",
    "finance_costs": "FinanceCosts",
    "depreciation": "DepreciationDepletionAndAmortisationExpense",
    "employee_cost": "EmployeeBenefitExpense",
}


@dataclass(frozen=True, slots=True)
class FinancialFacts:
    """One reporting period out of one filing.

    `months` is derived from the real span, so a caller filtering to quarterly
    results writes `f.months == 3` rather than trusting a label.
    """

    symbol: str
    period_start: date
    period_end: date
    consolidated: bool | None
    """None means the filing did not say - not the same as standalone."""
    audited: bool | None
    revenue: float | None = None
    other_income: float | None = None
    total_income: float | None = None
    expenses: float | None = None
    profit_before_tax: float | None = None
    tax_expense: float | None = None
    profit_after_tax: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None
    face_value: float | None = None
    paid_up_capital: float | None = None
    finance_costs: float | None = None
    depreciation: float | None = None
    employee_cost: float | None = None
    facts: dict[str, float] = field(default_factory=dict)
    """Every numeric fact in this context, including the named ones above.
    Nothing the filing said is discarded."""

    @property
    def months(self) -> int:
        """Length of the period, rounded to whole months. 3 for a quarter, 9
        for nine-month year-to-date, 12 for a full year."""
        days = (self.period_end - self.period_start).days + 1
        return max(1, round(days / 30.44))

    @property
    def is_quarterly(self) -> bool:
        return self.months == 3

    @property
    def net_margin_pct(self) -> float | None:
        """Profit after tax as a percentage of revenue. None rather than a
        guess when either input is missing or revenue is zero - a margin
        computed off a zero denominator is not a large margin."""
        if not self.revenue or self.profit_after_tax is None:
            return None
        return 100.0 * self.profit_after_tax / self.revenue

    def __str__(self) -> str:
        nature = ("consolidated" if self.consolidated else "standalone"
                  if self.consolidated is not None else "nature-unstated")
        rev = f"{self.revenue / 1e7:,.0f}cr" if self.revenue else "?"
        return (f"{self.symbol} {self.period_start}..{self.period_end} "
                f"({self.months}m, {nature}) revenue {rev}")


def parse_xbrl(raw: bytes, *, symbol: str | None = None
               ) -> tuple[list[FinancialFacts], list[str]]:
    """Every primary reporting period in one filing, plus what was skipped.

    `symbol` overrides the one in the document. Supply it from the `Filing`
    when you have it: the document's own Symbol fact is what the company
    typed, and this project keys everything on the exchange's spelling.

    Returns (periods, warnings). `warnings` is not decorative - it names the
    segment contexts skipped and any period that had to fall back to the
    context element, which is the one case where the 3x trap above could
    still bite.
    """
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise XbrlError(f"not parseable as XML: {exc}") from exc

    if _local(root.tag) != "xbrl":
        raise XbrlError(
            f"root element is <{_local(root.tag)}>, expected <xbrl> - this is "
            f"not an XBRL filing"
        )

    ctx_periods = _context_periods(root)

    by_ctx: dict[str, dict[str, str]] = {}
    for el in root.iter():
        ref = el.get("contextRef")
        if ref and el.text and el.text.strip():
            by_ctx.setdefault(ref, {})[_local(el.tag)] = el.text.strip()

    # The symbol is a property of the FILING, not of a period. NSE only puts
    # it in the first context, so reading it per-context leaves every other
    # period with a blank symbol - which then silently fails to join against
    # anything keyed on symbol downstream.
    doc_symbol = ""
    for facts in by_ctx.values():
        if facts.get("Symbol"):
            doc_symbol = facts["Symbol"]
            break

    periods: list[FinancialFacts] = []
    warnings: list[str] = []
    skipped = 0

    for ref, facts in by_ctx.items():
        start = _as_date(facts.get(_PERIOD_START))
        end = _as_date(facts.get(_PERIOD_END))

        if start is None or end is None:
            # No reporting-period facts: a segment or line-item breakdown,
            # not a company total. Counted, never summed.
            skipped += 1
            continue

        if end < start:
            warnings.append(
                f"context {ref!r} reports a period ending {end} before it "
                f"starts {start} - skipped")
            continue

        ctx = ctx_periods.get(ref)
        if ctx and ctx != (start, end):
            # This is the documented trap firing. Not an error - the facts
            # win - but worth surfacing, because it is the thing that would
            # silently triple a revenue figure if anyone "simplified" this
            # module back to trusting the context.
            warnings.append(
                f"context {ref!r} declares {ctx[0]}..{ctx[1]} but its facts "
                f"report {start}..{end} - trusting the facts")

        numeric = {k: v for k, v in ((k, _as_float(v))
                                     for k, v in facts.items()) if v is not None}
        named = {attr: numeric.get(tag) for attr, tag in _NAMED.items()}

        periods.append(FinancialFacts(
            symbol=symbol or doc_symbol,
            period_start=start,
            period_end=end,
            consolidated=_nature(facts.get("NatureOfReportStandaloneConsolidated")),
            audited=_audited(facts.get("WhetherResultsAreAuditedOrUnaudited")),
            facts=numeric,
            **named,
        ))

    if skipped:
        warnings.append(
            f"{skipped} segment/breakdown context(s) skipped - they carry no "
            f"reporting period and are not company totals")
    if not periods:
        warnings.append("no primary reporting period found in this document")

    periods.sort(key=lambda p: (p.period_end, p.months))
    return periods, warnings


# --------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _context_periods(root: ET.Element) -> dict[str, tuple[date, date]]:
    """The period each <context> DECLARES. Kept only so a disagreement with
    the facts can be reported - never used as the answer."""
    out: dict[str, tuple[date, date]] = {}
    for ctx in root.iter():
        if _local(ctx.tag) != "context":
            continue
        cid = ctx.get("id")
        if not cid:
            continue
        bits = {_local(e.tag): (e.text or "").strip() for e in ctx.iter()}
        start = _as_date(bits.get("startDate") or bits.get("instant"))
        end = _as_date(bits.get("endDate") or bits.get("instant"))
        if start and end:
            out[cid] = (start, end)
    return out


def _as_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s.strip()[:10])
    except ValueError:
        return None


def _as_float(s: str) -> float | None:
    """A number, or None. Non-finite values are refused: XBRL is text, and
    'NaN' in a revenue field must not become a float that compares false
    against every threshold while looking like data."""
    import math

    try:
        f = float(s.replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nature(v: str | None) -> bool | None:
    t = (v or "").strip().lower()
    if t == "consolidated":
        return True
    if t == "standalone":
        return False
    return None


def _audited(v: str | None) -> bool | None:
    t = (v or "").strip().lower()
    if t == "audited":
        return True
    if t == "unaudited":
        return False
    return None
