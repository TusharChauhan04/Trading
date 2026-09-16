"""R2 - the XBRL parser. Where fundamentals actually come from.

Pinned against a real NSE filing (RELIANCE Q3 FY25), and wherever possible
against figures that can be checked OUTSIDE this codebase: the published
results, and two internal accounting identities the filing must satisfy if we
are reading it coherently rather than just reading it successfully.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from desk.research.xbrl import FinancialFacts, XbrlError, parse_xbrl

FIXTURES = Path(__file__).parent / "fixtures"
CRORE = 1e7


@pytest.fixture(scope="module")
def reliance():
    raw = (FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes()
    return parse_xbrl(raw, symbol="RELIANCE")


@pytest.fixture(scope="module")
def quarter(reliance):
    periods, _ = reliance
    return next(p for p in periods if p.months == 3)


# ===========================================================================
# THE TRAP: the context element lies about the period
# ===========================================================================

def test_the_period_comes_from_the_facts_not_the_context_element(reliance):
    """REGRESSION target for this whole module, measured on the real filing.

    Both primary contexts DECLARE 2024-10-01..2024-12-31. The facts inside
    say otherwise: OneD really is the quarter, FourD is nine-month
    year-to-date. Trusting the context - which is what a general-purpose XBRL
    reader does - reports 9 months of revenue as 3 months, a silent 3.09x
    overstatement feeding every growth and margin calculation downstream.
    """
    periods, _ = reliance
    spans = {(p.period_start, p.period_end) for p in periods}
    assert (date(2024, 10, 1), date(2024, 12, 31)) in spans       # the quarter
    assert (date(2024, 4, 1), date(2024, 12, 31)) in spans        # 9m YTD

    q = next(p for p in periods if p.months == 3)
    ytd = next(p for p in periods if p.months == 9)
    assert ytd.revenue / q.revenue == pytest.approx(3.09, abs=0.02)


def test_the_disagreement_is_reported_not_silently_resolved(reliance):
    """The facts win, but a caller must be able to see that the document
    contradicted itself - that is the signal this module would be misread if
    anyone 'simplified' it back to trusting the context."""
    _, warnings = reliance
    assert any("declares" in w and "trusting the facts" in w for w in warnings)


def test_months_is_derived_from_the_real_span(reliance):
    periods, _ = reliance
    assert sorted(p.months for p in periods) == [3, 9]
    assert next(p for p in periods if p.months == 3).is_quarterly
    assert not next(p for p in periods if p.months == 9).is_quarterly


def test_both_periods_are_returned_because_the_parser_does_not_choose(reliance):
    """A quarterly filing normally carries the quarter AND the year-to-date,
    and which one a caller wants depends on what it is computing. Silently
    returning one would be making that decision invisibly - which is exactly
    where the 3x error above comes from."""
    periods, _ = reliance
    assert len(periods) == 2


# ===========================================================================
# the numbers, checked against the published results
# ===========================================================================

def test_headline_figures_match_reliances_published_q3_fy25(quarter):
    """Externally checkable: these are RIL's reported standalone Q3 FY25
    figures, in crore."""
    assert quarter.revenue / CRORE == pytest.approx(128_260, abs=1)
    assert quarter.other_income / CRORE == pytest.approx(3_214, abs=1)
    assert quarter.total_income / CRORE == pytest.approx(131_474, abs=1)
    assert quarter.expenses / CRORE == pytest.approx(119_877, abs=1)
    assert quarter.profit_before_tax / CRORE == pytest.approx(11_597, abs=1)
    assert quarter.profit_after_tax / CRORE == pytest.approx(8_721, abs=1)
    assert quarter.eps_basic == pytest.approx(6.44)


def test_the_income_statement_adds_up(quarter):
    """An accounting identity the filing must satisfy: total income minus
    total expenses is profit before exceptional items and tax. If this fails
    we are reading numbers from mismatched contexts - each individually
    plausible, collectively incoherent - which is the failure a per-field
    spot check would never catch."""
    identity = quarter.total_income - quarter.expenses
    assert identity / CRORE == pytest.approx(
        quarter.facts["ProfitBeforeExceptionalItemsAndTax"] / CRORE, abs=1)
    # and revenue + other income is total income
    assert (quarter.revenue + quarter.other_income) / CRORE == pytest.approx(
        quarter.total_income / CRORE, abs=1)
    # and PBT less tax is PAT
    assert (quarter.profit_before_tax - quarter.tax_expense) / CRORE == \
        pytest.approx(quarter.profit_after_tax / CRORE, abs=1)


def test_eps_is_consistent_with_profit_and_the_share_count(quarter):
    """A second, independent identity: paid-up capital over face value gives
    the share count, and profit over shares must reproduce the EPS the filing
    states. This catches reading the right-shaped number from the wrong
    place - the share count and the profit come from different facts."""
    shares = quarter.paid_up_capital / quarter.face_value
    assert quarter.profit_after_tax / shares == pytest.approx(
        quarter.eps_basic, abs=0.01)


def test_net_margin_is_computed_not_stated(quarter):
    assert quarter.net_margin_pct == pytest.approx(6.80, abs=0.01)


def test_a_zero_revenue_gives_no_margin_rather_than_a_huge_one(quarter):
    """A margin computed off a zero denominator is not a large margin."""
    from dataclasses import replace
    assert replace(quarter, revenue=0.0).net_margin_pct is None
    assert replace(quarter, revenue=None).net_margin_pct is None
    assert replace(quarter, profit_after_tax=None).net_margin_pct is None


# ===========================================================================
# segment contexts
# ===========================================================================

def test_segment_breakdowns_are_skipped_and_counted(reliance):
    """A filing carries dozens of per-segment and per-line-item contexts.
    They are not company totals and must never be summed as if they were -
    but they are also not silently discarded."""
    periods, warnings = reliance
    assert len(periods) == 2
    skipped = next(w for w in warnings if "segment/breakdown" in w)
    assert "44" in skipped


def test_segment_contexts_are_recognised_by_content_not_by_name():
    """They are told apart by what they LACK - the reporting-period facts -
    rather than by pattern-matching context ids, which is a naming convention
    NSE has never promised to keep."""
    doc = _xbrl("""
      <ctx id="Whatever01D"><OtherExpenses ctx="Whatever01D">1.0</OtherExpenses></ctx>
    """, extra_facts='<OtherExpenses contextRef="SegmentXYZ">99.0</OtherExpenses>')
    periods, warnings = parse_xbrl(doc.encode(), symbol="X")
    assert len(periods) == 1                      # only the real one
    assert any("segment/breakdown" in w for w in warnings)


# ===========================================================================
# refusing bad input
# ===========================================================================

def test_non_xml_is_refused_with_a_named_error():
    with pytest.raises(XbrlError, match="not parseable as XML"):
        parse_xbrl(b"<html>404 not found</html' ")


def test_xml_that_is_not_xbrl_is_refused():
    with pytest.raises(XbrlError, match="not an XBRL filing"):
        parse_xbrl(b"<html><body>Page not found</body></html>")


def test_a_document_with_no_reporting_period_says_so_rather_than_returning_nothing():
    doc = ('<xbrl xmlns="http://www.xbrl.org/2003/instance">'
           '<Revenue contextRef="C1">5</Revenue></xbrl>')
    periods, warnings = parse_xbrl(doc.encode(), symbol="X")
    assert periods == []
    assert any("no primary reporting period" in w for w in warnings)


def test_a_backwards_period_is_refused_rather_than_given_negative_months():
    doc = _xbrl("", start="2024-12-31", end="2024-10-01")
    periods, warnings = parse_xbrl(doc.encode(), symbol="X")
    assert periods == []
    assert any("before it starts" in w for w in warnings)


def test_a_non_finite_number_never_becomes_a_financial_fact():
    """XBRL is text. 'NaN' in a revenue field must not become a float that
    compares false against every threshold while looking like data."""
    doc = _xbrl('<RevenueFromOperations contextRef="C1">NaN</RevenueFromOperations>'
                '<OtherIncome contextRef="C1">Infinity</OtherIncome>'
                '<Income contextRef="C1">1234.5</Income>')
    periods, _ = parse_xbrl(doc.encode(), symbol="X")
    assert periods[0].revenue is None
    assert periods[0].other_income is None
    assert periods[0].total_income == 1234.5


# ===========================================================================
# the three-state fields
# ===========================================================================

def test_consolidated_and_audited_are_three_state(quarter):
    """None means the filing did not say, which is not the same as no."""
    assert quarter.consolidated is False        # this filing says Standalone
    assert quarter.audited is False             # and Unaudited

    unstated, _ = parse_xbrl(_xbrl("").encode(), symbol="X")
    assert unstated[0].consolidated is None
    assert unstated[0].audited is None

    con = _xbrl('<NatureOfReportStandaloneConsolidated contextRef="C1">'
                'Consolidated</NatureOfReportStandaloneConsolidated>'
                '<WhetherResultsAreAuditedOrUnaudited contextRef="C1">'
                'Audited</WhetherResultsAreAuditedOrUnaudited>')
    got, _ = parse_xbrl(con.encode(), symbol="X")
    assert got[0].consolidated is True and got[0].audited is True


def test_the_supplied_symbol_overrides_the_documents_own(reliance):
    """The document's Symbol fact is what the company typed; this project
    keys everything on the exchange's spelling."""
    raw = (FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes()
    periods, _ = parse_xbrl(raw, symbol="OVERRIDDEN")
    assert all(p.symbol == "OVERRIDDEN" for p in periods)
    without, _ = parse_xbrl(raw)
    assert all(p.symbol == "RELIANCE" for p in without)     # falls back to the doc


def test_nothing_the_filing_said_is_discarded(quarter):
    """`facts` carries every numeric value, so a caller needing something the
    named fields do not cover is never blocked on this module changing."""
    assert len(quarter.facts) == 52
    assert "CostOfMaterialsConsumed" in quarter.facts
    assert quarter.facts["RevenueFromOperations"] == quarter.revenue


# --------------------------------------------------------------------------

def _xbrl(facts: str, *, start: str = "2024-10-01", end: str = "2024-12-31",
          extra_facts: str = "") -> str:
    """A minimal but structurally real filing: one primary context carrying
    the reporting-period facts that mark it as a company total."""
    return (
        '<xbrl xmlns="http://www.xbrl.org/2003/instance">'
        '<context id="C1"><period>'
        f'<startDate>{start}</startDate><endDate>{end}</endDate>'
        '</period></context>'
        f'<DateOfStartOfReportingPeriod contextRef="C1">{start}</DateOfStartOfReportingPeriod>'
        f'<DateOfEndOfReportingPeriod contextRef="C1">{end}</DateOfEndOfReportingPeriod>'
        f'{facts}{extra_facts}</xbrl>'
    )


def test_the_symbol_is_taken_document_wide_not_per_context():
    """REGRESSION: NSE puts the Symbol fact only in the FIRST context, so
    reading it per-context left the year-to-date period with a blank symbol -
    which then silently fails to join against anything keyed on symbol."""
    raw = (FIXTURES / "nse_xbrl_RELIANCE.xml").read_bytes()
    periods, _ = parse_xbrl(raw)
    assert len(periods) == 2
    assert all(p.symbol == "RELIANCE" for p in periods), \
        [(p.months, p.symbol) for p in periods]
