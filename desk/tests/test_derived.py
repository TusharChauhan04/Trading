"""Derived valuation inputs: correct, or None. Never a plausible zero.

Every field here feeds a long-term screen, so the failure that matters is not
a crash but a number that looks reasonable and is wrong - an EBIT that quietly
equals PBT because the finance cost was missing, or a NOPAT that turns a loss
into a gain through a negative tax rate.
"""

from __future__ import annotations

import math

from desk.research.derived import DCF_MISSING_INPUTS, derive


# -- shares outstanding: the field present for all 186 symbols ---------------

def test_shares_outstanding_matches_reliance() -> None:
    """Pinned to a real filing so the arithmetic is anchored outside this file.

    RELIANCE 2024-03-31 standalone: paid-up Rs 67,660,000,000 at a Rs 10 face
    value is 6.766 bn shares, against roughly 6.77 bn actual.
    """
    d = derive(paid_up_capital=67_660_000_000.0, face_value=10.0)
    assert d.shares_outstanding is not None
    assert math.isclose(d.shares_outstanding, 6.766e9, rel_tol=1e-9)


def test_a_zero_face_value_yields_none_not_infinity() -> None:
    """inf survives isna() and sorts to the top of any ranking."""
    d = derive(paid_up_capital=1e9, face_value=0.0)
    assert d.shares_outstanding is None


def test_missing_either_half_yields_none() -> None:
    assert derive(paid_up_capital=1e9).shares_outstanding is None
    assert derive(face_value=10.0).shares_outstanding is None


# -- EBIT: the derivation that must not silently flatter a leveraged firm ----

def test_ebit_is_pbt_plus_finance_costs() -> None:
    d = derive(profit_before_tax=150_510_000_000.0,
               finance_costs=36_130_000_000.0)
    assert d.ebit == 186_640_000_000.0


def test_a_missing_finance_cost_does_not_become_zero() -> None:
    """THE important one.

    Treating an absent finance cost as zero would report PBT as EBIT, making
    every leveraged company's operating profit look better than it is - and
    interest cover uncomputable in a way nobody notices.
    """
    d = derive(profit_before_tax=100.0)
    assert d.ebit is None, "EBIT needs both terms, not a defaulted zero"
    assert d.ebitda is None
    assert d.interest_cover is None


def test_a_genuinely_reported_zero_finance_cost_still_gives_ebit() -> None:
    """Debt-free is a real reading and must not be confused with missing."""
    d = derive(profit_before_tax=100.0, finance_costs=0.0)
    assert d.ebit == 100.0
    # But cover is undefined rather than infinite - nothing to cover.
    assert d.interest_cover is None


# -- the tax rate, and the sign error it could hide -------------------------

def test_effective_tax_rate() -> None:
    d = derive(profit_before_tax=150_510_000_000.0,
               tax_expense=37_680_000_000.0)
    assert d.effective_tax_rate_pct is not None
    assert math.isclose(d.effective_tax_rate_pct, 25.0, abs_tol=0.05)


def test_a_loss_making_quarter_has_no_tax_rate_and_no_nopat() -> None:
    """A negative tax rate fed through (1 - t) turns a loss into a gain.

    That is the specific arithmetic accident this guard exists for, so it is
    asserted rather than assumed.
    """
    d = derive(profit_before_tax=-500.0, finance_costs=50.0, tax_expense=10.0)
    assert d.effective_tax_rate_pct is None
    assert d.nopat is None
    assert d.ebit == -450.0, "EBIT is still meaningful on a loss"


def test_nopat_applies_the_rate_to_ebit_not_to_pbt() -> None:
    d = derive(profit_before_tax=1000.0, finance_costs=200.0,
               tax_expense=250.0)
    assert d.ebit == 1200.0
    assert d.effective_tax_rate_pct == 25.0
    assert math.isclose(d.nopat, 900.0)      # 1200 x 0.75, not 1000 x 0.75


# -- EBITDA ------------------------------------------------------------------

def test_ebitda_adds_depreciation_to_ebit() -> None:
    d = derive(profit_before_tax=1000.0, finance_costs=200.0,
               depreciation=300.0)
    assert d.ebitda == 1500.0


def test_ebitda_is_none_without_depreciation() -> None:
    d = derive(profit_before_tax=1000.0, finance_costs=200.0)
    assert d.ebit == 1200.0
    assert d.ebitda is None


# -- interest cover ----------------------------------------------------------

def test_interest_cover() -> None:
    d = derive(profit_before_tax=1000.0, finance_costs=200.0)
    assert math.isclose(d.interest_cover, 6.0)      # 1200 / 200


def test_a_company_that_cannot_cover_its_interest_reads_below_one() -> None:
    d = derive(profit_before_tax=-50.0, finance_costs=100.0)
    assert d.ebit == 50.0
    assert math.isclose(d.interest_cover, 0.5)


# -- the empty case and the coverage report ---------------------------------

def test_nothing_in_nothing_out() -> None:
    d = derive()
    assert d.derivable == ()
    for f in ("shares_outstanding", "ebit", "ebitda",
              "effective_tax_rate_pct", "nopat", "interest_cover"):
        assert getattr(d, f) is None


def test_derivable_reports_only_what_came_out() -> None:
    d = derive(paid_up_capital=1e9, face_value=10.0)
    assert d.derivable == ("shares_outstanding",)


def test_the_dcf_blocker_is_data_and_it_is_named() -> None:
    """The blocker must be reportable precisely, not as "unavailable".

    All three missing inputs are balance-sheet or cash-flow items, and NSE's
    quarterly results XBRL is a P&L. Naming them is what lets the API say why
    rather than shrugging.
    """
    assert len(DCF_MISSING_INPUTS) == 3
    joined = " ".join(DCF_MISSING_INPUTS).lower()
    for needed in ("capex", "working capital", "net debt"):
        assert needed in joined


def test_a_nan_input_is_treated_as_absent() -> None:
    """pandas hands out NaN where a fact was missing, and NaN propagates
    silently through arithmetic into a NaN result that looks computed."""
    nan = float("nan")
    d = derive(paid_up_capital=nan, face_value=10.0,
               profit_before_tax=nan, finance_costs=1.0)
    assert d.shares_outstanding is None
    assert d.ebit is None
