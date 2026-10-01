"""TTM valuation: never mix natures, never show a negative P/E, never a fake bargain.

The failure mode here is specific and it already happened: BAJFINANCE came out at
a trailing P/E of 3.75, which reads as a screaming bargain for a lender that
normally trades near 25x. Its share count came from a filing 637 days old and its
price from today, with a split and bonus in between - so market cap was
understated roughly tenfold and the P/E divided by the same.

A missing number is recoverable. A fake bargain at the top of a value screen is
not, so every ambiguous case here resolves to None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import pytest

from desk.research.valuation import (
    TTM_COLUMNS, suspect_share_counts, trailing_twelve,
)


@dataclass
class _Q:
    """A quarter, shaped like what StoredFiling.period(3) returns."""
    period_end: date
    consolidated: bool | None
    revenue: float | None = 1000.0
    profit_after_tax: float | None = 100.0
    eps_basic: float | None = 5.0
    face_value: float | None = 10.0
    paid_up_capital: float | None = 1_000_000.0
    months: int = 3


@dataclass
class _Filing:
    q: _Q

    def period(self, months: int = 3):
        return self.q if self.q.months == months else None


def _quarters(n: int, consolidated: bool, start=date(2024, 3, 31), **kw):
    return [_Filing(_Q(period_end=start + timedelta(days=91 * i),
                       consolidated=consolidated, **kw)) for i in range(n)]


# -- the nature trap, inherited from fundamentals.py -------------------------

def test_four_same_nature_quarters_make_a_ttm() -> None:
    t = trailing_twelve(_quarters(4, True))
    assert t is not None
    assert t.quarters == 4 and t.nature == "consolidated"
    assert t.revenue_ttm == 4000.0
    assert t.pat_ttm == 400.0


def test_natures_are_never_mixed() -> None:
    """RELIANCE Q3 FY25 was Rs 128,260cr standalone against Rs 243,865cr
    consolidated. Summing across natures invents a company's worth of revenue.

    Two of each here, so neither nature can make four and the answer must be
    None rather than four mixed quarters.
    """
    recs = _quarters(2, True) + _quarters(2, False, start=date(2024, 6, 30))
    assert trailing_twelve(recs) is None


def test_consolidated_is_preferred_when_both_qualify() -> None:
    recs = _quarters(4, True) + _quarters(4, False)
    t = trailing_twelve(recs)
    assert t is not None and t.nature == "consolidated"


def test_standalone_is_used_when_only_it_qualifies() -> None:
    recs = _quarters(4, False) + _quarters(2, True)
    t = trailing_twelve(recs)
    assert t is not None and t.nature == "standalone"


# -- partial and gapped windows ---------------------------------------------

def test_three_quarters_is_not_a_year() -> None:
    """Summing three and calling it a year understates earnings by a quarter,
    which makes everything look expensive. None, not a best effort."""
    assert trailing_twelve(_quarters(3, True)) is None


def test_a_gap_in_the_window_is_refused() -> None:
    """A missing filing means the year has a hole in it. Bridging the hole while
    reporting a full-year figure is the silent kind of wrong."""
    recs = _quarters(3, True)
    far = _Filing(_Q(period_end=date(2026, 6, 30), consolidated=True))
    assert trailing_twelve(recs + [far]) is None


def test_quarter_drift_within_tolerance_is_accepted() -> None:
    """Real quarter-ends drift by days; 91 +/- 35 catches the neighbour without
    reaching the one beyond, which is 182 days away."""
    ends = [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30),
            date(2024, 12, 31)]
    recs = [_Filing(_Q(period_end=e, consolidated=True)) for e in ends]
    assert trailing_twelve(recs) is not None


def test_a_missing_field_makes_that_sum_none_not_zero() -> None:
    """A filing that omitted revenue has not reported zero revenue."""
    recs = _quarters(4, True)
    recs[1].q.revenue = None
    t = trailing_twelve(recs)
    assert t is not None
    assert t.revenue_ttm is None
    assert t.pat_ttm == 400.0, "one missing field must not void the others"


# -- shares outstanding comes from the LATEST filing ------------------------

def test_shares_come_from_the_latest_filing_not_an_average() -> None:
    """RELIANCE's count doubled on the Oct 2024 bonus. Averaging across that
    invents a company with a share count that never existed."""
    recs = _quarters(4, True)
    recs[0].q.paid_up_capital = 1_000_000.0      # oldest
    recs[-1].q.paid_up_capital = 2_000_000.0     # newest, post-bonus
    t = trailing_twelve(recs)
    assert t is not None
    assert t.shares_outstanding == 200_000.0     # 2,000,000 / 10


def test_a_zero_face_value_yields_no_share_count() -> None:
    recs = _quarters(4, True, face_value=0.0)
    t = trailing_twelve(recs)
    assert t is not None and t.shares_outstanding is None


# -- the discontinuity guard, which is doing nearly all the work ------------

def _history(gaps: dict[str, float], n: int = 40) -> pd.DataFrame:
    """A long frame like the bar store yields: symbol, date, open, close."""
    rows = []
    for sym, gap_pct in gaps.items():
        px = 100.0
        for i in range(n):
            op = px * (1 + gap_pct / 100.0) if i == n // 2 else px
            rows.append({"symbol": sym, "date": date(2025, 1, 1)
                         + timedelta(days=i), "open": op, "close": op})
            px = op
    return pd.DataFrame(rows)


def test_a_split_sized_gap_is_flagged() -> None:
    """A 1:2 split halves the price overnight - a -50% gap, far outside the 20%
    circuit band, which no cash-equity scrip can do legitimately."""
    sus = suspect_share_counts(_history({"SPLITTER": -50.0}))
    assert "SPLITTER" in sus


def test_a_normal_mover_is_not_flagged() -> None:
    """A 12% gap is inside the circuit band and is just a bad day."""
    assert suspect_share_counts(_history({"NORMAL": -12.0})) == set()


def test_the_nse_suffix_is_normalised() -> None:
    """Stage 1 indexes by RELIANCE.NS, the filing store by RELIANCE. That
    mismatch once made a fundamentals join match ZERO rows in production while
    reporting an honest-looking '438 of 438 have no filing'."""
    sus = suspect_share_counts(_history({"SPLITTER.NS": -50.0}))
    assert "SPLITTER" in sus


def test_history_without_the_needed_columns_raises() -> None:
    with pytest.raises(ValueError, match="open"):
        suspect_share_counts(pd.DataFrame({"symbol": ["X"], "close": [1.0]}))


# -- the column contract ---------------------------------------------------

def test_the_suspect_flag_is_part_of_the_contract() -> None:
    """A caller must be able to see the refusal, not just a NaN that could mean
    anything."""
    assert "share_count_suspect" in TTM_COLUMNS
    assert "earnings_age_days" in TTM_COLUMNS, \
        "a P/E on 637-day-old earnings must carry its own staleness"
