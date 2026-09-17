"""R3 - fundamentals into the scanner.

The central test here is the standalone/consolidated one. Everything else in
this file is ordinary; that one guards against a mistake that produces a
number nothing about the output marks as wrong.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from desk.research.fundamentals import build_table, latest_comparable
from desk.research.models import Filing, ResultPeriod
from desk.research.store import FilingStore, StoredFiling
from desk.research.xbrl import FinancialFacts


def _facts(end: date, *, revenue: float, pat: float, consolidated: bool):
    start = date(end.year, end.month, 1) - timedelta(days=61)
    return FinancialFacts(
        symbol="AAA", period_start=start.replace(day=1), period_end=end,
        consolidated=consolidated, audited=False,
        revenue=revenue, profit_after_tax=pat,
        facts={"RevenueFromOperations": revenue, "ProfitLossForPeriod": pat},
    )


def _stored(end: date, *, revenue, pat, consolidated, disclosed=None):
    d = disclosed or datetime.combine(end + timedelta(days=20),
                                      datetime.min.time()).replace(hour=19)
    return StoredFiling(
        filing=Filing(symbol="AAA", disclosed_at=d, period_start=None,
                      period_end=end, period=ResultPeriod.QUARTERLY,
                      audited=False, consolidated=consolidated),
        periods=(_facts(end, revenue=revenue, pat=pat,
                        consolidated=consolidated),),
    )


# ===========================================================================
# THE TRAP
# ===========================================================================

def test_growth_is_never_computed_across_reporting_natures():
    """Measured on real data: RELIANCE Q3 FY25 standalone revenue is
    Rs 128,260 crore and consolidated is Rs 243,865 crore - 1.9x apart.
    Comparing consolidated-this-year against standalone-last-year yields +87%
    revenue growth where the real standalone figure is -1.8%. Nothing about
    the output looks wrong; it is a different company's worth of revenue."""
    recs = [
        _stored(date(2023, 12, 31), revenue=130_579e7, pat=1e10,
                consolidated=False),                       # only standalone
        _stored(date(2024, 12, 31), revenue=128_260e7, pat=1e10,
                consolidated=False),
        _stored(date(2024, 12, 31), revenue=243_865e7, pat=2e10,
                consolidated=True),                        # no year-ago cons
    ]
    current, prior, nature = latest_comparable(recs)
    assert nature == "standalone", "paired across natures"
    assert current.revenue == pytest.approx(128_260e7)
    assert prior.revenue == pytest.approx(130_579e7)


def test_consolidated_is_preferred_when_BOTH_periods_have_it():
    recs = [
        _stored(date(2023, 12, 31), revenue=100e7, pat=1e7, consolidated=False),
        _stored(date(2023, 12, 31), revenue=200e7, pat=2e7, consolidated=True),
        _stored(date(2024, 12, 31), revenue=110e7, pat=1e7, consolidated=False),
        _stored(date(2024, 12, 31), revenue=240e7, pat=2e7, consolidated=True),
    ]
    current, prior, nature = latest_comparable(recs)
    assert nature == "consolidated"
    assert current.revenue == pytest.approx(240e7)
    assert prior.revenue == pytest.approx(200e7)


def test_a_symbol_with_no_year_ago_quarter_still_reports_its_margin():
    """A company's latest margin is usable even when its growth is not.
    Refusing the whole row would discard information we actually have."""
    recs = [_stored(date(2024, 12, 31), revenue=100e7, pat=10e7,
                    consolidated=False)]
    current, prior, nature = latest_comparable(recs)
    assert current is not None and prior is None
    assert nature == "standalone"


def test_the_year_ago_window_does_not_reach_the_adjacent_quarter():
    """Quarters are 91 days apart and the tolerance is 45, so the window
    cannot accidentally pick up Q2 when looking for Q3."""
    recs = [
        _stored(date(2024, 3, 31), revenue=90e7, pat=1e7, consolidated=False),
        _stored(date(2024, 12, 31), revenue=110e7, pat=1e7, consolidated=False),
    ]
    _current, prior, _n = latest_comparable(recs)
    assert prior is None, "matched a quarter 9 months away as 'year ago'"


# ===========================================================================
# derived numbers
# ===========================================================================

def test_growth_from_a_loss_making_base_is_none_not_a_spectacular_number():
    """A company that swung from a loss to a profit has no meaningful percent
    growth, and the arithmetic gives a sign-flipped artefact that reads as an
    extraordinary result."""
    from desk.research.fundamentals import _growth

    assert _growth(100.0, -50.0) is None
    assert _growth(100.0, 0.0) is None
    assert _growth(100.0, None) is None
    assert _growth(None, 50.0) is None
    assert _growth(110.0, 100.0) == pytest.approx(10.0)
    assert _growth(90.0, 100.0) == pytest.approx(-10.0)


def test_the_table_matches_a_hand_computed_growth(tmp_path):
    store = FilingStore(tmp_path)
    for rec in (_stored(date(2023, 12, 31), revenue=100e7, pat=10e7,
                        consolidated=False),
                _stored(date(2024, 12, 31), revenue=125e7, pat=15e7,
                        consolidated=False)):
        store.write(rec.filing, rec.periods)

    t, cov = build_table(store, ["AAA"], as_of=date(2026, 1, 1))
    row = t.loc["AAA"]
    assert row["nature"] == "standalone"
    assert row["revenue_growth_yoy_pct"] == pytest.approx(25.0)
    assert row["profit_growth_yoy_pct"] == pytest.approx(50.0)
    assert row["net_margin_pct"] == pytest.approx(12.0)
    assert row["margin_change_pp"] == pytest.approx(2.0)       # 12% - 10%
    assert cov["complete"] == 1


def test_coverage_separates_no_filings_from_no_comparison(tmp_path):
    """"No filing on file" and "filed but nothing to compare against" are
    different facts, and a filter must be able to tell them apart."""
    store = FilingStore(tmp_path)
    rec = _stored(date(2024, 12, 31), revenue=100e7, pat=10e7,
                  consolidated=False)
    store.write(rec.filing, rec.periods)

    _t, cov = build_table(store, ["AAA", "MISSING"], as_of=date(2026, 1, 1))
    assert cov["no filings on file"] == 1
    assert cov["no year-ago comparison"] == 1
    assert cov["complete"] == 0


def test_the_table_is_point_in_time(tmp_path):
    """A filing disclosed after as_of must not appear - the store's guarantee,
    carried through."""
    store = FilingStore(tmp_path)
    rec = _stored(date(2024, 12, 31), revenue=100e7, pat=10e7,
                  consolidated=False,
                  disclosed=datetime(2025, 1, 16, 20, 20, 21))
    store.write(rec.filing, rec.periods)

    before, _ = build_table(store, ["AAA"], as_of=date(2025, 1, 16))
    after, _ = build_table(store, ["AAA"], as_of=date(2025, 1, 17))
    assert before.empty
    assert not after.empty


def test_the_shape_is_pinned(tmp_path):
    from desk.research.fundamentals import COLUMNS

    store = FilingStore(tmp_path)
    rec = _stored(date(2024, 12, 31), revenue=100e7, pat=10e7,
                  consolidated=False)
    store.write(rec.filing, rec.periods)
    t, _ = build_table(store, ["AAA"], as_of=date(2026, 1, 1))
    assert list(t.columns) == list(COLUMNS)
    assert t.index.name == "symbol"


# ===========================================================================
# Stage 2 integration
# ===========================================================================

def _stage1(symbols):
    from desk.scanner.stage1 import FEATURE_COLUMNS, Stage1Result
    from desk.store import Coverage

    df = pd.DataFrame(index=pd.Index(symbols, name="symbol"),
                      columns=list(FEATURE_COLUMNS), dtype="float64")
    for i, s in enumerate(symbols):
        df.loc[s, "ret_20d_pct"] = float(i)
        df.loc[s, "dist_sma50_pct"] = float(i)
        df.loc[s, "pos_52w_pct"] = float(50 + i)
        df.loc[s, "rel_volume"] = 1.0 + i * 0.1
        df.loc[s, "atr_pct_rank"] = float(90 - i * 5)
    for flag in ("compressed", "unusual_volume", "unusual_move",
                 "near_52w_high", "extended"):
        df[flag] = False
    df["flag_count"] = 1
    df["bars"] = 250
    return Stage1Result(as_of=date(2026, 1, 1), features=df,
                        coverage=Coverage(start=None, end=date(2026, 1, 1),
                                          days_loaded=250,
                                          sessions_expected=250,
                                          calendar_checked=True),
                        universe_in=len(df), universe_out=len(df))


def _funds(rows):
    from desk.research.fundamentals import COLUMNS

    f = pd.DataFrame.from_dict(rows, orient="index", columns=list(COLUMNS))
    f.index.name = "symbol"
    return f


def test_a_hard_filter_removes_a_name_rather_than_ranking_it_lower():
    """A FactorSpec ranks and weights into a score; a filter takes the name
    off the list. "Its margin ranks bottom decile" and "it is loss-making"
    are different statements, and merging them lets a strong technical setup
    outvote a company that should not be there at all."""
    from desk.contracts.enums import Regime
    from desk.scanner.stage2 import run_stage2

    s1 = _stage1(["GOOD.NS", "THIN.NS"])
    funds = _funds({
        "GOOD.NS": {"nature": "standalone", "period_end": date(2024, 12, 31),
                    "disclosed_at": datetime(2025, 1, 20), "days_since_filing": 30,
                    "revenue": 100.0, "profit_after_tax": 12.0, "eps_basic": 1.0,
                    "net_margin_pct": 12.0, "revenue_growth_yoy_pct": 20.0,
                    "profit_growth_yoy_pct": 25.0, "margin_change_pp": 1.0},
        "THIN.NS": {"nature": "standalone", "period_end": date(2024, 12, 31),
                    "disclosed_at": datetime(2025, 1, 20), "days_since_filing": 30,
                    "revenue": 100.0, "profit_after_tax": -5.0, "eps_basic": -0.1,
                    "net_margin_pct": -5.0, "revenue_growth_yoy_pct": 60.0,
                    "profit_growth_yoy_pct": None, "margin_change_pp": -3.0},
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP, flagged_only=False,
                   min_factors=1, fundamentals=funds, min_net_margin_pct=0.0)
    assert r.ranked.index.tolist() == ["GOOD.NS"]
    assert r.excluded["net margin below 0.0%"] == 1


def test_a_stale_filing_is_excluded_with_a_named_reason():
    from desk.contracts.enums import Regime
    from desk.scanner.stage2 import run_stage2

    s1 = _stage1(["FRESH.NS", "STALE.NS"])
    base = {"nature": "standalone", "period_end": date(2024, 12, 31),
            "disclosed_at": datetime(2025, 1, 20), "revenue": 100.0,
            "profit_after_tax": 10.0, "eps_basic": 1.0, "net_margin_pct": 10.0,
            "revenue_growth_yoy_pct": 5.0, "profit_growth_yoy_pct": 5.0,
            "margin_change_pp": 0.0}
    funds = _funds({"FRESH.NS": dict(base, days_since_filing=40),
                    "STALE.NS": dict(base, days_since_filing=400)})
    r = run_stage2(s1, regime=Regime.TRENDING_UP, flagged_only=False,
                   min_factors=1, fundamentals=funds, max_filing_age_days=200)
    assert r.ranked.index.tolist() == ["FRESH.NS"]
    assert r.excluded["last filing older than 200 days"] == 1


def test_no_data_is_declared_and_never_counted_as_passing_a_filter():
    """R3's stated requirement. A name with no filing was not checked, and
    saying so is different from saying it passed."""
    from desk.contracts.enums import Regime
    from desk.scanner.stage2 import run_stage2

    s1 = _stage1(["HAVE.NS", "NONE.NS"])
    funds = _funds({
        "HAVE.NS": {"nature": "standalone", "period_end": date(2024, 12, 31),
                    "disclosed_at": datetime(2025, 1, 20), "days_since_filing": 30,
                    "revenue": 100.0, "profit_after_tax": 10.0, "eps_basic": 1.0,
                    "net_margin_pct": 10.0, "revenue_growth_yoy_pct": 5.0,
                    "profit_growth_yoy_pct": 5.0, "margin_change_pp": 0.0},
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP, flagged_only=False,
                   min_factors=1, fundamentals=funds, min_net_margin_pct=0.0)
    note = next(u for u in r.unavailable if "no filing on file" in u)
    assert "not checked, not cleared" in note
    # The unfiltered name survives - absence of data is not a failure.
    assert "NONE.NS" in r.ranked.index


def test_fundamental_factors_rank_when_supplied():
    from desk.contracts.enums import Regime
    from desk.scanner.stage2 import (DEFAULT_FACTORS, FUNDAMENTAL_FACTORS,
                                     run_stage2)

    s1 = _stage1(["SLOW.NS", "FAST.NS"])
    base = {"nature": "standalone", "period_end": date(2024, 12, 31),
            "disclosed_at": datetime(2025, 1, 20), "days_since_filing": 30,
            "revenue": 100.0, "profit_after_tax": 10.0, "eps_basic": 1.0,
            "net_margin_pct": 10.0}
    funds = _funds({
        "SLOW.NS": dict(base, revenue_growth_yoy_pct=2.0,
                        profit_growth_yoy_pct=1.0, margin_change_pp=-1.0),
        "FAST.NS": dict(base, revenue_growth_yoy_pct=40.0,
                        profit_growth_yoy_pct=55.0, margin_change_pp=3.0),
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP, flagged_only=False,
                   min_factors=1, fundamentals=funds,
                   factors=DEFAULT_FACTORS + FUNDAMENTAL_FACTORS)
    assert "rank_revenue_growth" in r.ranked.columns
    assert r.ranked.loc["FAST.NS", "rank_revenue_growth"] > \
        r.ranked.loc["SLOW.NS", "rank_revenue_growth"]


def test_fundamental_factors_are_not_on_by_default():
    """A factor that scores NaN for the whole universe is worse than an
    absent one: it dilutes every other factor's weight while contributing
    nothing."""
    from desk.scanner.stage2 import DEFAULT_FACTORS, FUNDAMENTAL_FACTORS

    names = {f.name for f in DEFAULT_FACTORS}
    assert not names & {f.name for f in FUNDAMENTAL_FACTORS}


def test_stage2_without_fundamentals_is_unchanged():
    from desk.contracts.enums import Regime
    from desk.scanner.stage2 import run_stage2

    s1 = _stage1(["A.NS", "B.NS"])
    r = run_stage2(s1, regime=Regime.TRENDING_UP, flagged_only=False,
                   min_factors=1)
    assert len(r.ranked) == 2
    assert not any("filing" in u for u in r.unavailable)


# ===========================================================================
# the trap, on entirely real data
# ===========================================================================

def test_the_nature_trap_against_four_real_reliance_filings(tmp_path):
    """The constructed test above proves the rule; this proves it holds on
    four documents fetched from NSE - RELIANCE's Q3 FY24 and Q3 FY25, each in
    both standalone and consolidated form.

    The numbers are the whole argument:
        standalone    FY24 130,579cr -> FY25 128,260cr   =  -1.8%
        consolidated  FY24 227,970cr -> FY25 243,865cr   =  +7.0%
        MIXED         FY24 130,579cr -> FY25 243,865cr   = +86.8%

    All three are arithmetically fine. Only two are real, and the wrong one
    is the most flattering.
    """
    from pathlib import Path as _P

    from desk.research.xbrl import parse_xbrl

    fixtures = _P(__file__).parent / "fixtures"
    store = FilingStore(tmp_path)
    disclosed = {date(2023, 12, 31): datetime(2024, 1, 19, 19, 14, 39),
                 date(2024, 12, 31): datetime(2025, 1, 16, 20, 20, 21)}

    for end in (date(2023, 12, 31), date(2024, 12, 31)):
        for tag, cons in (("S", False), ("C", True)):
            raw = (fixtures / f"nse_xbrl_RELIANCE_{end:%Y%m%d}{tag}.xml").read_bytes()
            periods, _ = parse_xbrl(raw, symbol="RELIANCE")
            quarter = [p for p in periods if p.months == 3]
            store.write(Filing(symbol="RELIANCE",
                               disclosed_at=disclosed[end] + timedelta(
                                   minutes=0 if not cons else 3),
                               period_start=None, period_end=end,
                               period=ResultPeriod.QUARTERLY, audited=False,
                               consolidated=cons),
                        quarter)

    t, cov = build_table(store, ["RELIANCE"], as_of=date(2026, 1, 1))
    row = t.loc["RELIANCE"]

    assert cov["complete"] == 1
    assert row["nature"] == "consolidated"
    assert row["revenue"] / 1e7 == pytest.approx(243_865, abs=1)
    assert row["revenue_growth_yoy_pct"] == pytest.approx(6.97, abs=0.05)

    # And explicitly NOT the cross-nature figure.
    mixed = 100 * (243_865 / 130_579 - 1)
    assert mixed == pytest.approx(86.8, abs=0.5)
    assert row["revenue_growth_yoy_pct"] != pytest.approx(mixed, abs=1.0)


def test_standalone_only_history_forces_the_standalone_comparison(tmp_path):
    """IRCTC's real shape: consolidated filings begin in 2024, so the latest
    quarter HAS a consolidated figure and the year-ago one does not. A
    'prefer consolidated' rule with a fallback does the cross-nature
    comparison here - on real data, not a contrived case."""
    from pathlib import Path as _P

    from desk.research.xbrl import parse_xbrl

    fixtures = _P(__file__).parent / "fixtures"
    store = FilingStore(tmp_path)

    # Year-ago: standalone only.
    raw = (fixtures / "nse_xbrl_RELIANCE_20231231S.xml").read_bytes()
    store.write(Filing(symbol="AAA", disclosed_at=datetime(2024, 1, 19, 19, 0),
                       period_start=None, period_end=date(2023, 12, 31),
                       period=ResultPeriod.QUARTERLY, audited=False,
                       consolidated=False),
                [p for p in parse_xbrl(raw, symbol="AAA")[0] if p.months == 3])
    # Current: both.
    for tag, cons, mins in (("S", False, 0), ("C", True, 3)):
        raw = (fixtures / f"nse_xbrl_RELIANCE_20241231{tag}.xml").read_bytes()
        store.write(Filing(symbol="AAA",
                           disclosed_at=datetime(2025, 1, 16, 20, 20 + mins),
                           period_start=None, period_end=date(2024, 12, 31),
                           period=ResultPeriod.QUARTERLY, audited=False,
                           consolidated=cons),
                    [p for p in parse_xbrl(raw, symbol="AAA")[0]
                     if p.months == 3])

    t, _ = build_table(store, ["AAA"], as_of=date(2026, 1, 1))
    row = t.loc["AAA"]
    assert row["nature"] == "standalone", "fell back across natures"
    assert row["revenue"] / 1e7 == pytest.approx(128_260, abs=1)
    assert row["revenue_growth_yoy_pct"] == pytest.approx(-1.78, abs=0.05)
