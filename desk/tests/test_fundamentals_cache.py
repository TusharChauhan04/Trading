"""The pre-open fundamentals cache, and its wiring into Stage 2.

Two properties carry this file.

1. LOOK-AHEAD IS STRUCTURAL, NOT CHECKED. The cache selects by FILENAME, so
   a table built for a later date is never opened during an earlier scan.
   That is the same guarantee BarStore and FilingStore already give, and it
   is tested the same way: by asserting the later file is not merely
   filtered out of the answer but never chosen.

2. "NO DATA", "DESCRIBED BUT NOT SCREENED" AND "FILTERED OUT" ARE THREE
   DIFFERENT STATES. Collapsing any two of them makes the plan claim a
   check it did not perform. There is a regression test for exactly that:
   the "no fundamentals source is wired" caveat used to be hardcoded, so it
   kept firing after the source WAS wired.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from desk.contracts.enums import Regime
from desk.research.fundamentals import (
    COLUMNS, DEFAULT_MAX_STALENESS_DAYS, CachedFundamentals, FundamentalsCache,
)
from desk.scanner.stage1 import Stage1Result
from desk.store.bars import Coverage
from desk.scanner.stage2 import (
    UNAVAILABLE_NO_FUNDAMENTALS, UNAVAILABLE_NO_THRESHOLDS, run_stage2,
)

DAY = date(2026, 9, 11)


def _table(symbols=("AAA", "BBB"), *, age=10, margin=12.0, growth=8.0):
    rows = {}
    for i, s in enumerate(symbols):
        rows[s] = {
            "nature": "consolidated", "period_end": date(2026, 6, 30),
            "disclosed_at": datetime(2026, 7, 20, 18, 0),
            "days_since_filing": age,
            "revenue": 1_000_000.0 + i, "profit_after_tax": 100_000.0,
            "eps_basic": 10.0, "net_margin_pct": margin,
            "revenue_growth_yoy_pct": growth, "profit_growth_yoy_pct": growth,
            "margin_change_pp": 0.5,
        }
    f = pd.DataFrame.from_dict(rows, orient="index", columns=list(COLUMNS))
    f.index.name = "symbol"
    return f


def _coverage():
    return {"no filings on file": 3, "complete": 2,
            "no year-ago comparison": 0, "no quarterly numbers": 0}


# --- round trip ----------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=DAY, coverage=_coverage())

    got = c.load(DAY)
    assert got is not None
    assert got.table_as_of == DAY
    assert got.staleness_days == 0
    assert not got.stale
    assert got.caveat is None
    assert list(got.frame.index) == ["AAA", "BBB"]
    assert got.frame.loc["AAA", "net_margin_pct"] == 12.0
    assert got.coverage == _coverage()
    assert isinstance(got.built_at, datetime)


def test_every_column_survives_the_round_trip(tmp_path):
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=DAY, coverage={})
    assert list(c.load(DAY).frame.columns) == list(COLUMNS)


def test_an_empty_table_round_trips(tmp_path):
    """The normal state before any filings are fetched. It must load as an
    empty table, not as a missing one - those mean different things."""
    empty = pd.DataFrame(columns=list(COLUMNS))
    empty.index.name = "symbol"
    c = FundamentalsCache(tmp_path)
    c.save(empty, as_of=DAY, coverage={"no filings on file": 1598})

    got = c.load(DAY)
    assert got is not None and got.frame.empty
    assert got.coverage["no filings on file"] == 1598


def test_saving_is_atomic(tmp_path):
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=DAY, coverage={})
    c.save(_table(), as_of=DAY, coverage={})
    assert not list(tmp_path.glob("*.tmp"))


def test_available_ignores_files_that_are_not_ours(tmp_path):
    (tmp_path / "notes.parquet").write_bytes(b"not a date")
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    FundamentalsCache(tmp_path).save(_table(), as_of=DAY, coverage={})
    assert FundamentalsCache(tmp_path).available() == [DAY]


def test_a_missing_directory_is_none_not_a_crash(tmp_path):
    c = FundamentalsCache(tmp_path / "nothing-here")
    assert c.available() == []
    assert c.load(DAY) is None


# --- the point-in-time guarantee -----------------------------------------

def test_a_table_built_after_the_scan_date_is_never_loaded(tmp_path):
    """Not filtered out of the result - NEVER OPENED. A table built on the
    18th holds filings disclosed after the 11th, and using it to scan the
    11th would screen candidates on numbers the market had not yet seen."""
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=date(2026, 9, 18), coverage={})
    assert c.load(date(2026, 9, 11)) is None


def test_the_newest_table_at_or_before_the_scan_date_wins(tmp_path):
    c = FundamentalsCache(tmp_path)
    c.save(_table(margin=1.0), as_of=date(2026, 9, 4), coverage={})
    c.save(_table(margin=2.0), as_of=date(2026, 9, 9), coverage={})
    c.save(_table(margin=3.0), as_of=date(2026, 9, 18), coverage={})

    got = c.load(date(2026, 9, 11))
    assert got.table_as_of == date(2026, 9, 9)
    assert got.frame.loc["AAA", "net_margin_pct"] == 2.0


def test_an_exact_match_is_preferred_over_an_earlier_one(tmp_path):
    c = FundamentalsCache(tmp_path)
    c.save(_table(margin=1.0), as_of=date(2026, 9, 4), coverage={})
    c.save(_table(margin=9.0), as_of=DAY, coverage={})
    assert c.load(DAY).frame.loc["AAA", "net_margin_pct"] == 9.0


# --- staleness -----------------------------------------------------------

@pytest.mark.parametrize("lag,stale", [
    (0, False), (1, False), (DEFAULT_MAX_STALENESS_DAYS, False),
    (DEFAULT_MAX_STALENESS_DAYS + 1, True), (60, True),
])
def test_staleness_boundary(tmp_path, lag, stale):
    from datetime import timedelta
    built_for = DAY - timedelta(days=lag)
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=built_for, coverage={})
    got = c.load(DAY)
    assert got.staleness_days == lag
    assert got.stale is stale


def test_a_slightly_old_table_says_what_it_costs(tmp_path):
    from datetime import timedelta
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=DAY - timedelta(days=2), coverage=_coverage())
    cav = c.load(DAY).caveat
    assert "2 day(s) before" in cav
    assert "previous quarter" in cav


def test_a_too_old_table_names_the_fix(tmp_path):
    from datetime import timedelta
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=DAY - timedelta(days=90), coverage={})
    cav = c.load(DAY).caveat
    assert "NOT applied" in cav
    assert "desk.research.refresh fundamentals" in cav


def test_staleness_is_never_negative(tmp_path):
    """A later table is not loaded, so the type cannot represent one."""
    c = FundamentalsCache(tmp_path)
    c.save(_table(), as_of=DAY, coverage={})
    assert c.load(date(2026, 9, 30)).staleness_days >= 0


# --- Stage 2: three distinct states --------------------------------------

def _stage1(symbols, *, flagged=True) -> Stage1Result:
    n = len(symbols)
    feats = pd.DataFrame({
        "close": [100.0] * n, "volume": [1e6] * n,
        "ret_20d_pct": [5.0 - i for i in range(n)],
        "dist_sma50_pct": [3.0] * n,
        "pos_52w_pct": [80.0] * n,
        "rel_volume": [1.5] * n,
        "atr_pct": [2.0] * n,
        "atr_pct_rank": [50.0] * n,
        "extended": [False] * n,
        "flag_count": [1 if flagged else 0] * n,
    }, index=list(symbols))
    feats.index.name = "symbol"
    return Stage1Result(as_of=DAY, features=feats,
                        coverage=Coverage(start=DAY, end=DAY, days_loaded=60,
                                  sessions_expected=60),
                        universe_in=n, universe_out=n)


def test_no_table_says_so_and_nothing_else():
    out = run_stage2(_stage1(["AAA", "BBB", "CCC"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=None)
    assert any(u in out.unavailable for u in UNAVAILABLE_NO_FUNDAMENTALS)
    assert not any(u in out.unavailable for u in UNAVAILABLE_NO_THRESHOLDS)


def test_a_table_with_no_thresholds_reports_described_not_screened():
    """REGRESSION. This caveat was hardcoded into UNAVAILABLE, so once the
    fundamentals cache was wired the plan went on claiming 'no fundamentals
    source is wired' while using one. A caveat that contradicts what the
    code did is worse than none: it reports a skipped check that ran."""
    out = run_stage2(_stage1(["AAA", "BBB"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=_table())
    assert not any(u in out.unavailable for u in UNAVAILABLE_NO_FUNDAMENTALS)
    assert any(u in out.unavailable for u in UNAVAILABLE_NO_THRESHOLDS)


def test_a_table_with_a_threshold_claims_neither():
    out = run_stage2(_stage1(["AAA", "BBB"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=_table(),
                     min_net_margin_pct=0.0)
    assert not any(u in out.unavailable for u in UNAVAILABLE_NO_FUNDAMENTALS)
    assert not any(u in out.unavailable for u in UNAVAILABLE_NO_THRESHOLDS)


def test_the_table_is_actually_joined_and_can_be_ranked_on():
    """`ranked` deliberately carries only score and rank_* columns, so the
    proof that the join happened is that a FUNDAMENTAL factor can rank -
    which is impossible unless its column arrived from the table."""
    from desk.scanner.stage2 import DEFAULT_FACTORS, FUNDAMENTAL_FACTORS

    table = _table(("AAA", "BBB"))
    table.loc["AAA", "revenue_growth_yoy_pct"] = 30.0
    table.loc["BBB", "revenue_growth_yoy_pct"] = 2.0

    out = run_stage2(_stage1(["AAA", "BBB"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=table,
                     factors=DEFAULT_FACTORS + FUNDAMENTAL_FACTORS)

    assert "rank_revenue_growth" in out.ranked.columns
    assert "revenue_growth" not in out.silenced, out.silenced
    assert (out.ranked.loc["AAA", "rank_revenue_growth"]
            > out.ranked.loc["BBB", "rank_revenue_growth"])


def test_fundamental_factors_are_silenced_when_no_table_is_supplied():
    """A factor scoring NaN across the universe dilutes every other
    factor's weight while contributing nothing - so it must be silenced
    and SAID, not quietly averaged in."""
    from desk.scanner.stage2 import DEFAULT_FACTORS, FUNDAMENTAL_FACTORS
    out = run_stage2(_stage1(["AAA", "BBB"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=None,
                     factors=DEFAULT_FACTORS + FUNDAMENTAL_FACTORS)
    assert "revenue_growth" in out.silenced
    assert "rank_revenue_growth" not in out.ranked.columns


def test_candidates_with_no_filing_are_counted_as_unchecked_not_cleared():
    """The table covers AAA and BBB; CCC has nothing. CCC must survive - it
    was not screened - and the plan must SAY it was not screened."""
    out = run_stage2(_stage1(["AAA", "BBB", "CCC"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=_table(("AAA", "BBB")))
    assert "CCC" in out.ranked.index
    assert any("no filing on file" in u and "SKIPPED" in u
               for u in out.unavailable)


def test_a_failed_filter_excludes_and_is_counted_separately_from_no_data():
    """'Loss-making' and 'we have no numbers' are different facts."""
    table = _table(("AAA", "BBB"), margin=-5.0)
    out = run_stage2(_stage1(["AAA", "BBB", "CCC"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=table, min_net_margin_pct=0.0)
    assert "AAA" not in out.ranked.index and "BBB" not in out.ranked.index
    assert "CCC" in out.ranked.index, "no data must not mean excluded"
    assert out.excluded.get("net margin below 0.0%") == 2


def test_a_stale_filing_is_excluded_by_the_age_filter():
    out = run_stage2(_stage1(["AAA", "BBB"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=_table(age=600),
                     max_filing_age_days=200)
    assert out.ranked.empty
    assert out.excluded.get("last filing older than 200 days") == 2


def test_a_filter_that_empties_the_shortlist_still_names_the_filter():
    """REGRESSION. The `no active factor` exit rebuilt `excluded` from
    scratch, discarding what the fundamental filter had counted. An empty
    shortlist then blamed the REGIME for a removal the filter had done,
    which sends the reader to fix the wrong thing."""
    out = run_stage2(_stage1(["AAA", "BBB"]), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=_table(age=600),
                     max_filing_age_days=200)
    assert out.excluded.get("last filing older than 200 days") == 2
    assert "no factor could run in this regime" not in out.excluded


def test_an_empty_stage1_still_reports_the_fundamentals_state():
    out = run_stage2(_stage1(["AAA"], flagged=False), regime=Regime.TRENDING_UP,
                     min_factors=1, fundamentals=None)
    assert any(u in out.unavailable for u in UNAVAILABLE_NO_FUNDAMENTALS)


# --- the API helper ------------------------------------------------------

def test_the_api_says_so_when_there_is_no_table(tmp_path, monkeypatch):
    from desk.api import main as api
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    frame, caveats = api._fundamentals(DAY)
    assert frame is None
    assert caveats and "no fundamentals table on file" in caveats[0]
    assert "desk.research.refresh fundamentals" in caveats[0]


def test_the_api_treats_a_too_old_table_as_absent(tmp_path, monkeypatch):
    from datetime import timedelta
    from desk.api import main as api
    FundamentalsCache(tmp_path / "research" / "fundamentals").save(
        _table(), as_of=DAY - timedelta(days=90), coverage={})
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    frame, caveats = api._fundamentals(DAY)
    assert frame is None, "a table 90 days behind must not be used"
    assert caveats and "NOT applied" in caveats[0]


def test_the_api_uses_a_current_table_without_a_caveat(tmp_path, monkeypatch):
    from desk.api import main as api
    FundamentalsCache(tmp_path / "research" / "fundamentals").save(
        _table(), as_of=DAY, coverage=_coverage())
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    frame, caveats = api._fundamentals(DAY)
    assert frame is not None and len(frame) == 2
    assert caveats == []


def test_the_api_never_reaches_forward_for_a_table(tmp_path, monkeypatch):
    from desk.api import main as api
    FundamentalsCache(tmp_path / "research" / "fundamentals").save(
        _table(), as_of=date(2026, 9, 18), coverage={})
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    frame, caveats = api._fundamentals(date(2026, 9, 11))
    assert frame is None
    assert "no fundamentals table on file" in caveats[0]
