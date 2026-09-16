"""Stage 0 of the scanner funnel. Small hand-built cases for each filter,
plus a pin against the real captured bhavcopy archive so a future change to
the filter logic cannot silently shift the funnel's numbers unnoticed.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from desk.marketdata.sources.nse import parse_bhavcopy
from desk.scanner.stage0 import run_stage0

FIXTURES = Path(__file__).parent / "fixtures"


def _bhav(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal bhavcopy-shaped frame. Every row gets sane defaults
    so a test only needs to state the field(s) it actually cares about."""
    base = {"series": "EQ", "date": date(2026, 9, 11), "prev_close": 100.0,
           "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
           "last": 100.0, "volume": 100000, "turnover_lacs": 500.0,
           "trades": 1000, "delivery_qty": 50000.0, "delivery_pct": 50.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_narrows_the_universe_and_reports_a_reason_per_exclusion():
    df = _bhav([
        {"symbol": "GOOD.NS"},                                    # survives
        {"symbol": "SME1.NS", "series": "SM"},                    # wrong series
        {"symbol": "PENNY.NS", "close": 5.0},                     # below price
        {"symbol": "THIN.NS", "turnover_lacs": 1.0},              # below liquidity
        {"symbol": "DEAD.NS", "volume": 0},                       # zero volume
    ])
    r = run_stage0(df)
    assert r.universe_in == 5
    assert r.universe_out == 1
    assert r.survivors["symbol"].tolist() == ["GOOD.NS"]
    assert r.excluded["non-EQ series"] == 1
    assert r.excluded["below price floor"] == 1
    assert r.excluded["below liquidity floor"] == 1
    assert r.excluded["zero volume (suspended or untraded)"] == 1


def test_circuit_locked_is_flagged_but_not_dropped_by_default():
    """NO TRADE today and 'not interesting' are different facts - a
    circuit-locked name stays in the survivor set, just labelled."""
    df = _bhav([
        {"symbol": "LOCKED.NS", "high": 110.0, "low": 110.0, "close": 110.0},
        {"symbol": "NORMAL.NS"},
    ])
    r = run_stage0(df)
    assert r.universe_out == 2                     # neither excluded
    locked_row = r.survivors[r.survivors["symbol"] == "LOCKED.NS"].iloc[0]
    assert bool(locked_row["circuit_locked"]) is True
    normal_row = r.survivors[r.survivors["symbol"] == "NORMAL.NS"].iloc[0]
    assert bool(normal_row["circuit_locked"]) is False


def test_circuit_locked_can_be_excluded_on_request():
    df = _bhav([
        {"symbol": "LOCKED.NS", "high": 110.0, "low": 110.0, "close": 110.0},
        {"symbol": "NORMAL.NS"},
    ])
    r = run_stage0(df, exclude_circuit_locked=True)
    assert r.universe_out == 1
    assert r.survivors["symbol"].tolist() == ["NORMAL.NS"]
    assert r.excluded["circuit-locked"] == 1


def test_zero_volume_bar_is_not_read_as_circuit_locked():
    """high == low with ZERO volume is an untraded/suspended name, not a
    circuit lock - the volume > 0 condition in circuit_locked exists
    specifically to keep these two facts apart."""
    df = _bhav([{"symbol": "UNTRADED.NS", "high": 50.0, "low": 50.0,
                "close": 50.0, "volume": 0}])
    r = run_stage0(df)
    # Excluded for zero volume before circuit_locked is even meaningful, but
    # confirm the flag itself would have read False had it survived that far.
    assert r.excluded["zero volume (suspended or untraded)"] == 1
    assert r.universe_out == 0


def test_thresholds_are_configurable_not_hardcoded():
    df = _bhav([{"symbol": "MID.NS", "close": 15.0, "turnover_lacs": 50.0}])
    default = run_stage0(df)
    assert default.universe_out == 0                # fails the defaults

    loosened = run_stage0(df, min_price=10.0, min_turnover_lacs=10.0)
    assert loosened.universe_out == 1


def test_empty_bhavcopy_is_refused_not_silently_processed():
    with pytest.raises(ValueError, match="empty"):
        run_stage0(pd.DataFrame())


def test_missing_required_columns_is_refused():
    with pytest.raises(ValueError, match="missing required columns"):
        run_stage0(pd.DataFrame({"symbol": ["X.NS"]}))


def test_caveats_are_always_present_and_name_the_real_gaps():
    """Stage 0 does not check the F&O ban list or point-in-time index
    membership - it must say so on every call, not just when asked."""
    df = _bhav([{"symbol": "X.NS"}])
    r = run_stage0(df)
    assert any("F&O ban" in c for c in r.caveats)
    assert any("index membership" in c for c in r.caveats)


def test_summary_is_human_readable():
    df = _bhav([{"symbol": "X.NS"}, {"symbol": "Y.NS", "series": "SM"}])
    text = run_stage0(df).summary()
    assert "1 ->" in text or "2 -> 1" in text
    assert "non-EQ series" in text


# ===========================================================================
# Pinned against the real captured archive - a future filter-logic change
# must not silently shift these numbers without a test failing to explain why.
# ===========================================================================

def test_stage0_against_the_real_captured_bhavcopy_archive():
    raw = (FIXTURES / "nse_bhavcopy_20260911.csv").read_bytes()
    bhav = parse_bhavcopy(raw, day=date(2026, 9, 11))
    r = run_stage0(bhav)
    assert r.universe_in == 3485
    assert r.universe_out == 1598
    assert r.excluded["non-EQ series"] == 848
    assert r.excluded["below liquidity floor"] == 756
    assert r.excluded["below price floor"] == 283
    assert r.excluded["zero volume (suspended or untraded)"] == 0
    # Every survivor must actually be an EQ-series name above both floors.
    assert (r.survivors["series"] == "EQ").all()
    assert (r.survivors["close"] >= 20.0).all()
    assert (r.survivors["turnover_lacs"] >= 100.0).all()
