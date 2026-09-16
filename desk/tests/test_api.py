"""API-layer tests. Uses FastAPI's TestClient - no live server, no network.

Exercises what a running desk actually depends on: the calendar cache
invalidating on refresh, /health reflecting real state rather than a bare
"process is up", and /calendar/{day} not 503ing for the day it was actually
asked about.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from desk.api import main as api_main

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point the API at a throwaway config dir so these tests never touch
    the real configs/holidays_nse.json, and clear the process-wide lru_cache
    between tests so one test's calendar can't leak into the next."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    monkeypatch.setattr(api_main, "CONFIGS", cfg_dir)
    api_main._calendar_cached.cache_clear()
    yield cfg_dir
    api_main._calendar_cached.cache_clear()


def _write_calendar(cfg_dir, years, holidays):
    (cfg_dir / "holidays_nse.json").write_text(json.dumps({
        "exchange": "NSE", "years": years, "holidays": holidays,
        "special_sessions": [],
    }), encoding="utf-8")


def test_calendar_cache_invalidates_when_the_file_is_refreshed(isolated_config):
    """REGRESSION target: a bare @lru_cache() would serve the calendar loaded
    at process start forever. Keying on the file's mtime must pick up a
    refresh with zero restart."""
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-01-26", "name": "Republic Day"}])
    cal1 = api_main._calendar()
    assert cal1.loaded_years == [2026]
    assert api_main._calendar() is cal1               # same mtime -> cache hit

    time.sleep(0.05)      # ensure a strictly later mtime on fast filesystems
    _write_calendar(isolated_config, [2026, 2027],
                    [{"date": "2026-01-26", "name": "Republic Day"},
                     {"date": "2027-01-26", "name": "Republic Day"}])
    cal2 = api_main._calendar()
    assert cal2 is not cal1                            # new mtime -> reloaded
    assert cal2.loaded_years == [2026, 2027]


def test_calendar_raises_a_named_error_when_the_file_is_absent(isolated_config):
    from desk.marketdata.calendar_in import CalendarError
    with pytest.raises(CalendarError, match="refresh calendar"):
        api_main._calendar()


def test_health_reflects_real_calendar_status_not_just_process_up(isolated_config):
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-01-26", "name": "Republic Day"}])
    client = TestClient(api_main.app)
    body = client.get("/health").json()
    assert body["calendar"]["loaded_years"] == [2026]
    # covers_current_year depends on the real clock, so just assert the shape
    # rather than the value - the point under test is that it's DERIVED, not
    # hardcoded "ok".
    assert "covers_current_year" in body["calendar"]


def test_health_degrades_when_the_calendar_is_missing(isolated_config):
    client = TestClient(api_main.app)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["calendar"] == {"error": "unavailable"}


def test_calendar_endpoint_does_not_503_for_the_requested_day_at_a_year_boundary(isolated_config):
    """REGRESSION: querying Jan 1 or Dec 31 of the ONLY loaded year 503'd,
    because the neighbour-day lookups crossed into an unloaded year and that
    failure took down the whole response - for a date the caller never asked
    about."""
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-01-26", "name": "Republic Day"}])
    client = TestClient(api_main.app)

    r = client.get("/calendar/2026-01-01")
    assert r.status_code == 200
    body = r.json()
    assert body["is_trading_day"] is True
    assert body["previous_trading_day"] is None      # crosses into 2025, unloaded
    assert body["next_trading_day"] == "2026-01-02"

    r2 = client.get("/calendar/2026-12-31")
    assert r2.status_code == 200
    assert r2.json()["next_trading_day"] is None      # crosses into 2027, unloaded


def test_calendar_endpoint_503s_for_a_genuinely_unloaded_year(isolated_config):
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-01-26", "name": "Republic Day"}])
    client = TestClient(api_main.app)
    r = client.get("/calendar/2025-06-15")
    assert r.status_code == 503


def test_open_positions_list_is_bounded(isolated_config):
    """REGRESSION: open_positions had no max_length. Verified pre-fix that a
    50,000-entry (23.5MB) payload was accepted and fully processed in 0.57s
    with no ceiling at all."""
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-01-26", "name": "Republic Day"}])
    client = TestClient(api_main.app)
    cfg = client.get("/risk/config/default").json()
    huge = [{"symbol": "A", "qty": 1, "entry": 100, "stop": 90}] * 201
    r = client.post("/risk/size", json={
        "symbol": "X.NS", "entry": 100, "stop": 95, "target": 130,
        "config": cfg, "open_positions": huge,
    })
    assert r.status_code == 422


def test_symbol_field_is_bounded_against_log_injection_and_flooding(isolated_config):
    """REGRESSION: SizeRequest.symbol had no max_length and was interpolated
    directly into a log.info() call - an unbounded value there is both a
    log-flooding vector and a way to forge log lines with embedded newlines."""
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-01-26", "name": "Republic Day"}])
    client = TestClient(api_main.app)
    cfg = client.get("/risk/config/default").json()
    r = client.post("/risk/size", json={
        "symbol": "X" * 5000, "entry": 100, "stop": 95, "target": 130,
        "config": cfg,
    })
    assert r.status_code == 422


def test_calendar_error_body_does_not_leak_the_filesystem_path(isolated_config):
    """REGRESSION: the 503 body for /calendar/{day} returned str(exc) verbatim,
    which for a missing/malformed config includes the full absolute path -
    OS username, OneDrive folder, project directory layout - to an
    unauthenticated caller."""
    # No calendar file written at all -> _calendar() raises CalendarError
    # carrying the absolute path in its message.
    client = TestClient(api_main.app)
    r = client.get("/calendar/2026-01-26")
    assert r.status_code == 503
    body = r.json()["detail"]
    assert str(isolated_config) not in body
    assert "holidays_nse.json" not in body


# ===========================================================================
# /scanner/stage0
# ===========================================================================

def test_scanner_stage0_404s_when_no_snapshot_has_been_fetched(isolated_config):
    client = TestClient(api_main.app)
    r = client.get("/scanner/stage0?day=2026-09-11")
    assert r.status_code == 404
    assert "refresh bhavcopy" in r.json()["detail"]


def test_scanner_stage0_reads_an_already_fetched_snapshot(isolated_config):
    import json as _json
    from datetime import date as _date

    from desk.marketdata.sources.nse import parse_bhavcopy

    fixtures = FIXTURES
    raw = (fixtures / "nse_bhavcopy_20260911.csv").read_bytes()
    bhav = parse_bhavcopy(raw, day=_date(2026, 9, 11))

    bhav_dir = isolated_config / "bhavcopy"
    bhav_dir.mkdir()
    bhav.to_parquet(bhav_dir / "2026-09-11.parquet", index=False)

    client = TestClient(api_main.app)
    r = client.get("/scanner/stage0?day=2026-09-11")
    assert r.status_code == 200
    body = r.json()
    assert body["universe_in"] == 3485
    assert body["universe_out"] == 1598
    assert len(body["survivors"]) == 1598
    assert any("F&O ban" in c for c in body["caveats"])


def test_scanner_stage0_thresholds_are_adjustable_via_query_params(isolated_config):
    from datetime import date as _date

    from desk.marketdata.sources.nse import parse_bhavcopy

    fixtures = FIXTURES
    raw = (fixtures / "nse_bhavcopy_20260911.csv").read_bytes()
    bhav = parse_bhavcopy(raw, day=_date(2026, 9, 11))
    bhav_dir = isolated_config / "bhavcopy"
    bhav_dir.mkdir()
    bhav.to_parquet(bhav_dir / "2026-09-11.parquet", index=False)

    client = TestClient(api_main.app)
    loose = client.get("/scanner/stage0?day=2026-09-11&min_price=0&min_turnover_lacs=0")
    tight = client.get("/scanner/stage0?day=2026-09-11&min_turnover_lacs=100000")
    assert loose.json()["universe_out"] > tight.json()["universe_out"]


# ===========================================================================
# /scanner/stage1
# ===========================================================================

def _seed_history(cfg_dir, days, symbols=("AAA.NS", "BBB.NS")):
    """Write `days` snapshots so Stage 1 has something to compute over."""
    import numpy as np
    import pandas as pd

    bhav_dir = cfg_dir / "bhavcopy"
    bhav_dir.mkdir(exist_ok=True)
    for i, day in enumerate(days):
        close = [100.0 + i, 250.0 + i]
        pd.DataFrame({
            "symbol": list(symbols), "series": ["EQ"] * len(symbols),
            "date": [day] * len(symbols),
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": [500_000, 900_000],
            "turnover_lacs": [5000.0, 9000.0],
            "delivery_pct": [50.0, 55.0],
        }).to_parquet(bhav_dir / f"{day.isoformat()}.parquet", index=False)


def _weekdays(n, end=date(2026, 9, 11)):
    from datetime import timedelta
    out, cur = [], end
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur -= timedelta(days=1)
    return sorted(out)


def test_scanner_stage1_404s_when_the_day_has_no_snapshot(isolated_config):
    client = TestClient(api_main.app)
    r = client.get("/scanner/stage1?day=2026-09-11")
    assert r.status_code == 404
    assert "refresh bhavcopy" in r.json()["detail"]


def test_scanner_stage1_computes_features_from_accumulated_snapshots(isolated_config):
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    r = client.get("/scanner/stage1?day=2026-09-11&lookback=60&min_bars=30"
                   "&flagged_only=false")
    assert r.status_code == 200
    body = r.json()
    assert body["as_of"] == "2026-09-11"
    assert body["universe_out"] == 2
    syms = {row["symbol"] for row in body["rows"]}
    assert syms == {"AAA.NS", "BBB.NS"}


def test_scanner_stage1_reports_coverage_so_gaps_are_visible(isolated_config):
    """A feature table computed over a window with missing fetches must not
    be indistinguishable from one computed over clean data."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-01-26", "name": "Republic Day"}])
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                      "&flagged_only=false").json()
    cov = body["coverage"]
    assert cov["calendar_checked"] is True
    assert cov["days_loaded"] == 60
    assert "describe" in cov


def test_scanner_stage1_degrades_to_unchecked_coverage_with_no_calendar(
        isolated_config):
    """A missing holiday file must not 503 a scan that is otherwise perfectly
    computable - it downgrades the gap check, nothing else."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)      # no calendar written
    client = TestClient(api_main.app)
    r = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                   "&flagged_only=false")
    assert r.status_code == 200
    cov = r.json()["coverage"]
    assert cov["calendar_checked"] is False
    assert cov["complete"] is False


def test_scanner_stage1_declares_what_it_could_not_check(isolated_config):
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                      "&flagged_only=false").json()
    joined = " ".join(body["unavailable"])
    assert "Sector strength" in joined and "News" in joined


def test_scanner_stage1_nan_features_serialise_as_null_not_zero(isolated_config):
    """A 200-day distance over 60 bars is UNKNOWN. Serialised as 0.0 it reads
    as 'exactly at the 200-DMA', which is a completely different claim."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                      "&flagged_only=false").json()
    row = body["rows"][0]
    assert row["dist_sma200_pct"] is None
    assert row["rs_rank"] is None
    assert row["dist_sma20_pct"] is not None


def test_scanner_stage1_response_is_bounded_by_limit(isolated_config):
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                      "&flagged_only=false&limit=1").json()
    assert len(body["rows"]) == 1
    assert body["universe_out"] == 2          # true count still reported


# ===========================================================================
# /scanner/stage2
# ===========================================================================

def test_scanner_stage2_ranks_and_explains(isolated_config):
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    r = client.get("/scanner/stage2?day=2026-09-11&lookback=60&min_bars=30"
                   "&regime=trending_up&min_factors=1&flagged_only=false")
    assert r.status_code == 200
    body = r.json()
    assert body["regime"] == "trending_up"
    assert body["universe_out"] == 2
    row = body["rows"][0]
    # Every score must carry its own derivation - no second call needed.
    assert set(row["factor_ranks"]) == {f["name"] for f in body["factors"]}
    assert row["symbol"] in row["explain"]
    assert body["rows"][0]["score"] >= body["rows"][-1]["score"]


def test_scanner_stage2_in_a_crisis_returns_no_ranking_and_says_why(
        isolated_config):
    """NO TRADE is a successful outcome. A crisis silences enough of the
    default factor set that no honest ranking can be produced, and the
    response must say that rather than returning a thin one."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage2?day=2026-09-11&lookback=60"
                      "&regime=crisis&flagged_only=false").json()
    assert body["universe_out"] == 0
    assert body["rows"] == []
    assert "momentum" in body["silenced"]
    assert any("crisis" in k for k in body["excluded"])


def test_scanner_stage2_404s_without_a_snapshot(isolated_config):
    client = TestClient(api_main.app)
    r = client.get("/scanner/stage2?day=2026-09-11")
    assert r.status_code == 404


def test_scanner_stage2_declares_that_untrusted_strategies_did_not_vote(
        isolated_config):
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage2?day=2026-09-11&lookback=60"
                      "&min_factors=1&flagged_only=false").json()
    assert any("maturity ladder" in u for u in body["unavailable"])


# ===========================================================================
# /plan/today - the whole funnel
# ===========================================================================

def test_plan_today_runs_the_whole_funnel_when_a_snapshot_exists(isolated_config):
    """Stage 0 -> 1 -> 2 -> 4 end to end through the HTTP surface. The counts
    must be real, not zeros-by-construction."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _write_calendar(isolated_config, [2026], [])
    client = TestClient(api_main.app)
    body = client.get("/plan/today?day=2026-09-11&regime=trending_up"
                      "&lookback=60&capital=10000000").json()
    assert body["as_of"] == "2026-09-11"
    assert body["universe_scanned"] == 2
    # Whatever the outcome, the plan must carry the funnel's own caveats.
    joined = " ".join(body["warnings"])
    assert "F&O ban" in joined
    assert "Sector strength" in joined
    assert any("sessions" in w for w in body["warnings"])


def test_plan_today_says_how_to_fix_a_missing_snapshot(isolated_config):
    """No network call from inside a request handler. A missing snapshot is a
    NO TRADE naming the fetch command, not a silent empty plan."""
    _write_calendar(isolated_config, [2026], [])
    client = TestClient(api_main.app)
    body = client.get("/plan/today?day=2026-09-11").json()
    assert "no scan was run" in body["no_trade_reason"]
    assert "refresh bhavcopy" in body["no_trade_reason"]
    assert body["universe_scanned"] == 0


def test_plan_today_returns_no_trade_on_a_holiday_regardless_of_the_scan(
        isolated_config):
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _write_calendar(isolated_config, [2026],
                    [{"date": "2026-09-11", "name": "Test Holiday"}])
    client = TestClient(api_main.app)
    body = client.get("/plan/today?day=2026-09-11&lookback=60").json()
    assert "Test Holiday" in body["no_trade_reason"]
    assert body["trades"] == []


def test_plan_today_never_500s_when_the_scan_cannot_run(isolated_config):
    """The plan endpoint's job is to answer honestly EVERY day. 'The scan
    could not run' is an answer; a 500 is not."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _write_calendar(isolated_config, [2026], [])
    client = TestClient(api_main.app)
    # lookback far beyond what was seeded - the funnel must degrade, not crash
    r = client.get("/plan/today?day=2026-09-11&lookback=300")
    assert r.status_code == 200
    assert r.json()["no_trade_reason"] is not None or r.json()["trades"]


# ===========================================================================
# REGRESSION: the two gaps the architecture review found on 2026-09-15
# ===========================================================================

def _seed_actions(cfg_dir, symbol="AAA", actions=()):
    d = cfg_dir / "corporate_actions"
    d.mkdir(exist_ok=True)
    (d / f"{symbol}.json").write_text(json.dumps({
        "symbol": f"{symbol}.NS", "actions": list(actions),
    }), encoding="utf-8")


def test_corporate_actions_on_file_actually_reach_stage1(isolated_config):
    """REGRESSION: run_stage1 has a guard that drops a symbol whose raw prices
    contain a split inside the lookback window - an unadjusted 1:2 split reads
    as a -50% day and poisons every rolling statistic. The guard was written,
    the data was being fetched to configs/corporate_actions/, and NO endpoint
    passed the two to each other. A feature that exists and is never called is
    indistinguishable from one that was never written."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _seed_actions(isolated_config, "AAA", [
        {"ex_date": days[30].isoformat(), "type": "split",
         "factor": 2.0, "amount": 0.0, "note": "Split 1:2"},
    ])
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                      "&min_bars=30&flagged_only=false").json()
    assert body["unadjusted_actions"] == {"AAA.NS": 1}
    assert "AAA.NS" not in {r["symbol"] for r in body["rows"]}
    assert "BBB.NS" in {r["symbol"] for r in body["rows"]}


def test_symbols_with_no_action_file_are_named_as_unchecked(isolated_config):
    """"No actions on file" and "no file" are different facts and only one of
    them is reassuring. A name with no file has not been cleared of splits -
    it has not been looked at."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                      "&min_bars=30&flagged_only=false").json()
    joined = " ".join(body["unavailable"])
    assert "no corporate-action file" in joined
    assert "refresh actions" in joined


def test_a_corrupt_action_file_does_not_pass_as_no_actions(isolated_config):
    """It must not take down the scan, and it must not read as a clean bill
    either - that would silently re-enable the exact failure the guard exists
    to prevent."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    d = isolated_config / "corporate_actions"
    d.mkdir(exist_ok=True)
    (d / "AAA.json").write_text("{ not json", encoding="utf-8")
    client = TestClient(api_main.app)
    r = client.get("/scanner/stage1?day=2026-09-11&lookback=60"
                   "&min_bars=30&flagged_only=false")
    assert r.status_code == 200
    assert "no corporate-action file" in " ".join(r.json()["unavailable"])


def test_plan_today_can_be_sized_against_a_portfolio_you_already_hold(
        isolated_config):
    """REGRESSION: the GET form sized every plan against an EMPTY book, which
    makes the sector cap, correlated-cluster cap, portfolio heat,
    max-open-positions and daily-loss gates all inert - in the one endpoint
    that is meant to be the actual daily decision surface. /risk/size had
    always accepted a portfolio; the plan endpoint must not be the one place
    that quietly ignores one."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _write_calendar(isolated_config, [2026], [])
    client = TestClient(api_main.app)

    flat = client.post("/plan/today", json={
        "day": "2026-09-11", "regime": "trending_up", "lookback": 60,
        "capital": 10_000_000,
    }).json()

    # A book already at the max-open-positions limit must block everything.
    full = client.post("/plan/today", json={
        "day": "2026-09-11", "regime": "trending_up", "lookback": 60,
        "capital": 10_000_000,
        "open_positions": [
            {"symbol": f"HELD{i}.NS", "qty": 10, "entry": 100, "stop": 90}
            for i in range(5)
        ],
    }).json()
    assert full["trades"] == []
    assert full["no_trade_reason"] is not None
    # And the flat-book call must have been able to reach the gate at all,
    # or this test would pass for the wrong reason.
    assert flat["analysed"] >= full["analysed"]


def test_the_post_and_get_plan_forms_agree_on_a_flat_book(isolated_config):
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _write_calendar(isolated_config, [2026], [])
    client = TestClient(api_main.app)
    get = client.get("/plan/today?day=2026-09-11&regime=trending_up"
                     "&lookback=60&capital=10000000").json()
    post = client.post("/plan/today", json={
        "day": "2026-09-11", "regime": "trending_up", "lookback": 60,
        "capital": 10_000_000,
    }).json()
    assert get["universe_scanned"] == post["universe_scanned"]
    assert get["analysed"] == post["analysed"]
    assert len(get["trades"]) == len(post["trades"])


def test_lookback_is_capped_so_one_request_cannot_read_years_of_history(
        isolated_config):
    """REGRESSION: lookback was bounded at 1000, which a security review
    measured as 1,045,500 rows / 576MB / ~12s of CPU for a single
    unauthenticated GET - synchronously, inside the request handler, and
    growing with every trading day the store accumulates."""
    client = TestClient(api_main.app)
    assert api_main.LOOKBACK_CEILING <= 400
    for path in ("/scanner/stage1", "/scanner/stage2", "/plan/today"):
        r = client.get(f"{path}?day=2026-09-11&lookback=1000")
        assert r.status_code == 422, path


def test_scanner_endpoints_do_not_leak_the_filesystem_path_on_a_bad_snapshot(
        isolated_config):
    """REGRESSION: StoreError messages embed the absolute path of the file -
    OS username, OneDrive folder, project layout - and both scanner endpoints
    re-raised them verbatim as HTTPException(400, str(exc)). /calendar/{day}
    had already been hardened against exactly this; the newer endpoints had
    not."""
    bhav_dir = isolated_config / "bhavcopy"
    bhav_dir.mkdir(exist_ok=True)
    (bhav_dir / "2026-09-11.parquet").write_bytes(b"this is not parquet")
    client = TestClient(api_main.app)
    for path in ("/scanner/stage1", "/scanner/stage2"):
        r = client.get(f"{path}?day=2026-09-11&lookback=60")
        assert r.status_code == 503, path
        body = r.json()["detail"]
        assert str(isolated_config) not in body, path
        assert ".parquet" not in body, path


def test_history_is_read_with_a_column_projection(isolated_config, monkeypatch):
    """REGRESSION: no call site passed `columns=`, so every scan read the
    whole bhavcopy row - prev_close, last, turnover_lacs, trades,
    delivery_qty, delivery_pct - across every day in the window, for fields
    Stage 1 never touches."""
    from desk.store import BarStore

    days = _weekdays(60)
    _seed_history(isolated_config, days)
    seen = {}
    original = BarStore.history

    def spy(self, **kw):
        seen.update(kw)
        return original(self, **kw)

    monkeypatch.setattr(BarStore, "history", spy)
    client = TestClient(api_main.app)
    client.get("/scanner/stage1?day=2026-09-11&lookback=60&min_bars=30")
    assert seen.get("columns"), "history() was called without a projection"
    assert "turnover_lacs" not in seen["columns"]
    assert "close" in seen["columns"]


def test_stage2_carries_the_corporate_action_caveat_forward(isolated_config):
    """REGRESSION: when the Stage 0+1 setup was extracted into a shared
    helper, Stage 2's copy of the caveat was briefly computed as an
    always-empty slice of stage1.unavailable. The caveat names which symbols
    could NOT be checked for splits - dropping it silently is exactly the
    failure the caveat exists to prevent."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    client = TestClient(api_main.app)
    body = client.get("/scanner/stage2?day=2026-09-11&lookback=60"
                      "&min_factors=1&flagged_only=false").json()
    assert any("no corporate-action file" in u for u in body["unavailable"])


def test_the_plan_reports_all_four_funnel_stages(isolated_config):
    """survived_stage0 was populated into ScanSummary and then discarded -
    the funnel has four stages and the plan was only ever showing three."""
    days = _weekdays(60)
    _seed_history(isolated_config, days)
    _write_calendar(isolated_config, [2026], [])
    client = TestClient(api_main.app)
    body = client.get("/plan/today?day=2026-09-11&lookback=60").json()
    for field in ("universe_scanned", "survived_stage0", "survived_stage1",
                  "survived_stage2", "analysed"):
        assert field in body, field
    assert body["survived_stage0"] == 2
