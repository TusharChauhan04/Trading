"""FastAPI surface. The web app talks only to this - never to an agent directly.

Run:  .venv/Scripts/python.exe -m uvicorn desk.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from desk.contracts.enums import Regime
from desk.marketdata.calendar_in import (
    CalendarError,
    CalendarNotLoaded,
    TradingCalendar,
)
from desk.marketdata.corporate_actions import ActionLoadError, load_actions
from desk.marketdata.providers import provider_status
from desk.marketdata.sectors import SectorMap
from desk.marketdata.symbols import SymbolError, Symbol
from desk.plan.build import build_plan
from desk.plan.models import DailyPlan, ScanSummary
from desk.registry.fleet import fleet_status
from desk.risk.engine import Portfolio, Position, RiskConfig, Sizing, size_position
from desk.scanner.stage0 import run_stage0
from desk.scanner.stage1 import REQUIRED_BARS, Stage1Result, run_stage1
from desk.llm.budget import CostMeter
from desk.llm.client import MeteredClient
from desk.llm.providers.openai import OpenAIProvider
from desk.regime.engine import compute_regime
from desk.research.events import load_calendar
from desk.research.fundamentals import FundamentalsCache
from desk.scanner.stage2 import run_stage2
from desk.scanner.stage3 import run_stage3
from desk.scanner.stage4 import run_stage4
from desk.store import BarStore, StoreError
from desk.strategies.catalog import catalog_status, eligible

log = logging.getLogger("desk.api")

#: Where the generated configs live. Overridable because on any machine that
#: is not this laptop, the config directory is a mounted volume, not a path
#: two levels up from the source tree - and a value that can only be changed
#: by editing code is not configuration.
CONFIGS = Path(os.environ.get("DESK_CONFIG_DIR")
               or Path(__file__).resolve().parents[2] / "configs")

#: Hard ceiling on how much history one synchronous request may ask for.
#: Measured: a 1000-session full-universe scan reads 1,045,500 rows, peaks at
#: 576MB and takes ~12s IN THE REQUEST HANDLER - and the store is designed to
#: keep accumulating, so that figure grows with every trading day. 300
#: sessions is comfortably more than any feature here needs (the longest
#: window is the 252-session ATR percentile) while keeping the worst case a
#: single caller can force to roughly a third of that.
#: This is a mitigation, not a fix. The real fix is precomputing the scan
#: once per day into a cached artifact, the same way the bhavcopy snapshot
#: itself is fetched once rather than per request.
LOOKBACK_CEILING = 300

# Every date computation in this system is about NSE, which trades in IST and
# nowhere else. date.today() alone reads server-local time - fine on a laptop
# in India, silently wrong by 5:30h the moment this runs on a UTC-clock VPS.
IST = ZoneInfo("Asia/Kolkata")


def _today_ist() -> date:
    from datetime import datetime
    return datetime.now(IST).date()


def _calendar_uncached(mtime: float) -> TradingCalendar:
    """`mtime` is part of the cache key on purpose - see `_calendar()`."""
    log.info("loading trading calendar from %s (mtime=%s)",
             CONFIGS / "holidays_nse.json", mtime)
    cal = TradingCalendar.from_file(CONFIGS / "holidays_nse.json")
    log.info("calendar loaded: years=%s", cal.loaded_years)
    return cal


_calendar_cached = lru_cache(maxsize=2)(_calendar_uncached)


def _calendar() -> TradingCalendar:
    """Reloads automatically when the config file changes on disk.

    `python -m desk.marketdata.refresh calendar` rewrites this exact file, and
    that is the documented, expected workflow - not a hypothetical "year ever
    hot-added". Keying purely on `lru_cache()` with no arguments would serve
    the calendar loaded at process start forever, and at more than one worker
    the workers could disagree about which year is even loaded. Keying on the
    file's mtime costs one `stat()` per request (microseconds) and converges
    every caller, in-process or across workers, on the same file version.
    """
    try:
        mtime = (CONFIGS / "holidays_nse.json").stat().st_mtime
    except OSError as exc:
        raise CalendarError(
            f"holiday file not found: {CONFIGS / 'holidays_nse.json'}. Run "
            f"python -m desk.marketdata.refresh calendar first."
        ) from exc
    return _calendar_cached(mtime)


def _calendar_or_none() -> TradingCalendar | None:
    """For callers where the calendar improves the answer but is not required.

    The store uses it to tell a genuine data gap from a market holiday. With
    no calendar it reports gaps as UNCHECKED, which is an honest degraded
    answer - so a missing holiday file must not 503 a scan that is otherwise
    perfectly computable.
    """
    try:
        return _calendar()
    except CalendarError:
        return None

CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "DESK_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

app = FastAPI(
    title="The India Desk",
    description="NSE/BSE decision-support. Research and risk only - no order routing.",
    version="0.1.0",
)

# Vite dev server by default. DESK_CORS_ORIGINS (comma-separated) widens it
# without a code edit - but note that widening CORS is NOT the same as making
# this safe to expose: there is still no authentication on any endpoint, so a
# non-localhost origin needs network-level restriction (VPN, allowlist) or an
# auth layer first. CORS is a browser convention, not an access control.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- health ---

@app.get("/health", tags=["system"])
def health() -> dict:
    """Reflects the process's actual dependency, not just that it is running.

    The only dependency this service has is the calendar file - so "the
    process is up" while that file is missing or stale is exactly the false
    green light a liveness-only check gives. This checks whether the CURRENT
    year is actually loaded, since a server can be "up" for months on a
    calendar that quietly stopped covering the present.
    """
    try:
        cal = _calendar()
        current_year = _today_ist().year
        calendar_ok = current_year in cal.loaded_years
        data = _data_freshness(cal if calendar_ok else None)
        return {
            "status": "ok" if (calendar_ok and not data["stale"]) else "degraded",
            "service": "india-desk", "version": app.version,
            "calendar": {"loaded_years": cal.loaded_years,
                        "covers_current_year": calendar_ok},
            "data": data,
        }
    except CalendarError as exc:
        log.warning("health check: calendar unavailable: %s", exc)
        return {"status": "degraded", "service": "india-desk",
                "version": app.version, "calendar": {"error": "unavailable"},
                "data": _data_freshness(None)}


def _data_freshness(cal: TradingCalendar | None) -> dict:
    """Is the market data current, or is the desk quietly trading on last
    week's prices?

    THIS EXISTS BECAUSE /health WAS LYING BY OMISSION. It checked the calendar
    and nothing else, so an uptime monitor polling it would report "ok"
    indefinitely while configs/bhavcopy/ sat days stale - the scanner's whole
    input depends on someone running the refresh every trading day, and there
    is no scheduler. The failure was visible ONLY to a human who happened to
    open the web app and read the NO TRADE banner. A liveness check that
    cannot see the one dependency most likely to break is worse than none,
    because it actively reassures.
    """
    store = BarStore(CONFIGS / "bhavcopy")
    days = store.available_days()
    latest = days[-1] if days else None

    expected = None
    if cal is not None:
        try:
            today = _today_ist()
            expected = today if cal.is_trading_day(today) else \
                cal.previous_trading_day(today)
        except CalendarError:
            expected = None

    # With no calendar we cannot say what SHOULD be there, so we refuse to
    # call it fresh - unknown is not the same as fine, the same rule the
    # store's Coverage and the risk engine's checks_skipped both follow.
    if latest is None:
        stale, behind = True, None
    elif expected is None:
        stale, behind = True, None
    else:
        behind = cal.session_count(latest, expected) - 1 if latest <= expected else 0
        stale = latest < expected

    return {
        "latest_snapshot": latest.isoformat() if latest else None,
        "expected_as_of": expected.isoformat() if expected else None,
        "sessions_behind": behind,
        "snapshots_on_disk": len(days),
        "stale": stale,
        "fix": (None if not stale else
                "python -m desk.marketdata.refresh bhavcopy --date "
                + (expected.isoformat() if expected else "<TRADING DAY>")),
    }


@app.get("/fleet", tags=["system"])
def fleet() -> list[dict]:
    """Which agents exist, what they can do, and whether they are wired in."""
    return fleet_status()


# -------------------------------------------------------------- providers ---

@app.get("/providers", tags=["marketdata"])
def providers() -> list[dict]:
    """Data providers scored against the twelve criteria, with the criteria
    still unassessed named rather than left to read as passing."""
    return provider_status()


@app.get("/calendar/{day}", tags=["marketdata"])
def calendar_day(day: date) -> dict:
    """Is the market open, and when does this trade settle?

    Returns 503 rather than a guess when the year has no holiday data - the
    calendar refuses to assume weekends-only, and so does this endpoint.
    """
    try:
        cal = _calendar()
    except CalendarError as exc:
        # The underlying message carries a full filesystem path - config
        # location, which on a real machine includes the OS username and the
        # project's directory layout. Log it server-side where it is actually
        # useful for debugging, and give an unauthenticated caller only the
        # fact that something is wrong, not where on disk to go looking.
        log.warning("calendar unavailable for /calendar/%s: %s", day, exc)
        raise HTTPException(503, "trading calendar unavailable") from exc
    try:
        trading = cal.is_trading_day(day)
        special = cal.special_session(day)
    except CalendarNotLoaded as exc:
        raise HTTPException(503, str(exc)) from exc
    except CalendarError as exc:
        raise HTTPException(400, str(exc)) from exc

    # The neighbour lookups are best-effort, not part of what was asked. NSE's
    # holiday master publishes exactly one year at a time, so `_loaded_years`
    # is almost always a singleton - which means EVERY query for the first or
    # last trading day of that year would 503, for a date the caller never
    # asked about, unless these are allowed to come back None instead of
    # failing the whole response.
    def _neighbour(fn) -> str | None:
        try:
            return fn(day).isoformat()
        except CalendarNotLoaded:
            return None

    return {
        "date": day.isoformat(),
        "exchange": cal.exchange,
        "is_trading_day": trading,
        "is_weekend": cal.is_weekend(day),
        "holiday": cal.holiday_name(day),
        "special_session": (
            {"name": special.name, "start": special.start.isoformat(),
             "end": special.end.isoformat()} if special else None
        ),
        # Same reasoning: settlement on the last trading day of the loaded
        # year needs next_trading_day() too, and can cross the same boundary.
        "settlement_date": (_neighbour(cal.settlement_date) if trading else None),
        "previous_trading_day": _neighbour(cal.previous_trading_day),
        "next_trading_day": _neighbour(cal.next_trading_day),
    }


def _require_snapshot(target: date) -> None:
    """404 with the command that fixes it, rather than an empty result that
    reads as "nothing qualified today"."""
    if not BarStore(CONFIGS / "bhavcopy").has(target):
        raise HTTPException(
            404,
            f"no bhavcopy snapshot for {target} - run 'python -m "
            f"desk.marketdata.refresh bhavcopy --date {target.isoformat()}' first"
        )


def _actions_for(symbols: list[str]) -> tuple[list, list[str]]:
    """Corporate actions on file for these names, plus the ones with NO file.

    Stage 1 drops a symbol whose raw prices contain a split inside the
    lookback window, because an unadjusted 1:2 split reads as a -50% day and
    poisons every rolling statistic over it. That protection is only as good
    as the actions actually handed to it, so this is threaded into every call
    site rather than left as an optional argument nobody passes.

    A name with no file has NOT been cleared of splits - it has not been
    looked at - and the caller is told which names those are.
    """
    try:
        return load_actions(CONFIGS / "corporate_actions", symbols=symbols)
    except ActionLoadError as exc:
        # A corrupt action file must not take down a scan, but it must not
        # pass as "no actions" either - that would silently re-enable the
        # exact failure Stage 1's guard exists to prevent.
        log.warning("corporate actions unreadable, scan runs UNGUARDED: %s", exc)
        return [], list(symbols)


def _action_caveat(unchecked: list[str], total: int) -> list[str]:
    if not unchecked:
        return []
    return [f"{len(unchecked)} of {total} scanned symbols have no "
            f"corporate-action file, so an unadjusted split or bonus inside "
            f"the lookback window would not have been caught for them. Run "
            f"'python -m desk.marketdata.refresh actions <SYMBOLS>'."]


def _regime_caveats(measured) -> list[str]:
    """What the measured regime could NOT see.

    A regime that is only partly measured must say so, or a plan reports
    "range" with the same confidence whether four dimensions were computed
    or one. `RegimeState.is_measured` already draws that line; this turns
    it into something the plan prints.
    """
    if measured is None:
        return []
    out = []
    if not measured.is_measured:
        unknown = [k for k, v in measured.sources.items()
                   if str(v).startswith("not measured")]
        out.append(
            f"the market regime is only PARTLY measured - "
            f"{', '.join(unknown) or 'some dimensions'} could not be "
            f"computed, so '{measured.label.value}' rests on the "
            f"dimensions that could.")
    for line in measured.explain():
        if line.startswith("leading:"):
            out.append("regime " + line)
    return out


def _event_calendar(as_of: date):
    """(calendar, caveats) for the earnings gate.

    Returns None for the calendar in BOTH the missing and the stale case,
    but with different caveats, because they are different failures. A
    missing calendar means nobody fetched it. A stale one means somebody did
    and then stopped, which is more dangerous: a calendar fetched two weeks
    ago answers "nothing scheduled" for every company that has announced a
    board meeting since, and the gate would report a clean check. Staleness
    is a FALSE CLEAR, so a stale file is treated as absent rather than used.
    """
    stored = load_calendar(CONFIGS / "research" / "event_calendar.json",
                           as_of=as_of)
    if stored is None:
        return None, ["no event calendar on file, so the earnings gate did "
                      "NOT run - a name reporting inside the holding window "
                      "would not have been caught. Run 'python -m "
                      "desk.research.refresh events'."]
    if stored.stale:
        return None, [stored.caveat]
    return stored.calendar, []


def _fundamentals(as_of: date):
    """(table, caveats) for Stage 2's fundamental layer.

    Read from the pre-open cache rather than computed here. build_table is
    ~12s for the full universe and this endpoint is stateless and recomputes
    the funnel on every request, so computing it inline would put 12 seconds
    on every /plan/today. See desk/research/fundamentals.py.

    A table too far behind the scan date is treated as ABSENT, the same way
    a stale event calendar is: reporting last quarter's revenue as this
    quarter's is a false statement rather than a slightly old one. A table
    dated AFTER the scan date is never even opened - the cache selects by
    filename, so look-ahead is structurally impossible rather than checked.
    """
    cached = FundamentalsCache(CONFIGS / "research" / "fundamentals").load(as_of)
    if cached is None:
        return None, ["no fundamentals table on file, so no fundamental "
                      "check ran - names were not screened on revenue, "
                      "profit or margin. Build it with 'python -m "
                      "desk.research.refresh fundamentals'."]
    if cached.stale:
        return None, [cached.caveat]
    return cached.frame, [c for c in (cached.caveat,) if c]


def _llm_client():
    """A metered client, or None when Stage 3 cannot run.

    None is the normal state today: no key is configured. Stage 3 reports
    itself as not run and the shortlist passes through untouched, which is
    why /plan/today keeps working unchanged. Nothing here can spend money
    without OPENAI_API_KEY being set deliberately, and even then only up to
    the ceiling in desk.llm.budget - which is still a placeholder awaiting a
    real number.
    """
    provider = OpenAIProvider()
    if not provider.configured:
        return None
    return MeteredClient(provider=provider,
                         meter=CostMeter(ledger_path=CONFIGS / "llm"
                                         / "spend.ledger.jsonl"))


def _stage1_for(target: date, *, lookback: int, min_price: float,
                min_turnover_lacs: float, min_bars: int
                ) -> tuple[Stage1Result, list[str]]:
    """Stage 0 + history + Stage 1 for one day - the shape both scanner
    endpoints need. Raises StoreError/ValueError for the caller to map;
    /plan/today deliberately does NOT use this, because it turns a failure
    into a NO TRADE rather than an HTTP error and does not expose the stage
    thresholds as overrides.
    """
    store = BarStore(CONFIGS / "bhavcopy", calendar=_calendar_or_none())
    stage0 = run_stage0(store.load_day(target), min_price=min_price,
                        min_turnover_lacs=min_turnover_lacs)
    survivors = stage0.survivors["symbol"].tolist()
    history = store.history(as_of=target, lookback=lookback,
                            symbols=survivors, columns=list(REQUIRED_BARS))
    actions, unchecked = _actions_for(survivors)
    stage1 = run_stage1(history, as_of=target, min_bars=min_bars,
                        actions=actions)
    # Returned separately rather than only appended, so a caller reporting a
    # LATER stage's caveats can carry it forward too. Slicing it back off
    # stage1.unavailable afterwards is the obvious shortcut and it is wrong -
    # nothing records where the stage's own entries end.
    extra = _action_caveat(unchecked, len(survivors))
    stage1.unavailable.extend(extra)
    return stage1, extra


class Stage0Survivor(BaseModel):
    symbol: str
    series: str
    close: float
    volume: int
    turnover_lacs: float
    delivery_pct: float | None
    circuit_locked: bool


class Stage0Response(BaseModel):
    as_of: date
    universe_in: int
    universe_out: int
    excluded: dict[str, int]
    caveats: list[str]
    survivors: list[Stage0Survivor]


@app.get("/scanner/stage0", response_model=Stage0Response, tags=["scanner"])
def scanner_stage0(
    day: date | None = None,
    min_price: float = Query(default=20.0, ge=0),
    min_turnover_lacs: float = Query(default=100.0, ge=0),
    exclude_circuit_locked: bool = False,
) -> Stage0Response:
    """Universe hygiene for one day - see desk.scanner.stage0.run_stage0.

    Reads an ALREADY-FETCHED bhavcopy snapshot; this endpoint makes no network
    call of its own, the same discipline as /calendar/{day} reading the
    already-fetched holiday file.
    """
    target = day or _today_ist()
    _require_snapshot(target)
    store = BarStore(CONFIGS / "bhavcopy", calendar=_calendar_or_none())
    try:
        result = run_stage0(
            store.load_day(target), min_price=min_price,
            min_turnover_lacs=min_turnover_lacs,
            exclude_circuit_locked=exclude_circuit_locked,
        )
    except StoreError as exc:
        log.warning("store unreadable for %s: %s", target, exc)
        raise HTTPException(503, "market data snapshot unreadable") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    log.info("scanner/stage0 %s: %d -> %d survivors",
             result.as_of, result.universe_in, result.universe_out)
    return Stage0Response(
        as_of=result.as_of, universe_in=result.universe_in,
        universe_out=result.universe_out, excluded=result.excluded,
        caveats=result.caveats,
        survivors=[
            Stage0Survivor(
                symbol=row.symbol, series=row.series, close=float(row.close),
                volume=int(row.volume), turnover_lacs=float(row.turnover_lacs),
                delivery_pct=(None if pd.isna(row.delivery_pct)
                              else float(row.delivery_pct)),
                circuit_locked=bool(row.circuit_locked),
            )
            for row in result.survivors.itertuples()
        ],
    )


class Stage1Row(BaseModel):
    symbol: str
    bars: int
    close: float | None = None
    ret_1d_pct: float | None = None
    ret_5d_pct: float | None = None
    ret_20d_pct: float | None = None
    rel_volume: float | None = None
    gap_pct: float | None = None
    atr_pct: float | None = None
    atr_pct_rank: float | None = None
    dist_sma20_pct: float | None = None
    dist_sma50_pct: float | None = None
    dist_sma200_pct: float | None = None
    pos_52w_pct: float | None = None
    bb_width_pct: float | None = None
    rs_rank: float | None = None
    compressed: bool
    unusual_volume: bool
    unusual_move: bool
    near_52w_high: bool
    extended: bool
    flag_count: int


class CoverageOut(BaseModel):
    """Whether the history behind these features is actually complete. A
    caller that ignores this can read a 200-day average computed over a
    window with three missing fetches as if it were clean."""
    start: date | None
    end: date | None
    days_loaded: int
    sessions_expected: int | None
    missing: list[date]
    calendar_checked: bool
    truncated: bool
    complete: bool
    describe: str


class Stage1Response(BaseModel):
    as_of: date
    universe_in: int
    universe_out: int
    flagged: int
    returned: int
    excluded: dict[str, int]
    unavailable: list[str]
    unadjusted_actions: dict[str, int]
    coverage: CoverageOut
    rows: list[Stage1Row]


@app.get("/scanner/stage1", response_model=Stage1Response, tags=["scanner"])
def scanner_stage1(
    day: date | None = None,
    lookback: int = Query(default=252, ge=30, le=LOOKBACK_CEILING),
    min_price: float = Query(default=20.0, ge=0),
    min_turnover_lacs: float = Query(default=100.0, ge=0),
    min_bars: int = Query(default=30, ge=2, le=500),
    flagged_only: bool = True,
    limit: int = Query(default=200, ge=1, le=2000),
) -> Stage1Response:
    """Per-symbol features and opportunity flags for the Stage 0 survivors.

    Reads ALREADY-FETCHED snapshots only - no network call, same discipline
    as /scanner/stage0 and /calendar/{day}. Needs `lookback` days of history
    accumulated in configs/bhavcopy/, so run the bhavcopy refresh daily; the
    response's `coverage` says exactly what was actually available.

    `limit` bounds the response because the full universe is ~2,600 rows of
    24 columns and an unbounded scan is a large payload on every call.
    `flagged` and `universe_out` report the true counts regardless.
    """
    target = day or _today_ist()
    _require_snapshot(target)
    try:
        result, _ = _stage1_for(target, lookback=lookback,
                                min_price=min_price,
                                min_turnover_lacs=min_turnover_lacs,
                                min_bars=min_bars)
    except StoreError as exc:
        # StoreError messages carry the absolute path of the offending file,
        # which on a real machine includes the OS username and the project's
        # directory layout. Same leak /calendar/{day} was already hardened
        # against - log it server-side where it is useful for debugging, and
        # tell an unauthenticated caller only that something is wrong.
        log.warning("store unreadable for %s: %s", target, exc)
        raise HTTPException(503, "market data snapshot unreadable") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    table = result.flagged if flagged_only else result.features
    log.info("scanner/stage1 %s: %d -> %d, %d flagged (%s)",
             result.as_of, result.universe_in, result.universe_out,
             len(result.flagged), result.coverage.describe())

    cov = result.coverage
    return Stage1Response(
        as_of=result.as_of,
        universe_in=result.universe_in,
        universe_out=result.universe_out,
        flagged=len(result.flagged),
        returned=min(len(table), limit),
        excluded=result.excluded,
        unavailable=result.unavailable,
        unadjusted_actions=result.unadjusted_actions,
        coverage=CoverageOut(
            start=cov.start, end=cov.end, days_loaded=cov.days_loaded,
            sessions_expected=cov.sessions_expected, missing=list(cov.missing),
            calendar_checked=cov.calendar_checked, truncated=cov.truncated,
            complete=cov.complete, describe=cov.describe(),
        ),
        rows=[_stage1_row(sym, row)
              for sym, row in table.head(limit).iterrows()],
    )


def _stage1_row(symbol: str, row) -> Stage1Row:
    """NaN is not valid JSON, and `float('nan')` silently becomes `NaN` in
    some encoders and `null` in others. Map it to None explicitly so a
    feature that could not be computed reads as absent rather than as zero."""
    def num(name: str) -> float | None:
        v = row.get(name)
        return None if v is None or pd.isna(v) else float(v)

    return Stage1Row(
        symbol=symbol, bars=int(row["bars"]),
        close=num("close"),
        ret_1d_pct=num("ret_1d_pct"), ret_5d_pct=num("ret_5d_pct"),
        ret_20d_pct=num("ret_20d_pct"), rel_volume=num("rel_volume"),
        gap_pct=num("gap_pct"), atr_pct=num("atr_pct"),
        atr_pct_rank=num("atr_pct_rank"),
        dist_sma20_pct=num("dist_sma20_pct"),
        dist_sma50_pct=num("dist_sma50_pct"),
        dist_sma200_pct=num("dist_sma200_pct"),
        pos_52w_pct=num("pos_52w_pct"), bb_width_pct=num("bb_width_pct"),
        rs_rank=num("rs_rank"),
        compressed=bool(row["compressed"]),
        unusual_volume=bool(row["unusual_volume"]),
        unusual_move=bool(row["unusual_move"]),
        near_52w_high=bool(row["near_52w_high"]),
        extended=bool(row["extended"]),
        flag_count=int(row["flag_count"]),
    )


class Stage2Row(BaseModel):
    symbol: str
    score: float
    factors_used: int
    penalty: float
    factor_ranks: dict[str, float | None]
    """Percentile rank per factor, so every score can be read back to its
    inputs without a second call - section 32's explainability requirement."""
    explain: str


class Stage2Response(BaseModel):
    as_of: date
    regime: Regime
    universe_in: int
    universe_out: int
    returned: int
    factors: list[dict]
    silenced: dict[str, str]
    excluded: dict[str, int]
    unavailable: list[str]
    coverage: CoverageOut
    rows: list[Stage2Row]


@app.get("/scanner/stage2", response_model=Stage2Response, tags=["scanner"])
def scanner_stage2(
    day: date | None = None,
    regime: Regime = Regime.UNKNOWN,
    lookback: int = Query(default=252, ge=30, le=LOOKBACK_CEILING),
    min_price: float = Query(default=20.0, ge=0),
    min_turnover_lacs: float = Query(default=100.0, ge=0),
    min_bars: int = Query(default=30, ge=2, le=500),
    min_factors: int = Query(default=3, ge=1, le=10),
    flagged_only: bool = True,
    limit: int = Query(default=25, ge=1, le=500),
) -> Stage2Response:
    """The ranked, explained shortlist - the last stage before anything costs
    money per call.

    `regime` is a REQUIRED INPUT, not an inferred one. Nothing in this project
    computes a regime from market data yet (that needs India VIX, Nifty
    breadth and a sectoral feed), so it defaults to `unknown` and the caller
    states it. A hostile regime SILENCES a factor rather than down-weighting
    it, and with the default factor set a `crisis` silences enough of them
    that no ranking is produced at all - which is NO TRADE, a first-class
    successful outcome, not a failure.
    """
    target = day or _today_ist()
    _require_snapshot(target)
    try:
        stage1, extra = _stage1_for(target, lookback=lookback,
                                    min_price=min_price,
                                    min_turnover_lacs=min_turnover_lacs,
                                    min_bars=min_bars)
        result = run_stage2(stage1, regime=regime, min_factors=min_factors,
                            flagged_only=flagged_only)
        result.unavailable.extend(extra)
    except StoreError as exc:
        # StoreError messages carry the absolute path of the offending file,
        # which on a real machine includes the OS username and the project's
        # directory layout. Same leak /calendar/{day} was already hardened
        # against - log it server-side where it is useful for debugging, and
        # tell an unauthenticated caller only that something is wrong.
        log.warning("store unreadable for %s: %s", target, exc)
        raise HTTPException(503, "market data snapshot unreadable") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    log.info("scanner/stage2 %s regime=%s: %d -> %d ranked, factors=%s",
             result.as_of, regime.value, result.universe_in,
             result.universe_out, [f.name for f in result.factors])

    cov = stage1.coverage
    head = result.ranked.head(limit)
    return Stage2Response(
        as_of=result.as_of, regime=result.regime,
        universe_in=result.universe_in, universe_out=result.universe_out,
        returned=len(head),
        factors=[{"name": f.name, "column": f.column,
                  "direction": f.direction, "weight": f.weight,
                  "rationale": f.rationale} for f in result.factors],
        silenced=result.silenced, excluded=result.excluded,
        unavailable=result.unavailable,
        coverage=CoverageOut(
            start=cov.start, end=cov.end, days_loaded=cov.days_loaded,
            sessions_expected=cov.sessions_expected, missing=list(cov.missing),
            calendar_checked=cov.calendar_checked, truncated=cov.truncated,
            complete=cov.complete, describe=cov.describe(),
        ),
        rows=[
            Stage2Row(
                symbol=sym, score=float(row["score"]),
                factors_used=int(row["factors_used"]),
                penalty=float(row["penalty"]),
                factor_ranks={
                    f.name: (None if pd.isna(row.get(f"rank_{f.name}"))
                             else float(row[f"rank_{f.name}"]))
                    for f in result.factors
                },
                explain=result.explain(sym),
            )
            for sym, row in head.iterrows()
        ],
    )


# ------------------------------------------------------------ strategies ---

@app.get("/strategies", tags=["strategies"])
def strategies() -> list[dict]:
    """The six shared strategies, with their open defects stated up front."""
    return catalog_status()


@app.get("/strategies/eligible/{regime}", tags=["strategies"])
def strategies_eligible(regime: Regime, trusted_only: bool = True) -> list[str]:
    """Who may speak in a given regime. Hostile-regime strategies are silenced."""
    return [s.key for s in eligible(regime, trusted_only=trusted_only)]


# --------------------------------------------------------------- symbols ---

@app.get("/symbols/{raw}", tags=["marketdata"])
def resolve_symbol(raw: str) -> dict:
    """One truth, every dialect. RELIANCE -> RELIANCE.NS -> RELIANCE.NSE ..."""
    try:
        s = Symbol.parse(raw)
    except SymbolError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "canonical": s.canonical, "base": s.base, "exchange": s.exchange.value,
        "dialects": {"yfinance": s.yfinance, "kite": s.kite,
                     "kite_instrument": s.kite_instrument,
                     "nautilus": s.nautilus, "vibe": s.vibe},
    }


# ------------------------------------------------------------------ risk ---

class PositionIn(BaseModel):
    # Every string is bounded. An unbounded value here is both a memory
    # amplifier on a 200-entry list and, for `symbol`, a way to forge log
    # lines - it is interpolated straight into a log.info() call.
    symbol: str = Field(max_length=40)
    qty: int = Field(ge=0)          # magnitude only; direction is not sized here
    entry: float = Field(gt=0)
    stop: float = Field(gt=0)
    sector: str = Field(default="UNKNOWN", max_length=40)
    corr_group: str | None = Field(default=None, max_length=40)
    beta: float = 1.0

    def to_position(self) -> Position:
        # Keyword construction: Position's field order is not part of its
        # contract, and positional args here would break silently if it
        # changed.
        return Position(symbol=self.symbol, qty=self.qty, entry=self.entry,
                        stop=self.stop, sector=self.sector, beta=self.beta,
                        corr_group=self.corr_group)


class SizeRequest(BaseModel):
    """Mirrors `size_position`. Every optional gate input is exposed here -
    a parameter the API cannot send is a gate that can never run in production,
    which is indistinguishable from not having written the gate at all."""

    # Bounded because it is interpolated into a log line: unbounded here is
    # both a log-flooding vector and a way to forge entries with embedded
    # newlines.
    symbol: str = Field(max_length=40)
    entry: float = Field(gt=0)
    stop: float = Field(gt=0)
    target: float | None = Field(default=None, gt=0)
    sector: str = "UNKNOWN"
    corr_group: str | None = None
    adv_shares: float | None = None
    atr_pct: float | None = None
    expected_slippage_pct: float | None = None
    market_risk_off: bool = False
    tradeable: bool = True
    data_as_of: date | None = None
    config: RiskConfig
    # Verified pre-bound that a 50,000-entry (23.5MB) payload was accepted
    # and fully processed with no ceiling at all.
    open_positions: list[PositionIn] = Field(default_factory=list,
                                             max_length=200)
    realised_pnl_today: float = 0.0


@app.post("/risk/size", response_model=Sizing, tags=["risk"])
def risk_size(req: SizeRequest) -> Sizing:
    """Deterministic. Same inputs, same answer. No LLM involved at any point."""
    pf = Portfolio(
        positions=[p.to_position() for p in req.open_positions],
        realised_pnl_today=req.realised_pnl_today,
    )
    return size_position(
        symbol=req.symbol, entry=req.entry, stop=req.stop, target=req.target,
        cfg=req.config, portfolio=pf, sector=req.sector,
        corr_group=req.corr_group, adv_shares=req.adv_shares,
        atr_pct=req.atr_pct, expected_slippage_pct=req.expected_slippage_pct,
        market_risk_off=req.market_risk_off, tradeable=req.tradeable,
        data_as_of=req.data_as_of,
        today=_today_ist() if req.data_as_of else None,
    )


@app.get("/risk/config/default", response_model=RiskConfig, tags=["risk"])
def default_config() -> RiskConfig:
    return RiskConfig(capital=1_000_000)


# ------------------------------------------------------------------ plan ---

class PlanRequest(BaseModel):
    """POST body for /plan/today.

    `open_positions` exists because sizing a day's plan against an EMPTY book
    is wrong for anyone holding anything overnight: the sector cap, the
    correlated-cluster cap, portfolio heat, max-open-positions and the daily
    loss cap are all real, tested gates that are completely inert without it.
    /risk/size has always accepted a portfolio; the plan endpoint must not be
    the one place that quietly ignores one.
    """
    day: date | None = None
    regime: Regime = Regime.UNKNOWN
    capital: float = Field(default=1_000_000.0, gt=0, le=1e12)
    max_trades: int = Field(default=3, ge=1, le=10)
    lookback: int = Field(default=252, ge=30, le=LOOKBACK_CEILING)
    open_positions: list[PositionIn] = Field(default_factory=list,
                                             max_length=200)
    realised_pnl_today: float = 0.0


@app.post("/plan/today", response_model=DailyPlan, tags=["plan"])
def plan_today_with_portfolio(req: PlanRequest) -> DailyPlan:
    """The day's plan, sized against a portfolio you already hold.

    Same pipeline as the GET form - that one exists for a flat book and for
    browsing /docs; this is the one a trader with open positions should call.
    """
    return _plan(
        as_of=req.day or _today_ist(), regime=req.regime, capital=req.capital,
        max_trades=req.max_trades, lookback=req.lookback,
        portfolio=Portfolio(
            positions=[p.to_position() for p in req.open_positions],
            realised_pnl_today=req.realised_pnl_today,
        ),
    )


@app.get("/plan/today", response_model=DailyPlan, tags=["plan"])
def plan_today(
    day: date | None = None,
    regime: Regime = Regime.UNKNOWN,
    capital: float = Query(default=1_000_000.0, gt=0, le=1e12),
    max_trades: int = Query(default=3, ge=1, le=10),
    lookback: int = Query(default=252, ge=30, le=LOOKBACK_CEILING),
) -> DailyPlan:
    """The day's plan: the whole funnel, or an explicit NO TRADE.

    Stage 0 -> 1 -> 2 -> 4, every stage deterministic. Stage 3 (LLM research)
    is not built, so this is the DETERMINISTIC path end to end - which is ~98%
    of the funnel by design; the AI reads and explains, it does not compute.

    Runs only if the day's bhavcopy snapshot is already on disk. It makes no
    network call: a missing snapshot is a NO TRADE with the fetch command in
    the reason, not a silent empty plan and not a live download from inside a
    request handler.
    """
    return _plan(as_of=day or _today_ist(), regime=regime, capital=capital,
                 max_trades=max_trades, lookback=lookback, portfolio=None)


def _plan(*, as_of: date, regime: Regime, capital: float, max_trades: int,
          lookback: int, portfolio: Portfolio | None) -> DailyPlan:
    try:
        cal = _calendar()
    except CalendarError:
        cal = None
    scan = _run_funnel(as_of, regime=regime, capital=capital,
                       max_trades=max_trades, lookback=lookback,
                       portfolio=portfolio)
    return build_plan(as_of=as_of, calendar=cal, scan=scan)


def _run_funnel(as_of: date, *, regime: Regime, capital: float,
                max_trades: int, lookback: int,
                portfolio: Portfolio | None,
                today: date | None = None) -> ScanSummary | None:
    """The whole scanner, reduced to the primitives build_plan takes.

    Returns None when no snapshot exists for the day - build_plan turns that
    into a NO TRADE naming the fetch command. Any other failure is also None
    plus a logged warning rather than a 500: the plan endpoint's job is to
    answer honestly every day, and "the scan could not run" is an answer.

    `today` exists for the BACKTEST, and it is not cosmetic. The risk
    engine refuses data more than 5 days old, measured against "today" - so
    with the wall clock hard-coded, every historical session is stale by
    definition and a backtest returns NO TRADE on every single day while
    looking like it ran correctly. In a replay, today IS the simulated
    date. Live callers omit it and get the IST clock.
    """
    store = BarStore(CONFIGS / "bhavcopy", calendar=_calendar_or_none())
    if not store.has(as_of):
        return None

    try:
        stage0 = run_stage0(store.load_day(as_of))
        survivors = stage0.survivors["symbol"].tolist()
        history = store.history(as_of=as_of, lookback=lookback,
                                symbols=survivors, columns=list(REQUIRED_BARS))

        # MEASURE the regime rather than being told it, unless a caller
        # deliberately overrode it. A hand-chosen regime is worse than
        # none: it silences factors and gates strategies on an opinion,
        # and the plan then prints that opinion as if it were a finding.
        # Computed from the SAME history the scan uses, so the breadth
        # figure and the ranking cannot disagree about what the market did.
        measured = None
        if regime is Regime.UNKNOWN:
            measured = compute_regime(
                history, as_of=as_of,
                sectors=SectorMap.load(CONFIGS / "sectors.json"),
                events=None)
            regime = measured.label
        actions, unchecked = _actions_for(survivors)
        stage1 = run_stage1(history, as_of=as_of, actions=actions)

        # R3's fundamental layer. The table is JOINED whether or not a filter
        # is set, so Stage 2 can report honestly how many candidates it could
        # not check - and so Stage 3 sees real figures instead of the
        # "not available" placeholder.
        #
        # THE HARD FILTERS ARE DELIBERATELY NOT SET HERE. max_filing_age_days
        # and min_net_margin_pct decide which companies are tradeable, which
        # is a policy choice rather than an engineering default, and picking
        # one silently would change what the desk trades without anyone
        # choosing it. Measured on real data: every filing currently on file
        # is 582-610 days old, so a plausible-looking age filter of 200 days
        # would exclude the ENTIRE universe and return NO TRADE every day
        # while looking like it was working.
        fundamentals, fundamental_caveats = _fundamentals(as_of)
        stage2 = run_stage2(stage1, regime=regime, fundamentals=fundamentals)

        # R4's event gate and R7's narrative pass. Both are OPTIONAL by
        # construction and both declare themselves when absent, so this
        # endpoint answers every day whether or not a calendar has been
        # refreshed and whether or not an LLM key is configured.
        events, event_caveats = _event_calendar(as_of)
        stage3 = run_stage3(stage2, client=_llm_client(), events=events,
                            fundamentals=fundamentals)
        stage4 = run_stage4(stage3.narrow(stage2), stage1,
                            cfg=RiskConfig(capital=capital),
                            portfolio=portfolio, events=events,
                            max_trades=max_trades,
                            today=today or _today_ist())
    except (ValueError, StoreError) as exc:
        log.warning("scan for %s could not run: %s", as_of, exc)
        return None

    log.info("plan %s regime=%s: %d -> %d -> %d -> %d -> %d -> %d trades",
             as_of, regime.value, stage0.universe_in, stage0.universe_out,
             len(stage1.flagged), stage2.universe_out, len(stage3.kept),
             len(stage4.approved))

    return ScanSummary(
        universe_scanned=stage0.universe_in,
        survived_stage0=stage0.universe_out,
        survived_stage1=len(stage1.flagged),
        survived_stage2=stage2.universe_out,
        considered=stage4.considered,
        trades=stage4.approved,
        no_trade_reason=stage4.no_trade_reason(),
        caveats=[*_regime_caveats(measured),
                 *stage0.caveats, *stage1.unavailable, *stage2.unavailable,
                 *stage3.unavailable, *stage4.unavailable, *event_caveats,
                 *fundamental_caveats,
                 *_action_caveat(unchecked, len(survivors))],
        coverage_note=stage1.coverage.describe(),
    )
