"""FastAPI surface. The web app talks only to this - never to an agent directly.

Run:  .venv/Scripts/python.exe -m uvicorn desk.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
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
from desk.journal import JournalError, JournalStore
from desk.journal.store import decision_from_plan
from desk.marketdata.calendar_in import (
    CalendarError,
    CalendarNotLoaded,
    TradingCalendar,
)
from desk.marketdata.corporate_actions import ActionLoadError, load_actions
from desk.marketdata import quality
from desk.marketdata.providers import provider_status
from desk.marketdata.sectors import SectorMap
from desk.marketdata.symbols import SymbolError, Symbol
from desk.plan.build import build_plan
from desk.plan.models import DailyPlan, ScanSummary
from desk.registry.fleet import fleet_status
from desk.settings import Settings
from desk.risk.engine import Portfolio, Position, RiskConfig, Sizing, size_position
from desk.scanner.stage0 import run_stage0
from desk.scanner.stage1 import REQUIRED_BARS, Stage1Result, run_stage1
from desk.llm.budget import CostMeter
from desk.llm.client import MeteredClient
from desk.llm.providers.openai import OpenAIProvider
from desk.regime.engine import compute_regime
from desk.research.events import load_calendar
from desk.research.fundamentals import FundamentalsCache
from desk.research.news import load_news
from desk.scanner.stage2 import run_stage2
from desk.scanner.stage3 import VerdictCache, run_stage3
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


def _vet_candidates(history, symbols: list[str]) -> tuple[list[str], list[str]]:
    """Screen the shortlist's own bars. Returns (survivors, caveats).

    A FATAL finding DROPS the name. That is deliberate and it is the one
    place in this funnel where bad data removes a candidate outright: the
    entry, the stop and the size are all computed from these bars, so a
    duplicated index or an unexplained 60% gap does not make the trade
    riskier, it makes every number attached to it meaningless.

    A WARNING is reported and the name SURVIVES. An extreme return is
    usually a real move and sometimes an unadjusted corporate action, and
    this screen cannot tell which - so it says so and lets the risk gate
    and a human decide. Dropping on a warning would silently discard the
    biggest movers, which is most of what a momentum shortlist is.
    """
    if not symbols:
        return symbols, []
    try:
        frame = history.frame
        sub = frame[frame["symbol"].isin(symbols)].copy()
        if sub.empty:
            return symbols, []
        sub["date"] = pd.to_datetime(sub["date"])
        reports = quality.check_panel(sub.set_index("date"), by="symbol")
    except Exception as exc:                        # noqa: BLE001
        # The gate failing must not take the plan with it - but it must
        # not pass silently either, or "checked and clean" and "the
        # checker crashed" become the same output.
        log.warning("candidate quality screen could not run: %s", exc)
        return symbols, [f"the data-quality screen on the shortlist could "
                         f"not run ({exc}), so these bars were NOT vetted."]

    caveats: list[str] = []
    dropped: set[str] = set()
    for symbol, rep in reports.items():
        for issue in rep.fatal:
            dropped.add(symbol)
            caveats.append(f"{symbol} DROPPED - its price history failed a "
                           f"data-quality check: {issue}")
        for issue in rep.warnings:
            caveats.append(f"{symbol}: {issue}")

    survivors = [s for s in symbols if s not in dropped]
    if not caveats:
        caveats.append(f"the {len(symbols)} shortlisted name(s) passed the "
                       f"bar-level data-quality screen.")
    return survivors, caveats


def _news():
    """(clusters, caveats) for Stage 3's market context.

    Read from the snapshot the refresh writes. Fetching three RSS feeds
    and clustering them inside a request would add seconds to every page
    load to recompute something that changes a few times an hour.

    A STALE SNAPSHOT IS DISCARDED, not used with a warning, and the bar is
    much tighter than elsewhere - 18 hours rather than the calendar's
    three days. A fundamentals table a week behind is merely old; news a
    week old is actively misleading, because anything labelled "market
    context" will be read as describing today.
    """
    stored = load_news(CONFIGS / "research" / "news.json")
    if stored is None:
        return None, ["no news snapshot on file, so Stage 3 saw no market "
                      "context. Run 'python -m desk.research.refresh news'."]
    if stored.stale:
        return None, [stored.caveat]
    out = [f"market context: {len(stored.clusters)} deduplicated headline(s), "
           f"fetched {stored.age_hours:.1f}h ago"]
    out.extend(stored.caveats[:3])
    return stored.clusters, out


def _crosscheck_caveats(as_of: date) -> list[str]:
    """What the second source said about the prices this plan is built on.

    Read from the report the refresh writes, never computed here: the
    answer to "do NSE and BSE agree on the 17th" is fixed once that day
    closes, and recomputing it per request would mean a BSE fetch and a
    5,000-row join on every page load to arrive at a constant.

    ABSENCE IS REPORTED. A plan whose prices were never cross-checked and
    a plan whose prices were checked and agreed are different states, and
    only one of them has a second source behind it.
    """
    path = CONFIGS / "crosscheck" / f"{as_of.isoformat()}.json"
    if not path.is_file():
        return ["prices were NOT cross-checked against BSE for this day - "
                "the plan rests on a single source. Run 'python -m "
                "desk.marketdata.refresh crosscheck'."]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["the BSE cross-check report for this day is unreadable, so "
                "prices rest on a single source."]

    cov = report.get("coverage", {})
    checkable = int(cov.get("checkable", 0))
    total = int(cov.get("nse_symbols", 0))
    bad = report.get("disagreements", [])
    out = [
        f"prices cross-checked against BSE on {checkable} of {total} "
        f"symbols ({100.0 * checkable / total:.0f}%) - the rest are thin on "
        f"BSE or not listed there, and were NOT checked rather than checked "
        f"and cleared."
    ]
    for row in bad[:5]:
        out.append(
            f"NSE and BSE DISAGREE on {row.get('symbol')}: "
            f"{row.get('nse_close')} vs {row.get('bse_close')} "
            f"({row.get('diff_pct')}%). One of them is wrong.")
    if len(bad) > 5:
        out.append(f"...and {len(bad) - 5} more price disagreements.")
    return out


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


def _event_calendar(as_of: date, today: date | None = None):
    """(calendar, caveats) for the earnings gate.

    `today` is the DECISION date and it is the one staleness is measured
    against - not `as_of`, which is the date of the DATA being scanned.
    Getting that backwards made this gate fire exactly never.

    The two dates are always different in live use: bhavcopy for day D is
    published after D closes, so a plan built on D's bars is decided on
    D+1 at the earliest. Comparing the calendar's fetch time against D
    therefore made every calendar look like it came "from the future",
    and the look-ahead guard refused all of them - silently removing the
    gate described as the highest safety-per-hour item in the plan.

    The distinction that was collapsed: THE CALENDAR IS FORWARD-LOOKING.
    For a live plan today's calendar is the actual schedule of what is
    coming, which is the whole point of consulting it. Look-ahead only
    arises in a HISTORICAL REPLAY, where a calendar fetched later really
    does contain intimations published after the simulated date - and the
    backtest passes its simulated date as `today`, so it stays protected.

    Returns None for the calendar in BOTH the missing and the stale case,
    but with different caveats, because they are different failures. A
    missing calendar means nobody fetched it. A stale one means somebody did
    and then stopped, which is more dangerous: a calendar fetched two weeks
    ago answers "nothing scheduled" for every company that has announced a
    board meeting since, and the gate would report a clean check. Staleness
    is a FALSE CLEAR, so a stale file is treated as absent rather than used.
    """
    stored = load_calendar(CONFIGS / "research" / "event_calendar.json",
                           as_of=today or _today_ist())
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

    None is the normal state until a key is configured. Stage 3 then
    reports itself as not run and the shortlist passes through untouched,
    which is why /plan/today keeps working unchanged.

    Everything comes from the environment (see .env.example): the key, the
    model, the monthly rupee ceiling and the USD rate. Settings are re-read
    on every call rather than cached at import, so editing .env and
    reloading the page takes effect without restarting the server - which
    is what someone actually does when they first paste a key in.
    """
    cfg = Settings.from_env()
    provider = OpenAIProvider(model=cfg.llm_model, api_key=cfg.openai_api_key)
    if not provider.configured:
        return None
    return MeteredClient(
        provider=provider,
        meter=CostMeter(ledger_path=CONFIGS / "llm" / "spend.ledger.jsonl",
                        monthly_ceiling_inr=cfg.llm_monthly_ceiling_inr,
                        usd_inr=cfg.usd_inr))


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

    Stage 0 -> 1 -> 2 -> 3 -> 4. Stages 0, 1, 2 and 4 are deterministic and
    free; Stage 3 is the narrative pass and the only one that costs money.

    Stage 3 CAN ONLY REMOVE NAMES - it cannot add a candidate, raise a
    score, move a stop or change a size - so the ~98% of the funnel that
    decides what a trade looks like stays deterministic either way. When no
    key is configured, or the account has no credits, or the model errors,
    Stage 3 reports itself as not run and the shortlist passes through
    untouched. The plan still answers.

    Runs only if the day's bhavcopy snapshot is already on disk. It makes no
    network call: a missing snapshot is a NO TRADE with the fetch command in
    the reason, not a silent empty plan and not a live download from inside a
    request handler.
    """
    return _plan(as_of=day or _today_ist(), regime=regime, capital=capital,
                 max_trades=max_trades, lookback=lookback, portfolio=None)



# ---------------------------------------------------------------- regime ---

@app.get("/regime", tags=["regime"])
def regime_today(day: date | None = None,
                 lookback: int = Query(default=200, ge=60,
                                       le=LOOKBACK_CEILING)) -> dict:
    """The market regime as MEASURED, with what each dimension came from.

    Exposed because a regime silences factors and gates strategies, so it
    has to be arguable rather than asserted. `sources` says what every
    dimension was computed from and over what window; a dimension with no
    data reads UNKNOWN and `measured` is false.
    """
    target = day or _latest_snapshot_day()
    if target is None:
        raise HTTPException(404, "no market snapshot on file - run "
                                 "'python -m desk.marketdata.refresh bhavcopy'")
    store = BarStore(CONFIGS / "bhavcopy", calendar=_calendar_or_none())
    if not store.has(target):
        raise HTTPException(404, f"no snapshot for {target}")

    stage0 = run_stage0(store.load_day(target))
    history = store.history(as_of=target, lookback=lookback,
                            symbols=stage0.survivors["symbol"].tolist(),
                            columns=list(REQUIRED_BARS))
    state = compute_regime(history, as_of=target,
                           sectors=SectorMap.load(CONFIGS / "sectors.json"))
    return {
        "as_of": target.isoformat(),
        "label": state.label.value,
        "measured": state.is_measured,
        "risk_off": state.risk_off,
        "trend": state.trend.value,
        "volatility": state.volatility.value,
        "breadth": state.breadth.value,
        "risk_appetite": state.risk_appetite.value,
        "leading_sectors": state.leading_sectors,
        "lagging_sectors": state.lagging_sectors,
        "max_concurrent_positions_hint": state.max_concurrent_positions_hint,
        "explain": state.explain(),
        # The whole point: every number, and where it came from.
        "sources": state.sources,
        "universe": len(history.symbols),
    }


# --------------------------------------------------------------- journal ---

@app.get("/journal", tags=["journal"])
def journal_index(limit: int = Query(default=120, ge=1, le=2000)) -> dict:
    """Every day the desk has recorded a decision for.

    `unrecorded` is the journal's own to-do list - days with trades whose
    outcomes were never written down. Without it, "we have no losing
    trades" and "nobody recorded how the trades went" look identical, and
    the flattering reading wins.
    """
    store = JournalStore(CONFIGS / "journal")
    # ONE directory listing, shared by every lookup below. This handler
    # used to call days(), then latest() per day, then unrecorded() and
    # open_positions() - each of which looped over every day AGAIN - so
    # it read every decision file twice and every outcomes file twice,
    # with each lookup globbing the whole directory. O(D^2), measured at
    # 3.4s by 750 recorded days.
    index = store.index()
    days = sorted(index)
    # `limit` bounds the response the way /scanner/stage1, /news and
    # /fundamentals already do. This was the one list endpoint with no
    # bound at all, on the one collection that grows every trading day.
    shown = list(reversed(days))[:limit]
    rows = []
    for d in shown:
        dec = store.latest(d, index)
        if dec is None:
            rows.append({"as_of": d.isoformat(), "unreadable": True})
            continue
        rows.append({
            "as_of": d.isoformat(),
            "regime": dec.regime,
            "trades": len(dec.trades),
            "symbols": list(dec.symbols),
            "no_trade_reason": dec.no_trade_reason,
            "digest": dec.digest(),
            "versions": len(index.get(d, ())),
            "capital": dec.capital,
            "outcomes_recorded": len(store.outcomes(d)),
        })
    return {
        "days": rows,
        "total": len(days),
        "shown": len(rows),
        "truncated": len(rows) < len(days),
        "no_trade_days": sum(1 for r in rows if not r.get("trades")),
        "unrecorded_outcomes": [d.isoformat()
                                for d in store.unrecorded(index)],
        "open_positions": [
            {"decision_date": o.decision_date.isoformat(), "symbol": o.symbol}
            for o in store.open_positions(index)
        ],
    }


@app.get("/journal/{day}", tags=["journal"])
def journal_day(day: date) -> dict:
    """One day's decision, its amendment chain, and what came of it."""
    store = JournalStore(CONFIGS / "journal")
    dec = store.latest(day)
    if dec is None:
        raise HTTPException(404, f"nothing recorded for {day}")
    return {
        "as_of": dec.as_of.isoformat(),
        "recorded_at": dec.recorded_at.isoformat(),
        "regime": dec.regime,
        "capital": dec.capital,
        "digest": dec.digest(),
        "amends": dec.amends,
        "is_no_trade": dec.is_no_trade,
        "no_trade_reason": dec.no_trade_reason,
        "funnel": {
            "universe_scanned": dec.universe_scanned,
            "survived_stage0": dec.survived_stage0,
            "survived_stage1": dec.survived_stage1,
            "survived_stage2": dec.survived_stage2,
            "considered": dec.considered,
        },
        "trades": [
            {"symbol": t.symbol, "stance": t.stance, "entry": t.entry,
             "stop": t.stop, "target": t.target, "qty": t.qty,
             "capital_at_risk": t.capital_at_risk,
             "reward_to_risk": t.reward_to_risk, "rationale": t.rationale}
            for t in dec.trades
        ],
        # Part of the decision, not metadata about it: "why did it pick
        # that" and "what did it not know" are the same question later.
        "caveats": list(dec.caveats),
        "coverage_note": dec.coverage_note,
        "versions": [
            {"digest": v.digest(), "recorded_at": v.recorded_at.isoformat(),
             "amends": v.amends, "note": v.note}
            for v in store.history(day)
        ],
        "outcomes": [
            {"symbol": o.symbol, "status": o.status, "exit_price": o.exit_price,
             "exit_date": o.exit_date.isoformat() if o.exit_date else None,
             "exit_reason": o.exit_reason, "r_multiple": o.r_multiple,
             "pnl": o.pnl}
            for o in store.outcomes(day)
        ],
    }


# ------------------------------------------------------------------ news ---

@app.get("/news", tags=["research"])
def news_context(limit: int = Query(default=40, ge=1, le=200)) -> dict:
    """Market-wide headlines, deduplicated by story rather than by outlet.

    NOT per-symbol. Mapping a headline to a ticker is entity resolution,
    and a wrong mapping attaches someone else's news to your trade. A
    cluster carries ONE vote weighted by its best source tier, never
    scaled by how many outlets ran it.
    """
    stored = load_news(CONFIGS / "research" / "news.json")
    if stored is None:
        return {"available": False, "clusters": [],
                "caveat": "no news snapshot on file - run 'python -m "
                          "desk.research.refresh news'"}
    return {
        "available": not stored.stale,
        "fetched_at": stored.fetched_at.isoformat(),
        "age_hours": round(stored.age_hours, 2),
        "stale": stored.stale,
        "caveat": stored.caveat,
        "feed_caveats": list(stored.caveats),
        "total": len(stored.clusters),
        "clusters": [
            {"title": c.title, "published_at": c.published_at.isoformat(),
             "tier": c.tier, "weight": c.weight, "duplicated": c.duplicated,
             "sources": list(c.sources), "session_phase": c.session_phase,
             "url": c.url}
            for c in stored.clusters[:limit]
        ],
    }


# ------------------------------------------------------------ crosscheck ---

@app.get("/crosscheck/{day}", tags=["marketdata"])
def crosscheck_day(day: date) -> dict:
    """Did BSE agree with NSE on that day's closes?

    Only about a fifth of NSE symbols are checkable: the rest trade too
    thinly on BSE for its close to be independent evidence, or are not
    listed there. Those are reported as NOT CHECKED rather than as
    agreeing - a thin print that matches is not corroboration.
    """
    path = CONFIGS / "crosscheck" / f"{day.isoformat()}.json"
    if not path.is_file():
        raise HTTPException(
            404, f"no cross-check on file for {day} - run 'python -m "
                 f"desk.marketdata.refresh crosscheck --date {day}'")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"cross-check report unreadable: {exc}") from exc


# ---------------------------------------------------------- fundamentals ---

@app.get("/fundamentals", tags=["research"])
def fundamentals_table(day: date | None = None,
                       limit: int = Query(default=50, ge=1, le=500)) -> dict:
    """The pre-open fundamentals table, as the scanner sees it.

    Point-in-time by filename: a table built for a later date is never
    opened for an earlier scan, so what is returned here is what was
    knowable that morning.
    """
    target = day or _latest_snapshot_day()
    if target is None:
        raise HTTPException(404, "no market snapshot on file")
    cached = FundamentalsCache(CONFIGS / "research" / "fundamentals").load(target)
    if cached is None:
        raise HTTPException(
            404, f"no fundamentals table for {target} or earlier - run "
                 f"'python -m desk.research.refresh fundamentals'")
    frame = cached.frame.head(limit)
    return {
        "requested_as_of": target.isoformat(),
        "table_as_of": cached.table_as_of.isoformat(),
        "staleness_days": cached.staleness_days,
        "stale": cached.stale,
        "caveat": cached.caveat,
        "coverage": cached.coverage,
        "total": len(cached.frame),
        "rows": [
            {"symbol": str(sym),
             **{k: (None if pd.isna(v) else
                    (v.isoformat() if hasattr(v, "isoformat") else
                     (float(v) if isinstance(v, (int, float)) else str(v))))
                for k, v in row.items()}}
            for sym, row in frame.iterrows()
        ],
    }


def _latest_snapshot_day() -> date | None:
    store = BarStore(CONFIGS / "bhavcopy", calendar=_calendar_or_none())
    days = store.available_days()
    return days[-1] if days else None

def _plan(*, as_of: date, regime: Regime, capital: float, max_trades: int,
          lookback: int, portfolio: Portfolio | None) -> DailyPlan:
    try:
        cal = _calendar()
    except CalendarError:
        cal = None
    scan = _run_funnel(as_of, regime=regime, capital=capital,
                       max_trades=max_trades, lookback=lookback,
                       portfolio=portfolio)
    plan = build_plan(as_of=as_of, calendar=cal, scan=scan)
    _journal(plan, scan, capital)
    return plan


def _journal(plan: DailyPlan, scan: ScanSummary | None, capital: float) -> None:
    """Record what the desk decided. Never fails the request.

    THE FIRST PLAN OF THE DAY IS THE ONE RECORDED. JournalStore refuses to
    overwrite, so re-requesting - browsing, a different capital, a page
    reload - leaves the original standing. That is the journal working:
    the decision is what was decided first, and a later run knows more
    than the morning did. A genuine correction goes through amend(), which
    keeps both versions and demands a reason.

    Wrapped so a journal problem can never break the plan endpoint. The
    desk answering today matters more than the record of it, and a full
    disk must not turn into a 500 on the one page the trader needs.
    """
    try:
        JournalStore(CONFIGS / "journal").record(
            decision_from_plan(plan, summary=scan, capital=capital))
    except JournalError:
        pass                    # already recorded for this day - correct
    except Exception as exc:    # noqa: BLE001
        log.warning("could not journal the plan for %s: %s", plan.as_of, exc)


def _run_funnel(as_of: date, *, regime: Regime, capital: float,
                max_trades: int, lookback: int,
                portfolio: Portfolio | None,
                today: date | None = None,
                target_r: float | None = None,
                stop_atrs: float = 2.0,
                holding_days: int = 5,
                use_llm: bool = True,
                shuffle: int | None = None,
                prefetched=None) -> ScanSummary | None:
    """The whole scanner, reduced to the primitives build_plan takes.

    Returns None when no snapshot exists for the day - build_plan turns that
    into a NO TRADE naming the fetch command. Any other failure is also None
    plus a logged warning rather than a 500: the plan endpoint's job is to
    answer honestly every day, and "the scan could not run" is an answer.

    `use_llm=False` disables Stage 3 entirely and is what the backtest
    passes. See the comment at the call site: without it a replay would
    make live, paid, look-ahead-contaminated calls as soon as a key
    existed.

    `target_r`, `stop_atrs` and `holding_days` are the strategy's three
    real tunables and they are exposed here so a BACKTEST CAN SWEEP THEM.
    Without that the harness can only ever measure one configuration, which
    is the one question it is least useful for - the first real run showed
    3 of 60 trades reaching a 2.5R target, and there was no way to ask what
    2.0R would have done. `target_r` defaults to DESK_RISK_REWARD.

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
        # `prefetched` is the BACKTEST's preloaded range, sliced in memory
        # instead of re-read per session. Consecutive replay days overlap
        # by 119 of 120 files, so reading each day cost 373ms to fetch
        # bars it had already fetched. Live callers pass nothing and read
        # from disk as before.
        if prefetched is not None:
            history = prefetched.window(as_of=as_of, lookback=lookback,
                                        symbols=survivors)
        else:
            history = store.history(as_of=as_of, lookback=lookback,
                                    symbols=survivors,
                                    columns=list(REQUIRED_BARS))

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
        # THE HARD FILTERS COME FROM THE ENVIRONMENT AND DEFAULT TO OFF.
        # max_filing_age_days and min_net_margin_pct decide which companies
        # are tradeable, which is a policy choice rather than an engineering
        # default - so they are settable in .env without a code change and
        # nothing is applied until someone chooses a value.
        #
        # Measured, and the reason a "sensible" default would be wrong:
        # every filing currently on disk is 555-616 days old, because NSE's
        # results endpoint returns nothing newer than Jan 2025. A
        # plausible-looking 200-day age filter would exclude the ENTIRE
        # universe and return NO TRADE every day while appearing to work.
        fundamentals, fundamental_caveats = _fundamentals(as_of)
        cfg = Settings.from_env()
        stage2 = run_stage2(stage1, regime=regime, fundamentals=fundamentals,
                            max_filing_age_days=cfg.max_filing_age_days,
                            min_net_margin_pct=cfg.min_net_margin_pct)

        # R4's event gate and R7's narrative pass. Both are OPTIONAL by
        # construction and both declare themselves when absent, so this
        # endpoint answers every day whether or not a calendar has been
        # refreshed and whether or not an LLM key is configured.
        events, event_caveats = _event_calendar(as_of, today=today)
        news, news_caveats = _news()
        # use_llm=False is how the BACKTEST switches Stage 3 off, and it
        # is not a convenience. run_backtest reaches Stage 3 through this
        # function, so _llm_client() would hand a replay a live provider
        # the moment OPENAI_API_KEY was set: hundreds of paid calls, and
        # a model that partly REMEMBERS the answer for the simulated date.
        # stage3.py's "never call this inside a backtest loop" was true of
        # run_stage3 and false of the path that actually reaches it.
        stage3 = run_stage3(stage2,
                            client=_llm_client() if use_llm else None,
                            events=events, fundamentals=fundamentals,
                            news=news,
                            # Without this every page reload is a fresh
                            # billed call. Keyed on the shortlist, so a
                            # changed set of names still re-asks.
                            cache=VerdictCache(CONFIGS / "llm"))
        # R6's data-quality gate, on the SHORTLIST rather than the
        # universe. check_panel over all 1,548 survivors measured 3.59s -
        # too slow for a request that already takes ten - and it is the
        # wrong scope anyway: a corrupt print on a name we are not about
        # to trade changes nothing. Run on the ~8 names that survived
        # Stage 3, it costs milliseconds and guards exactly the bars a
        # position would be sized from.
        # RESEARCH CONTROL, not a trading feature. `shuffle` reorders the
        # Stage 2 ranking with a seeded RNG, so Stage 4 sizes RANDOM names
        # from the same survivor set under identical sizing, gates, exits
        # and costs. That is the null hypothesis the real ranking has to
        # beat, and without it "-0.35R" cannot be read: a random baseline
        # of -0.35R would mean the exits and costs are the problem, while
        # a random baseline near zero would mean the ranking is actively
        # harmful. Default None changes nothing.
        if shuffle is not None and not stage2.ranked.empty:
            stage2.ranked = stage2.ranked.sample(
                frac=1.0, random_state=shuffle)
            stage3.kept = [s for s in stage2.ranked.index
                           if s in set(stage3.kept)]

        vetted, quality_caveats = _vet_candidates(history, stage3.kept)
        if vetted != stage3.kept:
            stage3.kept = vetted

        stage4 = run_stage4(stage3.narrow(stage2), stage1,
                            cfg=RiskConfig(capital=capital),
                            portfolio=portfolio, events=events,
                            max_trades=max_trades,
                            target_r=target_r or cfg.risk_reward,
                            stop_atrs=stop_atrs, holding_days=holding_days,
                            today=today or _today_ist())
    except (ValueError, StoreError) as exc:
        log.warning("scan for %s could not run: %s", as_of, exc)
        return None

    log.info("plan %s regime=%s: %d -> %d -> %d -> %d -> %d -> %d trades",
             as_of, regime.value, stage0.universe_in, stage0.universe_out,
             len(stage1.flagged), stage2.universe_out, len(stage3.kept),
             len(stage4.approved))

    return ScanSummary(
        regime_state=measured,
        vetoed=dict(stage3.vetoed),
        rejected=[(sz.symbol, ", ".join(r.value for r in sz.reasons))
                  for sz in stage4.rejected],
        universe_scanned=stage0.universe_in,
        survived_stage0=stage0.universe_out,
        survived_stage1=len(stage1.flagged),
        survived_stage2=stage2.universe_out,
        considered=stage4.considered,
        trades=stage4.approved,
        no_trade_reason=stage4.no_trade_reason(),
        caveats=[*_regime_caveats(measured), *_crosscheck_caveats(as_of),
                 *stage0.caveats, *stage1.unavailable, *stage2.unavailable,
                 *stage3.unavailable, *stage4.unavailable, *event_caveats,
                 *fundamental_caveats, *news_caveats, *quality_caveats,
                 *_action_caveat(unchecked, len(survivors))],
        coverage_note=stage1.coverage.describe(),
    )
