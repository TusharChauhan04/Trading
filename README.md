# The India Desk

NSE/BSE decision-support. Research and risk only — there is no order routing in
this repository, by design.

## Run it

One-time setup — `desk` is an installed package, not a `sys.path` hack:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Then two processes. Backend first:

```powershell
# from the project root
.\.venv\Scripts\python.exe -m uvicorn desk.api.main:app --reload --port 8000
```

```powershell
cd web
npm run dev          # http://localhost:5173
```

API docs: <http://localhost:8000/docs>

## The daily run, in order

Nine commands, and the order matters. Each one writes a file the next
step or the plan reads; skipping one does not break the plan, it makes
the plan quietly report that a check did not run.

Run them before the market opens. Everything is resumable — a rerun skips
what is already on disk.

```bash
# 1. PRICES. The whole exchange in one request per day. --from/--to
#    backfills a range on a single session; days already on disk are
#    skipped, so this is safe to repeat.
python -m desk.marketdata.refresh bhavcopy --date 2026-09-17
python -m desk.marketdata.refresh bhavcopy --from 2026-09-01 --to 2026-09-17

# 2. CALENDAR. Changes a few times a year, not daily. Without it the
#    store cannot tell a data gap from a market holiday and says so.
python -m desk.marketdata.refresh calendar

# 3. SECOND SOURCE. Reconciles the day's NSE closes against BSE and
#    stores the verdict. Without it the plan rests on one price source
#    and says so.
python -m desk.marketdata.refresh crosscheck --date 2026-09-17

# 4. CORPORATE ACTIONS, per symbol. An unadjusted split inside the
#    lookback window is an unexplained 50% gap the scanner reads as a
#    signal.
python -m desk.marketdata.refresh actions RELIANCE TCS INFY

# --- the research layer: a DIFFERENT module. `desk.research.refresh`. ---

# 5. EVENT CALENDAR. ONE market-wide request. Feeds the earnings gate,
#    which refuses a trade whose holding window contains a result.
#    Treated as stale after 3 days, because notice can be that short.
python -m desk.research.refresh events

# 6. FILINGS, per symbol. THE SLOW ONE: ~1,598 symbols is hours at the
#    1s throttle, and the XBRL documents dominate it. Resumable per
#    symbol AND per kind. --since bounds the documents fetched.
python -m desk.research.refresh research RELIANCE TCS --kinds filings --since 2023-06-01

# 7. FUNDAMENTALS TABLE. No network - recomputes from the filings
#    already on disk, so a parser fix does not mean refetching. Prints
#    the filing-age distribution; read it before setting any age filter.
python -m desk.research.refresh fundamentals

# 8. NEWS. Three RSS feeds, deduplicated by story. Stale after 18 hours
#    - a headline from yesterday's close describes yesterday's market.
python -m desk.research.refresh news

# 9. OUTCOMES. Closes the loop on what was decided earlier. `settle`
#    derives them from price history using the same exit simulator the
#    backtest uses; `close` records what you actually did.
python -m desk.research.refresh settle
python -m desk.research.refresh close --day 2026-09-17 --symbol RIR.NS \
    --price 245.30 --reason target
```

Then start the API and the web app:

```bash
.venv/Scripts/python.exe -m uvicorn desk.api.main:app --reload --port 8000
cd web && npm run dev          # http://localhost:5173
```

### What happens if you skip a step

Nothing crashes. Every missing input becomes a named caveat on the plan,
because a check that did not run must never read like a check that
passed:

| Skipped | The plan says |
|---|---|
| `bhavcopy` | NO TRADE, naming the fetch command |
| `calendar` | gaps UNCHECKED — a holiday and a missing file look alike |
| `crosscheck` | prices rest on a single source |
| `actions` | an unadjusted split would not have been caught |
| `events` | the earnings gate did NOT run |
| `research` / `fundamentals` | candidates were not screened on fundamentals |
| `news` | Stage 3 judged the shortlist with no market context |
| `settle` / `close` | days with trades and no recorded outcome, in red |

Exit codes are meaningful, so a scheduled task can detect failure:
**0** success, **1** something genuinely failed, **2** NSE rate-limited —
come back later rather than treating it as broken.

### Back it up

`configs/journal/` is tracked in git deliberately: prices, filings and
cross-checks can all be refetched, but what the desk decided on a given
morning exists nowhere else. This repo has no remote — `git remote add
origin <url>` and push, or the one artifact that cannot be regenerated
lives on a single disk.

## The trading calendar needs real data before the API is fully healthy

`desk/marketdata/calendar_in.py` refuses to guess NSE holidays — see its
module docstring. Fetch the real list once:

```powershell
.\.venv\Scripts\python.exe -m desk.marketdata.refresh calendar
```

This writes `configs/holidays_nse.json`, merging with whatever is already
there (pass `--replace` to discard prior years instead). `GET /health` reports
`"degraded"` if the current year isn't covered; `/calendar/{day}` returns 503
for a year with no data at all rather than assuming weekends-only.

Corporate actions, per symbol:

```powershell
.\.venv\Scripts\python.exe -m desk.marketdata.refresh actions RELIANCE INFY TCS
```

Writes `configs/corporate_actions/<SYMBOL>.json`. Every line it prints marked
`REVIEW` is an action that moves the price and could not be reduced to one
factor (a rights issue, a demerger) — it needs a human, not a retry.

## Daily prices: one request for the whole exchange

```powershell
.\.venv\Scripts\python.exe -m desk.marketdata.refresh bhavcopy --date 2026-09-11
```

NSE's full bhavcopy is a single file covering every listed scrip — 3,485 rows
on a typical day, of which 2,637 are the mainstream `EQ` series. Fetching
per-symbol history instead would cost ~2,000 requests for the same
information. Saved as `configs/bhavcopy/<date>.parquet` (gitignored — it is a
regenerable cache, not configuration).

## The scanner

The funnel runs Stage 0 -> 1 -> 2 -> 4, all deterministic, all reading
already-fetched snapshots with **no network call of its own** - the same
discipline as `/calendar/{day}`. Stage 3 (LLM research) is not built, so what
exists today is the whole deterministic path, which is ~98% of the funnel by
design: the AI reads and explains, it does not compute.

```powershell
curl "http://localhost:8000/plan/today?day=2026-09-11&regime=trending_up"
```

That single call is the project's actual output: one plan, or NO TRADE with a
reason that distinguishes *nothing was flagged* from *forty names reached the
risk gate and every one was refused*.

The stages are individually addressable too:

| Endpoint | What it does |
| --- | --- |
| `/scanner/stage0` | Universe hygiene from one day's snapshot. On real 2026-09-11 data: **3,485 -> 1,598** (non-EQ 848, liquidity floor 756, price floor 283). |
| `/scanner/stage1` | Per-symbol features and opportunity flags, vectorised across the whole universe. Measured: **2,637 symbols x 215 sessions in 3.3s**. |
| `/scanner/stage2` | Cross-sectional ranking with per-factor percentile contributions, so every score reads back to its inputs. |
| `/plan/today` | The whole funnel plus the risk gate. |

Thresholds are query parameters everywhere, never hardcoded.

**What the scanner says it cannot do**, on every single call, rather than
omitting it quietly: the F&O ban list and point-in-time index membership (no
endpoint found for either); sector strength and news/event flags (no feed);
relative strength vs Nifty unless you pass a benchmark; strategy votes (the
six scripts are off disk, and 0 of 6 are `trusted` - an untrusted strategy may
not influence a live shortlist even when its code is present).

Three properties worth not breaking:

- **A crisis regime silences momentum, trend and range-position** rather than
  down-weighting them, which leaves too few factors to rank and yields NO
  TRADE. That is correct: momentum in a crisis is not a weak signal, it is a
  wrong one.
- **A feature that could not be computed never raises a flag.** An unknown is
  not an opportunity.
- **The risk engine has veto power.** A 100/100 Stage 2 score buys a name the
  right to be considered and nothing else.

### Where the stop comes from

The stop is placed at the level that would prove the setup **wrong**, not at
a fixed multiple of anything. Stage 1 finds two levels per symbol and Stage 4
picks between them:

| Basis | Meaning |
| --- | --- |
| `swing_low` | A confirmed fractal pivot - the market turned here. |
| `low_20` | The 20-session low: a range floor, the weaker second choice. |
| `noise_floor` | Structure was found and then **overruled** for sitting inside the stock's ordinary daily range. Declared, never silent. |
| `atr` | No structure in the loaded history. The old volatility stop, as a **labelled fallback**. |

The **nearer** level wins, because the question is where the setup fails
*first*. The stop then sits `0.25 x ATR` beneath it, so a test of the level
is not a stop-out. Every trade carries an `invalidation` sentence naming the
level, in the API, the web plan and the journal - a number alone cannot be
argued with before the order is placed, and a named level can.

Nothing pulls a distant stop in to make a trade fit. A structural level 20%
away produces a 20% stop and `size_position` refuses it as `STOP_TOO_WIDE`;
on 2026-09-17 that refusal removed RIR.NS, whose last pivot was 23.6% below
the price. Trimming it would have put the stop somewhere the thesis was
still intact, which is the arbitrary stop this replaces.

**It has not been shown to make money, and that was measured rather than
assumed.** Replaying 2026-06-01 to 2026-09-17:

| Stop | n | win % | expectancy |
| --- | --- | --- | --- |
| ATR 2.0x (the old default) | 170 | 32.9 | -0.248R |
| ATR 3.5x | 209 | 44.0 | -0.097R |
| **Structural** (median 3.12x ATR) | 169 | 39.6 | -0.114R |

Structural beats the old 2.0x stop by +0.134R, but at **z = 1.42 that is not
significant**, and a plain 3.5x ATR stop matches it (z = -0.29). The honest
reading is that the 2.0x stop was too TIGHT and most of the gain is width,
not placement. What structural stops definitely buy is explainability; the
edge is still missing and it is not in the exits - see below.


## Health and staleness

The scanner's entire input depends on someone running the bhavcopy refresh
every trading day, and there is no scheduler. `GET /health` therefore reports
data freshness, not just "the process is up":

```json
{ "status": "degraded",
  "data": { "latest_snapshot": "2026-09-11", "expected_as_of": "2026-09-15",
            "sessions_behind": 1, "stale": true,
            "fix": "python -m desk.marketdata.refresh bhavcopy --date 2026-09-15" } }
```

Before this, `/health` checked only the calendar — so an uptime monitor would
report `ok` indefinitely while the desk sat on week-old prices, and the
failure was visible only to a human who happened to open the web app.

`DESK_CONFIG_DIR` and `DESK_CORS_ORIGINS` override the config directory and
the allowed origins. Note that widening CORS is **not** the same as making
this safe to expose: there is still no authentication on any endpoint.

## Accumulating history

Stage 1 onwards needs multi-day history, which `desk/store/` provides by
reading across the daily snapshots. Run the bhavcopy refresh every trading
day; the store reports exactly what it had:

```
215 sessions, 2025-11-17 to 2026-09-11, complete
215 sessions, ..., 3 MISSING: 2026-08-14, 2026-08-21, 2026-09-02
215 sessions, ... (gaps UNCHECKED - no calendar supplied)
```

That third line is not the same as the first. `Coverage.complete` is False
when the check could not run, never True by default.

## Performance

Measured on 2,637 EQ symbols x 215 sessions (566,955 rows):

| | naive | now |
| --- | --- | --- |
| `BarStore.history` | 5.6s | **0.49s** |
| Stage 1 | 6.3s | **3.26s** |
| Stage 2 | 0.02s | 0.013s |

Two changes got that. The store reads every day file in **one** batched
pyarrow dataset call with the column projection and symbol predicate pushed
down, rather than a `read_parquet` per day — at 215 files the per-file Python
overhead dominated, not the I/O. And `History.wide_many` factorises the
`(date, symbol)` index **once** for all five fields instead of `pivot`
rebuilding it per field.

Neither weakened a guarantee. The file list is still point-in-time filtered
before anything is opened, and the filename-versus-contents check still runs
on every file — from the parquet footer's own min/max statistics, metadata
only, no column data read.

`/plan/today` still runs the whole funnel synchronously per request. That is
the remaining structural gap: the funnel is a pure function of a snapshot
that changes once a day, so it should be precomputed into a cached artifact
the same way the snapshot itself is fetched once rather than per request.
`lookback` is capped at 300 sessions as a mitigation, not a fix.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest desk/tests -q
```

965 tests. Symbol translation, corporate-action point-in-time correctness,
every gate in the risk engine, the NSE source's free-text parser and bhavcopy
parser (both tested against real payloads captured from the live API — see
`desk/tests/fixtures/`), the trading calendar's refusal to guess, the
indicator layer, and the scanner's Stage 0 pinned against the real archive.

## Static analysis

```powershell
$env:PYTHONIOENCODING = "utf-8"   # skylos crashes on the cp1252 console without this
skylos desk strategies
```

Config lives in `[tool.skylos]` in [pyproject.toml](pyproject.toml). Vendored
repos are excluded because we do not edit them.

**How to read the current output.** It reports 17 dead-code items and every one
is the same true-but-mislabelled signal: `AnalysisRequest` and `AgentManifest`
have no callers because no agent adapter is written yet, and the `strategy_lib`
indicators have none because the six strategy scripts that call them sit outside
the scanned tree. That is unfinished integration, not dead code. **Do not run
`--gate` until the adapters exist** — it would fail the build on a contract
deliberately built ahead of its consumers.

## Layout

| Path | What lives there |
| --- | --- |
| `desk/contracts/` | The shared vocabulary. Every agent speaks `AnalysisResult`. |
| `desk/marketdata/` | Symbols, corporate actions, the trading calendar, data quality, provider registry. |
| `desk/marketdata/sources/nse.py` | The only network-touching module. Fetch/parse split; parsers tested against real captured payloads. |
| `desk/indicators/` | Indicators as their own layer — VWAP, MACD, gaps, market structure, volatility compression, relative strength. |
| `desk/scanner/` | The daily funnel: stage0 (hygiene), stage1 (features), stage2 (ranking), stage4 (risk gate). Stage 3 (LLM) not built. |
| `desk/store/` | Reads across daily snapshots. Point-in-time by construction; reports gaps rather than hiding them. |
| `desk/plan/` | `DailyPlan` and how it's assembled — deliberately outside the API layer so the scanner can import it without FastAPI. |
| `desk/risk/` | Deterministic position sizing. No LLM touches this file. |
| `desk/strategies/` | The six strategies as data: params, defects, maturity. |
| `desk/registry/` | What exists, and whether it is actually wired in. |
| `desk/api/` | FastAPI. The web app talks only to this. |
| `strategies/india/` | The corrected shared indicator + backtest library. |
| `agents/*/upstream/` | Vendored repositories. **Never edit these.** |
| `web/` | Vite + React + TypeScript control centre. |

## Standing rules

These are not style preferences. Each one was learned by something breaking.

- **One virtualenv per agent, permanently.** NautilusTrader needs `pandas<3`;
  Freqtrade runs on 3.0.5. They cannot share an environment.
- **Never `import freqtrade` from desk code.** GPL-3.0 — CLI subprocess or REST only.
- **Never run Python from a repository's `upstream/`** (except Vibe-Trading's
  `upstream/agent/`, which is its package root). The source shadows the
  compiled install.
- **Never relax a pin without re-running that agent's smoke test.** The two that
  matter: `pandas==2.3.3` (Nautilus), `joblib==1.5.3` (Freqtrade).
- **Never call an LLM inside a backtest loop.** Signals are precomputed, dated
  and replayed, or the result is not a backtest.
- **No LLM may relax a risk gate.** Sizing is a pure, unit-tested function.
- **NO TRADE is a successful outcome**, not a failure state or an empty screen.
