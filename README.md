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
discipline as `/calendar/{day}`. Stage 3 (LLM research) **is** built and
wired - `_llm_client()` returns a live `MeteredClient` - but the configured
account has no credit, so in practice every run today takes the deterministic
path, which is ~98% of the funnel by design: the AI reads and explains, it
does not compute. Stage 3 may only ever REMOVE names, and `narrow()` iterates
Stage 2's own index so that holds even if the model misbehaves.

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
assumed.** Every figure below is NET of costs over the full 738 sessions on
disk (2023-09-01 to 2026-09-17) with the regime measured per day. Earlier
versions of this table quoted gross numbers from a 3.5-month window with the
regime hardcoded, which flattered them by roughly 4x.

| | n | win % | net expectancy |
| --- | --- | --- | --- |
| The funnel, structural stops | 1,250 | 45.4 | **-0.0817R** (SE 0.0205, z -3.98) |
| Random selection, same machinery | ~1,246 | - | -0.031R *gross* |

Three separate sources of loss, and the ordering is the reverse of where
intuition pointed:

| Source | Size | What moves it |
| --- | --- | --- |
| Round-trip costs | 0.041R | **Only** the stop width as a share of price - see below |
| Machinery floor, which random pays too | ~0.031R | The horizon and the exits; 88% of trades time out |
| Inverted factor signs | ~0.010R | `REVERSAL_FACTORS`, built and measured, not adopted |

**Cost drag is arithmetic, not a mystery.** Every component of `CostModel`
is a percentage of turnover, so the quantity cancels:

```
cost_R = round_trip_% / stop_distance_%
```

0.422% over a 4% stop is 0.105R; over a 12% stop it is 0.035R. Nothing else
moves it - not trading less often, not sizing bigger, both of which scale
cost and risk together. Confirmed independently on the strategy adapters:
`bollinger_rsi` at a 2.90% median stop measured 0.1435R against 0.145
predicted, `donchian_breakout` at 10.30% measured 0.0394R against 0.041.

**A wider stop is not a free win, and one experiment that looked like it was
turned out not to be.** Widening the structural buffer to 3.0x ATR produced
gross +0.134R at a 57.9% win rate - and it was not the stops. Median stop
distance barely moved (12.03% -> 12.65%); what changed was the UNIVERSE.
Breaching the 15% `STOP_TOO_WIDE` ceiling needs
`structural_% + buffer x atr_pct > 15%`, so a large buffer preferentially
removes high-volatility names: median ATR of trades taken fell 3.63% ->
2.21% and p90 fell 4.85% -> 2.98%, at the cost of 73% of the trade count
(1,251 -> 337). It was a volatility filter wearing a stop's clothes, which
matches the strongest signal in the feature study (`atr_pct`, t = -3.4).

What structural stops definitely buy is explainability. The edge is still
missing, and it is not in the exits.


## More than one strategy

The funnel is one opinion. `desk/strategies/catalog.py` describes six more -
family, parameters, regime gates, maturity, known defects - and for a long
time describing was all it did: a `StrategySpec` holds no entry logic, and
nothing turned one into a trade idea.

`desk/strategies/adapters.py` closes that. Each adapter reads the Stage 1
feature table that has already been computed and returns `AnalysisResult`
objects - the shared envelope, carrying entry zone, trigger, stop,
T1/T2/T3 with partial fractions, invalidations and timestamped evidence.

```powershell
curl "http://localhost:8000/strategies/proposals?day=2026-09-17"
```

**The measured regime is the router.** A strategy that names today's regime
as hostile is silenced, not down-weighted - a mean-reversion rule in a crisis
is not a weak opinion, it is a wrong one. On 2026-09-17 the regime measured
`range`, so both trend strategies correctly said nothing and `bollinger_rsi`
produced 12 proposals; on a trending day that flips. This is why more than
one strategy matters: a single rule is idle most of the time by construction.

| Strategy | Speaks in | Status |
| --- | --- | --- |
| `donchian_breakout` | trending up | **implemented** - 31 proposals on 2026-09-17 |
| `bollinger_rsi` | range | **implemented** - 12 proposals, every R:R clearing the 1.5 floor |
| `supertrend_adx` | trending up/down | catalogued, no adapter |
| `pairs_trading` | - | BUG-03 open: hedge ratio fitted on the whole sample |
| `ml_classifier` | - | BUG-04 open: random train/test split leaks the future |
| `opening_range_breakout` | - | parked: no intraday data exists in the system |

Three outcomes are reported separately and never conflated: a strategy that
**proposed** (possibly nothing), one **silenced** by the regime, and one
**unimplemented** so it could not be asked. Asking for an unimplemented
strategy raises rather than returning an empty list, because empty reads as
"the market offered nothing".

**Nothing here is sizeable.** Three sit at `AUDITED` and three at `DRAFT`
("code exists, nothing verified") - every one below the trusted threshold,
so `tradeable_now` comes back empty and every proposal carries its own
maturity. A strategy becomes sizeable by surviving
walk-forward, not by appearing in a response.

### The maturity ladder, and the machine that climbs it

```
DRAFT -> AUDITED -> REVALIDATED -> WALK_FORWARD -> PAPER -> LIVE
                                   ^ sizing starts here
```

`desk/backtest/walkforward.py` is what earns a rung. It measures a strategy
on the funnel's own yardstick - Stage 0's universe, Stage 1's features,
`size_position` with every gate, `simulate_trade` for exits, `BacktestResult`
for costs - so a strategy at -0.05R and the funnel at -0.082R are comparable
because nothing between them differs.

It **refuses to promote on an aggregate**. A rule that earns its whole result
in one window and loses the rest has shown one good quarter, not an edge, and
a pooled mean cannot tell those apart. Windows are disjoint rather than
rolling: overlapping windows share trades, so "positive in 3 of 4" would
count the same trades twice and read as corroboration when it is repetition.

Two things it does not simulate, declared on every result rather than left
implicit: a strategy's own exit (Donchian's channel exit is not modelled, so
its number is a lower bound on a rule whose stated edge includes cutting
failures early), and portfolio interaction between strategies.

Walk-forward is labelled honestly here. These strategies have **no fitted
parameters** - 20/10/2.0 and 20/2.0/14/30 are catalog constants, not values
optimised on this data - so splitting the history buys no protection against
overfitting, because no fitting happened. It buys a test of *stability*.


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
