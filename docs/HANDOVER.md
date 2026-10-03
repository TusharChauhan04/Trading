# HANDOVER — read this before touching anything

You are picking up **The India Desk**: an AI-assisted daily decision-support
system for Indian equities (NSE/BSE), personal use, one Windows laptop, free
data sources only. This file exists so a fresh agent can continue at the same
level of understanding as the one that stopped. It is written to be read
top-to-bottom once, then used as a reference.

**Last updated:** 2026-10-03. **Suite:** 1,333 tests passing. **Daily store:**
1,715 sessions (2019-10-01 → 2026-09-29). **Commits:** 93.

---

## 0. The single most important thing to understand

**This project has no validated trading edge, and its value is that it can
prove that honestly.** Every avenue tested so far has returned a measured
negative - §5 lists them with their numbers. The tooling is the asset: it
has repeatedly caught the previous agent's own errors, including published
ones.

If you arrive wanting to find an edge, you will find one — and it will be
wrong. The three times this project published a positive result, all three
were retracted by its own later measurements - and a fourth flattering
number was caught before it was believed:

1. "+0.071R, the first thing that makes money" → permutation test showed it
   was market beta (p = 0.574).
2. "+0.0505R, smaller and defensible" (20-session hold) → the walk-forward
   on disjoint windows killed it.
3. "5 of 8 windows positive and timing-significant" on 1,241 sessions → on
   1,715 sessions including the COVID crash, 2 of 8.

And the one caught in time: **DSR 0.8654** (2026-10-03), the best statistic
this project has produced, against a 0.95 bar - an artifact of
**window-boundary choice**, since the same data at 11 windows gives 0.2268.
It was never acted on, because the acceptance criterion was a count of
windows rather than a risk-adjusted aggregate. That is what the criterion is
for.

A related trap, and a different one: sample depth changes QUALITATIVE
conclusions, not just confidence intervals. "The whole edge is one
nine-month period" was published on 4 windows and became "works in 5 of 8
regimes" on 8. So do not characterise WHY a walk-forward failed on a thin
sample - establish that it failed, then deepen the sample.

**The discipline that catches this is pre-registration.** Decide the acceptance
criterion before the run, write it in the script's docstring, and judge on it
even when a different number looks better. The standing criterion for a
strategy cell is **the count of walk-forward windows that are BOTH positive AND
timing-significant** — never pooled net R (which rises mechanically whenever
losing periods are excluded) and never DSR alone (which believes whatever
window slicing you hand it).

---

## 1. What the user wants

A daily answer: **Primary / Secondary / Watchlist / NO TRADE**, with a
**1:2 risk-to-reward** target — "Risk ₹500, potential profit ₹1000". Stops must
come from where the trade is *invalidated* (a swing low, a structure break),
never an arbitrary percentage.

Constraints the user has stated and that are not negotiable:

- **Free data sources only.** No paid subscriptions.
- Screener.in is **manual human use only**, never automated — not for cost but
  because it shows *today's* restated financials with no point-in-time view, so
  any backtest fed from it is contaminated.
- Trendlyne/Tijori/Bloomberg/Reuters are out (paid; user confirmed no
  subscription).
- Moneycontrol/ET/Business Standard: **RSS only**, never article scraping.
- **Do not work around a 403.** Find a route the publisher offers.
- **Do not hammer NSE** — it rate-limits. Throttle and cache.

---

## 2. Hard engineering rules (violating these breaks the project)

- **Never `import freqtrade`** from desk code — GPL-3.0. CLI subprocess or
  REST only.
- **Never edit anything under `agents/*/upstream/`.** It is a read-only clone.
- **Never run Python from any repo's `upstream/`** (one exception:
  Vibe-Trading's `upstream/agent/`).
- **Never relax pins** without re-running smoke tests. `pandas==2.3.3` and
  `joblib==1.5.3` are load-bearing for every measurement taken so far.
- One virtualenv per agent, permanently. The desk's is `.venv`.
- The package is **`desk/`**, never `platform/`.
- **Never log, echo or commit the OpenAI API key.** `.env` stays gitignored.
- **No LLM may relax a risk gate.** Never call an LLM inside a backtest loop.
- **Point-in-time correctness is enforced by FILENAME**, never a runtime
  filter. `configs/bhavcopy/YYYY-MM-DD.parquet` — a scan for a past day
  physically cannot read a later file.

---

## 3. The architecture in one page

A **staged funnel**, cheap work first, expensive work only on survivors:

| Stage | What it does | Cost |
| --- | --- | --- |
| 0 | hygiene — price/turnover floors, data quality | whole universe |
| 1 | features — 35 columns, matrix-native | ~1,100 names |
| 2 | ranking — regime-routed factor scores | ~1,100 → ~600 |
| 3 | LLM narrative check — **can only REMOVE names** | ~8 names |
| 4 | risk gate + sizing — every hard limit | ~8 → 0-3 trades |

The economic argument for the funnel is that only Stage 3 costs per-call.
Running fundamentals+news on 1,598 Stage-0 survivors is the
₹40,000–80,000/month explosion the funnel exists to prevent.

Key modules:

- `desk/scanner/stage0..4.py` — the funnel
- `desk/risk/engine.py` — **owns every gate**; Stage 4 proposes prices and
  hands over. Nothing may loosen a limit, retry with a wider stop, or convert
  a refusal into a warning.
- `desk/journal/` — Decision (immutable) and Outcome (mutable observation)
- `desk/backtest/` — walkforward, preset_wf, permutation, ruin, slippage,
  costs, simulate, intraday
- `desk/robustness/scorecard.py` — PSR / DSR / MinTRL
- `desk/api/main.py` — FastAPI surface, ~2,400 lines
- `desk/marketdata/` — NSE, BSE, Yahoo sources; refresh commands; store

---

## 4. The measurement discipline — this is the actual methodology

**Four independent controls, each catching a different failure. Passing one
licenses nothing about the others.**

1. **Random-entry arm on the same calendar.** Catches market drift/beta. This
   is what retired the "+0.071R" claim.
2. **Permutation test on entry timing** — shuffle each symbol's signals
   *within that symbol*. Catches "the signal adds nothing over random timing".
   Run it **inside each window**, not only on the pooled sample: a
   full-sample permutation passed a cell at p=0.0033 that per-window analysis
   killed.
3. **Disjoint (never rolling) walk-forward windows.** Catches a single good
   period. With rolling windows "positive in 3 of 4" can be one good period
   counted three times.
4. **Deflated Sharpe (DSR)** with `num_trials` **REQUIRED, never defaulted**.
   Catches multiple testing.

**Traps already paid for, do not re-learn them:**

- **Sign-flip nulls are wrong for R-multiples.** Flipping +2R gives −2R, which
  is impossible under a stop, and the error inflates significance. (It is
  defensible for daily *equity-curve* returns, which is a different thing.)
- **IID bootstrap understates drawdown.** It breaks losing streaks: reported
  p95 DD 16.8% where a block bootstrap on the same data gave 33.2%. Use
  `desk/backtest/ruin.py`.
- **Window count is a free parameter that moves the verdict.** Report two.
- **`cost_R = round_trip% ÷ stop%`** — quantity cancels; only stop *width*
  matters. Every optimiser therefore drifts toward wide stops. Do not re-test
  narrow stops without also lengthening the hold; the two are coupled.
- **Three outcomes are never conflated**: a strategy that *proposed* (possibly
  nothing), one *silenced* by the regime, one *unimplemented*. Asking for an
  unimplemented strategy raises, because an empty list reads as "the market
  offered nothing".
- **`checks_skipped` / `unavailable` is a first-class third state.** A gate
  whose input is missing is *skipped and said so*, never silently passed.

---

## 5. What has been measured and FAILED (do not redo these)

| Hypothesis | Result |
| --- | --- |
| Donchian/Bollinger breakout, 12% stop / 20 sessions | DRAFT. 5/8 windows on 1,241 sessions; **2/8 or 5/11 on 1,715**. Does not survive COVID. |
| Regime gate on trend (EW index vs 50-day mean) | Window-level correlation did **not** transfer to day level. BOTH fell 5→4. |
| Markov volatility regime as a gate | Working and failing windows overlap (0.346 vs 0.469). |
| Volume confirmation on the breakout | Monotonically worse: BOTH 5/8 → 5/8 → 4/8 → 3/8 as the threshold rises. |
| Pairs / cointegration | 278 liquid within-industry pairs: p-value distribution **is** the null at every window. Survivors unstable across windows; hedge ratios non-economic (one negative). |
| Univariate mean reversion (Hurst) | 3.5%–4.4% of names pass at p<0.05 against a 5.0% null — **at or below chance**. |
| **Intraday 1:2 on 5-minute bars** | **Gross negative at every stop width.** Required round-trip cost comes out *negative*. |
| Intraday on 1-hour bars, 2–10 session holds | Gross turns positive (+0.047R) but 0 net-positive cells. |

**The unifying explanation**, and the most useful thing in this file:

| timescale | stop | cost_R | gross edge | net |
| --- | --- | --- | --- | --- |
| 5m, 1 session | 0.5% | 0.844 | −0.150R | −0.994 |
| 1h, 10 sessions | 1.5% | 0.281 | +0.047R | −0.235 |
| daily, 20 sessions | 12% | 0.035 | +0.080R | **+0.045** |

The gross edge is small and roughly flat (+0.05 to +0.08R) once the horizon
exceeds a week. **`cost_R` decides viability, and it is set entirely by stop
width.** Solving: a +0.05R edge needs a stop wider than **8.4%**, which puts a
2R target ~17% away — a multi-*week* hold. **The search direction is up in
timescale, not down.** Every move toward shorter timeframes makes `cost_R`
worse hyperbolically.

---

## 6. The upstream repositories — and the rule that matters

`agents/openterminal_ui/upstream/` is a 795-file MIT clone (OpenTerminalUI).
`agents/vibe_trading/` has a 206-factor library. **Nothing is imported
wholesale.** Each module is read, verified against our data, and wired
individually.

**THE RULE: read the file before importing it. Six for six, reading reversed
the decision.** Not one problem was visible from the module name or README.

**FOUR MODULES FABRICATE DATA** and are quarantined by an enforced test
(`desk/tests/test_upstream_quarantine.py`):

| module | what it invents |
| --- | --- |
| `core/historical_data_service` | `_synthetic_ohlcv()` returns a **seeded random walk as OHLCV** on an empty fetch — no warning, no flag, no exception. The worst and quietest. |
| `tca/service` | builds ten "trades" from an md5 hash. There is no real TCA in that repo. |
| `services/sector_rotation` | `_generate_mock_rrg()`; at least it warns. |
| `services/stress_test_service` | every factor beta from **sha256 of the ticker string**; for an unknown ticker the *sector* comes from `_infer_sector_from_hash` too. |

The map had marked three of those "take". The quarantine test also asserts the
fabricating code is **still present upstream**, so if it is ever fixed the test
fails and the quarantine gets reconsidered deliberately rather than outliving
the problem.

**20 desk modules derive from the extraction.** `agents/openterminal_ui/
manifest.yaml` records every decision: 6 verified, 17 refusals/caveats with
measured reasons, 3 scope exclusions. **Refusals outnumber extractions ~2:1**,
and that ratio is the honest part of the work.

**Scope honesty:** ~38 of 795 files have been read. About 360 are out of scope
(the web application: tests, api routes, alembic, auth, db, oms). That leaves
~220 quant-relevant files, so **under a fifth has been examined.** Never claim
otherwise — this project has twice published a completeness claim it could not
support. Inventory with `find . -name "*.py"` grouped by directory, compared
against the manifest, **at the moment of claiming**.

---

## 7. Things that look like bugs and are not

- `_extreme()` uses `series.shift(1).rolling(length)` — the `shift(1)` is
  **load-bearing**. A channel breakout must compare against the channel as it
  stood *before* the current bar. Without it the rule fires never and reads as
  a quiet market. The same off-by-one appeared in three places.
- Pine export emits `ta.highest(high, channelLen)[1]` — the `[1]` is the same
  fix.
- `aroon()` maps window position straight to recency. Do **not** "correct" it
  to `(period-1-pos)`; that is bars-since and inverts the indicator.
- Ichimoku `senkou_span_a/b` are returned **unshifted**. A chart draws them 26
  bars ahead; shifting them here would hand a caller a value at bar *t*
  computed at *t−26*.
- The drawdown cap has **no default**, deliberately. A 20% cap at 1% risk fires
  one run in three while the strategy behaves exactly as measured.
- `pct_change(fill_method=None)` everywhere — pandas' default pads gaps into
  **fabricated zero returns**, which manufactures correlation between names
  that merely share non-trading days.

---

## 8. Known open problems

1. **`risk_pct` default is 1.0 and the drawdown work supports 0.5.** At 1.0%
   there is a 32.6% chance of a 20% drawdown; at 0.5% it is 2.5%. **User
   decision pending.**
2. **`/plan/today` writes a decision to the journal as a side effect**, so any
   regression audit that calls it forges a record. `Decision.reconstructed`
   now makes that visible. Whether the endpoint should *refuse* for a past
   date is the operator's call. **User decision pending.**
3. **The paper track record is not a paper record.** `refresh settle` works and
   has produced two real outcomes (BHAGYANGR +0.4517R, GAYAPROJ −0.3954R) —
   but both come from a decision recorded **7 days after its date**. 2026-09-17
   is unsettled and is also a reconstruction.
4. **Stage 3 has no provider configured.** A free OpenRouter key (3 lines in
   `.env`) unblocks it at zero cost; a local LM Studio endpoint also works.
5. **F&O verdict was reopened and never revisited.** The "₹12 lakh" ruling was
   about futures *margin*; a defined-risk option spread costs the premium. Ten
   derivatives services upstream are unexamined. Today's cost arithmetic makes
   this *more* interesting: options define risk without needing a wide stop.
6. **DCF is blocked on data, not code.** NSE quarterly XBRL is a P&L; FCFF
   needs capex, change in working capital and net debt, which it does not
   carry. Computing it anyway yields EBITDA wearing a cash flow's name.
7. Corporate actions cover 3 of ~4,243 symbols. An unadjusted split inside a
   lookback window is an unexplained 50% gap the scanner reads as a signal.

---

## 9. How to work here

**Daily data refresh** (order matters):

```bash
python -m desk.marketdata.refresh bhavcopy --date YYYY-MM-DD   # or --from/--to
python -m desk.marketdata.refresh calendar
python -m desk.marketdata.refresh crosscheck
python -m desk.marketdata.refresh actions SYMBOL...
python -m desk.marketdata.refresh intraday --interval 5m       # accumulates
python -m desk.research.refresh events|filings|news|settle
```

**Before every commit, both of these:**

```bash
.venv/Scripts/python.exe -m pytest desk/tests/ -q          # must stay green
```

and the **regression audit** — a worktree at the pre-extraction baseline, run
with `DESK_CONFIG_DIR` pointed at the real configs, comparing: row count (995),
every pre-existing Stage 1 column bit-identical at atol=0, and the plan's
verdict/fingerprint/regime. A behaviour change must be *deliberate, measured
and reported* — never silent. **Check `git status` for
`configs/journal/decisions/` afterwards and delete any forged entry.**

**Commit messages carry the reasoning**, not just the change. Read `git log`
for the house style: what was measured, what number came out, what it means,
and what is explicitly not claimed.

**megamemory** is the project knowledge graph and the only continuity between
sessions. Call `megamemory` overview at session start, query before each task,
and **record after each task** with specific parameter names, defaults, file
paths and rationale.

---

## 10. Intraday data — a hard constraint worth knowing early

Measured, not from documentation:

| interval | max span per request | sessions obtained |
| --- | --- | --- |
| 5m / 15m | 45 days (60 → HTTP 422) | 31 |
| 1h | 365 days (730 → 422) | 245 |

**The window cannot be stitched.** Walking backwards in 40-day chunks, every
window but the most recent returns 422 — including one *inside* the advertised
60-day retention. So 5-minute history is hard-capped near 31 sessions and the
only way to deepen it is to accumulate daily.

**The last reliably traded 5-minute bar is 15:15.** Yahoo returns 15:20 and
15:25 as null on every symbol and day checked; 15:30 is absent or zero-volume;
09:15 always carries zero volume. An intraday exit "at the close" means 15:15.

`configs/intraday/` is gitignored like `configs/bhavcopy/` — but unlike the
daily store it **cannot be refetched for past dates**, so those files are the
only copy.

---

## 10b. Reproducing anything in this file

**`research/` holds the script behind every number quoted here**, grouped as
walkforward / intraday / calibration / audit / extraction, with a README
table mapping each script to what it established. Until 2026-10-03 those
scripts lived in a session-scoped temp directory and were therefore lost
between sessions - the project could state its conclusions and not
reproduce them. `research/audit/audit_now.py` is the pre-commit regression
harness referred to throughout.

## 11. If you only do one thing

Run the suite, read `agents/openterminal_ui/manifest.yaml` and this project's
`README.md`, then query megamemory for the concept list. The README carries the
measured tables; the manifest carries every extraction decision with its
reason; megamemory carries the narrative of why each conclusion was reached and
which earlier claims it superseded.

Then pick up from §8 — the open problems are ordered roughly by how much they
block.
