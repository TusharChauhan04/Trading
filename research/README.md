# research/ — the scripts that produced every number this project reports

**Why this directory exists.** Every measurement in the README, in
`docs/HANDOVER.md` and in the module docstrings was produced by a script that
lived in a session-scoped temp directory. Those directories are wiped between
sessions, and one was wiped mid-session. So the project could state its
conclusions and had no way to **reproduce** them — a new agent (or the same one
after a restart) would have had to rebuild each harness from scratch, including
the exact panel construction, buffer logic and `num_trials` bookkeeping that
make the numbers comparable.

These are **research scripts, not library code**. They are deliberately not
imported by `desk/` and not covered by the test suite. They are kept because a
measurement you cannot re-run is an anecdote.

**Run them from the project root** with the desk venv:

```bash
.venv/Scripts/python.exe research/walkforward/h11_covid.py
```

Each carries a docstring stating what it tests, what it found, and — for the
hypothesis scripts — the **pre-registered acceptance criterion** that was fixed
before the run. Read that docstring before re-running; it is the record of what
the measurement was for.

---

## walkforward/ — the strategy hypotheses, H6 onward

The surviving cell throughout is **donchian_breakout, 12% stop / 24% target /
20 sessions / 1% buffer**, with `bollinger_breakout` as the agreeing second
rule.

| script | question | result |
| --- | --- | --- |
| `h6.py` | does the cell survive disjoint walk-forward windows? | 2 of 4 windows. Killed the "defensible" claim. |
| `h7_rerun.py` | re-walk on a deeper sample after the 5-year backfill | 5 of 8 windows positive and timing-significant; corrected H6's "one nine-month period" characterisation |
| `h8_regime.py` | do the failing windows separate on any regime measure? | "EW index above its 50-day mean" separated cleanly (65.3% vs 26.9% of days) |
| `h9_gate.py` | does gating on that regime, **point-in-time**, help? | **No.** BOTH fell 5→4. Window-level association did not transfer to the day level. |
| `h10_volume.py` | does volume confirmation help? (from `breakout_engine`) | **No.** Monotonically worse: BOTH 5/8 → 5/8 → 4/8 → 3/8 as the threshold rises |
| `h11_covid.py` | does the cell survive the COVID crash, on 1,715 sessions? | **No.** 2 of 8 (or 5 of 11). The crash window is −0.2029R with permutation p = 1.0000 |
| `h12_longhold.py` | holds of 20/40/60 against stops of 12/15/20% | the cost arithmetic points here; was running when last recorded — check `git log` |

**The acceptance criterion in every one of these** is the count of windows
**both positive and timing-significant** — never pooled net R (it rises
mechanically when losing periods are excluded) and never DSR alone. H11 is the
reason: the same data gave DSR 0.8654 at 8 windows and 0.2268 at 11, because
the 8-window boundary averaged the crash together with the recovery.

**`num_trials` bookkeeping matters and is in the scripts.** It was 41 at H9,
47 at H10/H11 (which added no new cells — the cell was pre-specified and only
the sample changed), and 56 at H12 (which searches nine new cells, so DSR has
to be told).

## intraday/ — the main-goal question

| script | what it does |
| --- | --- |
| `fetch_intraday.py` | fetches 5-minute bars for the most liquid names and stores one parquet per session. Superseded by `python -m desk.marketdata.refresh intraday`, kept because it documents the measured per-request limits |
| `intraday_scale.py` | the natural scale: 5-minute bar range (median 0.166% of price) and session range (median 2.18% of open). Everything else follows from these two numbers |
| `intraday_rr.py` | **the main-goal measurement.** 84,090 trades, every bar as an entry, confined to one session. Gross negative at every stop width |
| `intraday_opt.py` | the same with the optimistic convention (target checked before stop on a bar touching both). Moves the 0.5% cell from 16.2% to 16.3% — the convention is not what makes it negative |
| `hourly_rr.py` | 1-hour bars held 2/5/10 sessions, which 5-minute bars could not test. Gross turns positive (+0.047R) but zero net-positive cells |
| `rr_frontier.py` | varies `rr` from 0.5 to 3.0 — the parameter never previously moved. All 20 cells have a negative gap between achieved and required win rate |

## calibration/ — measuring the tool before trusting the tool

These produced the `_NULL` table in `desk/research/stationarity.py`. The
pattern is worth copying: before using a published threshold, simulate its null
at the actual sample length.

| script | what it established |
| --- | --- |
| `hurst_cal.py` | the lagged-variance estimator is biased **down**: a pure random walk scores a mean of 0.41 at n=250, so upstream's "H < 0.45 = mean-reverting" labels 61% of random walks mean-reverting |
| `hurst_null.py` | the null distribution across ten sample lengths, 4,000 paths each |
| `hurst_table.py` | the final shipped table, 20,000 paths per length with eleven quantiles |
| `hurst_robust.py` | **the check that had to pass first** — is the null the same under fat tails and volatility clustering? Across Gaussian, Student-t(3), GARCH(1,1) and this desk's measured two-state vol the 5% critical value moves at most 0.011 in H, under a tenth of a standard deviation |
| `selfcheck.py` | does the **shipped module** reproduce its advertised false-positive rate? P(p<0.05) = 0.060/0.057/0.051/0.046. Calibrating and validating-through-the-shipping-path are different steps |
| `real_hurst.py` | the universe answer: 3.5% of names pass at p<0.05 over a year, 4.4% over five — at or below the 5% null |
| `uniformity.py` | the binomial test behind those counts, without scipy |

## audit/ — the pre-commit regression harness

**The prompt in `docs/MEGA_PROMPT.md` instructs a new agent to run this before
every commit, so it must not live in a temp directory.**

`audit_now.py` dumps Stage 1 output and the day's plan as JSON. Run it in the
current tree and in a git worktree at the pre-extraction baseline
(`dd21486`), with `DESK_CONFIG_DIR` pointed at the real configs, then compare:

- row count (995)
- every pre-existing Stage 1 column **bit-identical at atol=0**
- the plan's verdict / fingerprint / regime / extra_notes

```bash
git worktree add --detach <tmp>/baseline dd21486
cp research/audit/audit_now.py <tmp>/baseline/
export DESK_CONFIG_DIR="<project>/configs"
.venv/Scripts/python.exe research/audit/audit_now.py > now.json
cd <tmp>/baseline && <project>/.venv/Scripts/python.exe audit_now.py > base.json
```

**Afterwards check `git status` for `configs/journal/decisions/` and delete any
entry it created** — `/plan/today` writes a decision as a side effect, so the
audit forges a journal record every time it runs. `regress.py` and
`regress2.py` are narrower variants that check the LLM route and the
fundamentals join.

## extraction/ — the measurements behind the upstream decisions

Each of these is why a module was taken or refused. `agents/openterminal_ui/
manifest.yaml` records the decisions; these record the evidence.

| script | what it decided |
| --- | --- |
| `bug03.py`, `bug03b.py` | BUG-03 measured in money: in-sample hedge ratio +2.698% per round trip vs point-in-time +0.279% — a 10x inflation |
| `fullpower.py` | the pairs screen at full power: 278 pairs, p-value distribution **is** the null at every window |
| `clust_probe.py` | `ward` linkage on a correlation distance is invalid here — 7 negative Gram eigenvalues, and it merges at height 1.5570 where the maximum possible distance is 1.0 |
| `thresh.py` | the correlation threshold: pairwise correlation runs mean 0.219, so 0.5 is about the 98th percentile. Industry explains only +0.089 of co-movement |
| `binds.py` | does the correlated cap actually bind? 53.3% of days had two of the top-8 in one cluster |
| `shrink2.py` | **Ledoit-Wolf shrinkage destroys threshold clustering** rather than denoising it — ARI falls to 0.0 at δ=0.5. The reason `covariance.py` was refused |
| `s4b_slippage.py` | the flat 15bps/side guess was 71% of the round trip; modelled against real turnover it is 2.6–3.9 bps |
| `s6_permutation.py` | the permutation test that retired the "+0.071R" claim (p = 0.574) |
| `s7_ruin.py` | block vs IID bootstrap: p95 drawdown 33.2% vs 16.8% on the same data |

---

## If you are adding a new hypothesis

1. Write the script here, with a docstring stating the question and the
   **pre-registered acceptance criterion** before you run it.
2. Increment `num_trials` honestly if you are searching new cells.
3. Report **two window counts**. A result that disagrees between them is not a
   result.
4. Commit the script with the finding in the message, update the README table
   above, and record the conclusion in megamemory.
