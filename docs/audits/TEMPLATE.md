# Repository intake audit — `<repo-name>`

> Copy this file to `docs/audits/<repo-name>.md` and fill it in **before** the
> repository is wired into the pipeline. The gate at the bottom is not advisory:
> a repo that cannot pass it does not get an adapter.

| | |
| --- | --- |
| **Repository** | `<url>` |
| **Pinned at** | tag / commit `<sha>` |
| **Licence** | |
| **Audited by / on** | |
| **Verdict** | Core / Useful / Sim-only / Wrong market / Reject |

---

## Stage 1 — Discover

*What is this, in two sentences, without marketing language?*

- **What problem does it solve:**
- **What it is not:**
- **Maintenance signal:** last commit, release cadence, open issue count, whether
  it has release tags at all.

---

## Stage 2 — Audit

The sixteen questions. **Answer every one.** "Unknown" is an acceptable answer
and a meaningful one — it is not the same as leaving the line blank.

| # | Question | Answer |
| --- | --- | --- |
| 1 | What does it actually do? | |
| 2 | What inputs does it require? | |
| 3 | What outputs does it produce? | |
| 4 | What assumptions does it make? | |
| 5 | What market conditions is it designed for? | |
| 6 | What timeframe does it target? | |
| 7 | **Is it suitable for Indian equities?** | |
| 8 | Does it require data we do not have? | |
| 9 | Does it contain look-ahead bias? | |
| 10 | Does it contain survivorship bias? | |
| 11 | Is there data leakage? | |
| 12 | Does it overfit? | |
| 13 | How does it behave across market regimes? | |
| 14 | What are its dependencies? | |
| 15 | Is the code reliable? | |
| 16 | Can results be reproduced? Is it computationally feasible? | |

### India compatibility — verified by grep, not by README

README claims do not count. Cite the file and line you actually read.

- [ ] Handles `.NS` / `.BO` symbols, or can be made to — *evidence:*
- [ ] Understands T+1 settlement — *evidence:*
- [ ] Models circuit bands — *evidence:*
- [ ] Models the Indian cost stack (STT, stamp duty, exchange txn, SEBI fee) — *evidence:*
- [ ] Handles the retail short-selling prohibition — *evidence:*
- [ ] Has any Indian broker/venue adapter — *evidence:*

### Dependency risk

| Package | Declared range | Actually works on | Risk |
| --- | --- | --- | --- |
| | | | |

> Loose upper bounds are the recurring defect across every repo audited so far.
> `pandas<4.0.0` is not a constraint, it is a wish. Pin what you tested.

**Silent-failure watch.** List any dependency whose failure mode is partial —
a subsystem that breaks while the main path keeps working. These are the ones
that get discovered months later.

---

## Stage 3 — Understand

- **Architecture, in one paragraph:**
- **Entry points:**
- **Where its decisions are actually made** (file:line):
- **What must never be modified** (and why):

---

## Stage 4 — Classify

- **Capabilities declared** (from `desk.contracts.enums.Capability`):
- **Asset classes:**
- **Horizons:**
- **Overlaps with** which existing fleet member, and **does it duplicate or differ**:
- **Unique capability it brings that nothing else has:**

> If it brings nothing unique, say so. Carrying a redundant integration costs
> maintenance forever and buys nothing.

---

## Stage 5 — Test

- [ ] Installed in **its own virtualenv** at `agents/<name>/.venv`
- [ ] `smoke_test.py` written and passing — *paste the output*
- [ ] Pinned in `requirements.txt` at versions that were actually run
- [ ] `manifest.yaml` written
- [ ] Confirmed it does **not** need to be run from `upstream/`

```
<smoke test output>
```

---

## Stage 6 — Backtest

**This stage is mandatory and is the one most often skipped.** A repository is
not integrated until its strategies have been backtested on Indian data with
the full cost stack.

| Metric | Result | Notes |
| --- | --- | --- |
| Period tested | | |
| Total return / CAGR | | |
| Sharpe / Sortino | | |
| Max drawdown | | |
| Win rate / loss rate | | |
| **Expectancy per trade** | | |
| Profit factor / payoff ratio | | |
| **Max consecutive losses** | | |
| Trades per year | | |
| Exposure | | |
| Costs applied | | STT, stamp duty, exchange txn, SEBI fee, slippage |

- [ ] Out-of-sample window held back and tested **once**
- [ ] Walk-forward (expanding window)
- [ ] Regime slices — results reported **per regime**, not blended
- [ ] Parameter sensitivity ±20%
- [ ] Cost stress (double slippage and impact)
- [ ] Data passed `desk.marketdata.quality.assert_clean()` first

> If fewer than 30 closed trades, the win rate is an anecdote. Say so in the
> notes rather than quoting it.

---

## Stage 7 — Compare

Against the incumbent that does the closest thing:

| | This repo | Incumbent | Winner |
| --- | --- | --- | --- |
| | | | |

**Bake-off result:**

---

## Stage 8 — Adapt

- **Adapter location:** `agents/<name>/adapter.py`
- **Maps its output onto** `AnalysisResult` — which fields can it fill, and which
  must stay `None`?
- **Fields it cannot supply:**
- **Licence constraint on the adapter** (e.g. GPL ⇒ subprocess or REST only,
  never `import`):

---

## Stage 9 — Integrate

- [ ] Registered in `desk/registry/fleet.py`
- [ ] `installed` / `verified` / `wired` / `india_ready` set **honestly**
- [ ] Blockers listed
- [ ] Standing constraints added to `README.md` if it introduced any

---

## Stage 10 — Orchestrate

- **Which regimes is it eligible in:**
- **Which regimes silence it:**
- **Initial weight:** 1.0 — and it stays there until the decision journal has
  resolved outcomes to justify moving it.
- **Correlation cluster:** which existing agents does it likely correlate with?

---

## The gate

A repository may be marked `wired = True` **only** when every line below is true.
If any is false, it stays at `verified` and the reason is recorded in
`blockers`.

- [ ] Licence understood and compatible with how we intend to call it
- [ ] Installed in its own venv with pinned, tested versions
- [ ] Smoke test passing from a neutral working directory
- [ ] India compatibility verified **by reading source**, not the README
- [ ] Look-ahead, survivorship and leakage each explicitly assessed
- [ ] Backtested on Indian data with the full cost stack
- [ ] Walk-forward survived
- [ ] Adapter speaks `AnalysisResult` and fills no field it cannot honestly fill
- [ ] Registered in the fleet with accurate flags
- [ ] Nothing under `upstream/` was modified

**Signed off:** _____________  **Date:** _____________
