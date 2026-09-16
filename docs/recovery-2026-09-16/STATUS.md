# Recovery — 2026-09-16 — COMPLETE

## What happened

Every file under `C:\Users\TUSHAR\OneDrive\Desktop\Trading` vanished. The
folders survived as OneDrive Files On-Demand placeholders (reparse tag
`0x9000e01a`), so the content was in the cloud rather than deleted — but
OneDrive restarted at 23:49:53 on 2026-09-15 and would not re-hydrate, and the
OneDrive-side restore did not work. Recycle Bin held none of it, no shadow
copies, no restore points.

There was no git repository. That is what turned a sync glitch into a
recovery project.

## How it was rebuilt

From the Claude Code session transcript (20MB, stored outside OneDrive, so it
survived). Not a snapshot — a **chronological replay**: every `Write`, every
`Edit` and every file-modifying shell command extracted and re-executed in
original order. Scripts in this folder; `replay.js` is the one that matters.

**Three traps worth knowing if this is ever needed again:**

1. A `Read` result is only a valid full-file base if it **starts at line 1 and
   passed no `offset`/`limit`**. Partial reads produce a perfectly
   valid-looking numbered listing of the *middle* of a file — treating one as
   a whole file silently truncated a 900-line `main.py` to 5KB.
2. `Write`/`Edit` calls are captured verbatim, but **shell-applied patches are
   not recorded as file content**. A snapshot-based recovery therefore lands
   mid-session: it compiles, it mostly passes, and it is missing a day's work.
   Only replaying the shell commands too reaches the real head.
3. **Subagent edits never appear in the parent transcript at all.**

## Repaired by hand afterwards

The replay reached ~90%. These were the gaps, all found by the test suite:

| file | what was missing |
| --- | --- |
| `marketdata/refresh.py` | the replay faithfully reproduced a *real* bug from the session (a bulk `sed` injected `main()`'s handler into `refresh_actions`); the manual repair that followed was an `Edit` that no longer matched. Also: `date` and `parse_bhavcopy` imports, the `ValueError`→`CalendarError` wrap, the failure counter, `_REVIEW_WORTHY`, and the whole `bhavcopy` CLI subcommand |
| `marketdata/quality.py` | `groupby(dropna=False)` and the per-symbol `actions` filtering — both halves of the F1 fix |
| `api/main.py` | worst hit (9 failed edits): missing imports, `_require_snapshot`, `_actions_for`, `_action_caveat`, `_stage1_for`, the whole `/scanner/stage0` route, and a plan section that had reverted to its pre-extraction state with `DailyPlan` defined inline. Also the `PositionIn`/`SizeRequest` `max_length` bounds and the `/calendar/{day}` path-leak hardening |
| `scanner/stage4.py` | `_rejection_counts()` — made by a review subagent, so never in the transcript |
| `pyproject.toml` | `httpx2` was required by `fastapi.testclient` but never declared, so a clean rebuild could not even *collect* the API tests |

## Verified

- **346 tests passing** — the exact pre-loss count
- Fresh `.venv` on Python 3.14.3, all pinned dependencies installed
- Live end-to-end: Stage 0 gives **3,485 → 1,598** on real 2026-09-11 data,
  matching the documented figure exactly
- `desk/tests/fixtures/nse_bhavcopy_20260911.csv` re-fetched from NSE:
  394,927 bytes, 3,485 rows — identical to the original
- `configs/holidays_nse.json`, six `configs/bhavcopy/*.parquet` days and three
  `configs/corporate_actions/*.json` all regenerated from NSE

## Still to do

- **`git init`.** Not done — it is the user's call, but it is the one change
  that would have made all of the above a thirty-second `git restore`.
- `TradingAgents/` needs re-cloning (`agents/*/upstream/` came back on its own
  when OneDrive hydrated).
- `web/` scaffolding from `npm create vite` (`package.json`, `vite.config.ts`,
  `index.html`, `src/main.tsx`). All eight files under `web/src/` that we
  wrote are restored.
- A backup copy of the reconstruction is at `C:\Users\TUSHAR\Trading-RESTORED`.
  Safe to delete once you are satisfied, but no rush.
