"""H11: does the surviving cell hold up through the COVID crash?

The cell is donchian 12% stop / 24% target / 20 sessions / 1% buffer. On
2021-09 to 2026-09 (1,241 sessions, 8 disjoint windows) it was pooled +0.0446R,
positive in 5 of 8, timing-significant in 5 of 8, DSR 0.08. Every one of those
sessions came AFTER the COVID crash, in a period that was mostly a bull market.

The store now reaches 2019-10-01 - 1,715 sessions - which adds the pre-COVID
market, the March 2020 crash and the recovery. The cell has never been tested
through a crash and this is the only cheap test left that could change its
verdict.

NUM_TRIALS STAYS AT 47. This is not a new search: the cell is pre-specified and
the only thing that changed is the sample. Incrementing the trial count for
re-testing one fixed cell on more data would penalise the honest thing to do.

WINDOW COUNT IS REPORTED TWICE on purpose. The project has already been bitten
by this: on the same five years, 4 windows gave donchian 1 of 4
timing-significant while 8 gave 5 of 8. Too few slices hides intermittency, too
many starve each slice of trades. 8 windows keeps the count comparable to the
previous run (with longer windows); 11 keeps the window LENGTH comparable
(~155 sessions) so the two runs measure the same thing at the same resolution.

THE ACCEPTANCE TEST, pre-registered and unchanged: the count of windows BOTH
positive AND timing-significant. Pooled net R rises whenever losing periods are
excluded and falls whenever they are added, so it cannot be the criterion - and
this run ADDS a crash, which will drag the pooled figure down whatever the cell
is worth.
"""
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))

from desk.backtest.preset_wf import walk_forward_preset
from desk.scanner.stage0 import run_stage0
from desk.store.bars import BarStore
from desk.strategies.presets import preset_rules

AS_OF = date(2026, 9, 29)
STOP, HOLD, BUF = 12.0, 20, 1.0
COST_R = (0.1222 + 0.300) / STOP
NUM_TRIALS = 47

store = BarStore(Path("configs/bhavcopy"))
days = sorted(store.available_days())
print(f"store: {len(days)} sessions, {days[0]} -> {days[-1]}", flush=True)

s0 = run_stage0(store.load_day(AS_OF), min_price=50.0, min_turnover_lacs=500.0)
hist = store.history(as_of=AS_OF, start=days[0],
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}", flush=True)

prior_high = panel["high"].shift(1).rolling(20, min_periods=20).max()
excess = (close / prior_high - 1.0).to_numpy(float)

for pid in ("donchian_breakout", "bollinger_breakout"):
    base = preset_rules()[pid].entries(panel).to_numpy(bool)
    sig = base & (excess >= BUF / 100.0)
    for nwin in (8, 11):
        wf = walk_forward_preset(
            panel, sig, label=f"{pid} [{nwin}w]", stop_pct=STOP,
            hold_bars=HOLD, rr=2.0, buffer_pct=BUF, cost_r=COST_R,
            windows=nwin, num_trials=NUM_TRIALS, n_permutations=200)
        rows = wf.scored
        n = len(rows)
        both = sum(1 for w in rows if w.positive and w.timed)
        print(f"\n{pid}  {nwin} windows  ({n} scored)", flush=True)
        print(f"  pooled {wf.pooled_net_r:+.4f}R   positive "
              f"{wf.windows_positive}/{n}   timing {wf.windows_timed}/{n}   "
              f"BOTH {both}/{n}   DSR {wf.robustness().dsr:.4f}", flush=True)
        for w in rows:
            flag = ""
            if w.start <= "2020-06-30" and w.end >= "2020-02-01":
                flag = "  <-- CONTAINS THE COVID CRASH"
            nr = "n/a" if w.net_r is None else f"{w.net_r:+.4f}"
            pp = "n/a" if w.perm_p is None else f"{w.perm_p:.4f}"
            print(f"    {w.start} .. {w.end}  {w.trades:>5} trades  "
                  f"net {nr}  perm p {pp}{flag}", flush=True)
