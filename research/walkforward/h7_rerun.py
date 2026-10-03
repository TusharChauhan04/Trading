"""H7: re-run H6 on the deepened sample, with more windows and more cohorts.

Four windows and 36 monthly cohorts were the binding constraint on every verdict
- not the modelling. With roughly five years on disk instead of three there are
enough disjoint windows to ask whether window one's edge was a period or a fluke,
and enough monthly cohorts to let the deflated statistics mean something.

TWO THINGS CHANGE AND BOTH ARE REPORTED. The universe GREW over the period: NSE's
bhavcopy carried ~1,548 EQ symbols in Sep 2021, ~1,749 in Jun 2022 and ~2,200
now. So a symbol that listed in 2024 is simply NaN before it existed, which is
handled (no forward fill anywhere, min_periods on every window) but means the
early windows are genuinely narrower universes rather than the same one.

AND THE REGIME DIFFERS. 2021-2022 includes a sharp drawdown that 2023-2026 does
not. That is the POINT - a cell whose edge lived in one nine-month rally should
be tested against a period that was not one.
"""
import sys, warnings
from datetime import date
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from desk.backtest.preset_wf import walk_forward_preset
from desk.scanner.stage0 import run_stage0
from desk.store.bars import BarStore
from desk.strategies.presets import preset_rules

AS_OF = date(2026, 9, 29)
STOP, HOLD, BUF = 12.0, 20, 1.0
COST_R = (0.1222 + 0.300) / STOP
NUM_TRIALS = 39              # every cell tried to arrive here

store = BarStore(Path("configs/bhavcopy"))
have = sorted(p.stem for p in Path("configs/bhavcopy").glob("*.parquet"))
START = date.fromisoformat(have[0])
print(f"history on disk: {have[0]} -> {have[-1]}  ({len(have)} sessions)",
      flush=True)

s0 = run_stage0(store.load_day(AS_OF), min_price=50.0, min_turnover_lacs=500.0)
hist = store.history(as_of=AS_OF, start=START,
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}  {close.index[0]} -> {close.index[-1]}", flush=True)

# How much of the early panel actually has data - the universe grew.
cov = close.notna().sum(axis=1)
print(f"symbols with data: {int(cov.iloc[0])} on day one, "
      f"{int(cov.iloc[len(cov)//2])} midway, {int(cov.iloc[-1])} at the end\n",
      flush=True)

prior_high = panel["high"].shift(1).rolling(20, min_periods=20).max()
excess = (close / prior_high - 1.0).to_numpy(float)

for pid in ("donchian_breakout", "bollinger_breakout"):
    base = preset_rules()[pid].entries(panel).to_numpy(bool)
    sig = base & (excess >= BUF / 100.0)
    for nw in (4, 8):
        wf = walk_forward_preset(
            panel, sig, label=f"{pid} {STOP:g}%/{HOLD}b/{BUF:g}%buf [{nw}w]",
            stop_pct=STOP, hold_bars=HOLD, rr=2.0, buffer_pct=BUF,
            cost_r=COST_R, windows=nw, num_trials=NUM_TRIALS,
            n_permutations=200)
        print(wf.report(), flush=True)
        print()
