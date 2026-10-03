"""H6: walk the surviving cell through disjoint windows."""
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
# Cells tried to arrive here: 10 preset cells + 20 stop-width + 9 filter cells.
NUM_TRIALS = 39

store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(AS_OF), min_price=50.0, min_turnover_lacs=500.0)
hist = store.history(as_of=AS_OF, start=date(2023, 9, 1),
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}  cost {COST_R:.4f}R  trials {NUM_TRIALS}\n", flush=True)

prior_high = panel["high"].shift(1).rolling(20, min_periods=20).max()
excess = (close / prior_high - 1.0).to_numpy(float)

for pid in ("donchian_breakout", "bollinger_breakout"):
    base = preset_rules()[pid].entries(panel).to_numpy(bool)
    sig = base & (excess >= BUF / 100.0)
    wf = walk_forward_preset(panel, sig, label=f"{pid} {STOP:g}%/{HOLD}b/{BUF:g}%buf",
                             stop_pct=STOP, hold_bars=HOLD, rr=2.0,
                             buffer_pct=BUF, cost_r=COST_R, windows=4,
                             num_trials=NUM_TRIALS, n_permutations=200)
    print(wf.report(), flush=True)
    print()
