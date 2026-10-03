"""Does a POINT-IN-TIME trend gate turn 5-of-8 into something better?

H8 found the three failing windows separate cleanly from the five working ones on
"equal-weighted index above its own 50-day average" - working windows >=43.2% of
days, failing <=42.6%.

TWO HONEST CAVEATS BEFORE THE TEST, because the finding is weaker than it looks.
  1. The margin is 0.6 percentage points on EIGHT observations. One window either
     way destroys the separation. The chance of a clean 5/3 split on one measure
     is 2/C(8,5) = 3.6%, and FOUR measures were tried, so seeing two clean splits
     is notable rather than conclusive.
  2. "A long-only breakout rule loses when the market falls" is close to a
     TAUTOLOGY, not a discovery about the market. The useful question is not
     whether the regime explains the failures - it nearly must - but whether
     gating on it, using only information available ON THE DAY, improves the
     per-trade edge rather than merely removing trades.

So the gate here is strictly point-in-time: the equal-weighted close against its
own trailing 50-day mean, both computed from bars at or before the signal bar.
No window statistic, no hindsight.

THE TEST THAT MATTERS is not whether net R rises. Removing losing periods raises
net R by construction. It is whether the number of windows that are BOTH positive
AND timing-significant goes up - because that is the thing the walk-forward
refused the rule for.
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
NUM_TRIALS = 41            # 39 + the two gate variants tried here

store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(AS_OF), min_price=50.0, min_turnover_lacs=500.0)
hist = store.history(as_of=AS_OF, start=date(2021, 9, 1),
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}\n", flush=True)

prior_high = panel["high"].shift(1).rolling(20, min_periods=20).max()
excess = (close / prior_high - 1.0).to_numpy(float)

# POINT-IN-TIME trend state: the equal-weighted close against its own trailing
# 50-day mean. Both use bars at or before the signal bar; nothing is shifted
# backwards and no window-level statistic is used.
ew = close.mean(axis=1)
trend_on = (ew > ew.rolling(50, min_periods=50).mean()).to_numpy(bool)
print(f"trend gate ON for {100*trend_on.mean():.1f}% of sessions\n", flush=True)

for pid in ("donchian_breakout", "bollinger_breakout"):
    base = preset_rules()[pid].entries(panel).to_numpy(bool)
    buffered = base & (excess >= BUF / 100.0)
    for name, sig in (("ungated", buffered),
                      ("trend-gated", buffered & trend_on[:, None])):
        wf = walk_forward_preset(
            panel, sig, label=f"{pid} [{name}]", stop_pct=STOP,
            hold_bars=HOLD, rr=2.0, buffer_pct=BUF, cost_r=COST_R,
            windows=8, num_trials=NUM_TRIALS, n_permutations=200)
        n = len(wf.scored)
        print(f"{pid:<20}{name:<14}{wf.pooled_net_r:+.4f}R  "
              f"positive {wf.windows_positive}/{n}  "
              f"timing {wf.windows_timed}/{n}  "
              f"DSR {wf.robustness().dsr:.4f}", flush=True)
