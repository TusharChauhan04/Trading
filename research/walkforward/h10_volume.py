"""H10: does VOLUME CONFIRMATION improve the surviving breakout cell?

WHERE THE HYPOTHESIS COMES FROM, which matters because it is not mine.
OpenTerminalUI's breakout_engine/detectors.py makes volume a REQUIRED condition
for a breakout to trigger:

    triggered = bool(crossed and volume_ratio >= min_ratio and confidence >= 0.15)

with min_volume_ratio defaulting to 1.2 and forced to >= 1.6 when
require_volume_spike is set. volume_ratio is the signal bar's volume over the
mean of the preceding `lookback` bars - and that module computes its channel
from candles[-(lookback+1):-1], EXCLUDING the current bar, which is the
off-by-one three other places in this project got wrong. It is careful code,
so its one structural difference from our donchian rule is worth testing.

Our cell has no volume condition at all. Baseline, 8 disjoint windows over
1,241 sessions: pooled +0.0446R, positive 5/8, timing-significant 5/8,
DSR 0.0910.

THE PRE-REGISTERED ACCEPTANCE TEST, fixed before the run: the count of windows
that are BOTH POSITIVE AND TIMING-SIGNIFICANT must rise. Not pooled net R -
that rises by construction whenever trades are deleted, and a volume filter
deletes trades. This is the same criterion H9's trend gate failed, and the
reason it reads as a clean negative rather than an ambiguous one.

Reported as the intersection per window rather than two separate counts,
because "5 positive and 5 timing-significant" does not by itself say they are
the same five.

POINT-IN-TIME: the ratio divides the signal bar's volume by the mean of the 20
bars BEFORE it. The signal is evaluated at that bar's close, where its own
volume is known - the same standing as using its close.
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
LOOKBACK = 20
# 41 after H9, plus the six variants tried here (3 thresholds x 2 rules).
NUM_TRIALS = 47

store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(AS_OF), min_price=50.0, min_turnover_lacs=500.0)
hist = store.history(as_of=AS_OF, start=date(2021, 9, 1),
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}", flush=True)

prior_high = panel["high"].shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).max()
excess = (close / prior_high - 1.0).to_numpy(float)

vol = panel["volume"]
prior_vol = vol.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).mean()
vol_ratio = (vol / prior_vol).to_numpy(float)
import numpy as np

finite = np.isfinite(vol_ratio)
print(f"volume ratio: median {np.nanmedian(vol_ratio[finite]):.3f}  "
      f"share >=1.2 {np.nanmean(vol_ratio[finite] >= 1.2):.3f}  "
      f">=1.6 {np.nanmean(vol_ratio[finite] >= 1.6):.3f}  "
      f">=2.0 {np.nanmean(vol_ratio[finite] >= 2.0):.3f}\n", flush=True)


def report(wf, name, pid):
    rows = wf.scored
    n = len(rows)
    both = sum(1 for w in rows if w.positive and w.timed)
    print(f"{pid:<20}{name:<16}{wf.pooled_net_r:+.4f}R  "
          f"trades {sum(w.trades for w in rows):>5}  "
          f"positive {wf.windows_positive}/{n}  "
          f"timing {wf.windows_timed}/{n}  "
          f"BOTH {both}/{n}  "
          f"DSR {wf.robustness().dsr:.4f}", flush=True)
    return both, n


for pid in ("donchian_breakout", "bollinger_breakout"):
    base = preset_rules()[pid].entries(panel).to_numpy(bool)
    buffered = base & (excess >= BUF / 100.0)
    variants = []
    if pid == "bollinger_breakout":
        variants.append(("ungated", buffered))
    thresholds = (1.6, 2.0) if pid == "donchian_breakout" else (1.2, 1.6, 2.0)
    for thr in thresholds:
        variants.append((f"vol>={thr}", buffered & (vol_ratio >= thr)))
    for name, sig in variants:
        wf = walk_forward_preset(
            panel, sig, label=f"{pid} [{name}]", stop_pct=STOP,
            hold_bars=HOLD, rr=2.0, buffer_pct=BUF, cost_r=COST_R,
            windows=8, num_trials=NUM_TRIALS, n_permutations=200)
        report(wf, name, pid)
    print(flush=True)
