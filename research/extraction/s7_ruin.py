"""What does +0.043R per trade actually feel like to trade?

The surviving cell - donchian 12% stop, 20-session hold - passed the permutation
test and still only earns about +0.043R a trade with t +0.84. Expectancy says
nothing about whether a person can sit through the path, and the path is what
decides whether this ever gets traded.

Three questions, on the REAL trade sequence rather than on a fitted
distribution:
  1. How deep does the worst drawdown get?
  2. How likely is it to be deep enough to make someone stop?
  3. How many trades before the edge is distinguishable from zero?

Also prints the gap between a block bootstrap and the IID one the borrowed
monte_carlo module implements, because that gap is the argument for not having
imported it.
"""
import sys
import statistics as st
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from desk.backtest.permutation import outcome_lattice               # noqa: E402
from desk.backtest.ruin import (                                    # noqa: E402
    compare_iid, simulate_paths, trades_to_significance,
)
from desk.scanner.stage0 import run_stage0                          # noqa: E402
from desk.store.bars import BarStore                                # noqa: E402
from desk.strategies.presets import preset_rules                    # noqa: E402

AS_OF = date(2026, 9, 29)
STATUTORY, SLIP = 0.1222, 0.300          # pessimistic: the flat 15bps guess
STOP, HOLD = 12.0, 20
COST_R = (STATUTORY + SLIP) / STOP

store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(AS_OF), min_price=50.0,
                min_turnover_lacs=500.0)
hist = store.history(as_of=AS_OF, start=date(2023, 9, 1),
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}, cost {COST_R:.4f}R per trade "
      f"(pessimistic slippage)\n", flush=True)

r_lat, ex = outcome_lattice(panel["high"].to_numpy(float),
                            panel["low"].to_numpy(float),
                            close.to_numpy(float),
                            stop_pct=STOP, rr=2.0, hold=HOLD)

# The real trade sequence, IN TIME ORDER, non-overlapping per symbol. Order
# matters here in a way it did not for the permutation test: the block
# bootstrap resamples contiguous runs, so the sequence has to be the sequence.
rules = preset_rules()
for pid in ("donchian_breakout", "bollinger_breakout"):
    sig = rules[pid].entries(panel).to_numpy(bool)
    trades = []
    for c in range(sig.shape[1]):
        busy = -1
        col = sig[:, c]
        for i in range(sig.shape[0] - 1):
            if col[i] and i > busy:
                v = r_lat[i, c]
                if np.isfinite(v):
                    trades.append((i, v - COST_R))
                    busy = ex[i, c]
    trades.sort(key=lambda t: t[0])
    rs = [v for _i, v in trades]

    mean_r, sd_r = st.fmean(rs), st.stdev(rs)
    print("=" * 78)
    print(f"{pid}  stop {STOP:g}%  hold {HOLD}  ({len(rs)} trades)")
    print("=" * 78)
    print(f"  mean {mean_r:+.4f}R   sd {sd_r:.4f}R   "
          f"median {st.median(rs):+.4f}R")
    n_need = trades_to_significance(mean_r, sd_r)
    print(f"  trades needed for t=2: {n_need:,.0f}"
          + (f"  (~{n_need / 400:.0f} years at 400 trades/yr)"
             if np.isfinite(n_need) else ""))
    print()

    for risk in (0.5, 1.0, 2.0):
        rep = simulate_paths(rs, n_trades=200, n_paths=3000,
                             risk_per_trade_pct=risk, block=20, seed=7)
        print(f"  --- {risk:g}% risk per trade, 200 trades (~6 months) ---")
        for line in str(rep).splitlines()[1:]:
            print(f"  {line}")
        print()

    gap = compare_iid(rs, n_trades=200, n_paths=3000,
                      risk_per_trade_pct=1.0, block=20, seed=7)
    print(f"  block bootstrap p95 drawdown : "
          f"{gap['block_p95_dd_pct']:.1f}%")
    print(f"  IID bootstrap  p95 drawdown : "
          f"{gap['iid_p95_dd_pct']:.1f}%   <- what the borrowed module reports")
    print(f"  the IID model is kinder by {-gap['understatement_pp']:.1f} "
          f"percentage points")
    print(f"  P(DD>20%): block {100 * gap['block_prob_dd_over_20']:.1f}%  "
          f"vs IID {100 * gap['iid_prob_dd_over_20']:.1f}%")
    print()
