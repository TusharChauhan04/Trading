"""Same screen, full-power window. Is 'no pairs' power or truth?"""
import os, numpy as np, pandas as pd
os.environ.setdefault("DESK_CONFIG_DIR", r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
from pathlib import Path
from desk.store.bars import BarStore
from desk.scanner.stage0 import run_stage0
from desk.marketdata.sectors import SectorMap
from desk.research.cointegration import screen_pairs, COINT_TEST_WINDOW

C = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
store = BarStore(C / "bhavcopy")
days = sorted(store.available_days())
target = days[-1]
print(f"as_of {target}, {len(days)} sessions on file")

s0 = run_stage0(store.load_day(target), min_price=20.0, min_turnover_lacs=100.0)
surv = s0.survivors
hist = store.history(as_of=target, lookback=min(1240, len(days)),
                     symbols=surv["symbol"].tolist(), columns=["close"])
closes = hist.wide_many(["close"])["close"]
print(f"panel {closes.shape[0]} bars x {closes.shape[1]} symbols")

sec = SectorMap.load(C / "sectors.json")
turn = dict(zip(surv["symbol"], surv["turnover_lacs"]))
buckets = {}
for sym in closes.columns:
    ind = sec.sector_for(sym)
    if ind: buckets.setdefault(ind, []).append(sym)

cands = []
for ind, syms in sorted(buckets.items()):
    top = sorted(syms, key=lambda s: -float(turn.get(s, 0.0)))[:6]
    for i in range(len(top)):
        for j in range(i+1, len(top)): cands.append((top[i], top[j]))
print(f"{len(cands)} candidate pairs\n")

for win in (300, 600, min(1200, closes.shape[0])):
    res = screen_pairs(closes, cands, as_of=target, window=win, fdr=0.10)
    pv = res.pairs["pvalue"].dropna().to_numpy(float)
    if len(pv) == 0:
        print(f"window {win}: nothing estimable"); continue
    # Under the null, p-values are UNIFORM. Compare to what chance gives.
    exp_min = 1.0 / (len(pv) + 1)
    frac05 = float((pv < 0.05).mean())
    print(f"window {win:>4}: tested {res.n_tested:>3}  passed(FDR10%) "
          f"{res.n_passed:>3}  min p {pv.min():.5f} (chance ~{exp_min:.5f})  "
          f"frac p<0.05 {frac05:.3f} (null 0.050)  median p {np.median(pv):.3f}")
    if res.n_passed:
        for _, r in res.pairs[res.pairs["passes"]].head(10).iterrows():
            print(f"        {r['y']:<13}{r['x']:<13} p={r['pvalue']:.5f} "
                  f"q={float(r['qvalue']):.4f} beta={r['beta']:.3f} "
                  f"hl={r['half_life']:.1f}d z={r['zscore']:+.2f}")
