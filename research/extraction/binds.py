"""Does the correlated cap actually bind, or is it inert for a different reason?

A gate that can never fire is no better wired than unwired. Two questions:
1. Do the clusters look like anything a human would recognise?
2. On a realistic shortlist, do two candidates ever land in one cluster?
"""
import os, numpy as np, pandas as pd
os.environ.setdefault("DESK_CONFIG_DIR", r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
from pathlib import Path
from desk.store.bars import BarStore
from desk.scanner.stage0 import run_stage0
from desk.marketdata.sectors import SectorMap
from desk.risk.clusters import cluster_map

C = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
store = BarStore(C / "bhavcopy"); days = sorted(store.available_days())
target = days[-1]
s0 = run_stage0(store.load_day(target), min_price=20.0, min_turnover_lacs=100.0)
sec = SectorMap.load(C / "sectors.json")

# The whole liquid universe, so clusters are found against a real cross-section
# rather than within a shortlist of 8 (which would find nothing by construction).
surv = s0.survivors.nlargest(400, "turnover_lacs")
hist = store.history(as_of=target, lookback=250,
                     symbols=surv["symbol"].tolist(), columns=["close"])
closes = hist.wide_many(["close"])["close"]

for thr in (0.4, 0.5, 0.6):
    cm = cluster_map(closes, as_of=target, threshold=thr)
    multi = cm.multi_name_clusters
    print(f"\nthreshold {thr}: {len(cm.labels)} placed, "
          f"{len(cm.unclustered)} unclustered, {cm.n_obs} obs")
    print(f"  {len(multi)} clusters with >1 name, largest {cm.largest}")
    if thr == 0.5:
        inv = {}
        for s, l in cm.labels.items(): inv.setdefault(l, []).append(s)
        big = sorted(((l, v) for l, v in inv.items() if len(v) > 1),
                     key=lambda x: -len(x[1]))[:8]
        print("  biggest clusters and their industries:")
        for lab, syms in big:
            inds = {}
            for s in syms:
                i = sec.sector_for(s) or "?"
                inds[i] = inds.get(i, 0) + 1
            top = ", ".join(f"{k} x{v}" for k, v in
                            sorted(inds.items(), key=lambda x: -x[1])[:3])
            print(f"    {lab:<24} n={len(syms):<3} {top}")
            print(f"      {', '.join(sorted(s.replace('.NS','') for s in syms)[:10])}")

# 2. Would two shortlist names ever collide? Simulate many shortlists by
#    taking the top-8 by turnover from each of 60 past days.
cm = cluster_map(closes, as_of=target, threshold=0.5)
collide, total = 0, 0
for d in days[-60:]:
    try:
        day0 = run_stage0(store.load_day(d), min_price=20.0,
                          min_turnover_lacs=100.0)
    except Exception:
        continue
    picks = day0.survivors.nlargest(8, "turnover_lacs")["symbol"].tolist()
    labs = [cm.group_for(p) for p in picks]
    labs = [l for l in labs if l]
    total += 1
    if len(labs) != len(set(labs)):
        collide += 1
print(f"\nshortlist collisions: {collide}/{total} days had two of the top-8 "
      f"in one cluster  ({collide/max(total,1):.1%})")
