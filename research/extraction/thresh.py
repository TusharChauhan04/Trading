"""What correlation level means "one bet" on THIS market?

A threshold has to come from the distribution it is cutting. Indian equities
move together a lot; a cut that is right for US large caps may put the whole
universe in one cluster here.
"""
import os, numpy as np, pandas as pd
os.environ.setdefault("DESK_CONFIG_DIR", r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
from pathlib import Path
import scipy.cluster.hierarchy as sch, scipy.spatial.distance as ssd
from desk.store.bars import BarStore
from desk.scanner.stage0 import run_stage0
from desk.marketdata.sectors import SectorMap

C = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
store = BarStore(C / "bhavcopy"); days = sorted(store.available_days())
target = days[-1]
s0 = run_stage0(store.load_day(target), min_price=20.0, min_turnover_lacs=100.0)
surv = s0.survivors.nlargest(150, "turnover_lacs")
hist = store.history(as_of=target, lookback=260, symbols=surv["symbol"].tolist(),
                     columns=["close"])
closes = hist.wide_many(["close"])["close"]
rets = closes.pct_change()
# Only names with near-full history, so no fillna is needed at all.
good = rets.columns[rets.notna().sum() >= 200]
rets = rets[good].dropna()
print(f"{rets.shape[0]} returns x {rets.shape[1]} names (full-history only)")

corr = rets.corr()
iu = np.triu_indices_from(corr, k=1)
pw = corr.to_numpy()[iu]
print(f"\npairwise correlation: mean {pw.mean():.3f}  median {np.median(pw):.3f}")
for q in (50, 75, 90, 95, 99, 99.9):
    print(f"  p{q:<5} {np.percentile(pw, q):.3f}")
print(f"  max   {pw.max():.3f}")

sec = SectorMap.load(C / "sectors.json")
same, diff = [], []
cols = list(corr.columns)
for a in range(len(cols)):
    for b in range(a+1, len(cols)):
        sa, sb = sec.sector_for(cols[a]), sec.sector_for(cols[b])
        if sa and sb:
            (same if sa == sb else diff).append(corr.iloc[a, b])
print(f"\nsame industry : n={len(same):>5} mean {np.mean(same):.3f}")
print(f"diff industry : n={len(diff):>5} mean {np.mean(diff):.3f}")
print(f"  -> industry explains {np.mean(same)-np.mean(diff):+.3f} of correlation")

d = np.sqrt(np.clip((1 - corr.to_numpy()) / 2, 0, 1)); np.fill_diagonal(d, 0)
link = sch.linkage(ssd.squareform(d, checks=False), method="average")
print(f"\ncut level -> clusters (average linkage), n={len(cols)}")
for rho in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    t = np.sqrt((1 - rho) / 2)
    lab = sch.fcluster(link, t=t, criterion="distance")
    sizes = pd.Series(lab).value_counts()
    singles = int((sizes == 1).sum())
    print(f"  rho>={rho}  dist<={t:.3f}  {sizes.size:>3} clusters  "
          f"largest {sizes.max():>3}  singletons {singles:>3}")
