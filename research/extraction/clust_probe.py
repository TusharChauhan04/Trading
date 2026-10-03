"""Two claims about upstream's clustering, measured on real data.

1. `ward` linkage on a correlation distance. Ward assumes EUCLIDEAN distances;
   sqrt((1-rho)/2) is a metric but the matrix from pairwise-deleted
   correlations need not be Euclidean. Does it actually violate it here?
2. corr().fillna(0) turns "unknown" into "uncorrelated" - the most favourable
   diversification assumption there is. How often is it NaN?
"""
import os, numpy as np, pandas as pd
os.environ.setdefault("DESK_CONFIG_DIR", r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
from pathlib import Path
import scipy.cluster.hierarchy as sch, scipy.spatial.distance as ssd
from desk.store.bars import BarStore
from desk.scanner.stage0 import run_stage0

C = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
store = BarStore(C / "bhavcopy")
days = sorted(store.available_days()); target = days[-1]
s0 = run_stage0(store.load_day(target), min_price=20.0, min_turnover_lacs=100.0)
# Take the most liquid 120 names - a realistic "shortlist universe".
surv = s0.survivors.nlargest(120, "turnover_lacs")
hist = store.history(as_of=target, lookback=260,
                     symbols=surv["symbol"].tolist(), columns=["close"])
closes = hist.wide_many(["close"])["close"]
rets = closes.pct_change().dropna(how="all")
print(f"panel {rets.shape[0]} returns x {rets.shape[1]} names")

corr_raw = rets.corr()
nan_frac = float(corr_raw.isna().to_numpy().mean())
print(f"\n2) NaN cells in corr(): {nan_frac:.4%}  "
      f"({int(corr_raw.isna().to_numpy().sum())} of {corr_raw.size})")
print(f"   upstream fillna(0) would call those pairs UNCORRELATED")

corr = corr_raw.fillna(0).to_numpy()
# PSD check
ev = np.linalg.eigvalsh((corr + corr.T) / 2)
print(f"   smallest eigenvalue of the filled matrix: {ev.min():+.6f} "
      f"({'NOT PSD' if ev.min() < -1e-9 else 'PSD'})")

d = np.sqrt(np.clip((1 - corr) / 2, 0, 1)); np.fill_diagonal(d, 0)
# Euclidean-embeddability: the Gram matrix of a Euclidean distance matrix is PSD
n = d.shape[0]; J = np.eye(n) - np.ones((n, n)) / n
G = -0.5 * J @ (d ** 2) @ J
gev = np.linalg.eigvalsh((G + G.T) / 2)
print(f"\n1) Euclidean embeddability of the distance matrix:")
print(f"   most negative Gram eigenvalue: {gev.min():+.6f}")
print(f"   {'NOT Euclidean -> ward is invalid here' if gev.min() < -1e-8 else 'Euclidean'}")
neg = int((gev < -1e-8).sum())
print(f"   {neg} of {n} eigenvalues negative")

cond = ssd.squareform(d, checks=False)
for m in ("single", "average", "complete", "ward"):
    link = sch.linkage(cond, method=m)
    # cophenetic correlation: how faithfully the tree reproduces the distances
    from scipy.cluster.hierarchy import cophenet
    c, _ = cophenet(link, cond)
    heights = link[:, 2]
    print(f"   {m:<9} cophenetic r={c:.4f}  max merge height={heights.max():.4f}"
          f"  {'(exceeds max distance 1.0 - inversion)' if heights.max() > 1.0 else ''}")
