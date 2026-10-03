"""Does shrinking the correlation matrix change the CLUSTERS?

235 observations against 1,398 names is badly under-determined, so sample
correlations are noisy and shrinkage is the textbook fix. Ledoit-Wolf's
scaled-identity target scales every off-diagonal by the same factor. A
monotone transform of distances cannot reorder a dendrogram's merges, but
AVERAGE linkage averages distances, and averaging does not commute with a
nonlinear transform - so this needs measuring, not arguing.

No sklearn (absent, and installing it would pull the pinned joblib).
"""
import collections
import os

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

os.environ.setdefault("DESK_CONFIG_DIR",
                      r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
from pathlib import Path

from desk.scanner.stage0 import run_stage0
from desk.store.bars import BarStore

C = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
store = BarStore(C / "bhavcopy")
d = sorted(store.available_days())[-1]
s0 = run_stage0(store.load_day(d))
surv = s0.survivors.nlargest(300, "turnover_lacs")
h = store.history(as_of=d, lookback=250, symbols=surv["symbol"].tolist(),
                  columns=["close"])
cl = h.wide_many(["close"])["close"]
r = cl.pct_change(fill_method=None)
cnt = r.notna().sum()
need = int(0.98 * (len(r) - 1))
r = r[[c for c in r.columns if int(cnt[c]) >= need]].dropna()
print(f"{r.shape[0]} obs x {r.shape[1]} names  "
      f"(obs/names = {r.shape[0]/r.shape[1]:.2f})")


def clusters_from(corr, thr):
    dd = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    dd = (dd + dd.T) / 2
    np.fill_diagonal(dd, 0)
    link = sch.linkage(ssd.squareform(dd, checks=False), method="average")
    return sch.fcluster(link, t=float(np.sqrt((1 - thr) / 2)),
                        criterion="distance")


def shrink(corr, delta):
    out = (1.0 - delta) * corr.copy()
    np.fill_diagonal(out, 1.0)
    return out


def ari(a, b):
    """Adjusted Rand index, hand-rolled."""
    ct = collections.Counter(zip(a, b))
    ra = collections.Counter(a)
    rb = collections.Counter(b)

    def c2(k):
        return k * (k - 1) / 2

    sij = sum(c2(v) for v in ct.values())
    sa = sum(c2(v) for v in ra.values())
    sb = sum(c2(v) for v in rb.values())
    n = c2(len(a))
    exp = sa * sb / n
    mx = (sa + sb) / 2
    return (sij - exp) / (mx - exp) if mx != exp else 1.0


S = r.corr().to_numpy(float)
iu = np.triu_indices_from(S, k=1)
print(f"sample off-diagonal mean {S[iu].mean():.4f}")

for delta in (0.1, 0.3, 0.5):
    L = shrink(S, delta)
    print(f"\ndelta={delta}  off-diagonal mean {L[iu].mean():.4f}")
    for thr in (0.4, 0.5, 0.6):
        a = clusters_from(S, thr)
        b = clusters_from(L, thr)
        matched = 1.0 - (1.0 - thr) * (1.0 - delta)
        bm = clusters_from(L, matched)
        print(f"   thr {thr}: sample {len(set(a)):>3} clusters | "
              f"same-thr {len(set(b)):>3} ARI {ari(a, b):.4f} | "
              f"matched-thr({matched:.3f}) {len(set(bm)):>3} "
              f"ARI {ari(a, bm):.4f}")
