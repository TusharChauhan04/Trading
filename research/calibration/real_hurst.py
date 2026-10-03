"""Is anything in the real universe detectably mean-reverting?

The null says 5% of random walks pass at p<0.05. If the universe gives ~5%,
there is nothing here. Anything well above 5% is a real finding - and the
binomial tells us which.
"""
import math, os
import numpy as np, pandas as pd
os.environ.setdefault("DESK_CONFIG_DIR", r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
from pathlib import Path
from desk.store.bars import BarStore
from desk.scanner.stage0 import run_stage0
from desk.research.stationarity import hurst

C = Path(r"C:\Users\TUSHAR\OneDrive\Desktop\Trading\configs")
store = BarStore(C / "bhavcopy")
days = sorted(store.available_days()); target = days[-1]
s0 = run_stage0(store.load_day(target), min_price=20.0, min_turnover_lacs=100.0)
hist = store.history(as_of=target, lookback=1240,
                     symbols=s0.survivors["symbol"].tolist(), columns=["close"])
closes = hist.wide_many(["close"])["close"]
print(f"as_of {target}, panel {closes.shape[0]} x {closes.shape[1]}")

def binom_tail(k, n, p):
    """P(X >= k), normal approximation with continuity correction.
    n=1490, p=0.05 gives mean 74.5 sd 8.4 - comfortably normal."""
    mu = n * p; sd = math.sqrt(n * p * (1 - p))
    z = (k - 0.5 - mu) / sd
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))

for win in (250, 1240):
    rows = []
    for sym in closes.columns:
        s = closes[sym].dropna().tail(win)
        e = hurst(s)
        if e.h is None: continue
        rows.append((sym, e.h, e.pvalue, e.n_obs, e.upstream_verdict))
    df = pd.DataFrame(rows, columns=["symbol","h","p","n","upstream"])
    n = len(df); k = int((df["p"] < 0.05).sum())
    print(f"\nwindow {win}: {n} names estimable, mean h {df['h'].mean():.3f}")
    z = (k - n*0.05) / math.sqrt(n*0.05*0.95)
    print(f"  p<0.05: {k}/{n} = {k/n:.3%}   (null 5.000%, i.e. {n*0.05:.0f} "
          f"names)   z={z:+.2f}   P(>= that | all random walks) = "
          f"{binom_tail(k, n, 0.05):.4f}")
    up = df["upstream"].value_counts()
    print(f"  upstream's fixed threshold would say: "
          + "  ".join(f"{i} {v} ({v/n:.1%})" for i, v in up.items()))
    if k:
        best = df.nsmallest(8, "p")
        print("  most mean-reverting:")
        for _, r in best.iterrows():
            print(f"    {r['symbol']:<14} h={r['h']:.3f} p={r['p']:.4f} n={r['n']}")
