import numpy as np
def h_lv(x, max_lag=None):
    n = len(x)
    if max_lag is None: max_lag = max(4, min(n // 4, 100))
    tau, keep = [], []
    for lag in range(2, max_lag + 1):
        d = x[lag:] - x[:-lag]; s = d.std()
        if s > 0: tau.append(s); keep.append(lag)
    if len(keep) < 4: return None
    return float(np.polyfit(np.log(keep), np.log(tau), 1)[0])

QS = [0.1, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0]
rng = np.random.default_rng(20260102)
print("#: n: (mean, sd, " + ", ".join(f"p{q:g}" for q in QS) + ")")
print("_NULL: dict[int, tuple[float, ...]] = {")
for n in (120, 180, 250, 375, 500, 750, 1000, 1250, 1750, 2500):
    hs = np.array([v for v in (h_lv(np.cumsum(rng.normal(0, 1, n)))
                               for _ in range(20000)) if v is not None])
    qs = np.percentile(hs, QS)
    print(f"    {n:>5}: ({hs.mean():.4f}, {hs.std():.4f}, "
          + ", ".join(f"{v:.4f}" for v in qs) + "),")
print("}")
