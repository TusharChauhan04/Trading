"""The random-walk null for the lagged-variance H, across length.

Upstream compares H to a fixed 0.45. The null's MEAN is 0.40-0.48 depending on
n, so the fixed threshold is inside the null. This builds the table the module
needs, and checks whether a normal approximation to the null is good enough to
turn into a p-value.
"""
import numpy as np

def hurst_lagvar(x, max_lag=None):
    n = len(x)
    if max_lag is None:
        max_lag = max(4, min(n // 4, 100))
    tau, keep = [], []
    for lag in range(2, max_lag + 1):
        d = x[lag:] - x[:-lag]
        s = d.std()
        if s > 0:
            tau.append(s); keep.append(lag)
    if len(keep) < 4:
        return None
    return float(np.polyfit(np.log(keep), np.log(tau), 1)[0])

rng = np.random.default_rng(20260102)
print(f"{'n':>6}{'mean':>8}{'sd':>7}{'p01':>8}{'p05':>8}{'p50':>8}"
      f"{'skew':>7}{'  normal-approx 5% vs empirical':>32}")
table = {}
for n in (60, 90, 120, 180, 250, 375, 500, 750, 1000, 1250):
    hs = np.array([h for h in (hurst_lagvar(np.cumsum(rng.normal(0, 1, n)))
                               for _ in range(4000)) if h is not None])
    m, s = hs.mean(), hs.std()
    p01, p05, p50 = np.percentile(hs, [1, 5, 50])
    skew = float(((hs - m) ** 3).mean() / s ** 3)
    # how close is the normal 5% quantile to the empirical one?
    approx = m - 1.645 * s
    print(f"{n:>6}{m:>8.4f}{s:>7.4f}{p01:>8.4f}{p05:>8.4f}{p50:>8.4f}"
          f"{skew:>7.2f}{approx:>20.4f} vs {p05:.4f}")
    table[n] = (round(float(m), 4), round(float(s), 4))
print("\nNULL = {")
for k, (m, s) in table.items():
    print(f"    {k}: ({m}, {s}),")
print("}")
# Where does the fixed 0.45 threshold sit inside this null?
print("\nupstream's 0.45 expressed as a null quantile:")
for n in (120, 250, 500, 1250):
    m, s = table[n]
    z = (0.45 - m) / s
    from math import erf, sqrt
    pct = 0.5 * (1 + erf(z / sqrt(2)))
    print(f"  n={n:>5}: 0.45 is {z:+.2f} sd from the null mean "
          f"-> {pct*100:.0f}% of random walks fall below it")
