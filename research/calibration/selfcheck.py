"""Does the MODULE's hurst() hit its own advertised false-positive rate?

The table was built with a standalone estimator on cumsum(normal). The module
takes PRICES and logs them. If those differ at all the p-values are wrong, and
nothing else in the module would reveal it.
"""
import numpy as np
from desk.research.stationarity import hurst

rng = np.random.default_rng(4242)
print(f"{'n':>6}{'paths':>7}{'P(p<0.05)':>11}{'P(p<0.10)':>11}"
      f"{'P(p<0.25)':>11}{'mean h':>9}{'  verdict MEAN-REVERTING':>26}")
for n in (120, 250, 500, 1250):
    ps, hs, mr = [], [], 0
    N = 1200
    for _ in range(N):
        # A GEOMETRIC random walk: real prices, so log(prices) is the walk.
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        e = hurst(px)
        if e.h is None: continue
        ps.append(e.pvalue); hs.append(e.h)
        mr += e.mean_reverting
    ps, hs = np.array(ps), np.array(hs)
    print(f"{n:>6}{len(ps):>7}{(ps<0.05).mean():>11.3f}{(ps<0.10).mean():>11.3f}"
          f"{(ps<0.25).mean():>11.3f}{hs.mean():>9.3f}{mr/len(ps):>26.3f}")

print("\nand the thing it must catch: a genuinely mean-reverting price series")
for phi in (0.99, 0.97, 0.94, 0.85):
    det = 0; N = 400; hh = []
    for _ in range(N):
        s, out = 0.0, []
        for _ in range(250):
            s = phi * s + rng.normal(0, 0.015); out.append(s)
        px = 100 * np.exp(np.array(out))
        e = hurst(px)
        if e.h is None: continue
        hh.append(e.h); det += e.mean_reverting
    print(f"  AR(1) phi={phi}: mean h {np.mean(hh):.3f}  "
          f"detected mean-reverting {det/N:.2%}")

print("\nand what upstream's fixed threshold would have said on the walks:")
for n in (250, 1250):
    up = {"mean-reverting": 0, "trending": 0, "random walk": 0}
    for _ in range(600):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        up[hurst(px).upstream_verdict] += 1
    tot = sum(up.values())
    print(f"  n={n}: " + "  ".join(f"{k} {v/tot:.2%}" for k, v in up.items()))
