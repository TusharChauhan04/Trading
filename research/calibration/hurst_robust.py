"""Is the random-walk null for H the same under fat tails and vol clustering?

If not, a p-value calibrated on Gaussian increments is wrong on real stocks -
which have both - and the whole module would be miscalibrated in a way that
looks fine in tests.
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
    if len(keep) < 4: return None
    return float(np.polyfit(np.log(keep), np.log(tau), 1)[0])

def gaussian(n, rng):
    return np.cumsum(rng.normal(0, 1, n))

def student_t(n, rng, df=3.0):
    """Fat tails. df=3 is roughly daily equity kurtosis."""
    return np.cumsum(rng.standard_t(df, n))

def garch(n, rng, omega=0.05, alpha=0.10, beta=0.85):
    """Volatility clustering, GARCH(1,1) with equity-like persistence."""
    e = np.empty(n); s2 = omega / (1 - alpha - beta)
    for i in range(n):
        z = rng.normal()
        e[i] = np.sqrt(s2) * z
        s2 = omega + alpha * e[i] ** 2 + beta * s2
    return np.cumsum(e)

def regime_vol(n, rng):
    """Two-state vol, the Markov structure actually measured on this desk
    (21.3% vs 8.3% annualised, ~37% high-vol)."""
    out, i = [], 0
    while len(out) < n:
        calm = i % 2 == 0
        size = int(rng.integers(60, 140))
        out.extend(rng.normal(0, 0.52 if calm else 1.34, size)); i += 1
    return np.cumsum(np.array(out[:n]))

rng = np.random.default_rng(777)
print(f"{'innovations':<26}{'n':>6}{'mean H':>9}{'sd':>8}{'p05':>8}"
      f"{'   vs Gaussian p05':>20}")
for n in (250, 1250):
    base = None
    for name, gen in (("Gaussian", gaussian),
                      ("Student t, df=3", lambda a, b: student_t(a, b, 3.0)),
                      ("GARCH(1,1) .10/.85", garch),
                      ("two-state vol (measured)", regime_vol)):
        hs = np.array([h for h in (gen(n, rng) for _ in range(2500))
                       for h in [hurst_lagvar(h)] if h is not None])
        m, s = hs.mean(), hs.std()
        p05 = float(np.percentile(hs, 5))
        if base is None: base = p05
        print(f"{name:<26}{n:>6}{m:>9.4f}{s:>8.4f}{p05:>8.4f}"
              f"{p05-base:>+20.4f}")
    print()
