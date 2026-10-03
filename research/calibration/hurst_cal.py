"""Calibrate the lagged-variance Hurst estimator on series with KNOWN H.

Upstream classifies H<0.45 mean-reverting, H>0.55 trending, with no error bar.
If the estimator's standard error at realistic lengths is bigger than 0.05,
that band is noise and the labels are coin flips. Measured, not assumed.
"""
import numpy as np

def hurst_lagvar(x, max_lag=None):
    """Upstream's method, lags scaled to length. x = LOG prices."""
    n = len(x)
    if max_lag is None:
        max_lag = max(4, min(n // 4, 100))
    lags = np.arange(2, max_lag + 1)
    tau = []
    keep = []
    for lag in lags:
        d = x[lag:] - x[:-lag]
        s = d.std()
        if s > 0:
            tau.append(s); keep.append(lag)
    if len(keep) < 4:
        return None
    return float(np.polyfit(np.log(keep), np.log(tau), 1)[0])

def rw(n, rng):                       # true H = 0.5
    return np.cumsum(rng.normal(0, 1, n))

def ar1(n, rng, phi):                 # mean-reverting, H < 0.5
    out, s = np.empty(n), 0.0
    for i in range(n):
        s = phi * s + rng.normal(0, 1); out[i] = s
    return out

def trend(n, rng, slope):             # drift added, H > 0.5
    return np.cumsum(rng.normal(slope, 1, n))

rng = np.random.default_rng(11)
print(f"{'series':<24}{'n':>6}{'mean H':>9}{'sd':>7}{'  P(labelled MR)':>17}"
      f"{'P(labelled TR)':>16}")
for name, gen in (("random walk (H=0.5)", lambda n, r: rw(n, r)),
                  ("AR(1) phi=0.94", lambda n, r: ar1(n, r, 0.94)),
                  ("AR(1) phi=0.70", lambda n, r: ar1(n, r, 0.70)),
                  ("walk + drift 0.05", lambda n, r: trend(n, r, 0.05)),
                  ("walk + drift 0.20", lambda n, r: trend(n, r, 0.20))):
    for n in (120, 250, 500, 1200):
        hs = []
        for _ in range(300):
            h = hurst_lagvar(gen(n, rng))
            if h is not None: hs.append(h)
        hs = np.array(hs)
        mr = float((hs < 0.45).mean()); tr = float((hs > 0.55).mean())
        print(f"{name:<24}{n:>6}{hs.mean():>9.3f}{hs.std():>7.3f}"
              f"{mr:>17.2f}{tr:>16.2f}")
    print()
