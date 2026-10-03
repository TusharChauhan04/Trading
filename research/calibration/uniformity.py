"""Is the observed p-value distribution distinguishable from the null?"""
import math
def binom_tail(k, n, p):
    """P(X >= k) exactly, no scipy."""
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))
n = 278
for win, frac in ((300, 0.068), (600, 0.054), (1200, 0.065)):
    k = round(frac * n)
    print(f"window {win:>4}: {k:>2}/{n} pairs at p<0.05 "
          f"(null expects {0.05*n:.1f})  P(>= that many | no pair "
          f"cointegrated) = {binom_tail(k, n, 0.05):.3f}")
