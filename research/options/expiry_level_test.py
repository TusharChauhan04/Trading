"""The options result, tested at the unit where observations are independent.

WHY THIS EXISTS. `spread_expectancy.py` reports t = +8.72 on 48,943 spreads.
THAT T-STATISTIC IS MEANINGLESS AND REPORTING IT WOULD BE THE PROJECT'S OLDEST
MISTAKE IN NEW CLOTHES. On any one expiry date every symbol settles against the
same market move, so the 48,943 trades are not 48,943 observations - they are
roughly 50, one per expiry, each containing hundreds of highly correlated bets.
Treating correlated draws as independent is exactly what inflated the pairs
screen until Benjamini-Hochberg was applied.

So: collapse to one number per expiry, then test on those.

AND THE DIRECTIONAL CONFOUND IS THE REAL STORY. Over this period the
equal-weighted index fell 6.5%. Bull call spreads returned -0.1023R and bear
put spreads +0.1967R. That is not two findings, it is one: the market went
down. A combined book of both sides nets +0.0469R, but that is the average of a
losing directional bet and a winning one, and which is which was decided by the
regime rather than by any signal.

The question this file asks is therefore narrow and honest: AFTER removing
direction, is anything left? Three tests:

  1. Per-expiry mean, both sides combined - the honest t.
  2. The same, regressed against the market's own move over the holding window.
     If the edge is direction, the intercept goes to zero.
  3. Call-side and put-side per-expiry means separately, to show how completely
     the two are mirror images.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("research/options")))
from spread_expectancy import load_days, settlement, spreads_on  # noqa: E402


def build() -> pd.DataFrame:
    days = load_days()
    dates = sorted(days)
    rows = []
    for d in dates:
        for side in ("CE", "PE"):
            for s in spreads_on(days[d], d, side=side):
                exp = s["expiry"]
                if exp not in days:
                    continue
                S = settlement(days[exp], s["symbol"])
                if S is None:
                    continue
                payoff = (max(0.0, min(S - s["K1"], s["width"]))
                          if side == "CE"
                          else max(0.0, min(s["K2"] - S, s["width"])))
                s["r"] = (payoff - s["premium"]) / s["premium"]
                s["move_pct"] = (S - s["spot"]) / s["spot"] * 100.0
                rows.append(s)
    return pd.DataFrame(rows)


def t_stat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def main() -> None:
    t = build()
    print(f"{len(t):,} spreads, {t['expiry'].nunique()} expiries, "
          f"{t['symbol'].nunique()} underlyings\n")

    per = t.groupby("expiry").agg(
        n=("r", "size"), mean_r=("r", "mean"),
        move=("move_pct", "mean")).reset_index()
    per = per[per["n"] >= 30]

    print("THE HONEST TEST - one observation per expiry, both sides combined")
    print(f"  expiries used (n>=30 spreads): {len(per)}")
    print(f"  mean of per-expiry means: {per['mean_r'].mean():+.4f}R")
    print(f"  sd across expiries:       {per['mean_r'].std(ddof=1):.4f}")
    print(f"  t on {len(per)} independent observations: "
          f"{t_stat(per['mean_r'].to_numpy()):+.2f}")
    print(f"  expiries positive: {(per['mean_r'] > 0).sum()}/{len(per)}")
    print(f"  (the per-TRADE t was +8.72 on 48,943 correlated draws)")

    print("\nIS IT JUST DIRECTION? per-expiry return against the market's move")
    x = per["move"].to_numpy(float)
    y = per["mean_r"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) > 2:
        beta, alpha = np.polyfit(x, y, 1)
        pred = alpha + beta * x
        resid = y - pred
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        se_a = float(np.sqrt(ss_res / (len(x) - 2) *
                             (1 / len(x) + x.mean() ** 2 /
                              ((x - x.mean()) ** 2).sum())))
        print(f"  slope (sensitivity to the market move): {beta:+.4f}")
        print(f"  R^2: {r2:.3f}  <- how much of the result direction explains")
        print(f"  ALPHA (what is left after direction): {alpha:+.4f}R  "
              f"t {alpha / se_a if se_a else float('nan'):+.2f}")

    print("\nTHE TWO SIDES, per expiry - mirror images by construction")
    for side, name in (("CE", "bull call"), ("PE", "bear put")):
        sub = t[t["side"] == side].groupby("expiry")["r"].mean()
        print(f"  {name:<10} mean {sub.mean():+.4f}R  "
              f"t {t_stat(sub.to_numpy()):+.2f}  "
              f"positive {int((sub > 0).sum())}/{len(sub)}")

    print("\nPER-EXPIRY DETAIL")
    print(f"  {'expiry':<12}{'n':>6}{'mean R':>10}{'mkt move':>10}")
    for _, r in per.sort_values("expiry").iterrows():
        print(f"  {str(r['expiry'].date()):<12}{int(r['n']):>6}"
              f"{r['mean_r']:>+10.4f}{r['move']:>+9.2f}%")


if __name__ == "__main__":
    main()
