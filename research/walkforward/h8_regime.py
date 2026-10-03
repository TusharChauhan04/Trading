"""Do the failing windows share a REGIME? The desk already has a router.

H7 on five years and eight windows: five positive with timing significant at
p=0.0050, three negative with p=1.0000 - timing WORSE than shuffling. Same five
in both columns. That is not one lucky period; it is a rule that works in some
market conditions and actively fails in others.

If the failures cluster in a measurable regime, the desk's existing regime engine
can gate the rule - which is what it was built for and what the catalogue's
per-strategy eligibility already expresses. If they do not cluster, the rule is
just unstable and gating buys nothing.

Measured on the equal-weighted cross-section, not asserted: the same breadth the
regime engine reads.
"""
import sys, warnings, statistics as st
from datetime import date
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
from desk.scanner.stage0 import run_stage0
from desk.store.bars import BarStore

WINDOWS = [
    ("2021-09-01", "2022-04-19", -0.0647, 1.0000),
    ("2022-04-20", "2022-12-02", +0.0207, 0.0050),
    ("2022-12-05", "2023-07-19", +0.1331, 0.0050),
    ("2023-07-20", "2024-03-21", +0.3295, 0.0050),
    ("2024-03-22", "2024-11-07", +0.1303, 0.0050),
    ("2024-11-08", "2025-06-26", -0.1305, 1.0000),
    ("2025-06-27", "2026-02-10", -0.1234, 1.0000),
    ("2026-02-11", "2026-09-29", +0.0617, 0.0050),
]

store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(date(2026, 9, 29)), min_price=50.0,
                min_turnover_lacs=500.0)
hist = store.history(as_of=date(2026, 9, 29), start=date(2021, 9, 1),
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["close"])
close = hist.wide_many(["close"])["close"]
idx = pd.Index(close.index)

# The three breadth measures the regime engine reads, computed the same way.
ret = close.pct_change()
breadth_up = (ret > 0).sum(axis=1) / ret.notna().sum(axis=1)       # share advancing
ew = close.mean(axis=1)
ew_ret = ew.pct_change()
sma50 = ew.rolling(50, min_periods=50).mean()
above = (ew > sma50)                                               # index above its 50d
vol = ew_ret.rolling(20, min_periods=20).std() * np.sqrt(252) * 100

hdr = (f"{'window':<24}{'netR':>9}{'perm p':>8}  {'advance%':>9}"
       f"{'above50d%':>11}{'annvol%':>9}{'EW return%':>12}")
print(hdr); print("-" * len(hdr))
rows = []
for a, b, net, p in WINDOWS:
    m = (idx >= date.fromisoformat(a)) & (idx <= date.fromisoformat(b))
    adv = 100 * float(breadth_up[m].mean())
    ab = 100 * float(above[m].mean())
    vv = float(vol[m].mean())
    rr = 100 * (float(ew[m].iloc[-1]) / float(ew[m].iloc[0]) - 1)
    good = p < 0.05
    print(f"{a + ' - ' + b[5:]:<24}{net:>+9.4f}{p:>8.4f}  {adv:>9.1f}"
          f"{ab:>11.1f}{vv:>9.1f}{rr:>+12.1f}   {'OK' if good else 'FAIL'}")
    rows.append({"good": good, "adv": adv, "above": ab, "vol": vv, "ret": rr})

print()
g = [r for r in rows if r["good"]]
bad = [r for r in rows if not r["good"]]
print(f"  {len(g)} working windows vs {len(bad)} failing")
print()
print(f"{'measure':<22}{'working':>10}{'failing':>10}{'separates?':>13}")
print("-" * 55)
for key, name in (("adv", "advancing %"), ("above", "EW above 50d %"),
                  ("vol", "annualised vol %"), ("ret", "EW return %")):
    gm, bm = st.fmean([r[key] for r in g]), st.fmean([r[key] for r in bad])
    gmin, gmax = min(r[key] for r in g), max(r[key] for r in g)
    bmin, bmax = min(r[key] for r in bad), max(r[key] for r in bad)
    clean = (gmin > bmax) or (bmin > gmax)
    print(f"{name:<22}{gm:>10.1f}{bm:>10.1f}{'CLEANLY' if clean else 'overlaps':>13}")
print()
print("  'CLEANLY' means the working and failing windows do not overlap at all")
print("  on that measure - which is what a gate needs. 'overlaps' means a")
print("  threshold would misclassify at least one window, and with 8 windows")
print("  a clean split could still be chance.")
