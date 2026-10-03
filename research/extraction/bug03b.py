"""BUG-03's size, measured in MONEY this time.

The first attempt measured |entry z| - |exit z|, which is nearly fixed by the
trading rule itself (enter at 2, exit at 0.5) and therefore cannot detect a bad
hedge ratio at all. It returned 0.86x and meant nothing. This measures the
actual return on the pair position.

P&L on a spread trade: long the spread means long 1 unit of y and short beta
units of x. The return, as a fraction of the gross notional deployed, is

    (dy - beta * dx) / (|y0| + |beta * x0|)

Costs are applied on BOTH legs at both ends - four executions per round trip -
which is why a pairs trade needs a bigger move than a single-name trade to
break even.
"""
import sys, warnings, statistics as st
from datetime import date
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from desk.research.cointegration import DEFAULT_WINDOW, hedge_ratio, walk_spread
from desk.scanner.stage0 import run_stage0
from desk.store.bars import BarStore

ROUND_TRIP_PER_LEG = 0.422          # the desk's own cost model, per leg
store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(date(2026, 9, 29)), min_price=50.0,
                min_turnover_lacs=2000.0)
hist = store.history(as_of=date(2026, 9, 29), start=date(2021, 9, 1),
                     symbols=s0.survivors["symbol"].tolist(), columns=["close"])
close = hist.wide_many(["close"])["close"].dropna(axis=1, thresh=1100)

PAIRS = [("HDFCBANK.NS","ICICIBANK.NS"),("TCS.NS","INFY.NS"),
         ("ONGC.NS","OIL.NS"),("SBIN.NS","BANKBARODA.NS"),
         ("RELIANCE.NS","BPCL.NS"),("COALINDIA.NS","NTPC.NS")]
PAIRS = [(a,b) for a,b in PAIRS if a in close.columns and b in close.columns]

def pnl(y, x, z, betas, entry=2.0, exit_=0.5):
    """Realised % return per round trip, net of 2 legs x 2 sides of cost."""
    pos, ey, ex, eb = 0, 0.0, 0.0, 0.0
    out = []
    for d, v in z.dropna().items():
        if d not in y.index:
            continue
        b = float(betas.loc[d]) if betas is not None else eb
        if pos == 0 and abs(v) >= entry:
            pos = -1 if v > 0 else 1            # short spread if z high
            ey, ex, eb = float(y.loc[d]), float(x.loc[d]), b
        elif pos != 0 and abs(v) <= exit_:
            dy, dx = float(y.loc[d]) - ey, float(x.loc[d]) - ex
            gross = pos * (dy - eb * dx)
            notional = abs(ey) + abs(eb * ex)
            if notional > 0:
                # Cost on both legs, in and out, as a share of notional.
                cost = 2.0 * ROUND_TRIP_PER_LEG / 100.0 * (
                    (abs(ey) + abs(eb * ex)) / notional)
                out.append(100.0 * (gross / notional) - 100.0 * cost)
            pos = 0
    return out

hdr = (f"{'pair':<28}{'n':>4}{'IN-SAMPLE %':>13}{'POINT-IN-TIME %':>17}"
       f"{'difference':>12}")
print(hdr); print("-" * len(hdr))
bug_all, pit_all = [], []
for a, b in PAIRS:
    y, x = close[a], close[b]
    hr = hedge_ratio(y, x)
    if hr is None: continue
    beta, alpha = hr

    # ARM 1 - BUG-03: one whole-sample beta applied to all history.
    sp = y - beta * x - alpha
    z_bug = (sp - sp.rolling(DEFAULT_WINDOW).mean()) / sp.rolling(DEFAULT_WINDOW).std()
    r_bug = pnl(y, x, z_bug, pd.Series(beta, index=y.index))

    # ARM 2 - trailing beta, re-estimated, nothing from the future.
    wf = walk_spread(y, x, window=DEFAULT_WINDOW, refit_every=21)
    if wf.empty: continue
    r_pit = pnl(y, x, wf["zscore"], wf["beta"])

    mb = st.fmean(r_bug) if r_bug else float("nan")
    mp = st.fmean(r_pit) if r_pit else float("nan")
    bug_all += r_bug; pit_all += r_pit
    print(f"{a[:12]+'/'+b[:12]:<28}{len(r_pit):>4}{mb:>+13.3f}{mp:>+17.3f}"
          f"{mb-mp:>+12.3f}", flush=True)

print()
print(f"  pooled, {len(bug_all)} in-sample trades vs {len(pit_all)} point-in-time")
if bug_all and pit_all:
    mb, mp = st.fmean(bug_all), st.fmean(pit_all)
    print(f"  IN-SAMPLE beta      mean {mb:+.3f}% per round trip")
    print(f"  POINT-IN-TIME beta  mean {mp:+.3f}% per round trip")
    print(f"  BUG-03 inflation    {mb-mp:+.3f} percentage points per trade")
    print()
    print(f"  in-sample profitable:     {'YES' if mb > 0 else 'no'}")
    print(f"  point-in-time profitable: {'YES' if mp > 0 else 'no'}")
