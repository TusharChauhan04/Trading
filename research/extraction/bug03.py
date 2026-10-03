"""How much does BUG-03 flatter a pairs backtest? Same pair, both ways.

BUG-03 parked pairs_trading: the hedge ratio fitted on the whole sample, so
every historical spread knows its own future. OpenTerminalUI's cointegration
module has the same bug. This measures the size of it rather than asserting it
matters - the only honest basis for deciding whether to unpark the strategy.

Both arms trade the SAME rule on the SAME pair: enter at |z| >= 2, exit at
|z| <= 0.5, spread z-scored on a 250-bar window. The ONLY difference is where
beta comes from.
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

store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(date(2026, 9, 29)), min_price=50.0,
                min_turnover_lacs=2000.0)
syms = s0.survivors["symbol"].tolist()
hist = store.history(as_of=date(2026, 9, 29), start=date(2021, 9, 1),
                     symbols=syms, columns=["close"])
close = hist.wide_many(["close"])["close"].dropna(axis=1, thresh=1100)
print(f"{close.shape[1]} names with near-full history, "
      f"{close.shape[0]} sessions", flush=True)

# A few well-known Indian pairs that plausibly cointegrate, plus whatever of
# them is actually present.
CANDIDATES = [("HDFCBANK.NS","ICICIBANK.NS"), ("TCS.NS","INFY.NS"),
              ("ONGC.NS","OIL.NS"), ("SBIN.NS","BANKBARODA.NS"),
              ("RELIANCE.NS","BPCL.NS"), ("COALINDIA.NS","NTPC.NS")]
pairs = [(a,b) for a,b in CANDIDATES if a in close.columns and b in close.columns]
print(f"{len(pairs)} candidate pairs present\n", flush=True)

def trade(z: pd.Series, entry=2.0, exit_=0.5):
    """Mean-reversion on the z-score. Returns the realised z-moves captured."""
    pos, entry_z, out = 0, 0.0, []
    for v in z.dropna():
        if pos == 0 and abs(v) >= entry:
            pos, entry_z = (-1 if v > 0 else 1), v
        elif pos != 0 and abs(v) <= exit_:
            out.append(abs(entry_z) - abs(v) if pos * entry_z < 0 else 0.0)
            pos = 0
    return out

hdr = f"{'pair':<30}{'trades':>8}{'in-sample z':>13}{'point-in-time':>15}{'inflation':>11}"
print(hdr); print("-" * len(hdr))
infl = []
for a, b in pairs:
    y, x = close[a], close[b]
    # ARM 1: BUG-03 - one beta from the whole sample, applied to all history.
    hr = hedge_ratio(y, x)
    if hr is None:
        print(f"{a[:13]+'/'+b[:13]:<30}  beta not estimable"); continue
    beta, alpha = hr
    sp = y - beta * x - alpha
    rm = sp.rolling(DEFAULT_WINDOW).mean(); rs = sp.rolling(DEFAULT_WINDOW).std()
    z_bug = ((sp - rm) / rs)
    # ARM 2: trailing beta, re-estimated, nothing from the future.
    wf = walk_spread(y, x, window=DEFAULT_WINDOW, refit_every=21)
    if wf.empty:
        print(f"{a[:13]+'/'+b[:13]:<30}  no walk-forward spread"); continue

    t_bug, t_pit = trade(z_bug), trade(wf["zscore"])
    m_bug = st.fmean(t_bug) if t_bug else float("nan")
    m_pit = st.fmean(t_pit) if t_pit else float("nan")
    ratio = (m_bug / m_pit) if (t_pit and m_pit not in (0.0,) and
                                np.isfinite(m_pit) and m_pit > 0) else float("nan")
    if np.isfinite(ratio): infl.append(ratio)
    print(f"{a[:13]+'/'+b[:13]:<30}{len(t_pit):>8}{m_bug:>13.4f}{m_pit:>15.4f}"
          f"{(f'{ratio:.2f}x' if np.isfinite(ratio) else '   -'):>11}", flush=True)

print()
if infl:
    print(f"  median inflation from the in-sample beta: {st.median(infl):.2f}x")
print(f"  beta drift, point-in-time, per pair:")
for a, b in pairs[:4]:
    wf = walk_spread(close[a], close[b], window=DEFAULT_WINDOW, refit_every=21)
    if not wf.empty:
        print(f"    {a[:11]+'/'+b[:11]:<26} beta {wf['beta'].min():.3f} -> "
              f"{wf['beta'].max():.3f}  (whole-sample {hedge_ratio(close[a], close[b])[0]:.3f})")
print()
print("  A beta that moves this much is the reason the whole-sample estimate")
print("  flatters: it is the average of relationships that did not coexist.")
