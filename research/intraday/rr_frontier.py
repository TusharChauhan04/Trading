"""Which reward-to-risk ratio is actually achievable? The parameter never varied.

1:2 has now failed on daily bars (2R target 24% away, 26.6% hit at 60 sessions)
and intraday (gross negative at every stop width). `rr` is the one term in the
structure nobody has moved, and it pulls two ways at once:

    breakeven win rate = (1 + cost_R) / (1 + rr)

Raising rr LOWERS the win rate needed. It also moves the target further out, so
the win rate actually achieved falls. Lowering rr raises the required rate but
brings the target within reach. There is an optimum and it has never been
located.

Reported as the GAP: achieved win rate minus breakeven win rate. Positive
anywhere means a viable structure exists at that timescale.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import pandas as pd

from desk.backtest.intraday import breakeven_win_rate, session_outcomes

d = Path("configs/intraday/5m")
bars = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
                 ignore_index=True)
bars["session"] = bars["timestamp"].dt.date
# The 12 most-covered names keeps the run tractable and the sample honest.
keep = list(bars["symbol"].value_counts().head(12).index)
bars = bars[bars["symbol"].isin(keep)].sort_values(["symbol", "timestamp"])
print(f"{len(bars):,} bars  {bars['symbol'].nunique()} symbols  "
      f"{bars['session'].nunique()} sessions\n")

print(f"{'rr':>5}{'stop%':>7}{'target%':>9}{'trades':>8}{'win%':>8}"
      f"{'breakeven':>11}{'GAP':>9}{'gross R':>10}{'net R':>9}")
best = []
for rr in (0.5, 1.0, 1.5, 2.0, 3.0):
    for stop in (0.25, 0.5, 1.0, 2.0):
        o = session_outcomes(bars, stop_pct=stop, rr=rr, max_bars=73)
        if not o.trades:
            continue
        be = breakeven_win_rate(stop_pct=stop, rr=rr)
        gap = o.win_rate - be
        best.append((gap, rr, stop, o, be))
        print(f"{rr:>5.1f}{stop:>7.2f}{stop * rr:>8.2f}%{o.trades:>8,}"
              f"{o.win_rate:>7.1%}{be:>10.1%}{gap:>+9.1%}"
              f"{o.gross_r:>+10.4f}{o.net_r():>+9.3f}")
    print()

best.sort(reverse=True)
g, rr, stop, o, be = best[0]
print(f"BEST CELL: rr {rr}, stop {stop}% -> gap {g:+.1%}, "
      f"gross {o.gross_r:+.4f}R, net {o.net_r():+.3f}R")
print(f"  {'VIABLE' if o.net_r() > 0 else 'still negative'}")
print("\nGROSS-POSITIVE CELLS (structure works before costs):")
pos = [b for b in best if b[3].gross_r > 0]
if not pos:
    print("  none - no reward-to-risk ratio tested has positive gross "
          "expectancy at any stop width")
else:
    for g, rr, stop, o, be in pos[:8]:
        print(f"  rr {rr} stop {stop}%: gross {o.gross_r:+.4f}R  "
              f"cost {o.cost_r():.3f}R  net {o.net_r():+.3f}R")
