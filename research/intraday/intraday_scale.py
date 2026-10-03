"""What is the natural scale of a 5-minute bar, and of a session?

Everything about the intraday 1:2 question follows from this. The daily test
failed because a 2R target at a 12% stop is 24% away and the market rarely
travels that far in 20 sessions. Intraday the stop must be small - and
cost_R = round_trip% / stop% means a small stop is proportionally MORE
expensive. So the question is whether there is any stop width where the target
is reachable AND the cost is survivable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import pandas as pd

d = Path("configs/intraday/5m")
bars = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
                 ignore_index=True)
bars["session"] = bars["timestamp"].dt.date
print(f"{len(bars):,} bars  {bars['symbol'].nunique()} symbols  "
      f"{bars['session'].nunique()} sessions\n")

bars["bar_range_pct"] = (bars["high"] - bars["low"]) / bars["close"] * 100
print("5-MINUTE BAR RANGE, % of price:")
for q in (25, 50, 75, 90, 99):
    print(f"  p{q:<3} {bars['bar_range_pct'].quantile(q / 100):.3f}%")

g = bars.groupby(["symbol", "session"])
sess = pd.DataFrame({
    "open": g["open"].first(), "high": g["high"].max(),
    "low": g["low"].min(), "close": g["close"].last(), "n": g.size()})
sess["range_pct"] = (sess["high"] - sess["low"]) / sess["open"] * 100
sess["up_pct"] = (sess["high"] - sess["open"]) / sess["open"] * 100
sess["down_pct"] = (sess["open"] - sess["low"]) / sess["open"] * 100

print("\nSESSION RANGE (high-low as % of open):")
for q in (10, 25, 50, 75, 90):
    print(f"  p{q:<3} {sess['range_pct'].quantile(q / 100):.2f}%")

print("\nMAXIMUM FAVOURABLE EXCURSION from the open, % "
      "(the ceiling on any long target):")
for q in (10, 25, 50, 75, 90):
    print(f"  p{q:<3} {sess['up_pct'].quantile(q / 100):.2f}%")

print("\nTHE ARITHMETIC THAT DECIDES IT. "
      "round trip 0.422% (0.1222 statutory + 0.300 slippage):")
print(f"  {'stop%':>7}{'cost_R':>9}{'target%':>10}"
      f"{'breakeven win':>15}{'sessions reaching target':>26}")
for stop in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    cost_r = 0.422 / stop
    target = 2 * stop
    be = (1 + cost_r) / 3.0
    reach = float((sess["up_pct"] >= target).mean())
    print(f"  {stop:>7.2f}{cost_r:>9.3f}{target:>9.2f}%"
          f"{be:>14.1%}{reach:>25.1%}")
print("\n  (last column: share of SESSIONS whose high ever reached the target")
print("   measured from the open - an UPPER BOUND, since it ignores whether")
print("   the stop was hit first)")
