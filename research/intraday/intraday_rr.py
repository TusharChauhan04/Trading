"""THE MAIN GOAL: is a 1:2 target reachable on 5-minute bars, net of costs?

The scale measurement gave an UPPER BOUND - the share of sessions whose high
ever reached the target, ignoring whether the stop was hit first. This is the
real thing: bar by bar, which came first.

Entries are EVERY bar (unconditional), which answers "is the payoff structure
achievable at all" separately from "does some signal predict it". An
unconditional test that fails cannot be rescued by a better entry rule: the
signal would have to beat not just random but the structure itself.

TRADES ARE CONFINED TO ONE SESSION. Holding across the overnight gap is a
different trade with different risk, and the desk is long-only cash equity, so
an intraday rule that cannot close by 15:15 is not the rule being tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd

STATUTORY = 0.1222

d = Path("configs/intraday/5m")
bars = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
                 ignore_index=True)
bars["session"] = bars["timestamp"].dt.date
bars = bars.sort_values(["symbol", "timestamp"])
print(f"{len(bars):,} bars  {bars['symbol'].nunique()} symbols  "
      f"{bars['session'].nunique()} sessions")


def outcomes(grp: pd.DataFrame, stop_pct: float, rr: float,
             max_bars: int) -> tuple[int, int, int]:
    """(target_hits, stop_hits, timeouts) for every bar in one session.

    Entry at the bar's close. Then walk forward within the SAME session,
    checking the stop before the target on each bar - the conservative
    convention, since a bar touching both is unknowable without ticks and
    assuming the good outcome is how a backtest flatters itself.
    """
    c = grp["close"].to_numpy(float)
    h = grp["high"].to_numpy(float)
    lo = grp["low"].to_numpy(float)
    n = len(c)
    tgt = stop = out = 0
    for i in range(n - 1):
        entry = c[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        s_lvl = entry * (1 - stop_pct / 100.0)
        t_lvl = entry * (1 + rr * stop_pct / 100.0)
        last = min(i + max_bars, n - 1)
        hit = None
        for j in range(i + 1, last + 1):
            if lo[j] <= s_lvl:
                hit = "stop"
                break
            if h[j] >= t_lvl:
                hit = "tgt"
                break
        if hit == "tgt":
            tgt += 1
        elif hit == "stop":
            stop += 1
        else:
            out += 1
    return tgt, stop, out


print(f"\n{'stop%':>6}{'cost_R':>8}{'trades':>9}{'win%':>8}{'stop%':>8}"
      f"{'timeout%':>10}{'breakeven win':>15}{'net R':>9}{'verdict':>10}")
groups = list(bars.groupby(["symbol", "session"]))
for stop_pct in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    cost_r = STATUTORY / stop_pct + 0.300 / stop_pct
    t = s = o = 0
    for _, grp in groups:
        a, b, c_ = outcomes(grp, stop_pct, 2.0, 73)
        t += a
        s += b
        o += c_
    total = t + s + o
    if not total:
        continue
    win = t / total
    # A timeout exits at the last close; treated as 0R before costs, which is
    # generous - the realised move is usually slightly negative after costs.
    net = (t * 2.0 + s * -1.0 + o * 0.0) / total - cost_r
    be = (1 + cost_r) / 3.0
    print(f"{stop_pct:>6.2f}{cost_r:>8.3f}{total:>9,}{win:>7.1%}"
          f"{s / total:>7.1%}{o / total:>9.1%}{be:>14.1%}{net:>+9.3f}"
          f"{'  VIABLE' if net > 0 else '  no':>10}")

print("\nSLIPPAGE SENSITIVITY - what would costs have to be? "
      "(statutory 0.1222% is irreducible)")
print(f"{'stop%':>6}{'win%':>8}  required round-trip for net>0")
for stop_pct in (0.5, 1.0, 2.0):
    t = s = o = 0
    for _, grp in groups:
        a, b, c_ = outcomes(grp, stop_pct, 2.0, 73)
        t += a
        s += b
        o += c_
    total = t + s + o
    gross = (t * 2.0 + s * -1.0) / total
    needed_pct = gross * stop_pct
    print(f"{stop_pct:>6.2f}{t / total:>7.1%}  gross {gross:+.4f}R -> round "
          f"trip must be under {needed_pct:.4f}% "
          f"({'IMPOSSIBLE, below statutory' if needed_pct < STATUTORY else 'possible'})")
