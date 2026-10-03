"""1-HOUR bars, held across sessions. The intraday case 5m could not test.

WHY THIS IS A DIFFERENT QUESTION. The 5-minute result was bounded by the
SESSION: a trade had to resolve inside 2.18% of median range, which caps the
stop below intra-session noise and puts any 2x target outside the day. Hourly
bars held for several sessions are not bounded that way - available range grows
roughly with the square root of time, so ~2.18% over one session becomes ~4.9%
over five. That lets the stop widen, and cost_R = round_trip/stop falls
proportionally: 0.844 at a 0.5% stop becomes 0.281 at 1.5%.

So this is the intermediate timescale between 5-minute (dead) and daily
(marginal - 5 of 8 windows, DSR 0.08). The honest prior is that it resembles a
worse daily rather than something new, but it is nearly free to measure and the
sample is 245 sessions against 31.

TRADES ARE NOT CONFINED TO A SESSION HERE, which is the whole point, and it
means overnight gap risk is real and included: a stop can be jumped through at
the next open, and because the lattice checks the stop against each bar's LOW
it registers the gap as a stop-out at the stop level rather than at the gap
price. That FLATTERS the result slightly - a real gap fills worse than the stop
- so a negative verdict here is conservative and a positive one would need the
gap modelled properly before being believed.

Reuses desk.backtest.intraday.session_outcomes with `session_col` pointed at a
constant, so each symbol becomes one continuous group across all sessions. The
tested lattice, different grouping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import pandas as pd

from desk.backtest.intraday import breakeven_win_rate, session_outcomes

d = Path("configs/intraday/1h")
files = sorted(d.glob("*.parquet"))
if not files:
    raise SystemExit(f"no 1h bars under {d} yet")
bars = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
bars["session"] = bars["timestamp"].dt.date
keep = list(bars["symbol"].value_counts().head(12).index)
bars = bars[bars["symbol"].isin(keep)].sort_values(["symbol", "timestamp"])
bars["_all"] = "all"
per = bars.groupby("symbol").size()
print(f"{len(bars):,} hourly bars  {bars['symbol'].nunique()} symbols  "
      f"{bars['session'].nunique()} sessions  "
      f"median {per.median():.0f} bars/symbol\n")

# NSE trades ~6.25h, so Yahoo gives ~7 hourly bars a session.
BARS_PER_SESSION = 7
print(f"{'hold':>12}{'rr':>5}{'stop%':>7}{'target%':>9}{'trades':>8}"
      f"{'win%':>7}{'stopped':>9}{'timeout':>9}{'breakeven':>11}"
      f"{'GAP':>8}{'gross R':>10}{'net R':>9}")
rows = []
for hold_sessions in (2, 5, 10):
    hold = hold_sessions * BARS_PER_SESSION
    for rr in (1.0, 2.0):
        for stop in (1.0, 1.5, 2.0, 3.0):
            o = session_outcomes(bars, stop_pct=stop, rr=rr, max_bars=hold,
                                 session_col="_all")
            if not o.trades:
                continue
            be = breakeven_win_rate(stop_pct=stop, rr=rr)
            gap = o.win_rate - be
            rows.append((gap, hold_sessions, rr, stop, o, be))
            print(f"{hold_sessions:>9}sess{rr:>5.1f}{stop:>7.2f}"
                  f"{stop * rr:>8.2f}%{o.trades:>8,}{o.win_rate:>6.1%}"
                  f"{o.stopped / o.trades:>8.1%}{o.timed_out / o.trades:>8.1%}"
                  f"{be:>10.1%}{gap:>+8.1%}{o.gross_r:>+10.4f}"
                  f"{o.net_r():>+9.3f}")
    print()

rows.sort(reverse=True)
print("BEST BY GAP (achieved minus required win rate):")
for gap, hs, rr, stop, o, be in rows[:5]:
    print(f"  {hs}sess rr{rr} stop{stop}%: gap {gap:+.1%}  "
          f"gross {o.gross_r:+.4f}R  net {o.net_r():+.3f}R  "
          f"req RT {o.required_round_trip_pct():+.4f}%")
pos = [r for r in rows if r[4].net_r() > 0]
print(f"\nNET-POSITIVE CELLS: {len(pos)}")
for gap, hs, rr, stop, o, be in pos[:8]:
    print(f"  {hs}sess rr{rr} stop{stop}%: net {o.net_r():+.3f}R "
          f"on {o.trades:,} trades, win {o.win_rate:.1%}")
