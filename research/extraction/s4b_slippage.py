"""Two questions the preset result forces, and both are cheap to answer.

The presets came in at 0 of 10 positive, but 9 of 10 beat random on the same
calendar, and the ranking by net R was almost exactly the ranking by COST. The
desk's own cost module says what that means:

    "a result that only just clears zero is really a statement about the
     slippage assumption rather than about the strategy"

So the cost gets decomposed instead of being taken as one number.

    statutory   0.1222%   arithmetic, unavoidable, STT dominates
    slippage    0.3000%   15bps a side, AND IT IS LABELLED A GUESS

Seventy-one percent of the cost that sank these strategies is an assumption
nobody has measured. Two questions follow.

(1) BREAK-EVEN SLIPPAGE. At what per-side slippage does each cell reach zero?
    That reframes the result from "it loses" to "it is profitable if and only
    if you can trade better than X bps" - a question about the broker and the
    order type rather than about the strategy. Pure arithmetic from figures
    already measured: no refitting, no new search, so it adds no trials.

(2) STOP WIDTH. cost_R = round_trip% / stop%, so a wider stop dilutes the cost
    without the strategy changing at all. Donchian's 5% stop pays 0.084R; a
    10% stop would pay 0.042R. But widening also moves where the 2R target
    sits and which trades stop out, so gross moves too and the direction is
    not obvious. Pre-registered by the cost identity rather than data-mined -
    the hypothesis existed before the sweep, which is why it is run at all.
"""
import sys
import math
import statistics as st
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from desk.robustness import deflated_sharpe                      # noqa: E402
from desk.scanner.stage0 import run_stage0                       # noqa: E402
from desk.store.bars import BarStore                             # noqa: E402
from desk.strategies.presets import preset_rules                 # noqa: E402

STATUTORY = 0.1222      # % round trip; arithmetic, not a guess
SLIP_BPS = 15.0         # per side, the default guess
AS_OF = date(2026, 9, 29)
RR = 2.0

store = BarStore(Path("configs/bhavcopy"))
stage0 = run_stage0(store.load_day(AS_OF), min_price=50.0,
                    min_turnover_lacs=500.0)
syms = stage0.survivors["symbol"].tolist()
hist = store.history(as_of=AS_OF, start=date(2023, 9, 1), symbols=syms,
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
idx = list(close.index)
cl = close.to_numpy(float)
hi = panel["high"].to_numpy(float)
lo = panel["low"].to_numpy(float)
months = pd.Series([d.strftime("%Y-%m") for d in idx], index=range(len(idx)))
print(f"panel {close.shape}  {idx[0]} -> {idx[-1]}\n", flush=True)


def walk(i, c, stop_pct, hold):
    e = cl[i, c]
    if not np.isfinite(e) or e <= 0:
        return None, None
    stop_px = e * (1 - stop_pct / 100.0)
    take_px = e * (1 + RR * stop_pct / 100.0)
    last = min(i + hold, len(idx) - 1)
    for j in range(i + 1, last + 1):
        bh, bl = hi[j, c], lo[j, c]
        if not (np.isfinite(bh) and np.isfinite(bl)):
            continue
        if bl <= stop_px:
            return -1.0, j
        if bh >= take_px:
            return RR, j
    px = cl[last, c]
    if not np.isfinite(px):
        return None, None
    return (px - e) / (e * stop_pct / 100.0), last


def sequential(sig, stop_pct, hold):
    """One position per symbol at a time. No overlapping trades, ever."""
    out = []
    for c in range(sig.shape[1]):
        busy = -1
        col = sig[:, c]
        for i in range(sig.shape[0] - 1):
            if col[i] and i > busy:
                r, j = walk(i, c, stop_pct, hold)
                if r is not None:
                    out.append((i, r))
                    busy = j
    return out


def gross_cohorts(trades):
    """GROSS monthly cohort means. Costs are applied afterwards, so one walk
    serves every slippage assumption."""
    by_month = {}
    for i, r in trades:
        by_month.setdefault(months[i], []).append(r)
    return [st.fmean(v) for _k, v in sorted(by_month.items())]


def net_at(cohorts, stop_pct, slip_bps):
    cost_r = (STATUTORY + 2.0 * slip_bps / 100.0) / stop_pct
    return [c - cost_r for c in cohorts], cost_r


def breakeven_bps(gross_r, stop_pct):
    """Per-side slippage at which this cell reaches exactly zero.

    net = gross - (statutory + 2s/100) / stop  =>  s = (gross*stop - stat)*50
    A NEGATIVE answer means the gross edge does not even cover STT, so the
    cell loses at zero slippage and no broker can rescue it.
    """
    return (gross_r * stop_pct - STATUTORY) * 50.0


# ======================================================================
print("=" * 86)
print("(1) BREAK-EVEN SLIPPAGE - what each catalogued cell needs to reach zero")
print("=" * 86)
hdr = (f"{'preset':<20}{'hold':>5}{'stop':>6}{'grossR':>9}{'netR@15bps':>12}"
       f"{'break-even':>12}  verdict")
print(hdr)
print("-" * len(hdr))

rules = preset_rules()
cells = []
for pid, rule in rules.items():
    if rule.unsupported:
        continue
    sig = rule.entries(panel).to_numpy(bool)
    if not sig.sum():
        continue
    for hold in (20, 60):
        tr = sequential(sig, rule.stop_pct, hold)
        if len(tr) < 200:
            continue
        g = gross_cohorts(tr)
        if len(g) < 5:
            continue
        gross = st.fmean(g)
        net15, _ = net_at(g, rule.stop_pct, SLIP_BPS)
        be = breakeven_bps(gross, rule.stop_pct)
        if be < 0:
            verdict = "DEAD - loses at ZERO slippage"
        elif be < 5:
            verdict = "needs better than retail"
        elif be < 10:
            verdict = "plausible with limit orders"
        else:
            verdict = "comfortable"
        print(f"{pid:<20}{hold:>5}{rule.stop_pct:>5.1f}%{gross:>+9.4f}"
              f"{st.fmean(net15):>+12.4f}{be:>10.1f}bp  {verdict}", flush=True)
        cells.append({"id": pid, "hold": hold, "stop": rule.stop_pct,
                      "gross": gross, "be": be, "cohorts": g,
                      "n": len(tr)})

print("\n  The statutory 0.1222% is arithmetic. The 0.30% slippage is the")
print("  module's own labelled GUESS at 15bps a side, and it is 71% of the")
print("  cost. Every 'break-even' above is therefore a question about the")
print("  broker and the order type, not about the strategy.")

# ======================================================================
print("\n" + "=" * 86)
print("(2) STOP WIDTH - cost_R = round_trip / stop, so does a wider stop pay?")
print("=" * 86)
print("  Pre-registered by the cost identity. Both breakout rules, 2R target")
print("  throughout, so only the stop width changes.\n")

STOPS = (4.0, 5.0, 7.0, 10.0, 12.0)
hdr2 = (f"{'preset':<20}{'hold':>5}{'stop':>6}{'trades':>8}{'targ%':>7}"
        f"{'grossR':>9}{'costR':>7}{'netR':>9}{'t':>7}{'be bps':>9}")
print(hdr2)
print("-" * len(hdr2))

sweep = []
for pid in ("donchian_breakout", "bollinger_breakout"):
    rule = rules[pid]
    sig = rule.entries(panel).to_numpy(bool)
    for hold in (20, 60):
        for stop in STOPS:
            tr = sequential(sig, stop, hold)
            if len(tr) < 200:
                continue
            g = gross_cohorts(tr)
            if len(g) < 5:
                continue
            gross = st.fmean(g)
            net, cost_r = net_at(g, stop, SLIP_BPS)
            nm = st.fmean(net)
            se = st.stdev(net) / math.sqrt(len(net)) if len(net) > 1 else 0
            hits = sum(1 for _i, r in tr if r == RR)
            print(f"{pid:<20}{hold:>5}{stop:>5.1f}%{len(tr):>8}"
                  f"{100 * hits / len(tr):>7.1f}{gross:>+9.4f}{cost_r:>7.3f}"
                  f"{nm:>+9.4f}{(nm / se if se else 0):>+7.2f}"
                  f"{breakeven_bps(gross, stop):>7.1f}bp", flush=True)
            sweep.append({"id": pid, "hold": hold, "stop": stop, "net": nm,
                          "t": (nm / se if se else 0.0), "cohorts": net,
                          "gross": gross, "n": len(tr),
                          "be": breakeven_bps(gross, stop)})

# ======================================================================
print("\n" + "=" * 86)
print("VERDICT")
print("=" * 86)
total = len(cells) + len(sweep)
pos = [r for r in sweep if r["net"] > 0]
print(f"  cells tried in total (both sections): {total}")
print(f"  stop-width cells positive at the 15bps GUESS: "
      f"{len(pos)} of {len(sweep)}")
best_be = max(cells + sweep, key=lambda r: r["be"])
print(f"  most forgiving cell: {best_be['id']} hold {best_be['hold']} "
      f"stop {best_be['stop']:g}% -> break-even {best_be['be']:.1f}bps/side")
if pos:
    b = max(pos, key=lambda r: r["t"])
    print(f"\n  best positive by t: {b['id']} hold {b['hold']} stop "
          f"{b['stop']:g}% -> {b['net']:+.4f}R  t {b['t']:+.2f}  "
          f"({b['n']} trades)")
    rep = deflated_sharpe(b["cohorts"], num_trials=total, periods_per_year=12)
    print(f"  {rep}")
    for r in rep.reasons:
        print(f"    - {r}")
else:
    print("\n  nothing positive at 15bps even with a wider stop.")
    b = max(sweep, key=lambda r: r["net"])
    print(f"  closest: {b['id']} hold {b['hold']} stop {b['stop']:g}% -> "
          f"{b['net']:+.4f}R (break-even {b['be']:.1f}bps/side)")
