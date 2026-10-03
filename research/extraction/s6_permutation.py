"""Does donchian's entry TIMING carry information, or is it the window?

The 12% / 24% / 60-session configuration came in at +0.071R per trade, positive
even at the pessimistic slippage guess. It failed deflation on POWER rather
than on sign: DSR 0.146 and verdict "insufficient", because 36 monthly cohorts
is short of the 60 Bailey & Lopez de Prado's estimators want.

A permutation test does not need 60. It builds the null from this data using
this strategy's own trade mechanics, so the p-value is exact at any sample
size. The null here shuffles each symbol's signals WITHIN that symbol, which
preserves the price process, preserves how many times each name traded, and
destroys only the timing. So it answers the sharpest version of the question:
given these stocks and this market, did the rule enter on better days than
chance?

This is a stronger control than the cross-sectional random arm already run.
That one held the dates and randomised the symbols, so it could be beaten by a
rule that merely picks stocks that go up. This one holds the symbols and
randomises the dates, so a rising window cannot help it at all - every
permutation lives in the same window, in the same stocks.

Run at both the catalogued 5% stop and the 12% that measured positive, because
a result that is significant only where it is profitable is a different claim
from one that is significant throughout.
"""
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from desk.backtest.permutation import (                           # noqa: E402
    outcome_lattice, signal_timing_test,
)
from desk.backtest.slippage import SlippageModel                  # noqa: E402
from desk.scanner.stage0 import run_stage0                        # noqa: E402
from desk.store.bars import BarStore                              # noqa: E402
from desk.strategies.presets import preset_rules                  # noqa: E402

AS_OF = date(2026, 9, 29)
STATUTORY = 0.1222
SLIP_PESSIMISTIC = 0.300          # the flat 15bps-a-side guess, round trip
N_PERM = 400

store = BarStore(Path("configs/bhavcopy"))
s0 = run_stage0(store.load_day(AS_OF), min_price=50.0,
                min_turnover_lacs=500.0)
syms = s0.survivors["symbol"].tolist()
hist = store.history(as_of=AS_OF, start=date(2023, 9, 1), symbols=syms,
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}  {close.index[0]} -> {close.index[-1]}", flush=True)

# The modelled retail round trip, for the second cost column.
turn = s0.survivors.set_index("symbol")["turnover_lacs"] * 1e5
model = SlippageModel()
med_part = float((100_000 / turn[turn > 0]).median())
slip_retail = model.round_trip_pct(med_part)
print(f"slippage: pessimistic {SLIP_PESSIMISTIC:.3f}% round trip, "
      f"modelled at Rs 1L {slip_retail:.3f}%\n", flush=True)

hi = panel["high"].to_numpy(float)
lo = panel["low"].to_numpy(float)
cl = close.to_numpy(float)

rules = preset_rules()
hdr = (f"{'preset':<20}{'stop':>6}{'hold':>5}{'trades':>8}{'obs netR':>10}"
       f"{'null mean':>11}{'null sd':>9}{'p':>8}{'pct':>7}")
print(hdr)
print("-" * len(hdr))

t0 = time.time()
for pid in ("donchian_breakout", "bollinger_breakout"):
    sig = rules[pid].entries(panel).to_numpy(bool)
    for stop, hold in ((5.0, 60), (12.0, 60), (12.0, 20)):
        r, ex = outcome_lattice(hi, lo, cl, stop_pct=stop, rr=2.0, hold=hold)
        for label, slip in (("pessimistic", SLIP_PESSIMISTIC),
                            ("retail", slip_retail)):
            cost_r = (STATUTORY + slip) / stop
            res = signal_timing_test(sig, r, ex, cost_r=cost_r,
                                     n_permutations=N_PERM, seed=7)
            star = "  <-- p<0.05" if res.significant else ""
            print(f"{pid:<20}{stop:>5.1f}%{hold:>5}{res.n_trades:>8}"
                  f"{res.observed:>+10.4f}{res.null_mean:>+11.4f}"
                  f"{res.null_std:>9.4f}{res.p_value:>8.4f}"
                  f"{res.percentile:>6.1f}%{star}"
                  f"   [{label}]", flush=True)

print(f"\nwall clock {(time.time() - t0) / 60:.1f} min")
print("\n" + "=" * 86)
print("HOW TO READ THIS")
print("=" * 86)
print("  The null mean is what the SAME signals earn at random dates in the")
print("  same stocks. If observed and null sit on top of each other, the rule")
print("  is harvesting the window rather than choosing moments - and a wider")
print("  stop would still 'work', because the wider stop is what harvests the")
print("  window, not the signal.")
print()
print("  A low p-value here does NOT mean the strategy is validated. It means")
print("  the timing carries information. Whether that information survives")
print("  costs and the search is DSR's question, and DSR said no.")
