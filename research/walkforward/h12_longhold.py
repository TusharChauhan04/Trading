"""H12: longer holds and wider stops - the region the cost arithmetic points at.

WHERE THIS COMES FROM. Today's cost frontier (desk/backtest/intraday.py plus
the hourly run) established that the gross edge available to this kind of rule
is small and roughly flat - about +0.05 to +0.08R - once the horizon exceeds a
week, and that cost_R = round_trip/stop is what decides whether it survives.
Solving for the stop width a +0.05R edge can afford gives >8.4%, which puts a
2R target ~17% away. A 17% move needs WEEKS.

Everything tested so far tops out at a 20-session hold. That was never a
measured choice: 20 came from the earlier finding that 5-10 sessions was far
too short for a 12% stop, and nothing has looked further out. So this walks the
natural extension - holds of 20, 40 and 60 sessions against stops of 12%, 15%
and 20% - on the full 1,715-session panel including the COVID crash.

NUM_TRIALS = 56, not 47. This IS a new search: nine new cells per rule are
being tried, and the deflated Sharpe has to be told. Keeping 47 would be the
exact dishonesty DSR exists to prevent.

THE ACCEPTANCE TEST is unchanged and pre-registered: the count of windows BOTH
positive AND timing-significant. H11 showed why nothing else will do - the same
cell on the same data gave DSR 0.8654 at 8 windows and 0.2268 at 11, because
the 8-window boundary averaged the crash together with the recovery. So BOTH
window counts are reported again, and a result that disagrees between them is
not a result.

WHAT WOULD COUNT AS SUCCESS: better than 5 of 8 and 5 of 11, the best the
12%/20-session cell ever managed, on a sample that now contains a crash. A
longer hold mechanically reduces the trade count per window, so the 30-trade
floor is checked and reported rather than assumed.
"""
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))

from desk.backtest.preset_wf import walk_forward_preset
from desk.scanner.stage0 import run_stage0
from desk.store.bars import BarStore
from desk.strategies.presets import preset_rules

AS_OF = date(2026, 9, 29)
BUF = 1.0
NUM_TRIALS = 56

store = BarStore(Path("configs/bhavcopy"))
days = sorted(store.available_days())
s0 = run_stage0(store.load_day(AS_OF), min_price=50.0, min_turnover_lacs=500.0)
hist = store.history(as_of=AS_OF, start=days[0],
                     symbols=s0.survivors["symbol"].tolist(),
                     columns=["open", "high", "low", "close", "volume"])
panel = hist.wide_many(["open", "high", "low", "close", "volume"])
close = panel["close"]
print(f"panel {close.shape}  {days[0]} -> {days[-1]}", flush=True)

prior_high = panel["high"].shift(1).rolling(20, min_periods=20).max()
excess = (close / prior_high - 1.0).to_numpy(float)

print(f"\n{'rule':<20}{'stop%':>6}{'hold':>6}{'cost_R':>8}{'win':>6}"
      f"{'pooled':>10}{'pos':>7}{'timing':>8}{'BOTH':>7}{'DSR':>8}{'trades':>8}",
      flush=True)
rows = []
DONE = {("donchian_breakout", 12.0, 20, 8), ("donchian_breakout", 12.0, 20, 11),
        ("donchian_breakout", 12.0, 40, 8), ("donchian_breakout", 12.0, 40, 11),
        ("donchian_breakout", 12.0, 60, 8), ("donchian_breakout", 12.0, 60, 11),
        ("donchian_breakout", 15.0, 20, 8), ("donchian_breakout", 15.0, 20, 11),
        ("donchian_breakout", 15.0, 40, 8)}

for pid in ("donchian_breakout", "bollinger_breakout"):
    base = preset_rules()[pid].entries(panel).to_numpy(bool)
    sig = base & (excess >= BUF / 100.0)
    for stop in (12.0, 15.0, 20.0):
        cost_r = (0.1222 + 0.300) / stop
        for hold in (20, 40, 60):
            for nwin in (8, 11):
                if (pid, stop, hold, nwin) in DONE:
                    continue
                wf = walk_forward_preset(
                    panel, sig, label=f"{pid} {stop}/{hold} [{nwin}w]",
                    stop_pct=stop, hold_bars=hold, rr=2.0, buffer_pct=BUF,
                    cost_r=cost_r, windows=nwin, num_trials=NUM_TRIALS,
                    n_permutations=200)
                scored = wf.scored
                n = len(scored)
                both = sum(1 for w in scored if w.positive and w.timed)
                tr = sum(w.trades for w in scored)
                rows.append((both / max(n, 1), both, n, pid, stop, hold,
                             nwin, wf, tr))
                print(f"{pid[:18]:<20}{stop:>6.0f}{hold:>6}{cost_r:>8.4f}"
                      f"{nwin:>5}w{wf.pooled_net_r:>+10.4f}"
                      f"{wf.windows_positive:>4}/{n}{wf.windows_timed:>6}/{n}"
                      f"{both:>5}/{n}{wf.robustness().dsr:>8.4f}{tr:>8,}",
                      flush=True)
    print(flush=True)

rows.sort(reverse=True)
print("RANKED BY THE PRE-REGISTERED CRITERION (share of windows both positive "
      "and timing-significant):")
for frac, both, n, pid, stop, hold, nwin, wf, tr in rows[:8]:
    print(f"  {pid[:18]:<20} stop {stop:>4.0f}%  hold {hold:>2}  {nwin}w  "
          f"BOTH {both}/{n} ({frac:.0%})  pooled {wf.pooled_net_r:+.4f}R  "
          f"DSR {wf.robustness().dsr:.4f}")
print("\nBASELINE TO BEAT: the 12%/20-session cell managed 5/8 and 5/11 on "
      "1,241 sessions, and 2/8 and 5/11 once the crash was included.")
