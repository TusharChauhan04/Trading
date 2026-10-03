"""Fetch 60 days of 5-minute bars for the most liquid names, and store them.

Point-in-time by filename, the same contract as the daily store:
configs/intraday/5m/<session>.parquet, one file per session.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd
from desk.marketdata.sources.errors import RateLimited, SourceError
from desk.marketdata.sources.yahoo import YahooSession, session_quality
from desk.scanner.stage0 import run_stage0
from desk.store.bars import BarStore

C = Path("configs")
store = BarStore(C / "bhavcopy")
day = sorted(store.available_days())[-1]
s0 = run_stage0(store.load_day(day), min_price=50.0, min_turnover_lacs=500.0)
syms = s0.survivors.nlargest(40, "turnover_lacs")["symbol"].tolist()
print(f"{len(syms)} most liquid names as of {day}", flush=True)

sess = YahooSession(min_interval=1.6)
frames, failed = [], []
for i, sym in enumerate(syms, 1):
    try:
        df = sess.fetch(sym, interval="5m", days=60)
        frames.append(df)
        print(f"  [{i:>2}/{len(syms)}] {sym:<14} {len(df):>5} bars  "
              f"{df['session'].nunique():>2} sessions", flush=True)
    except RateLimited as exc:
        print(f"  RATE LIMITED at {sym}: {exc}", flush=True)
        break
    except SourceError as exc:
        failed.append((sym, str(exc)[:70]))
        print(f"  [{i:>2}/{len(syms)}] {sym:<14} FAILED {exc}", flush=True)

if not frames:
    raise SystemExit("nothing fetched")
allbars = pd.concat(frames, ignore_index=True)
print(f"\ntotal {len(allbars):,} bars, {allbars['symbol'].nunique()} symbols, "
      f"{allbars['session'].nunique()} sessions", flush=True)
print(f"failed: {failed}")

out = C / "intraday" / "5m"
out.mkdir(parents=True, exist_ok=True)
for s, grp in allbars.groupby("session"):
    grp.drop(columns=["session"]).to_parquet(out / f"{s}.parquet", index=False)
print(f"wrote {allbars['session'].nunique()} session files to {out}")

q = session_quality(allbars, expected=40 * 70)
print("\nsessions with the fewest bars (incomplete coverage):")
print(q.nsmallest(5, "bars").to_string(index=False))
