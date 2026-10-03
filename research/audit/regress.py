"""Did this session's extraction change anything it should not have?

Run from inside each tree so each imports ITS OWN desk package. Dumps the
Stage 1 feature frame and the daily plan's decisions to a parquet/json pair,
which the comparer then diffs column by column.
"""
import json, sys, warnings
from datetime import date
from pathlib import Path
warnings.filterwarnings("ignore")

out = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd()))

import desk.api.main as m                                    # noqa: E402
from desk.backtest.costs import CostModel                    # noqa: E402

DAY = date(2026, 9, 29)
r, caveats = m._stage1_for(DAY, lookback=300, min_price=50.0,
                           min_turnover_lacs=500.0, min_bars=200)
r.features.to_parquet(out / "features.parquet")

c = CostModel()
meta = {
    "symbols": int(len(r.features)),
    "columns": list(r.features.columns),
    "caveats": sorted(caveats),
    "cost_round_trip_pct": round(c.round_trip_pct(), 6),
    "cost_slippage_bps": c.slippage_bps,
    "cost_stt": c.stt_sell_pct,
}
# The daily plan: the decision surface that matters most.
try:
    from fastapi.testclient import TestClient
    resp = TestClient(m.app).get(f"/plan/today?day={DAY}")
    meta["plan_status"] = resp.status_code
    if resp.status_code == 200:
        p = resp.json()
        meta["plan_keys"] = sorted(p.keys())
        for k in ("decision", "regime", "no_trade", "caveats"):
            if k in p:
                meta[f"plan_{k}"] = p[k] if not isinstance(p[k], list) \
                    else sorted(str(x) for x in p[k])
        for bucket in ("primary", "secondary", "watchlist"):
            v = p.get(bucket)
            if isinstance(v, list):
                meta[f"plan_{bucket}"] = [
                    {kk: x.get(kk) for kk in ("symbol", "entry", "stop",
                                              "target", "qty", "trigger")}
                    for x in v]
except Exception as exc:                                     # noqa: BLE001
    meta["plan_error"] = f"{type(exc).__name__}: {exc}"

(out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(f"wrote {out}: {meta['symbols']} symbols, "
      f"{len(meta['columns'])} columns, plan {meta.get('plan_status')}")
