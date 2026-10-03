"""Stage 1 output + plan decisions, dumped for baseline-vs-now comparison."""
import json, sys, warnings
from datetime import date
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import desk.api.main as m

out = {"tree": str(Path.cwd())}
s1, extra = m._stage1_for(date(2026, 9, 29), lookback=300, min_price=50.0,
                          min_turnover_lacs=500.0, min_bars=200)
df = s1.features
out["extra_notes"] = extra
out["rows"] = int(len(df))
out["columns"] = list(df.columns)
num = [c for c in df.columns if str(df[c].dtype).startswith(("float","int"))]
out["checksums"] = {c: round(float(df[c].astype(float).sum(skipna=True)), 8)
                    for c in sorted(num)}
out["symbols_head"] = sorted(df["symbol"].tolist())[:10] if "symbol" in df else []
try:
    req = m.PlanRequest(day=date(2026, 9, 29))
    plan = m.plan_today_with_portfolio(req)
    d = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
    out["plan_keys"] = sorted(d.keys())
    for key in ("verdict", "decision", "regime", "notes", "no_trade_reason"):
        if key in d: out[f"plan_{key}"] = str(d[key])[:400]
    fp = []
    for bucket in ("primary", "secondary", "watchlist", "candidates", "trades"):
        rows = d.get(bucket)
        if isinstance(rows, list):
            fp.append((bucket, len(rows), [
                (r.get("symbol"), r.get("entry"), r.get("stop"),
                 r.get("target"), r.get("quantity"))
                for r in rows if isinstance(r, dict)]))
    out["plan_fingerprint"] = fp
except Exception as e:
    out["plan_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out, indent=2, default=str))
