import json, sys, time, warnings
from datetime import date
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import desk.api.main as m
from desk.settings import Settings

out = {}
cfg = Settings.from_env()
out["llm_configured"] = cfg.llm_configured
out["llm_local"] = getattr(cfg, "llm_local", "n/a (field did not exist)")
out["llm_model"] = cfg.llm_model
c = m._llm_client()
out["provider"] = type(c.provider).__name__ if c else None
out["provider_model"] = c.provider.model if c else None

# Fundamentals: the join stage2/stage3 depend on.
from desk.research.store import FilingStore
from desk.research.fundamentals import build_table, COLUMNS
root = Path(m.CONFIGS) / "research" / "filings"
syms = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
if syms:
    t, cov = build_table(FilingStore(root), syms, as_of=date(2026, 9, 29))
    out["fund_rows"] = len(t)
    out["fund_columns"] = list(COLUMNS)
    out["fund_coverage"] = cov
    # the pre-existing columns' VALUES, so added columns cannot have shifted them
    base = ["revenue", "profit_after_tax", "eps_basic", "net_margin_pct",
            "revenue_growth_yoy_pct", "profit_growth_yoy_pct", "margin_change_pp"]
    out["fund_checksums"] = {
        c2: round(float(t[c2].astype(float).sum(skipna=True)), 4)
        for c2 in base if c2 in t.columns}

# Timing of the funnel, three runs.
ts = []
for _ in range(3):
    t0 = time.perf_counter()
    m._stage1_for(date(2026, 9, 29), lookback=300, min_price=50.0,
                  min_turnover_lacs=500.0, min_bars=200)
    ts.append(time.perf_counter() - t0)
out["stage1_seconds"] = [round(x, 3) for x in ts]
print(json.dumps(out, indent=2, default=str))
