"""What exists, what it can do, and whether it is actually wired in.

The registry is the only thing the coordinator reads to decide who runs.
`wired` is deliberately separate from `installed`: a repository can be present
and verified on disk while still having no adapter, and pretending otherwise
is how a plan drifts from reality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from desk.contracts.enums import AssetClass, Capability, Horizon


@dataclass(frozen=True, slots=True)
class FleetEntry:
    name: str
    kind: str
    version: str
    capabilities: list[Capability]
    asset_classes: list[AssetClass]
    horizons: list[Horizon]
    licence: str
    installed: bool
    verified: bool
    wired: bool                    # has an adapter speaking the contract
    india_ready: bool
    note: str = ""
    blockers: list[str] = field(default_factory=list)


FLEET: list[FleetEntry] = [
    FleetEntry(
        name="vibe_trading", kind="hybrid", version="0.1.14",
        capabilities=[Capability.QUANT, Capability.BACKTEST, Capability.SCREENING,
                      Capability.FUNDAMENTAL, Capability.TECHNICAL],
        asset_classes=[AssetClass.EQUITY, AssetClass.CRYPTO, AssetClass.FX],
        horizons=[Horizon.SWING, Horizon.POSITION, Horizon.LONG_TERM],
        licence="MIT", installed=True, verified=True, wired=False, india_ready=True,
        note="india_equity engine models T+1, circuits, STT. 462 alpha factors. The spine.",
        blockers=["no adapter yet", "pinned to a bare commit - no release tags"],
    ),
    FleetEntry(
        name="nautilus_trader", kind="engine", version="1.231.0",
        capabilities=[Capability.BACKTEST, Capability.EXECUTION, Capability.RISK],
        asset_classes=[AssetClass.CRYPTO, AssetClass.FX, AssetClass.EQUITY],
        horizons=[Horizon.INTRADAY, Horizon.SWING, Horizon.POSITION],
        licence="LGPL-3.0", installed=True, verified=True, wired=False, india_ready=False,
        note="Event-driven truth gate. Same code path backtest/paper/live.",
        blockers=["no Indian broker adapter - simulation only", "pandas must stay < 3"],
    ),
    FleetEntry(
        name="tradingagents", kind="agent", version="0.4.1",
        capabilities=[Capability.FUNDAMENTAL, Capability.SENTIMENT, Capability.TECHNICAL],
        asset_classes=[AssetClass.EQUITY],
        horizons=[Horizon.SWING, Horizon.POSITION],
        licence="Apache-2.0", installed=False, verified=False, wired=False, india_ready=True,
        note="12-agent bull/bear debate. Has .NS/.BO benchmarks already.",
        blockers=["NOT INSTALLED", "US-shaped news and fundamentals",
                  "analysts run in series - slow"],
    ),
    FleetEntry(
        name="freqtrade", kind="bot", version="2026.8",
        capabilities=[Capability.SCREENING, Capability.BACKTEST, Capability.EXECUTION],
        asset_classes=[AssetClass.CRYPTO],
        horizons=[Horizon.INTRADAY, Horizon.SWING],
        licence="GPL-3.0", installed=True, verified=True, wired=False, india_ready=False,
        note="Crypto only. Excluded from the India pipeline by design.",
        blockers=["WRONG MARKET for NSE/BSE", "GPL - subprocess only, never import"],
    ),
    FleetEntry(
        name="india_strategies", kind="strategy", version="0.2.0",
        capabilities=[Capability.TECHNICAL, Capability.QUANT],
        asset_classes=[AssetClass.EQUITY, AssetClass.INDEX],
        horizons=[Horizon.INTRADAY, Horizon.SWING],
        licence="own", installed=True, verified=True, wired=False, india_ready=True,
        note="Supertrend+ADX, Bollinger+RSI, Donchian, Pairs, ORB, ML. Audited; 4 defects fixed.",
        blockers=["needs re-run on corrected strategy_lib",
                  "pairs hedge ratio still fitted in-sample (BUG-03)",
                  "ORB parked until intraday data exists"],
    ),
]


def fleet_status() -> list[dict]:
    out = []
    for e in FLEET:
        d = asdict(e)
        d["capabilities"] = [c.value for c in e.capabilities]
        d["asset_classes"] = [a.value for a in e.asset_classes]
        d["horizons"] = [h.value for h in e.horizons]
        d["status"] = (
            "wired" if e.wired else
            "verified" if e.verified else
            "installed" if e.installed else "absent"
        )
        out.append(d)
    return out
