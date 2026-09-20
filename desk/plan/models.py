"""The daily plan's shape. Moved out of desk/api/main.py.

Why this had to move: the API route handler and the plan-building logic used
to share one file, and desk/api/main.py imports FastAPI and its CORS
middleware at module scope. The scanner (the next real consumer of
`DailyPlan` - it needs the model to report universe_scanned/survived_stageN)
would otherwise have had to import a transport-layer module to reach a data
model, dragging FastAPI into a batch job, or duplicate the model. Neither is
acceptable, so the model lives here and `desk/api/main.py` imports it like
any other domain type.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from desk.contracts.enums import Regime, Stance
from desk.regime.state import RegimeState


class PlanTrade(BaseModel):
    symbol: str
    stance: Stance
    confidence: float
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    qty: int = 0
    capital_at_risk: float = 0.0
    supporting: list[str] = Field(default_factory=list)
    dissenting: list[str] = Field(default_factory=list)
    rationale: str = ""

    stop_basis: str = ""
    """WHAT THE STOP IS ANCHORED TO, as one token: "swing_low", "low_20",
    "noise_floor" or "atr". Machine-readable on purpose - it is the field
    the backtest groups by to answer whether structural stops actually
    survive better than volatility ones, which prose cannot be grouped by.

    Empty on a trade recorded before stops had a basis, and on the
    watchlist/avoid entries, which were never sized."""

    invalidation: str = ""
    """The same thing in a sentence: the price level that would prove the
    setup wrong, and why that level. This is the master spec's "Trade
    Invalidation" line, and the reason the stop is defensible rather than
    arbitrary - a reader who disagrees with the level can say so, which is
    not possible with "2 x ATR"."""


class ScanSummary(BaseModel):
    """What a funnel run produced, in primitives.

    Deliberately NOT a scanner type. `desk.scanner.stage4` imports `PlanTrade`
    from this module, so if `desk.plan` imported the scanner back the two
    packages would be circular. Keeping the handoff to plain counts plus
    already-built `PlanTrade` objects fixes the dependency direction at
    scanner -> plan and keeps `build_plan` a pure transform with no
    filesystem access of its own.
    """
    universe_scanned: int = 0
    survived_stage0: int = 0
    survived_stage1: int = 0
    survived_stage2: int = 0
    considered: int = 0
    trades: list["PlanTrade"] = Field(default_factory=list)
    no_trade_reason: str | None = None
    vetoed: dict[str, str] = Field(default_factory=dict)
    """Names Stage 3 removed, symbol -> the reason it gave.

    Carried through because it was being COMPUTED AND DISCARDED. On a day
    where 2 trades are approved out of 8 shortlisted, the plan showed the
    2 and nothing else - no way to see which 6 the narrative pass vetoed
    or why. This codebase is otherwise disciplined about the difference
    between "not checked" and "checked and rejected"; this was the one
    place it dropped it."""
    rejected: list[tuple[str, str]] = Field(default_factory=list)
    """(symbol, reason) for every name the RISK GATE refused. Same
    reasoning: `Stage4Result.rejected` names each one with a
    RejectReason, and none of it reached the reader."""
    regime_state: "RegimeState | None" = None
    """The regime the scan ACTUALLY measured.

    Carried through because build_plan used to construct a fresh empty
    RegimeState and show that instead - so every plan reported "unknown"
    on all six dimensions and a note saying the engine was not built,
    while the funnel had just measured range / flat / narrow and used it
    to silence factors. The plan displayed one regime and the scan acted
    on another."""
    caveats: list[str] = Field(default_factory=list)
    """Everything the funnel declared it could NOT check - Stage 0's missing
    F&O ban list, Stage 1's absent sector and news feeds, Stage 2's silenced
    strategy votes, Stage 4's unenforceable sector cap. These become plan
    warnings, because a plan that hides what it could not check reads more
    confident than it is."""
    coverage_note: str | None = None
    """How complete the price history behind the scan actually was."""


class DailyPlan(BaseModel):
    as_of: date
    regime: Regime
    regime_note: str = ""
    regime_detail: list[str] = Field(default_factory=list)
    """The six dimensions in words - see desk.regime.state.RegimeState.explain."""

    universe_scanned: int = 0
    survived_stage0: int = 0
    survived_stage1: int = 0
    survived_stage2: int = 0
    analysed: int = 0
    trades: list[PlanTrade] = Field(default_factory=list)
    watchlist: list[PlanTrade] = Field(default_factory=list)
    avoid: list[PlanTrade] = Field(default_factory=list)
    no_trade_reason: str | None = None

    market_risks: list[str] = Field(default_factory=list)
    """Events, news and data affecting the TRADING DAY - results, policy,
    expiry, global moves. Deliberately separate from `warnings`: one is about
    the market, the other is about the system, and conflating them buries
    whichever matters more today."""

    warnings: list[str] = Field(default_factory=list)
    """System-health caveats: missing feeds, unvalidated strategies, stale data."""

    @property
    def is_no_trade(self) -> bool:
        return not self.trades
