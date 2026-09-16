"""The envelope every agent fills in. This IS the integration strategy.

No agent imports another agent. Everything crosses this boundary, which is
what keeps N repositories from fusing into one codebase.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .enums import AssetClass, Capability, Horizon, Regime, Stance


class Evidence(BaseModel):
    """One claim, with provenance. Required so a decision can be audited later
    and so the backtest harness can detect look-ahead by inspecting `as_of`."""
    claim: str
    source: str                       # "nse_bhavcopy" | "kite" | "screener.in" ...
    as_of: datetime                   # when this fact became knowable
    url: str | None = None
    value: float | None = None


class CostEstimate(BaseModel):
    tokens: int = 0
    wall_clock_s: float = 0.0
    inr: float = 0.0


class AnalysisRequest(BaseModel):
    symbol: str                       # canonical form, see marketdata.symbols
    as_of: date                       # the simulated clock. NEVER "today" in a backtest.
    horizon: Horizon
    asset_class: AssetClass = AssetClass.EQUITY
    regime: Regime = Regime.UNKNOWN
    task: str = "analyse"             # what the coordinator is asking for
    constraints: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    budget: CostEstimate | None = None


class PriceZone(BaseModel):
    """An entry is a RANGE with a trigger, never a single number.

    A scalar entry produces exactly the vague output the brief rejects -
    "buy XYZ, it looks strong". A zone plus a trigger produces "only above
    1,432 on above-average volume".
    """
    low: float = Field(gt=0)
    high: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "PriceZone":
        if self.low > self.high:
            # Read BOTH before writing either. Writing low first and then
            # reading it back collapses the zone to a point at the lower
            # bound, which is silent and survives into the derived entry.
            lo, hi = self.high, self.low
            self.low, self.high = lo, hi
        return self

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0


class Target(BaseModel):
    """T1/T2/T3 with partial exits. `fraction` is the share of the position
    taken off at this level; the remainder rides to the next one."""
    price: float = Field(gt=0)
    fraction: float = Field(default=1.0, gt=0, le=1.0)
    rationale: str = ""


class RiskNote(BaseModel):
    """The agent's OWN view of what could go wrong. This is not the risk
    engine's arithmetic - it is qualitative context the engine cannot compute,
    and it may never relax a gate."""
    concerns: list[str] = Field(default_factory=list)
    liquidity_note: str = ""
    event_risk: str = ""              # results, policy, expiry in the window
    max_adverse_expected_pct: float | None = None


class AnalysisResult(BaseModel):
    """What every agent returns. Producers that cannot fill a field leave it None
    rather than guessing."""

    agent: str
    symbol: str
    as_of: date                       # the DATA date this was computed against
    produced_at: datetime = Field(default_factory=datetime.now)
    """Wall-clock. Distinct from as_of, which is the simulated clock - conflating
    the two is how a backtest quietly reads the present."""

    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: Horizon

    setup_type: str = ""              # "pullback to 20DMA", "range breakout", ...
    entry: float | None = None        # the reference level
    entry_zone: PriceZone | None = None
    trigger: str = ""                 # the objective event that activates it
    stop: float | None = None
    target: float | None = None       # kept: T1, or the only target
    targets: list[Target] = Field(default_factory=list)
    expected_holding_days: int | None = None
    invalidations: list[str] = Field(default_factory=list)
    """What makes the thesis wrong BEFORE the stop is hit. Distinct from the
    stop: the stop is where you are wrong about price, an invalidation is where
    you are wrong about the reason."""

    narrative: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    risk: RiskNote = Field(default_factory=RiskNote)
    cost: CostEstimate = Field(default_factory=CostEstimate)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> "AnalysisResult":
        # Plain assignment, NOT object.__setattr__. The model is not frozen, so
        # the bypass bought nothing and cost correctness: it skips pydantic's
        # fields-set tracking, so any producer serialising with
        # `exclude_unset=True` would silently drop every value derived here.
        # On the one envelope that crosses every agent boundary, that is the
        # worst possible place for a value to vanish.

        # A REVIEW is a demand for a human, so it must not carry conviction.
        if self.stance is Stance.REVIEW and self.confidence > 0.0:
            self.confidence = 0.0

        # Keep the scalar and the structured forms in step, so callers that
        # only understand one of them still see the truth.
        if self.entry is None and self.entry_zone is not None:
            self.entry = self.entry_zone.mid
        if self.target is None and self.targets:
            self.target = self.targets[0].price

        if self.targets:
            total = sum(t.fraction for t in self.targets)
            if total > 1.0 + 1e-9:
                self.errors.append(
                    f"partial exits sum to {total:.2f} of the position")

        # Every level a producer can set - entry, stop, target, and each entry
        # of targets[] - must be a real, finite number before anything below
        # compares it. Every comparison here is False for NaN, so an
        # unfiltered NaN would fall through every check and leave
        # is_tradeable=True on a setup with no usable price in it - exactly
        # the failure size_position() guards against on its side of this same
        # boundary, and this is the one place upstream of it that can let it
        # through unfiltered.
        for label, v in (("entry", self.entry), ("stop", self.stop),
                         ("target", self.target)):
            if v is not None and not math.isfinite(v):
                self.errors.append(f"{label} is {v!r}, not a finite number")
        for i, t in enumerate(self.targets):
            if not math.isfinite(t.price):
                self.errors.append(f"targets[{i}] is {t.price!r}, not a finite number")

        # A directional call with levels must have them the right way round,
        # otherwise position sizing silently produces a negative stop distance.
        if self.entry is not None and self.stop is not None \
                and math.isfinite(self.entry) and math.isfinite(self.stop):
            d = self.stance.direction
            if d > 0 and self.stop >= self.entry:
                self.errors.append("long setup has stop at or above entry")
            if d < 0 and self.stop <= self.entry:
                self.errors.append("short setup has stop at or below entry")

        # The target needs the same check. Without it an inverted target still
        # produces a plausible POSITIVE risk:reward, because both this model
        # and the risk engine take abs(target - entry) - so a long whose target
        # sits below its stop reads as a clean 2:1 setup and clears the floor.
        if self.entry is not None and self.target is not None \
                and math.isfinite(self.entry) and math.isfinite(self.target):
            d = self.stance.direction
            if d > 0 and self.target <= self.entry:
                self.errors.append("long setup has target at or below entry")
            if d < 0 and self.target >= self.entry:
                self.errors.append("short setup has target at or above entry")

        # And every T1/T2/T3 individually - a partial exit priced on the wrong
        # side of entry is exactly as dangerous as a single wrong target, and
        # was previously invisible because only the scalar `target` was checked.
        if self.entry is not None and math.isfinite(self.entry):
            d = self.stance.direction
            for i, t in enumerate(self.targets):
                if not math.isfinite(t.price):
                    continue
                if d > 0 and t.price <= self.entry:
                    self.errors.append(f"targets[{i}] ({t.price}) is at or "
                                       f"below entry on a long")
                if d < 0 and t.price >= self.entry:
                    self.errors.append(f"targets[{i}] ({t.price}) is at or "
                                       f"above entry on a short")

        # A `mode="after"` validator that appends is not idempotent by
        # construction: `AnalysisResult.model_validate(r.model_dump())` - the
        # natural JSON round-trip across an agent boundary - re-runs this
        # validator, so an unguarded append grows the list on every pass while
        # `is_tradeable` stays correct. Dedupe on the way out rather than
        # guarding every append site individually.
        if self.errors:
            self.errors = list(dict.fromkeys(self.errors))
        return self

    @property
    def risk_reward(self) -> float | None:
        if None in (self.entry, self.stop, self.target):
            return None
        risk = abs(self.entry - self.stop)
        if risk == 0:
            return None
        return abs(self.target - self.entry) / risk

    @property
    def is_tradeable(self) -> bool:
        return (
            self.stance.is_actionable
            and not self.errors
            and self.entry is not None
            and self.stop is not None
        )


class AgentManifest(BaseModel):
    """What the registry reads to decide who runs. Honest declaration is the
    price of admission - the coordinator cannot route on capabilities that lie."""
    name: str
    kind: str                          # agent | engine | bot | hybrid | strategy
    version: str
    capabilities: set[Capability]
    horizons: set[Horizon]
    asset_classes: set[AssetClass]
    cost_per_call: CostEstimate = Field(default_factory=CostEstimate)
    licence: str | None = None
    notes: str = ""
