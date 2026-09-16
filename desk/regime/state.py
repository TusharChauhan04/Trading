"""Regime as a vector, because a market is several things at once.

The design has always described regime along six dimensions - trend,
volatility, breadth, sector rotation, risk appetite and event proximity - but
the code carried a single `Regime` enum, which cannot say "trending up *and*
high-volatility *and* narrow breadth". That is not a cosmetic gap: strategy
gating, position sizing and regime-sliced performance all want different
dimensions, and collapsing them early throws away the distinction.

`RegimeState` is the vector. `Regime` survives as a derived summary label,
because the daily plan, the strategy catalog and the journal all want one word
to slice on. The derivation is deliberately explicit and boring - no model, no
weights to tune, just a documented precedence order you can argue with.

Nothing here computes a regime from market data yet; that is the regime engine,
and it needs India VIX, Nifty breadth and a sectoral-index feed. This module
defines the shape so everything downstream can be written against it now.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from desk.contracts.enums import Breadth, Regime, RiskAppetite, Trend, VolState


class EventProximity(BaseModel):
    """Scheduled events inside the intended holding window.

    Carried separately from the other dimensions because it is the only one
    that can force NO TRADE on its own, regardless of how good the setup is.
    """
    rbi_policy_days: int | None = None
    budget_days: int | None = None
    expiry_days: int | None = None
    results_season: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def blocks_new_risk(self) -> bool:
        """An event close enough that a fresh position is a coin flip on it."""
        for d in (self.rbi_policy_days, self.budget_days):
            if d is not None and 0 <= d <= 1:
                return True
        return False


class RegimeState(BaseModel):
    """The six dimensions, each independently `UNKNOWN` until measured."""

    as_of: date
    trend: Trend = Trend.UNKNOWN
    volatility: VolState = VolState.UNKNOWN
    breadth: Breadth = Breadth.UNKNOWN
    risk_appetite: RiskAppetite = RiskAppetite.UNKNOWN
    leading_sectors: list[str] = Field(default_factory=list)
    lagging_sectors: list[str] = Field(default_factory=list)
    events: EventProximity = Field(default_factory=EventProximity)

    # What each dimension was measured from, so a regime call can be audited.
    sources: dict[str, str] = Field(default_factory=dict)

    @property
    def is_measured(self) -> bool:
        """False while any dimension is still unknown. The plan must say so
        rather than presenting a half-measured regime as a finding."""
        return (
            self.trend is not Trend.UNKNOWN
            and self.volatility is not VolState.UNKNOWN
            and self.breadth is not Breadth.UNKNOWN
            and self.risk_appetite is not RiskAppetite.UNKNOWN
        )

    @property
    def risk_off(self) -> bool:
        """Feeds the risk engine's market_risk_off flag."""
        return (
            self.risk_appetite is RiskAppetite.RISK_OFF
            or self.volatility is VolState.EXTREME
        )

    @property
    def max_concurrent_positions_hint(self) -> int | None:
        """Narrow breadth means fewer things are genuinely working, so holding
        five positions is really holding one idea five times."""
        if self.breadth is Breadth.NARROW:
            return 2
        if self.breadth is Breadth.BROAD:
            return None          # no extra restriction beyond the config
        return None

    @property
    def label(self) -> Regime:
        """Collapse to one word. Precedence is deliberate and ordered by what
        would hurt you most if ignored, not by what is most common."""
        if self.volatility is VolState.EXTREME or self.risk_appetite is RiskAppetite.RISK_OFF:
            return Regime.CRISIS
        if self.volatility is VolState.ELEVATED:
            return Regime.HIGH_VOL
        if self.trend in (Trend.STRONG_UP, Trend.UP):
            return Regime.TRENDING_UP
        if self.trend in (Trend.STRONG_DOWN, Trend.DOWN):
            return Regime.TRENDING_DOWN
        if self.trend is Trend.FLAT:
            return Regime.RANGE
        return Regime.UNKNOWN

    def explain(self) -> list[str]:
        """Human-readable, one line per dimension. The regime is an input to
        every downstream decision, so it has to be arguable."""
        out = [
            f"trend: {self.trend.value}",
            f"volatility: {self.volatility.value}",
            f"breadth: {self.breadth.value}",
            f"risk appetite: {self.risk_appetite.value}",
        ]
        if self.leading_sectors:
            out.append("leading: " + ", ".join(self.leading_sectors))
        if self.events.blocks_new_risk:
            out.append("a scheduled event blocks new risk today")
        if not self.is_measured:
            out.append("NOT FULLY MEASURED - dimensions above marked unknown "
                       "have no feed behind them yet")
        return out
