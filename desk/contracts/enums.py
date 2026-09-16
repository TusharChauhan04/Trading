"""Shared vocabulary. Every agent, strategy and engine speaks these."""

from __future__ import annotations
from enum import Enum


class Stance(str, Enum):
    """The 5-tier rating, lifted from TradingAgents so the whole fleet agrees.

    REVIEW is deliberate: an unparseable or low-confidence result must never
    be silently downgraded to HOLD, because HOLD is a tradeable neutral and
    REVIEW is a demand for a human. See ADR-001.
    """
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"
    REVIEW = "REVIEW"

    @property
    def is_actionable(self) -> bool:
        return self in (Stance.BUY, Stance.OVERWEIGHT, Stance.UNDERWEIGHT, Stance.SELL)

    @property
    def direction(self) -> int:
        """+1 long, -1 short, 0 no position."""
        return {
            Stance.BUY: 1, Stance.OVERWEIGHT: 1,
            Stance.SELL: -1, Stance.UNDERWEIGHT: -1,
        }.get(self, 0)


class Horizon(str, Enum):
    INTRADAY = "intraday"
    SWING = "swing"          # days to ~2 weeks
    POSITION = "position"    # weeks to months
    LONG_TERM = "long_term"


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FX = "fx"
    COMMODITY = "commodity"
    INDEX = "index"


class Capability(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    NEWS = "news"                # distinct from sentiment: facts vs mood
    MACRO = "macro"
    QUANT = "quant"
    PATTERN = "pattern"          # chart / candlestick structure recognition
    SCREENING = "screening"
    REGIME = "regime"
    PORTFOLIO = "portfolio"
    RESEARCH = "research"        # qualitative synthesis
    RISK = "risk"
    BACKTEST = "backtest"
    EXECUTION = "execution"


class Regime(str, Enum):
    """The SUMMARY label only.

    Regime is genuinely multi-dimensional - a market can be trending up *and*
    high-volatility *and* narrow-breadth at once, and a single label cannot say
    so. See `desk.regime.state.RegimeState` for the six-dimensional vector this
    is derived from. This enum survives because strategy gating, the daily plan
    and the journal all want one word to slice on.
    """
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


class Trend(str, Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    FLAT = "flat"
    DOWN = "down"
    STRONG_DOWN = "strong_down"
    UNKNOWN = "unknown"


class VolState(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class Breadth(str, Enum):
    BROAD = "broad"              # most of the market participating
    NEUTRAL = "neutral"
    NARROW = "narrow"            # a handful of names carrying the index
    UNKNOWN = "unknown"


class RiskAppetite(str, Enum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    UNKNOWN = "unknown"


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


class RejectReason(str, Enum):
    """Why the risk engine refused. Every rejection is named, never silent."""
    STOP_TOO_WIDE = "stop_too_wide"
    QTY_ROUNDS_TO_ZERO = "qty_rounds_to_zero"
    POSITION_VALUE_CAP = "position_value_cap"
    SECTOR_CAP = "sector_cap"
    PORTFOLIO_HEAT_CAP = "portfolio_heat_cap"
    CORRELATED_CAP = "correlated_cap"
    RR_TOO_LOW = "risk_reward_too_low"
    ILLIQUID = "illiquid_vs_adv"
    DAILY_LOSS_CAP = "daily_loss_cap"
    MAX_POSITIONS = "max_open_positions"
    NOT_TRADEABLE = "not_tradeable"     # ban / circuit / suspended
    STALE_DATA = "stale_data"
    MALFORMED_INPUT = "malformed_input"   # NaN, inf, or a nonsense magnitude
    SLIPPAGE_TOO_HIGH = "slippage_too_high"
    EXTREME_VOLATILITY = "extreme_volatility"
    MARKET_RISK_OFF = "market_risk_off"
    EVENT_IN_WINDOW = "event_in_window"   # results/corporate action pending
