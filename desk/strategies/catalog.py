"""The six shared strategies, described as data rather than prose.

These came in as standalone scripts. A script is not a strategy the desk can
reason about: the coordinator needs to know what each one needs, what regime it
belongs in, what is still wrong with it, and - above all - whether its reported
numbers are trustworthy yet. So each one gets a record here, and `trusted` is
false until it has been re-run on the corrected library and survived
walk-forward.

Nothing in this file runs a strategy. It is the manifest the scanner and the
plan builder read to decide who is even eligible to speak on a given day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from desk.contracts.enums import AssetClass, Horizon, Regime


class Family(str, Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    STAT_ARB = "stat_arb"
    INTRADAY = "intraday"
    LEARNED = "learned"


class Maturity(str, Enum):
    """How far a strategy has come along the pipeline. Never skip a rung."""
    DRAFT = "draft"                 # code exists, nothing verified
    AUDITED = "audited"             # read line by line, defects listed
    REVALIDATED = "revalidated"     # re-run on the corrected library
    WALK_FORWARD = "walk_forward"   # survives out-of-sample windows
    PAPER = "paper"                 # traded on paper against live data
    LIVE = "live"                   # not until everything above is true


@dataclass(frozen=True, slots=True)
class StrategySpec:
    key: str
    name: str
    family: Family
    horizon: Horizon
    asset_classes: list[AssetClass]
    thesis: str                     # why this should make money, in one line
    params: dict[str, float | int | str]
    data_needed: list[str]
    favourable_regimes: list[Regime]
    hostile_regimes: list[Regime]
    maturity: Maturity
    defects: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def trusted(self) -> bool:
        """Is this allowed to influence a live plan?

        Only once it has cleared walk-forward. Anything below that rung may
        appear on screen, clearly labelled, and may not be sized.
        """
        return self.maturity in (Maturity.WALK_FORWARD, Maturity.PAPER, Maturity.LIVE)


CATALOG: list[StrategySpec] = [
    StrategySpec(
        key="supertrend_adx",
        name="Supertrend + ADX filter",
        family=Family.TREND,
        horizon=Horizon.SWING,
        asset_classes=[AssetClass.EQUITY, AssetClass.INDEX],
        thesis="Ride a confirmed trend; refuse to trade when ADX says there is none.",
        params={"st_period": 10, "st_multiplier": 3.0, "adx_period": 14, "adx_min": 20},
        data_needed=["daily OHLC", "corporate-action adjusted"],
        favourable_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN],
        hostile_regimes=[Regime.RANGE, Regime.HIGH_VOL],
        maturity=Maturity.AUDITED,
        defects=[
            "BUG-01 (fixed in lib): trade P&L was understated by the exit-bar slice",
            "BUG-02 (fixed in lib): ADX was SMA-smoothed, so signals fired on "
            "different bars than the TradingView chart being watched",
        ],
        notes="The most defensible of the six. Whipsaw in a range is the whole risk.",
    ),
    StrategySpec(
        key="bollinger_rsi",
        name="Bollinger band + RSI reversion",
        family=Family.MEAN_REVERSION,
        horizon=Horizon.SWING,
        asset_classes=[AssetClass.EQUITY],
        thesis="Stretched price snaps back, but only when momentum agrees it is stretched.",
        params={"bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
                "rsi_oversold": 30, "rsi_overbought": 70},
        data_needed=["daily OHLC", "corporate-action adjusted"],
        favourable_regimes=[Regime.RANGE],
        hostile_regimes=[Regime.TRENDING_DOWN, Regime.CRISIS],
        maturity=Maturity.AUDITED,
        defects=[
            "BUG-02 (fixed in lib): RSI was SMA-smoothed, not Wilder",
            "No regime gate: as written it buys every step of a downtrend",
        ],
        notes="Mean reversion in a crisis is how accounts die. This one needs the "
              "regime engine before it goes anywhere near a live plan.",
    ),
    StrategySpec(
        key="donchian_breakout",
        name="Donchian channel breakout",
        family=Family.BREAKOUT,
        horizon=Horizon.POSITION,
        asset_classes=[AssetClass.EQUITY, AssetClass.INDEX],
        thesis="New highs beget new highs; cut the ones that fail immediately.",
        params={"entry_lookback": 20, "exit_lookback": 10, "atr_period": 14,
                "atr_stop_mult": 2.0},
        data_needed=["daily OHLC", "corporate-action adjusted"],
        favourable_regimes=[Regime.TRENDING_UP],
        hostile_regimes=[Regime.RANGE],
        maturity=Maturity.AUDITED,
        defects=[
            "Unadjusted data turns every split into a fake breakdown",
            "Low win rate by construction - needs a large trade count to judge at all",
        ],
        notes="Cheapest of the six to validate, because the rule has almost no "
              "parameters to overfit.",
    ),
    StrategySpec(
        key="pairs_trading",
        name="Cointegrated pairs (stat arb)",
        family=Family.STAT_ARB,
        horizon=Horizon.SWING,
        asset_classes=[AssetClass.EQUITY],
        thesis="Two shares that share a driver should not drift apart for long.",
        params={"lookback": 60, "entry_z": 2.0, "exit_z": 0.5, "adf_p_max": 0.05},
        data_needed=["daily close for both legs", "borrow / F&O availability for the short"],
        favourable_regimes=[Regime.RANGE, Regime.TRENDING_UP],
        hostile_regimes=[Regime.CRISIS],
        maturity=Maturity.DRAFT,
        defects=[
            "BUG-03 STILL OPEN: the hedge ratio is fitted on the whole sample, so "
            "every backtested spread is built from data the trade could not have "
            "seen. The results are not yet meaningful.",
            "Retail cannot short cash equity beyond intraday - needs F&O legs or MIS",
            "Cointegration breaks silently; there is no re-test schedule",
        ],
        notes="The most attractive on paper and the least usable in an Indian retail "
              "account. Fix the rolling hedge ratio before believing any number here.",
    ),
    StrategySpec(
        key="opening_range_breakout",
        name="Opening range breakout (ORB)",
        family=Family.INTRADAY,
        horizon=Horizon.INTRADAY,
        asset_classes=[AssetClass.EQUITY, AssetClass.INDEX],
        thesis="The first fifteen minutes set the day's reference; a clean break of it runs.",
        params={"range_minutes": 15, "atr_stop_mult": 1.0, "square_off": "15:15"},
        data_needed=["1-minute or 5-minute intraday bars", "NSE session calendar"],
        favourable_regimes=[Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.HIGH_VOL],
        hostile_regimes=[Regime.RANGE],
        maturity=Maturity.DRAFT,
        defects=[
            "PARKED: there is no intraday data in the system, so it cannot be validated",
            "Intraday costs and slippage dominate; the daily cost model does not apply",
        ],
        notes="Parked deliberately. Running it on daily bars would produce a number "
              "that means nothing, which is worse than no number.",
    ),
    StrategySpec(
        key="ml_classifier",
        name="ML directional classifier",
        family=Family.LEARNED,
        horizon=Horizon.SWING,
        asset_classes=[AssetClass.EQUITY],
        thesis="A model over many weak technical features beats any one of them alone.",
        params={"features": "returns, RSI, ATR, volume z-score", "horizon_days": 5,
                "split": "walk-forward expanding"},
        data_needed=["daily OHLCV", "point-in-time index membership (survivorship)"],
        favourable_regimes=[Regime.TRENDING_UP, Regime.RANGE],
        hostile_regimes=[Regime.CRISIS],
        maturity=Maturity.DRAFT,
        defects=[
            "BUG-04: a random train/test split leaks the future into training. "
            "It must be an expanding walk-forward window.",
            "No survivorship-free universe, so the label set is biased upward",
            "No IC or feature-importance study - the edge is entirely unexplained",
        ],
        notes="The least trustworthy of the six until its validation is rebuilt. "
              "An unexplained edge is usually a leak.",
    ),
]

BY_KEY: dict[str, StrategySpec] = {s.key: s for s in CATALOG}


def catalog_status() -> list[dict]:
    """JSON-safe view for the API."""
    out = []
    for s in CATALOG:
        d = asdict(s)
        d["family"] = s.family.value
        d["horizon"] = s.horizon.value
        d["maturity"] = s.maturity.value
        d["asset_classes"] = [a.value for a in s.asset_classes]
        d["favourable_regimes"] = [r.value for r in s.favourable_regimes]
        d["hostile_regimes"] = [r.value for r in s.hostile_regimes]
        d["params"] = {k: str(v) for k, v in s.params.items()}
        d["trusted"] = s.trusted
        out.append(d)
    return out


def eligible(regime: Regime, *, trusted_only: bool = True) -> list[StrategySpec]:
    """Who is allowed to speak in this regime.

    A strategy that names the current regime as hostile is silenced, not
    down-weighted. A mean-reversion rule in a crisis is not a weak opinion,
    it is a wrong one.
    """
    return [
        s for s in CATALOG
        if regime not in s.hostile_regimes and (s.trusted or not trusted_only)
    ]
