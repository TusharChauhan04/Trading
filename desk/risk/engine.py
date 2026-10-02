"""Deterministic risk engine. No LLM touches this file, ever.

An LLM may argue for a trade. It may never change the arithmetic that decides
how much of your capital reaches a broker. Every gate below either passes or
names its reason for refusing - there is no "close enough" path.

This is a pure function of (setup, portfolio, config). Same inputs, same
answer, every time. That is what makes it testable, and testability is the
only reason to trust it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from desk.contracts.enums import RejectReason


class RiskConfig(BaseModel):
    """YOUR numbers. Nothing here is a recommendation - set them deliberately."""

    # Pydantic v2 accepts inf and NaN for float by default, and JSON's 1e400
    # parses to inf. That reached `math.floor` and raised OverflowError,
    # breaking the engine's "returns, never raises" contract - and with a large
    # but FINITE capital it did something worse, returning approved=True with a
    # 301-digit quantity. `approved` is the only field execution may read, so a
    # physically impossible size carrying it is an integrity failure.
    model_config = ConfigDict(allow_inf_nan=False)

    capital: float = Field(gt=0, le=1e12)
    risk_pct: float = Field(default=1.0, gt=0, le=5.0)          # % of capital per trade
    max_position_pct: float = Field(default=15.0, gt=0, le=100) # % of capital in one name
    max_sector_pct: float = Field(default=30.0, gt=0, le=100)
    max_open_risk_pct: float = Field(default=4.0, gt=0, le=20)  # portfolio heat
    max_correlated_pct: float = Field(default=25.0, gt=0, le=100)
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=20)
    max_drawdown_pct: float | None = Field(default=None, gt=0, le=90)
    """Peak-to-trough equity drawdown at which to stop opening positions.

    DELIBERATELY None BY DEFAULT, and not out of caution about the gate -
    there is no honest default, because this number and `risk_pct` are one
    decision. From `drawdown_outlook` on the donchian 12%/20-session
    sequence over 200 trades:

        0.5% risk   median DD  8.4%   p95 18.2%   P(DD>20%)  2.5%
        1.0% risk   median DD 16.2%   p95 33.2%   P(DD>20%) 32.6%
        2.0% risk   median DD 30.1%   p95 55.7%   P(DD>35%) 35.9%

    A 20% cap at the default 1% risk fires about ONE RUN IN THREE while
    the strategy is behaving exactly as measured. That is not a circuit
    breaker; it is a coin flip that halts trading, and it would halt it
    during the ordinary drawdown that precedes recovery. The same 20% cap
    at 0.5% risk fires 2.5% of the time, which is a genuine tail.

    So pick the pair, never the number: run `drawdown_outlook` on your own
    trade sequence and set this above the p95 for the `risk_pct` you intend
    to use. Left None, the gate is SKIPPED and says so - the same
    no-claim-no-gate rule the correlated cap follows.

    WHY IT IS NEEDED AT ALL: `max_daily_loss_pct` reads only
    `realised_pnl_today`, so it resets every morning. Losing 2% a day for
    ten sessions is an 18% drawdown that never once trips a 3% daily cap.
    """
    max_open_positions: int = Field(default=5, ge=1, le=50)
    min_risk_reward: float = Field(default=1.5, gt=0)
    max_adv_participation_pct: float = Field(default=1.0, gt=0, le=25)
    max_stop_distance_pct: float = Field(default=15.0, gt=0)    # sanity ceiling
    lot_size: int = Field(default=1, ge=1)                      # 1 for cash equity

    def drawdown_outlook(self, r_multiples, *, n_trades: int = 200,
                         n_paths: int = 2000):
        """What `risk_pct` implies for drawdown, given a real trade sequence.

        Sizing is the one decision this config makes that cannot be undone by a
        later gate, and `risk_pct` defaults to 1.0 without anything in the
        config knowing what that costs. This answers it from measured trades
        rather than from a rule of thumb.

        MEASURED on the donchian 12%/20-session sequence, 200 trades:

            0.5% risk   median DD  8.4%   p95 18.2%   P(DD>20%)  2.5%
            1.0% risk   median DD 16.2%   p95 33.2%   P(DD>20%) 32.6%
            2.0% risk   median DD 30.1%   p95 55.7%   P(DD>35%) 35.9%

        At 1% there is roughly a one-in-three chance of a 20% drawdown inside
        six months on an edge of +0.08R a trade. That is not a reason to avoid
        the edge; it is a reason the default is not obviously right for it.

        Uses a BLOCK bootstrap, which keeps losing streaks intact. The IID
        alternative reported 16.8% where this reports 33.2% on the same data -
        see desk.backtest.ruin for why that direction of error is the one that
        matters.
        """
        from desk.backtest.ruin import simulate_paths

        return simulate_paths(r_multiples, n_trades=n_trades,
                              n_paths=n_paths,
                              risk_per_trade_pct=self.risk_pct)

    # --- volatility and execution-quality safeguards -----------------------
    max_slippage_pct: float = Field(default=0.5, gt=0)
    """Slippage as a % of PRICE. Catches an outright illiquid name."""

    max_slippage_share_of_stop_pct: float = Field(default=25.0, gt=0, le=100)
    """Slippage as a % of the STOP DISTANCE. Catches a liquid name whose stop
    is too tight to survive its own fill - the case a price-denominated cap
    waves straight through."""

    max_instrument_atr_pct: float = Field(default=12.0, gt=0)
    """Extreme-volatility protection. ATR as a % of price, above which the
    name is simply not sized regardless of how good the setup looks."""

    vol_reference_atr_pct: float | None = Field(default=None, gt=0)
    """Volatility-adjusted sizing. When set, and an instrument ATR% is supplied,
    the risk budget is scaled by (reference / actual) so a name twice as jumpy
    as the reference gets half the budget. None disables the scaling entirely
    rather than guessing a reference for you."""

    risk_off_exposure_pct: float = Field(default=0.0, ge=0, le=100)
    """Market-wide risk-off protection: the fraction of normal risk permitted
    when the regime engine reports risk-off. 0.0 means stop trading."""


@dataclass(slots=True)
class Position:
    symbol: str
    qty: int
    entry: float
    stop: float
    sector: str = "UNKNOWN"
    beta: float = 1.0
    corr_group: str | None = None
    """Correlation cluster. Deliberately BROADER than sector - "RATE_SENSITIVE"
    spans banks, NBFCs, realty and autos, which all move together on a rate
    surprise no matter what their sector label says. Supplied by the caller
    today; a measured correlation matrix replaces the label once there is
    enough price history to compute one. Falls back to sector when unset."""

    @property
    def cluster(self) -> str:
        return self.corr_group or self.sector

    @property
    def value(self) -> float:
        # abs(): a short is coded with a negative qty in some callers, and a
        # negative exposure would create headroom that does not exist. Exposure
        # and risk are magnitudes - direction lives in the sign of the P&L.
        return abs(self.qty) * self.entry

    @property
    def risk(self) -> float:
        """Rupees lost if the stop is hit."""
        return abs(self.qty) * abs(self.entry - self.stop)


@dataclass(slots=True)
class Portfolio:
    positions: list[Position] = field(default_factory=list)
    realised_pnl_today: float = 0.0
    equity: float | None = None
    """Account equity now. None means the caller did not say, which is
    different from zero and must never be read as a total loss."""
    peak_equity: float | None = None
    """The highest equity this account has reached. Supplied by the caller
    because only the caller knows it - the engine sees one day at a time and
    the journal is what holds the history."""

    @property
    def drawdown_pct(self) -> float | None:
        """Current peak-to-trough drawdown as a positive percentage.

        None when either figure is missing - NOT 0.0, which would read as
        "at a high" and is exactly the state a missing input is not.
        """
        if self.equity is None or self.peak_equity is None:
            return None
        if not (self.peak_equity > 0):
            return None
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100.0
        return max(0.0, dd)

    @property
    def open_risk(self) -> float:
        return sum(p.risk for p in self.positions)

    @property
    def exposure(self) -> float:
        return sum(p.value for p in self.positions)

    def sector_exposure(self, sector: str) -> float:
        return sum(p.value for p in self.positions if p.sector == sector)

    def cluster_exposure(self, cluster: str) -> float:
        """Exposure to everything that would move together against you."""
        return sum(p.value for p in self.positions if p.cluster == cluster)

    def holds(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.positions)


class Sizing(BaseModel):
    """The answer. `approved` is the only field execution is allowed to read."""
    approved: bool
    symbol: str
    qty: int = 0
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    capital_at_risk: float = 0.0
    position_value: float = 0.0
    risk_reward: float | None = None
    reasons: list[RejectReason] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    checks_skipped: list[str] = Field(default_factory=list)
    """Gates that could not run because their input was not supplied. An
    approval with a long list here is weaker than one with an empty list, and
    the caller cannot tell the difference unless it is stated."""

    @property
    def rejected_because(self) -> str:
        return ", ".join(r.value for r in self.reasons) or "-"


def size_position(
    *,
    symbol: str,
    entry: float,
    stop: float,
    target: float | None,
    cfg: RiskConfig,
    portfolio: Portfolio | None = None,
    sector: str = "UNKNOWN",
    corr_group: str | None = None,
    adv_shares: float | None = None,
    atr_pct: float | None = None,
    expected_slippage_pct: float | None = None,
    market_risk_off: bool = False,
    tradeable: bool = True,
    data_as_of: date | None = None,
    today: date | None = None,
    max_staleness_days: int = 5,
) -> Sizing:
    """Size one trade and run every hard gate. Returns, never raises.

    Optional inputs are genuinely optional: a gate whose input is None is
    skipped and said so in `notes`, rather than being silently assumed safe.
    That distinction matters - "not checked" and "checked and passed" are
    different states and the caller deserves to know which it got.
    """

    pf = portfolio or Portfolio()
    out = Sizing(approved=False, symbol=symbol, entry=entry, stop=stop, target=target)

    def reject(reason: RejectReason, note: str = "") -> Sizing:
        out.reasons.append(reason)
        if note:
            out.notes.append(note)
        return out

    # --- gate 0: is this instrument even tradeable today ------------------
    if not tradeable:
        return reject(RejectReason.NOT_TRADEABLE,
                      "F&O ban, circuit, or suspended")

    if data_as_of and today and (today - data_as_of).days > max_staleness_days:
        return reject(RejectReason.STALE_DATA,
                      f"price data is {(today - data_as_of).days}d old")

    # Market-wide risk-off. This is a portfolio-level judgement handed down by
    # the regime engine, and it outranks any individual setup's merits.
    if market_risk_off and cfg.risk_off_exposure_pct <= 0:
        return reject(RejectReason.MARKET_RISK_OFF,
                      "regime is risk-off and risk_off_exposure_pct is 0")

    # Extreme volatility: a name whose daily range is this wide cannot be
    # sized sensibly, because the stop is either inside the noise or so far
    # away that the position rounds to nothing.
    if atr_pct is not None and atr_pct > cfg.max_instrument_atr_pct:
        return reject(RejectReason.EXTREME_VOLATILITY,
                      f"ATR is {atr_pct:.1f}% of price, cap is "
                      f"{cfg.max_instrument_atr_pct}%")

    # --- gate 1: the setup itself is coherent -----------------------------
    # NaN first, because every comparison below is False for NaN and the value
    # would fall through every gate to math.floor(), which raises. This
    # function's contract is that it returns and never raises, so a non-finite
    # input has to be refused by name like anything else.
    for label, v in (("entry", entry), ("stop", stop), ("target", target),
                     ("atr_pct", atr_pct),
                     ("expected_slippage_pct", expected_slippage_pct),
                     ("adv_shares", adv_shares)):
        if v is not None and not math.isfinite(v):
            return reject(RejectReason.MALFORMED_INPUT,
                          f"{label} is {v!r}, which is not a finite number")

    # A supplied-but-useless value must not silently disable its gate. Zero ADV
    # is a suspended scrip, not "liquidity unknown"; zero ATR is a broken feed.
    if adv_shares is not None and adv_shares <= 0:
        return reject(RejectReason.ILLIQUID,
                      f"ADV is {adv_shares:g} - suspended, or a broken feed")
    if atr_pct is not None and atr_pct <= 0:
        return reject(RejectReason.EXTREME_VOLATILITY,
                      f"ATR is {atr_pct:g}% - a broken feed, not a calm stock")

    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return reject(RejectReason.STOP_TOO_WIDE, "stop equals entry")
    if entry <= 0:
        return reject(RejectReason.STOP_TOO_WIDE, "non-positive entry")

    stop_pct = 100.0 * stop_distance / entry
    if stop_pct > cfg.max_stop_distance_pct:
        return reject(RejectReason.STOP_TOO_WIDE,
                      f"stop is {stop_pct:.1f}% away, cap is {cfg.max_stop_distance_pct}%")

    if target is not None:
        rr = abs(target - entry) / stop_distance
        out.risk_reward = round(rr, 2)
        if rr < cfg.min_risk_reward:
            return reject(RejectReason.RR_TOO_LOW,
                          f"R:R {rr:.2f} below minimum {cfg.min_risk_reward}")

    # Slippage is judged against the STOP DISTANCE, not against price. Losing
    # 0.3% to slippage is trivial on a 10% stop and ruinous on a 0.4% one, so
    # a price-denominated cap alone approves exactly the trades it should stop:
    # a tight-stop setup where slippage eats the entire risk budget.
    #
    # Both caps apply. The price cap catches an outright illiquid name; the
    # share cap catches a liquid name with a stop too tight to survive the fill.
    if expected_slippage_pct is not None:
        if expected_slippage_pct > cfg.max_slippage_pct:
            return reject(RejectReason.SLIPPAGE_TOO_HIGH,
                          f"expected slippage {expected_slippage_pct:.2f}% of price "
                          f"exceeds cap {cfg.max_slippage_pct}%")

        slip_share = 100.0 * (expected_slippage_pct / 100.0 * entry) / stop_distance
        if slip_share > cfg.max_slippage_share_of_stop_pct:
            return reject(RejectReason.SLIPPAGE_TOO_HIGH,
                          f"slippage would consume {slip_share:.1f}% of the stop "
                          f"distance, cap is {cfg.max_slippage_share_of_stop_pct}%")
        out.notes.append(f"slippage is {slip_share:.1f}% of the stop distance")

    # --- gate 2: the portfolio has room -----------------------------------
    if len(pf.positions) >= cfg.max_open_positions and not pf.holds(symbol):
        return reject(RejectReason.MAX_POSITIONS,
                      f"{len(pf.positions)} open, cap is {cfg.max_open_positions}")

    daily_loss_cap = cfg.capital * cfg.max_daily_loss_pct / 100.0
    if -pf.realised_pnl_today >= daily_loss_cap:
        return reject(RejectReason.DAILY_LOSS_CAP,
                      f"down {-pf.realised_pnl_today:,.0f} against cap {daily_loss_cap:,.0f}")

    # Peak-to-trough, which the daily cap above cannot see: it resets every
    # morning, so a slow bleed never trips it. Runs only when the caller
    # supplies both a cap and the equity state, because neither is
    # derivable from one day of positions - no claim, no gate.
    dd_now = pf.drawdown_pct
    if cfg.max_drawdown_pct is None:
        out.checks_skipped.append(
            "drawdown: no max_drawdown_pct configured, so peak-to-trough "
            "equity was not checked. The daily loss cap resets each morning "
            "and cannot see a slow bleed.")
    elif dd_now is None:
        out.checks_skipped.append(
            "drawdown: no equity/peak_equity supplied on the portfolio, so "
            f"the {cfg.max_drawdown_pct}% cap could not be applied.")
    elif dd_now >= cfg.max_drawdown_pct:
        return reject(RejectReason.DRAWDOWN_CAP,
                      f"down {dd_now:.1f}% from a peak of "
                      f"{pf.peak_equity:,.0f}, cap is "
                      f"{cfg.max_drawdown_pct}%")

    # --- the actual sizing -------------------------------------------------
    risk_pct = cfg.risk_pct

    # Volatility-adjusted size: a name twice as jumpy as the reference gets
    # half the budget. Only scales DOWN - a quiet stock does not earn extra
    # risk, because low realised volatility is not the same as low danger.
    if cfg.vol_reference_atr_pct is not None and atr_pct is not None and atr_pct > 0:
        scale = min(1.0, cfg.vol_reference_atr_pct / atr_pct)
        if scale < 1.0:
            risk_pct *= scale
            out.notes.append(
                f"risk scaled to {scale:.2f}x for {atr_pct:.1f}% ATR "
                f"(reference {cfg.vol_reference_atr_pct}%)")

    if market_risk_off:
        risk_pct *= cfg.risk_off_exposure_pct / 100.0
        out.notes.append(
            f"risk-off regime: risk cut to {cfg.risk_off_exposure_pct}% of normal")

    risk_budget = cfg.capital * risk_pct / 100.0
    raw_qty = risk_budget / stop_distance
    qty = int(math.floor(raw_qty / cfg.lot_size) * cfg.lot_size)

    if qty <= 0:
        return reject(RejectReason.QTY_ROUNDS_TO_ZERO,
                      f"risk budget {risk_budget:,.0f} / stop {stop_distance:.2f} "
                      f"= {raw_qty:.2f}, below one lot of {cfg.lot_size}")

    # --- gate 3: caps on the resulting position ---------------------------
    value = qty * entry
    max_value = cfg.capital * cfg.max_position_pct / 100.0

    # Existing exposure in THIS symbol counts against the cap. Gate 2 above
    # explicitly allows an add-on (`not pf.holds(symbol)`), and checking only
    # the new tranche's value let repeated add-ons walk a single name past the
    # cap in full view of it: a position already at 14% could add another 14%
    # tranche and clear a 15% cap, reaching 28%.
    existing_value = sum(p.value for p in pf.positions if p.symbol == symbol)
    if existing_value + value > max_value:
        room = max_value - existing_value
        if room <= 0:
            return reject(RejectReason.POSITION_VALUE_CAP,
                          f"already holding {100*existing_value/cfg.capital:.1f}% "
                          f"of {symbol}, at or past the {cfg.max_position_pct}% cap")
        qty = int(math.floor((room / entry) / cfg.lot_size) * cfg.lot_size)
        if qty <= 0:
            return reject(RejectReason.POSITION_VALUE_CAP,
                          f"one lot at {entry:.2f} would exceed the "
                          f"{cfg.max_position_pct}% cap given existing exposure")
        value = qty * entry
        out.notes.append(f"trimmed to {cfg.max_position_pct}% position cap "
                         f"(existing exposure counted)")

    sector_after = pf.sector_exposure(sector) + value
    max_sector = cfg.capital * cfg.max_sector_pct / 100.0
    if sector_after > max_sector:
        return reject(RejectReason.SECTOR_CAP,
                      f"{sector} would reach {100*sector_after/cfg.capital:.1f}%, "
                      f"cap is {cfg.max_sector_pct}%")

    # Correlated exposure. Broader than the sector cap and deliberately so:
    # five different sectors that all sell off on one rate decision are one
    # position wearing five names. Without this gate the sector cap gives a
    # false sense of diversification.
    # Runs ONLY on an explicitly supplied cluster. Falling back to `sector`
    # makes this the sector gate with a different number on it: both bucket by
    # the same key, so the tighter correlated cap (25%) always binds first and
    # the sector cap (30%) becomes unreachable. The rejection would then read
    # `correlated_cap` to someone who configured no correlations at all.
    #
    # A correlation cluster is a real claim about what moves together, and the
    # caller is the only one who can make it. No claim, no gate - recorded in
    # `checks_skipped` rather than approximated.
    cluster = corr_group
    if cluster is not None:
        cluster_after = pf.cluster_exposure(cluster) + value
        max_cluster = cfg.capital * cfg.max_correlated_pct / 100.0
        if cluster_after > max_cluster:
            return reject(RejectReason.CORRELATED_CAP,
                          f"cluster {cluster} would reach "
                          f"{100*cluster_after/cfg.capital:.1f}%, cap is "
                          f"{cfg.max_correlated_pct}%")

    trade_risk = qty * stop_distance
    heat_after = pf.open_risk + trade_risk
    max_heat = cfg.capital * cfg.max_open_risk_pct / 100.0
    if heat_after > max_heat:
        return reject(RejectReason.PORTFOLIO_HEAT_CAP,
                      f"portfolio heat would be {100*heat_after/cfg.capital:.2f}%, "
                      f"cap is {cfg.max_open_risk_pct}%")

    if adv_shares is not None and adv_shares > 0:
        participation = 100.0 * qty / adv_shares
        if participation > cfg.max_adv_participation_pct:
            capped = int(math.floor(
                (adv_shares * cfg.max_adv_participation_pct / 100.0) / cfg.lot_size
            ) * cfg.lot_size)
            if capped <= 0:
                return reject(RejectReason.ILLIQUID,
                              f"one lot is {participation:.2f}% of ADV, "
                              f"cap is {cfg.max_adv_participation_pct}%")
            qty, value = capped, capped * entry
            trade_risk = qty * stop_distance
            out.notes.append(f"trimmed to {cfg.max_adv_participation_pct}% of ADV")

    # --- approved ----------------------------------------------------------
    if adv_shares is None:
        out.checks_skipped.append("liquidity: no ADV supplied")
    if atr_pct is None:
        out.checks_skipped.append("extreme volatility: no ATR% supplied")
    if expected_slippage_pct is None:
        out.checks_skipped.append("slippage: no estimate supplied")
    if corr_group is None:
        out.checks_skipped.append(
            "correlation: no cluster supplied, so only the sector cap applied")
    if data_as_of is None:
        out.checks_skipped.append("staleness: no data date supplied")

    out.approved = True
    out.qty = qty
    out.position_value = round(value, 2)
    out.capital_at_risk = round(trade_risk, 2)
    return out
