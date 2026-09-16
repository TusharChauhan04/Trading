"""Core correctness tests. These are the gates the whole system leans on."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

# desk is an installed package (pip install -e .) - no sys.path hack needed.

from desk.contracts.enums import Exchange, Regime, RejectReason, Stance
from desk.contracts.envelope import AnalysisResult, Evidence, PriceZone, Target
from desk.marketdata import corporate_actions as ca
from desk.marketdata.symbols import Symbol, SymbolError, to_canonical, translate
from desk.risk.engine import Portfolio, Position, RiskConfig, size_position
from desk.strategies import catalog


# ===========================================================================
# Symbols - four dialects, one truth
# ===========================================================================

def test_every_dialect_resolves_to_one_canonical():
    for raw in ["RELIANCE.NS", "RELIANCE.NSE", "NSE:RELIANCE", "reliance.ns", " RELIANCE "]:
        assert to_canonical(raw) == "RELIANCE.NS"


def test_bse_is_kept_distinct_from_nse():
    assert to_canonical("TCS.BO") == "TCS.BO"
    assert Symbol.parse("TCS.BO").exchange is Exchange.BSE
    assert Symbol.parse("TCS.NS").exchange is Exchange.NSE


def test_translation_to_each_consumer():
    s = Symbol.parse("RELIANCE.NS")
    assert s.kite == "RELIANCE"                    # Zerodha: bare tradingsymbol
    assert s.kite_instrument == "NSE:RELIANCE"
    assert s.nautilus == "RELIANCE.NSE"
    assert s.vibe == "RELIANCE"
    assert translate("NSE:INFY", "nautilus") == "INFY.NSE"


def test_ambiguous_input_is_refused_not_guessed():
    for bad in ["", "REL!ANCE", "RELIANCE.XYZ"]:
        with pytest.raises(SymbolError):
            Symbol.parse(bad)


# ===========================================================================
# Corporate actions - the highest-risk module
# ===========================================================================

def _series(prices, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices,
         "close": prices, "volume": [1000] * len(prices)}, index=idx)


def test_split_makes_the_series_continuous():
    # 1-for-5 split on day 3: price legitimately drops 1000 -> 200.
    df = _series([1000, 1000, 1000, 200, 200])
    split = ca.CorporateAction.split("X.NS", df.index[3].date(), old_shares=1, new_shares=5)

    raw_ret = df["close"].pct_change().iloc[3]
    assert raw_ret == pytest.approx(-0.80)          # the lie we are removing

    out = ca.adjust(df, [split])
    adj_ret = out["adj_close"].pct_change().iloc[3]
    assert adj_ret == pytest.approx(0.0, abs=1e-12)  # continuous
    assert out["adj_close"].iloc[-1] == pytest.approx(200.0)  # latest = real price


def test_indian_bonus_convention_2_for_1_means_three_shares():
    # 2:1 bonus -> hold 1, receive 2, own 3. Price should third, not halve.
    b = ca.CorporateAction.bonus("X.NS", date(2024, 6, 3), free_shares=2, per_held=1)
    assert b.factor == pytest.approx(3.0)

    df = _series([300, 300, 100, 100])
    b = ca.CorporateAction.bonus("X.NS", df.index[2].date(), free_shares=2, per_held=1)
    out = ca.adjust(df, [b])
    assert out["adj_close"].pct_change().iloc[2] == pytest.approx(0.0, abs=1e-12)


def test_point_in_time_ignores_actions_that_have_not_happened():
    """The property that makes a backtest honest."""
    df = _series([1000, 1000, 1000, 200, 200])
    ex = df.index[3].date()
    split = ca.CorporateAction.split("X.NS", ex, old_shares=1, new_shares=5)

    # Simulating a date BEFORE the split: it must not touch the history.
    early = ca.adjust(df, [split], as_of=df.index[1].date())
    assert (early["adj_factor"] == 1.0).all()
    assert early["adj_close"].iloc[0] == pytest.approx(1000.0)

    # Simulating a date after it: the adjustment applies.
    later = ca.adjust(df, [split], as_of=ex)
    assert later["adj_factor"].iloc[0] == pytest.approx(0.2)


def test_raw_prices_survive_adjustment():
    """Adjusted is right for returns and wrong for absolutes. Keep both."""
    df = _series([1000, 200])
    split = ca.CorporateAction.split("X.NS", df.index[1].date(), 1, 5)
    out = ca.adjust(df, [split])
    assert out["close"].iloc[0] == 1000.0            # untouched
    assert out["adj_close"].iloc[0] == pytest.approx(200.0)


def test_missing_corporate_action_is_detected():
    df = _series([1000, 1000, 200, 200])             # split with no action on file
    found = ca.find_discontinuities(df)
    assert len(found) == 1
    assert found[0].implied_factor == pytest.approx(5.0)
    assert "1-for-5" in found[0].nearest_ratio

    with pytest.raises(ValueError, match="unexplained price discontinuity"):
        ca.assert_clean(df, actions=[], symbol="X.NS")


def test_adjustment_factors_rejects_an_unsorted_index():
    """PERFORMANCE FIX REGRESSION: adjustment_factors() switched from a
    boolean mask (order-independent) to index.searchsorted() (requires
    ascending order). A caller that skipped quality.check()'s ordering
    check must get a raised error, not a silently wrong slice."""
    df = _series([1000, 1000, 200, 200]).iloc[::-1]     # reversed, not sorted
    split = ca.CorporateAction.split("X.NS", df.index[0].date(), 1, 5)
    with pytest.raises(ValueError, match="sorted DatetimeIndex"):
        ca.adjust(df, [split])


def test_dividend_adjustment_uses_total_return_treatment():
    """The include_dividends branch was rewritten to index a plain numpy
    array (close_arr[:pos]) instead of a boolean-masked Series slice - had
    zero test coverage before this session, despite being real, shipped
    code path."""
    df = _series([100, 100, 100])
    div = ca.CorporateAction(symbol="X.NS", ex_date=df.index[2].date(),
                             type=ca.ActionType.DIVIDEND, amount=5.0)
    out = ca.adjust(df, [div], include_dividends=True)
    # Prior close was 100; dividend of 5 -> factor (100-5)/100 = 0.95 on
    # every bar strictly before the ex-date.
    assert out["adj_factor"].iloc[0] == pytest.approx(0.95)
    assert out["adj_factor"].iloc[1] == pytest.approx(0.95)
    assert out["adj_factor"].iloc[2] == pytest.approx(1.0)   # ex-date onward


def test_implausible_dividend_leaves_the_series_alone():
    """A dividend >= the reference close is nonsensical - the original
    guard (`a.amount >= ref: continue`) must survive the numpy rewrite."""
    df = _series([100, 100, 100])
    bad_div = ca.CorporateAction(symbol="X.NS", ex_date=df.index[2].date(),
                                 type=ca.ActionType.DIVIDEND, amount=500.0)
    out = ca.adjust(df, [bad_div], include_dividends=True)
    assert (out["adj_factor"] == 1.0).all()


def test_split_and_bonus_compose_correctly_across_many_actions():
    """PERFORMANCE FIX REGRESSION: the vectorized rewrite processes actions
    in a loop over a shared numpy array rather than a fresh boolean mask each
    time - verify multiple structural actions at different dates still
    compose multiplicatively and don't interfere with each other's slice."""
    df = _series([1000] * 6)
    a1 = ca.CorporateAction.split("X.NS", df.index[2].date(), 1, 2)   # factor 2
    a2 = ca.CorporateAction.bonus("X.NS", df.index[4].date(), free_shares=1, per_held=1)  # factor 2
    out = ca.adjust(df, [a1, a2])
    # Before a1: divided by both factors (2*2=4). Between a1 and a2: divided
    # by a2's factor only (2). From a2 onward: untouched (1.0).
    assert out["adj_factor"].iloc[0] == pytest.approx(0.25)
    assert out["adj_factor"].iloc[1] == pytest.approx(0.25)
    assert out["adj_factor"].iloc[2] == pytest.approx(0.5)
    assert out["adj_factor"].iloc[3] == pytest.approx(0.5)
    assert out["adj_factor"].iloc[4] == pytest.approx(1.0)
    assert out["adj_factor"].iloc[5] == pytest.approx(1.0)


def test_find_discontinuities_vectorized_scan_matches_row_by_row_semantics():
    """PERFORMANCE FIX REGRESSION: the Python row loop was replaced with a
    numpy prefilter (np.flatnonzero) feeding the same per-candidate object
    construction. Verify multiple simultaneous gaps, a below-threshold gap
    that must NOT appear, and the first-bar NaN are all still handled
    identically to the original row-by-row semantics."""
    df = _series([1000, 1000, 1000, 1000, 1000])
    df.loc[df.index[1], "open"] = 1000            # below threshold: no flag
    df.loc[df.index[3], "open"] = 500              # -50%: flag
    df.loc[df.index[4], "open"] = 100              # further -80%: flag
    found = ca.find_discontinuities(df)
    assert {f.date for f in found} == {df.index[3].date(), df.index[4].date()}
    assert all(abs(f.gap_pct) >= 20.0 for f in found)


def test_known_action_is_not_flagged():
    df = _series([1000, 1000, 200, 200])
    split = ca.CorporateAction.split("X.NS", df.index[2].date(), 1, 5)
    assert ca.find_discontinuities(df, known=[split]) == []
    ca.assert_clean(df, actions=[split])             # must not raise


# ===========================================================================
# Risk engine - every gate, and the arithmetic itself
# ===========================================================================

CFG = RiskConfig(capital=1_000_000, risk_pct=1.0)


def test_position_size_is_exactly_risk_over_stop_distance():
    """Pure arithmetic, on a case where no cap binds.

    1% of 10,00,000 = 10,000 risk budget. Stop is 2 away -> 5,000 shares
    at Rs.100 = Rs.5,00,000... which would breach the position cap, so use a
    capital large enough that only the risk maths is under test.
    """
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0,
                     max_position_pct=100.0, max_sector_pct=100.0)
    s = size_position(symbol="X.NS", entry=1000, stop=980, target=1060, cfg=cfg)
    assert s.approved
    assert s.qty == 500                              # 10,000 / 20
    assert s.capital_at_risk == pytest.approx(10_000)


def test_concentration_cap_binds_before_the_risk_budget_is_spent():
    """A tight stop on an expensive share wants a huge position. The 15%
    cap catches it - this is the gate doing its job, not a bug."""
    s = size_position(symbol="X.NS", entry=1000, stop=980, target=1060, cfg=CFG)
    assert s.approved
    assert s.qty == 150                              # 15% of 10,00,000 / 1000
    assert s.position_value == pytest.approx(150_000)
    assert s.capital_at_risk == pytest.approx(3_000)  # LESS risk than budgeted
    assert any("trimmed to 15.0% position cap" in n for n in s.notes)


def test_lot_size_rounds_down_never_up():
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0, lot_size=75)
    s = size_position(symbol="NIFTY", entry=22000, stop=21800, target=22400, cfg=cfg)
    assert s.qty % 75 == 0
    assert s.qty * 200 <= 10_000                     # never exceeds the risk budget


def test_risk_reward_floor_rejects():
    s = size_position(symbol="X.NS", entry=100, stop=95, target=102, cfg=CFG)
    assert not s.approved
    assert RejectReason.RR_TOO_LOW in s.reasons


def test_portfolio_heat_cap_rejects():
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0, max_open_risk_pct=2.0)
    pf = Portfolio(positions=[
        Position("A.NS", qty=1000, entry=100, stop=90),   # 10,000 risk
        Position("B.NS", qty=1000, entry=100, stop=90),   # 10,000 risk -> at cap
    ])
    s = size_position(symbol="C.NS", entry=100, stop=90, target=130, cfg=cfg, portfolio=pf)
    assert not s.approved
    assert RejectReason.PORTFOLIO_HEAT_CAP in s.reasons


def test_sector_cap_rejects():
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0, max_sector_pct=10.0)
    pf = Portfolio(positions=[Position("HDFCBANK.NS", 900, 100, 90, sector="BANK")])
    s = size_position(symbol="ICICIBANK.NS", entry=100, stop=90, target=130,
                      cfg=cfg, portfolio=pf, sector="BANK")
    assert not s.approved
    assert RejectReason.SECTOR_CAP in s.reasons


def test_daily_loss_cap_stops_the_day():
    cfg = RiskConfig(capital=1_000_000, max_daily_loss_pct=2.0)
    pf = Portfolio(realised_pnl_today=-20_000)
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=cfg, portfolio=pf)
    assert not s.approved
    assert RejectReason.DAILY_LOSS_CAP in s.reasons


def test_illiquid_names_are_trimmed_then_refused():
    # 1% of ADV. 500 shares wanted, ADV 10,000 -> capped at 100.
    s = size_position(symbol="SMALL.NS", entry=1000, stop=980, target=1060,
                      cfg=CFG, adv_shares=10_000)
    assert s.approved and s.qty == 100
    # ADV so thin that even one share breaches the cap.
    s2 = size_position(symbol="TINY.NS", entry=1000, stop=980, target=1060,
                       cfg=CFG, adv_shares=50)
    assert not s2.approved and RejectReason.ILLIQUID in s2.reasons


def test_untradeable_instrument_is_refused_first():
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130,
                      cfg=CFG, tradeable=False)
    assert not s.approved
    assert s.reasons == [RejectReason.NOT_TRADEABLE]


def test_stale_data_is_refused():
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=CFG,
                      data_as_of=date(2024, 1, 1), today=date(2024, 2, 1))
    assert not s.approved
    assert RejectReason.STALE_DATA in s.reasons


# --- the four safeguards that were declared but never enforced -------------

def test_correlated_cluster_cap_binds_across_sectors():
    """The whole point: five sectors that sell off together are one position.

    Sector caps alone would pass this - HDFCBANK is BANK and DLF is REALTY -
    so without the cluster gate the portfolio looks diversified and is not.
    """
    cfg = RiskConfig(capital=1_000_000, max_correlated_pct=20.0, max_sector_pct=100.0)
    pf = Portfolio(positions=[
        Position("HDFCBANK.NS", 1500, 100, 90, sector="BANK", corr_group="RATE_SENSITIVE"),
        Position("DLF.NS", 500, 100, 90, sector="REALTY", corr_group="RATE_SENSITIVE"),
    ])
    s = size_position(symbol="BAJFINANCE.NS", entry=100, stop=90, target=140,
                      cfg=cfg, portfolio=pf, sector="NBFC",
                      corr_group="RATE_SENSITIVE")
    assert not s.approved
    assert RejectReason.CORRELATED_CAP in s.reasons


def test_unknown_cluster_is_reported_not_guessed():
    """An unlabelled name must not silently become its own correlation bucket,
    and must not be lumped with every other unlabelled name either."""
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=CFG)
    assert s.approved
    assert any("correlation" in c for c in s.checks_skipped)


def test_the_cluster_cap_does_not_shadow_the_sector_cap():
    """REGRESSION: the cluster defaulted to `sector`, so both gates bucketed by
    the same key and the tighter cluster cap (25%) always bound first - making
    the 30% sector cap unreachable and reporting `correlated_cap` to someone
    who had configured no correlations at all."""
    cfg = RiskConfig(capital=1_000_000, max_sector_pct=30.0,
                     max_correlated_pct=25.0, max_position_pct=100.0)
    # 25% already held; the new position adds 10% (10,000 risk / 10 stop =
    # 1,000 shares at 100), taking BANK to 35% and past the 30% sector cap.
    pf = Portfolio(positions=[Position("A.NS", 2500, 100, 90, sector="BANK")])
    s = size_position(symbol="B.NS", entry=100, stop=90, target=140,
                      cfg=cfg, portfolio=pf, sector="BANK")
    assert not s.approved
    assert RejectReason.SECTOR_CAP in s.reasons
    assert RejectReason.CORRELATED_CAP not in s.reasons

    # State the claim, and the tighter cluster cap applies as intended.
    pf2 = Portfolio(positions=[
        Position("A.NS", 2000, 100, 90, sector="BANK", corr_group="RATE_SENSITIVE")])
    # 20% in the cluster + 10% new = 30%, past the 25% cluster cap, while the
    # two sit in DIFFERENT sectors so the sector gate is not what stops it.
    s2 = size_position(symbol="B.NS", entry=100, stop=90, target=140,
                       cfg=cfg, portfolio=pf2, sector="REALTY",
                       corr_group="RATE_SENSITIVE")
    assert not s2.approved
    assert RejectReason.CORRELATED_CAP in s2.reasons


def test_extreme_volatility_refuses_before_sizing():
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=CFG,
                      atr_pct=20.0)
    assert not s.approved
    assert RejectReason.EXTREME_VOLATILITY in s.reasons


def test_volatility_adjusted_sizing_only_scales_down():
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0, vol_reference_atr_pct=2.0,
                     max_position_pct=100.0, max_sector_pct=100.0)
    calm = size_position(symbol="X.NS", entry=100, stop=95, target=130,
                         cfg=cfg, atr_pct=1.0)      # quieter than reference
    jumpy = size_position(symbol="Y.NS", entry=100, stop=95, target=130,
                          cfg=cfg, atr_pct=4.0)     # twice the reference
    assert calm.qty == 2000                          # 10,000 / 5, unscaled
    assert jumpy.qty == 1000                         # half the budget
    assert any("risk scaled" in n for n in jumpy.notes)


def test_slippage_cap_refuses():
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=CFG,
                      expected_slippage_pct=2.0)
    assert not s.approved
    assert RejectReason.SLIPPAGE_TOO_HIGH in s.reasons


def test_market_risk_off_stops_trading_by_default():
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=CFG,
                      market_risk_off=True)
    assert not s.approved
    assert RejectReason.MARKET_RISK_OFF in s.reasons


def test_market_risk_off_can_scale_instead_of_stopping():
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0, risk_off_exposure_pct=50.0,
                     max_position_pct=100.0, max_sector_pct=100.0)
    normal = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=cfg)
    reduced = size_position(symbol="X.NS", entry=100, stop=95, target=130, cfg=cfg,
                            market_risk_off=True)
    assert reduced.approved
    assert reduced.qty == normal.qty // 2


def test_engine_is_deterministic():
    a = size_position(symbol="X.NS", entry=1000, stop=980, target=1060, cfg=CFG)
    b = size_position(symbol="X.NS", entry=1000, stop=980, target=1060, cfg=CFG)
    assert a.model_dump() == b.model_dump()


# ===========================================================================
# Contracts
# ===========================================================================

def test_review_can_never_carry_conviction():
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.REVIEW, confidence=0.9, horizon="swing")
    assert r.confidence == 0.0
    assert not r.is_tradeable


def test_inverted_stop_is_caught_before_it_reaches_sizing():
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.8, horizon="swing",
                       entry=100, stop=110)                    # stop above entry on a long
    assert r.errors and not r.is_tradeable


def test_risk_reward_is_computed_from_the_levels():
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.7, horizon="swing",
                       entry=100, stop=95, target=115)
    assert r.risk_reward == pytest.approx(3.0)
    assert r.is_tradeable


# ===========================================================================
# Regressions from the 2026-09-11 correctness review. Each of these passed
# silently before the fix, which is what made them worth a permanent test.
# ===========================================================================

def test_backwards_price_zone_swaps_instead_of_collapsing():
    """REGRESSION: the swap was written without a temporary, so `low = high`
    then `high = low` read the already-overwritten value and both ends
    collapsed to the lower bound."""
    z = PriceZone(low=1500, high=1400)
    assert (z.low, z.high) == (1400.0, 1500.0)
    assert z.mid == 1450.0


def test_derived_fields_survive_exclude_unset():
    """REGRESSION: object.__setattr__ skipped pydantic's fields-set tracking,
    so every value derived in the validator vanished from a serialised result -
    on the one envelope that crosses every agent boundary."""
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.6, horizon="swing",
                       entry_zone=PriceZone(low=100, high=110), stop=95)
    assert r.entry == 105.0
    assert "entry" in r.model_dump(exclude_unset=True)

    rev = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                         stance=Stance.REVIEW, confidence=0.9, horizon="swing")
    assert rev.model_dump(exclude_unset=True)["confidence"] == 0.0


def test_inverted_target_is_caught_not_scored_as_positive_rr():
    """REGRESSION: only the stop direction was checked. Because both the model
    and the engine use abs(target - entry), a long whose target sat BELOW its
    stop reported a clean 2:1 and cleared the risk-reward floor."""
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.8, horizon="swing",
                       entry=100, stop=95, target=90)
    assert r.errors and not r.is_tradeable


def test_multi_target_partial_exits_cannot_exceed_the_position():
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.8, horizon="swing",
                       entry=100, stop=95,
                       targets=[Target(price=110, fraction=0.6),
                                Target(price=120, fraction=0.6)])
    assert any("partial exits" in e for e in r.errors)


def test_engine_returns_on_nan_rather_than_raising():
    """REGRESSION: every gate compares with <, > or <=, all False for NaN, so a
    NaN fell through to math.floor() and raised - breaking the engine's
    "returns, never raises" contract."""
    for bad in ({"entry": float("nan")}, {"stop": float("inf")},
                {"target": float("nan")}):
        kw = {"symbol": "X.NS", "entry": 100.0, "stop": 95.0,
              "target": 130.0, "cfg": CFG, **bad}
        s = size_position(**kw)
        assert not s.approved
        assert RejectReason.MALFORMED_INPUT in s.reasons


def test_zero_adv_is_refused_not_treated_as_unknown():
    """REGRESSION: the gate needed adv > 0 and the skip note needed adv is
    None, so adv=0 satisfied neither - a suspended scrip was approved and the
    caller could not tell the liquidity gate had never run."""
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130,
                      cfg=CFG, adv_shares=0)
    assert not s.approved
    assert RejectReason.ILLIQUID in s.reasons


def test_zero_atr_is_refused_not_treated_as_calm():
    s = size_position(symbol="X.NS", entry=100, stop=95, target=130,
                      cfg=CFG, atr_pct=0.0)
    assert not s.approved
    assert RejectReason.EXTREME_VOLATILITY in s.reasons


def test_slippage_is_judged_against_the_stop_not_only_the_price():
    """REGRESSION: the comment said "against the stop distance"; the gate
    compared against price. 0.4% slippage on a 0.4% stop was APPROVED, with a
    note reading "slippage is 100.0% of the stop distance"."""
    cfg = RiskConfig(capital=1_000_000, max_slippage_pct=0.5)
    s = size_position(symbol="X.NS", entry=1000, stop=996, target=1030,
                      cfg=cfg, expected_slippage_pct=0.4)
    assert not s.approved
    assert RejectReason.SLIPPAGE_TOO_HIGH in s.reasons

    # A wide stop makes the same slippage harmless.
    ok = size_position(symbol="X.NS", entry=1000, stop=900, target=1300,
                       cfg=cfg, expected_slippage_pct=0.4)
    assert ok.approved


def test_negative_qty_cannot_manufacture_portfolio_headroom():
    """REGRESSION: value and risk were linear in qty with no sign guard, so a
    negative qty produced negative open_risk and every heat check passed."""
    pf = Portfolio(positions=[Position("A.NS", qty=-100_000, entry=100, stop=90)])
    assert pf.open_risk > 0
    assert pf.exposure > 0
    s = size_position(symbol="B.NS", entry=100, stop=90, target=130,
                      cfg=RiskConfig(capital=1_000_000, max_open_risk_pct=4.0),
                      portfolio=pf)
    assert not s.approved


def test_evidence_carries_provenance():
    e = Evidence(claim="closed above 200DMA", source="nse_bhavcopy",
                 as_of=pd.Timestamp("2024-01-01").to_pydatetime())
    assert e.source and e.as_of


# ===========================================================================
# Strategy catalog - the six, held as data
# ===========================================================================

def test_all_six_strategies_are_registered():
    assert len(catalog.CATALOG) == 6
    assert set(catalog.BY_KEY) == {
        "supertrend_adx", "bollinger_rsi", "donchian_breakout",
        "pairs_trading", "opening_range_breakout", "ml_classifier",
    }


def test_nothing_is_trusted_until_it_clears_walk_forward():
    """If this ever fails, something was promoted without evidence."""
    for s in catalog.CATALOG:
        if s.maturity in (catalog.Maturity.DRAFT, catalog.Maturity.AUDITED,
                          catalog.Maturity.REVALIDATED):
            assert not s.trusted, f"{s.key} is trusted at maturity {s.maturity}"


def test_a_strategy_with_open_defects_cannot_be_trusted():
    for s in catalog.CATALOG:
        if s.defects and s.trusted:
            raise AssertionError(f"{s.key} is trusted while carrying open defects")


def test_hostile_regime_silences_rather_than_downweights():
    picked = catalog.eligible(Regime.CRISIS, trusted_only=False)
    keys = {s.key for s in picked}
    assert "bollinger_rsi" not in keys          # mean reversion in a crisis
    assert "pairs_trading" not in keys
    assert "ml_classifier" not in keys
    assert "supertrend_adx" in keys


def test_trusted_only_is_the_default_and_is_currently_empty():
    """The live path must be empty while the whole library is unvalidated."""
    assert catalog.eligible(Regime.TRENDING_UP) == []


def test_catalog_is_json_safe_for_the_api():
    import json
    json.dumps(catalog.catalog_status())


def test_position_cap_counts_existing_exposure_on_an_add_on():
    """REGRESSION: the position cap checked only the NEW tranche's value, so a
    position already at 14% of a 15% cap could add another 14% tranche and
    clear the check, reaching 28% - the cap giving a false sense of protection
    for exactly the add-on path gate 2 explicitly permits."""
    cfg = RiskConfig(capital=1_000_000, risk_pct=5.0, max_position_pct=15.0,
                     max_sector_pct=100.0, max_correlated_pct=100.0)
    pf = Portfolio(positions=[Position("X.NS", qty=1000, entry=140, stop=130)])
    # Already holding 140,000 (14%). A wide-stop add-on would want a huge
    # tranche; the cap must trim it to the REMAINING room, not a fresh 15%.
    s = size_position(symbol="X.NS", entry=140, stop=130, target=200,
                      cfg=cfg, portfolio=pf)
    assert s.approved
    combined = pf.positions[0].value + s.position_value
    assert combined <= cfg.capital * 0.15 + 1e-6
    assert "existing exposure" in " ".join(s.notes)


def test_position_cap_refuses_outright_once_existing_exposure_is_at_the_cap():
    cfg = RiskConfig(capital=1_000_000, risk_pct=1.0, max_position_pct=15.0,
                     max_sector_pct=100.0, max_correlated_pct=100.0)
    pf = Portfolio(positions=[Position("X.NS", qty=1500, entry=100, stop=90)])
    s = size_position(symbol="X.NS", entry=100, stop=95, target=120,
                      cfg=cfg, portfolio=pf)
    assert not s.approved
    assert RejectReason.POSITION_VALUE_CAP in s.reasons


def test_target_list_entries_are_direction_checked():
    """REGRESSION: only the scalar target was checked for direction. A T2
    priced on the wrong side of entry in targets[] was invisible."""
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.7, horizon="swing",
                       entry=1400, stop=1350,
                       targets=[Target(price=1500, fraction=0.5),
                               Target(price=1300, fraction=0.5)])
    assert any("targets[1]" in e for e in r.errors)
    assert not r.is_tradeable


def test_non_finite_levels_are_rejected_not_silently_tradeable():
    """REGRESSION: every comparison is False for NaN, so an unfiltered NaN
    fell through every direction check and left is_tradeable=True."""
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.7, horizon="swing",
                       entry=float("nan"), stop=95)
    assert r.errors and not r.is_tradeable

    r2 = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                        stance=Stance.BUY, confidence=0.7, horizon="swing",
                        entry=100, stop=95,
                        targets=[Target(price=float("inf"), fraction=1.0)])
    assert r2.errors and not r2.is_tradeable


def test_errors_do_not_accumulate_across_a_round_trip():
    """REGRESSION: a mode='after' validator that appends is not idempotent.
    model_validate(model_dump()) - the natural JSON round-trip across an agent
    boundary - re-ran the validator and grew the errors list on every pass."""
    r = AnalysisResult(agent="x", symbol="X.NS", as_of=date(2024, 1, 1),
                       stance=Stance.BUY, confidence=0.7, horizon="swing",
                       entry=100, stop=110)
    n = len(r.errors)
    assert n > 0
    r2 = AnalysisResult.model_validate(r.model_dump())
    r3 = AnalysisResult.model_validate(r2.model_dump())
    assert len(r2.errors) == n
    assert len(r3.errors) == n


def test_assert_clean_scopes_actions_to_the_named_symbol():
    """REGRESSION (architecture re-verification, 2026-09-13): assert_clean
    took a `symbol` argument but never passed it to find_discontinuities,
    so calling it on a multi-symbol panel with a shared action list would
    have let one share's split explain a different share's bad print -
    reopening the exact cross-symbol contamination bug find_discontinuities
    was scoped to fix. Currently no production caller does this, but
    assert_clean is precisely the hard gate a backtest harness would use."""
    df = _series([1000, 1000, 200, 200])          # unexplained 80% gap
    unrelated_split = ca.CorporateAction.split(
        "OTHER.NS", df.index[2].date(), 1, 5)     # a DIFFERENT symbol's split
    with pytest.raises(ValueError, match="unexplained price discontinuit"):
        ca.assert_clean(df, actions=[unrelated_split], symbol="X.NS")

    # The same action, correctly attributed, must silence the gate.
    own_split = ca.CorporateAction.split("X.NS", df.index[2].date(), 1, 5)
    ca.assert_clean(df, actions=[own_split], symbol="X.NS")   # must not raise
