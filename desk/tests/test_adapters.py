"""Strategy adapters - turning a catalogued strategy into proposals.

The catalog described six strategies and produced none of them. These tests
pin the two things that make an adapter safe to add: it may propose but
never size, and a strategy that COULD NOT be asked must never be reported
the same way as one that was asked and had nothing to say.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from desk.contracts.enums import Regime, Stance
from desk.scanner.stage1 import run_stage1
from desk.store import Coverage, History
from desk.strategies.adapters import (ADAPTERS, donchian_breakout, propose,
                                      propose_all)
from desk.strategies.catalog import BY_KEY

AS_OF = date(2026, 3, 2)
SPEC = BY_KEY["donchian_breakout"]


def _history(**symbols: tuple[list[float], list[float], list[float]]) -> History:
    """(high, low, close) per symbol, oldest first, one bar per day."""
    n = len(next(iter(symbols.values()))[0])
    days = [AS_OF - timedelta(days=n - 1 - i) for i in range(n)]
    rows = []
    for sym, (hi, lo, cl) in symbols.items():
        rows += [{"date": d, "symbol": sym, "open": c, "high": h, "low": lw,
                  "close": c, "volume": 1_000_000.0}
                 for d, h, lw, c in zip(days, hi, lo, cl)]
    return History(
        frame=pd.DataFrame(rows),
        coverage=Coverage(start=days[0], end=days[-1], days_loaded=n,
                          sessions_expected=n, calendar_checked=True))


def _flat_then(move: list[float], base: float = 100.0, flat: int = 50):
    """`flat` quiet bars at `base`, then the closes in `move`."""
    cl = [base] * flat + move
    hi = [base + 0.5] * flat + [c + 0.3 for c in move]
    lo = [base - 0.5] * flat + [c - 0.3 for c in move]
    return hi, lo, cl


BREAKOUT = _flat_then([100.8, 102.8, 104.8, 106.8, 107.8,
                       108.2, 108.8, 109.2, 109.8, 110.2])
QUIET = _flat_then([100.0] * 10)


def _stage1(**symbols):
    return run_stage1(_history(**symbols), as_of=AS_OF, min_bars=30)


# ===========================================================================
# the channel itself
# ===========================================================================

def test_the_channel_excludes_today_or_it_could_never_be_broken():
    """The correctness of the whole rule, not a detail of it.

    A channel computed over a window CONTAINING the current bar can never
    be broken: the bar's own high is in the max and high >= close always,
    so `close > channel` is arithmetically impossible. The rule would fire
    exactly never and look like a market with no breakouts in it.
    """
    f = _stage1(BREAK=BREAKOUT).features.loc["BREAK"]
    assert f["prior_high_20"] < f["close"], (
        "the channel contains today's bar - this rule can never fire")
    # Yesterday closed 109.8 and _flat_then sets high = close + 0.3,
    # so the channel the breakout had to clear is 110.1 - and today
    # closed 110.2, one paisa-scale tick of real break above it.
    assert f["prior_high_20"] == pytest.approx(110.1)


def test_the_channel_is_nan_rather_than_short_when_history_is_thin():
    """A 20-session high computed from 12 sessions is not a 20-session high.
    min_periods makes it NaN, and the adapter then proposes nothing."""
    short = ([100.5] * 12, [99.5] * 12, [100.0] * 12)
    f = _stage1(A=short, B=short).features
    if len(f):                       # min_bars may drop them entirely
        assert f["prior_high_20"].isna().all()


# ===========================================================================
# the rule
# ===========================================================================

def test_only_a_close_above_the_channel_produces_a_proposal():
    s1 = _stage1(BREAK=BREAKOUT, QUIET=QUIET)
    out = propose("donchian_breakout", s1)
    assert [r.symbol for r in out] == ["BREAK"]


def test_a_non_breakout_returns_None_not_a_HOLD():
    """A breakout scanner has no opinion on a name that did not break out.
    HOLD is a TRADEABLE NEUTRAL (ADR-001), so emitting it on ~1,500
    non-breakouts a day would be 1,500 opinions nobody formed."""
    row = _stage1(QUIET=QUIET).features.loc["QUIET"]
    assert donchian_breakout("QUIET", row, SPEC, AS_OF) is None


def test_the_proposal_is_internally_coherent_and_tradeable():
    r = propose("donchian_breakout", _stage1(BREAK=BREAKOUT))[0]
    assert r.stance is Stance.BUY
    assert r.errors == [], r.errors
    assert r.stop < r.entry < r.target
    assert r.entry_zone.low <= r.entry <= r.entry_zone.high
    assert r.is_tradeable
    # 2R against the stop distance the strategy's own atr_stop_mult chose
    assert r.target - r.entry == pytest.approx(2.0 * (r.entry - r.stop))


def test_the_stop_honours_the_CATALOGUED_multiplier_not_stage4s():
    """Running a catalogued strategy means running ITS parameters. Stage 4's
    structural stop is a different question and applies to the funnel's own
    setups; substituting it here would mean the backtest measures something
    the catalog does not describe."""
    s1 = _stage1(BREAK=BREAKOUT)
    row = s1.features.loc["BREAK"]
    r = donchian_breakout("BREAK", row, SPEC, AS_OF)
    atr = row["close"] * row["atr_pct"] / 100.0
    assert r.stop == pytest.approx(
        row["close"] - float(SPEC.params["atr_stop_mult"]) * atr)


def test_a_missing_feature_produces_no_proposal_rather_than_a_guess():
    """NaN survives every comparison as False, so an unguarded one does not
    raise - it silently stops the rule firing, which is indistinguishable
    from a calm market."""
    base = _stage1(BREAK=BREAKOUT).features.loc["BREAK"]
    for col in ("close", "prior_high_20", "atr_pct"):
        row = base.copy()
        row[col] = float("nan")
        assert donchian_breakout("BREAK", row, SPEC, AS_OF) is None, col


def test_an_atr_that_puts_the_stop_under_zero_is_refused():
    row = _stage1(BREAK=BREAKOUT).features.loc["BREAK"].copy()
    row["atr_pct"] = 80.0            # 2 x 80% of price is below zero
    assert donchian_breakout("BREAK", row, SPEC, AS_OF) is None


# ===========================================================================
# what may be sized - the part that protects the account
# ===========================================================================

def test_nothing_catalogued_is_tradeable_yet_and_the_result_says_so():
    """Three sit at AUDITED and three at DRAFT - none trusted. A proposal
    must carry that on itself - a caller that has to look the maturity up
    elsewhere is a caller that will forget to."""
    r = propose("donchian_breakout", _stage1(BREAK=BREAKOUT))[0]
    assert r.metadata["trusted"] is False
    assert r.metadata["maturity"] == "audited"
    assert SPEC.trusted is False

    out = propose_all(_stage1(BREAK=BREAKOUT), regime=Regime.TRENDING_UP)
    assert out["tradeable_now"] == [], (
        "a strategy became sizeable without passing walk-forward")


# ===========================================================================
# silenced, unimplemented and empty are THREE different answers
# ===========================================================================

def test_a_hostile_regime_silences_a_strategy_with_a_stated_reason():
    """Donchian declares RANGE hostile. Silenced is not the same as
    'proposed nothing', and the plan must be able to say which."""
    out = propose_all(_stage1(BREAK=BREAKOUT), regime=Regime.RANGE)
    assert "donchian_breakout" in out["silenced"]
    assert "hostile" in out["silenced"]["donchian_breakout"]
    assert "donchian_breakout" not in out["proposals"]


def test_an_unimplemented_strategy_is_reported_not_skipped():
    """Five of six have no adapter. Dropping them silently would make the
    response indistinguishable from a day on which they all declined."""
    out = propose_all(_stage1(BREAK=BREAKOUT), regime=Regime.TRENDING_UP)
    assert set(out["unimplemented"]) == set(BY_KEY) - set(ADAPTERS)
    for why in out["unimplemented"].values():
        assert "no adapter" in why


def test_asking_for_an_unimplemented_strategy_raises_rather_than_returning_empty():
    """An empty list would read as 'the market offered nothing', which is
    the exact class of disguised failure this codebase keeps finding."""
    with pytest.raises(NotImplementedError):
        propose("supertrend_adx", _stage1(BREAK=BREAKOUT))
    with pytest.raises(KeyError):
        propose("not_a_strategy", _stage1(BREAK=BREAKOUT))


def test_an_eligible_strategy_with_no_setups_reports_an_empty_list():
    """The third state: asked, allowed to answer, and had nothing to say."""
    out = propose_all(_stage1(QUIET=QUIET, ALSO=QUIET),
                      regime=Regime.TRENDING_UP)
    assert out["proposals"]["donchian_breakout"] == []
    assert "donchian_breakout" not in out["silenced"]


# ===========================================================================
# bollinger_rsi - the reversion side, and the geometry its stop comes from
# ===========================================================================

BR_SPEC = BY_KEY["bollinger_rsi"]


def _reversion_row(close: float, lower: float, mid: float, rsi: float,
                   atr_pct: float = 2.0) -> pd.Series:
    return pd.Series({"close": close, "bb_lower": lower, "bb_mid": mid,
                      "rsi_14": rsi, "atr_pct": atr_pct})


def test_both_conditions_are_required_not_either():
    """The thesis is "stretched AND momentum agrees". A close below the band
    with neutral RSI is a trend leaving, not a rubber band stretching."""
    from desk.strategies.adapters import bollinger_rsi
    both = _reversion_row(94.0, 95.0, 100.0, rsi=25.0)
    assert bollinger_rsi("A", both, BR_SPEC, AS_OF) is not None

    band_only = _reversion_row(94.0, 95.0, 100.0, rsi=55.0)
    assert bollinger_rsi("A", band_only, BR_SPEC, AS_OF) is None
    rsi_only = _reversion_row(97.0, 95.0, 100.0, rsi=25.0)
    assert bollinger_rsi("A", rsi_only, BR_SPEC, AS_OF) is None


def test_the_stop_is_one_further_sigma_below_the_band():
    """Derived from bb_std and the band's own dispersion, because the
    catalog specifies no stop at all. sigma = (mid - lower) / bb_std, and
    the stop is mid - (bb_std + 1) * sigma."""
    from desk.strategies.adapters import bollinger_rsi
    r = bollinger_rsi("A", _reversion_row(94.0, 95.0, 100.0, 25.0),
                      BR_SPEC, AS_OF)
    sigma = (100.0 - 95.0) / 2.0             # bb_std is 2.0
    assert r.stop == pytest.approx(100.0 - 3.0 * sigma)
    assert r.metadata["sigma"] == pytest.approx(sigma)


def test_the_target_is_the_mean_because_that_is_the_thesis():
    from desk.strategies.adapters import bollinger_rsi
    r = bollinger_rsi("A", _reversion_row(94.0, 95.0, 100.0, 25.0),
                      BR_SPEC, AS_OF)
    assert r.target == pytest.approx(100.0)
    assert r.errors == [] and r.is_tradeable
    assert r.stop < r.entry < r.target


def test_a_close_already_past_its_invalidation_is_refused():
    """Below mid - 3 sigma the reversion thesis is already dead. Proposing
    it would put a stop ABOVE entry on a long - the envelope would catch
    that, but catching it downstream is worse than never proposing it."""
    from desk.strategies.adapters import bollinger_rsi
    # sigma 2.5, so mid - 3 sigma = 92.5; a close at 92.0 is through it
    assert bollinger_rsi("A", _reversion_row(92.0, 95.0, 100.0, 20.0),
                         BR_SPEC, AS_OF) is None


def test_a_stop_inside_the_noise_is_REFUSED_not_widened():
    """The degenerate case that produced a 251R round-trip cost in the
    backtest: R is divided by the stop distance, so a near-zero denominator
    makes every derived number nonsense. Widening is not available here -
    moving the stop off mid - 3 sigma discards the only thing justifying
    the level - so the setup is refused."""
    from desk.strategies.adapters import bollinger_rsi
    # sigma 2.5 -> stop 92.5; a close at 92.6 leaves 0.10 of risk on a
    # stock whose ATR is 2% of ~93, about 1.86. Far inside the noise.
    r = bollinger_rsi("A", _reversion_row(92.6, 95.0, 100.0, 20.0),
                      BR_SPEC, AS_OF)
    assert r is None


def test_a_flat_window_has_no_bands_and_no_setup():
    """Zero dispersion means no band worth the name and no distance to
    place a stop against - not a maximally stretched signal."""
    from desk.strategies.adapters import bollinger_rsi
    assert bollinger_rsi("A", _reversion_row(100.0, 100.0, 100.0, 20.0),
                         BR_SPEC, AS_OF) is None


def test_the_reported_risk_reward_matches_the_levels():
    """`rr` is carried so a walk-forward run can see the distribution
    rather than discovering later that the risk gate refused most of these
    on the 1.5 floor."""
    from desk.strategies.adapters import bollinger_rsi
    r = bollinger_rsi("A", _reversion_row(94.0, 95.0, 100.0, 25.0),
                      BR_SPEC, AS_OF)
    assert r.metadata["rr"] == pytest.approx(
        (r.target - r.entry) / (r.entry - r.stop))


def test_the_downtrend_defect_is_handled_by_ROUTING_not_by_the_rule():
    """The catalog's standing defect is "No regime gate: as written it buys
    every step of a downtrend". That is fixed where it belongs - a rule
    that second-guesses its own eligibility cannot be reasoned about from
    the catalog."""
    assert Regime.TRENDING_DOWN in BR_SPEC.hostile_regimes
    assert Regime.CRISIS in BR_SPEC.hostile_regimes
    out = propose_all(_stage1(BREAK=BREAKOUT), regime=Regime.TRENDING_DOWN)
    assert "bollinger_rsi" in out["silenced"]
    assert "bollinger_rsi" not in out["proposals"]


def test_each_strategy_ranks_by_its_own_measure_not_a_shared_one():
    """4% past a channel and 2.5 sigma below a band are different units.
    Sorting them together would be a cross-strategy ranking resting on a
    coincidence of scale - that is the coordinator's job, and it needs
    walk-forward expectancy, which does not exist yet."""
    from desk.strategies.adapters import _rank_key, bollinger_rsi
    d = propose("donchian_breakout", _stage1(BREAK=BREAKOUT))[0]
    b = bollinger_rsi("A", _reversion_row(94.0, 95.0, 100.0, 25.0),
                      BR_SPEC, AS_OF)
    assert _rank_key(d) == pytest.approx(d.metadata["break_pct"])
    assert _rank_key(b) == pytest.approx(b.metadata["stretch_sigma"])


def test_rsi_and_the_bands_match_the_reference_implementations():
    """PARITY. Stage 1 computes these across a date x symbol matrix; the
    catalog's spec is written against strategy_lib's per-symbol versions.
    Two implementations of one indicator drift silently, and RSI's
    zero-loss case is exactly where they would."""
    import numpy as np
    from desk.indicators.volatility import bollinger_bands as desk_bb
    from strategies.india.strategy_lib import rsi as ref_rsi

    rng = np.random.default_rng(11)
    n = 90
    cl = list(100 + np.cumsum(rng.normal(0, 1.4, n)))
    hi = [c + 1.0 for c in cl]
    lo = [c - 1.0 for c in cl]
    f = run_stage1(_history(A=(hi, lo, cl)), as_of=AS_OF,
                   min_bars=30).features.loc["A"]
    s = pd.Series(cl)

    assert f["rsi_14"] == pytest.approx(ref_rsi(s).iloc[-1], rel=1e-9)
    bb = desk_bb(s)
    assert f["bb_lower"] == pytest.approx(bb["lower"].iloc[-1], rel=1e-9)
    assert f["bb_mid"] == pytest.approx(bb["mid"].iloc[-1], rel=1e-9)

    # The awkward case both must agree on: no down bar means zero average
    # loss, and the reference returns NaN rather than 100. 100 would read as
    # "maximally overbought" when the truth is "undefined".
    assert pd.isna(ref_rsi(pd.Series(np.arange(100.0, 140.0))).iloc[-1])


def test_the_catalogued_maturities_are_what_the_docs_claim():
    """REGRESSION on a claim I got wrong in three files and several commit
    messages: "all six are AUDITED". Three are AUDITED and three are DRAFT.

    The safety-relevant half was true either way - NOTHING is trusted - but
    a doc that misstates where a strategy sits on the ladder is a doc that
    will eventually be used to argue it is further along than it is. This
    asserts the shape rather than each key, so adding a strategy does not
    break it, and it fails loudly if anything is ever promoted without the
    documentation moving with it."""
    from desk.strategies.catalog import CATALOG, Maturity
    by = {}
    for s in CATALOG:
        by.setdefault(s.maturity, []).append(s.key)
    assert len(by.get(Maturity.AUDITED, [])) == 3
    assert len(by.get(Maturity.DRAFT, [])) == 3
    assert not any(s.trusted for s in CATALOG), (
        "a strategy became trusted - the README, the adapters docstring and "
        "the roadmap all state that none are, and they must move together")
