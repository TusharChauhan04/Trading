"""Scanner Stage 1. The feature maths is vectorised across the whole universe
rather than looped per symbol, so the single most important test here is
PARITY: the fast path must agree, to floating-point tolerance, with the
per-symbol functions in desk/indicators/ that were each checked against a
hand-computed value. Everything else guards a way a screen can silently
report a fictional opportunity.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from desk.marketdata.corporate_actions import CorporateAction
from desk.scanner.stage1 import FEATURE_COLUMNS, run_stage1
from desk.store import BarStore

pytest.importorskip("pyarrow")


def _sessions(n: int, end: date = date(2026, 9, 11)) -> list[date]:
    out, cur = [], end
    while len(out) < n:
        if cur.weekday() < 5:
            out.append(cur)
        cur -= timedelta(days=1)
    return sorted(out)


def _build(store: BarStore, days, prices: dict[str, list[float]],
           volumes: dict[str, list[float]] | None = None) -> None:
    """Write one snapshot per day from per-symbol close series. Highs and lows
    are a fixed 1% band so ATR is predictable."""
    for i, day in enumerate(days):
        syms = [s for s, p in prices.items() if i < len(p)]
        close = [prices[s][i] for s in syms]
        vol = [volumes[s][i] if volumes else 1000.0 for s in syms]
        store.write_day(day, pd.DataFrame({
            "symbol": syms, "date": [day] * len(syms),
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close, "volume": vol,
        }))


@pytest.fixture
def store(tmp_path) -> BarStore:
    return BarStore(tmp_path)


# ===========================================================================
# parity with the reference indicators
# ===========================================================================

def test_vectorised_features_match_the_per_symbol_indicator_functions(store):
    """If these ever diverge, the scanner and the strategies are computing
    different numbers under the same names - the exact failure this project's
    indicator layer exists to prevent."""
    from desk.indicators.trend import sma
    from desk.indicators.volatility import atr_pct as ref_atr_pct
    from desk.indicators.volume import relative_volume as ref_rel_volume

    days = _sessions(120)
    rng = np.random.RandomState(11)
    px = list(100 * np.cumprod(1 + rng.normal(0, 0.012, 120)))
    vol = list(rng.randint(5_000, 50_000, 120).astype(float))
    _build(store, days, {"AAA.NS": px}, {"AAA.NS": vol})

    h = store.history(as_of=days[-1], lookback=120)
    r = run_stage1(h, min_bars=10)
    row = r.features.loc["AAA.NS"]
    one = h.series("AAA.NS")

    assert row["atr_pct"] == pytest.approx(ref_atr_pct(one).iloc[-1], rel=1e-9)
    assert row["rel_volume"] == pytest.approx(
        ref_rel_volume(one["volume"]).iloc[-1], rel=1e-9)
    for period in (20, 50):
        expected = 100.0 * (one["close"].iloc[-1]
                            / sma(one["close"], period).iloc[-1] - 1.0)
        assert row[f"dist_sma{period}_pct"] == pytest.approx(expected, rel=1e-9)


def test_relative_volume_excludes_todays_own_bar(store):
    """A 10x print must read as 10x, not as ~5x because it inflated its own
    baseline. Pinned here as well as in the indicator tests because the
    scanner recomputes it on the wide matrix rather than calling through."""
    days = _sessions(30)
    vol = [1000.0] * 29 + [10_000.0]
    _build(store, days, {"AAA.NS": [100.0] * 30}, {"AAA.NS": vol})
    r = run_stage1(store.history(as_of=days[-1], lookback=30), min_bars=10)
    assert r.features.loc["AAA.NS", "rel_volume"] == pytest.approx(10.0)


def test_true_range_falls_back_to_high_minus_low_on_the_first_bar(store):
    """np.fmax must prefer the defined operand over NaN. np.maximum would
    propagate the NaN and poison the whole Wilder average downstream."""
    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    r = run_stage1(store.history(as_of=days[-1], lookback=40), min_bars=10)
    # Flat closes with a fixed 1% band -> ATR is exactly the 2% band width.
    assert r.features.loc["AAA.NS", "atr_pct"] == pytest.approx(2.0, rel=1e-6)


# ===========================================================================
# point-in-time
# ===========================================================================

def test_as_of_truncates_the_window_even_when_history_holds_later_bars(store):
    days = _sessions(40)
    px = [100.0] * 39 + [500.0]            # a huge final bar
    _build(store, days, {"AAA.NS": px})
    h = store.history(as_of=days[-1], lookback=40)
    r = run_stage1(h, as_of=days[-2], min_bars=10)
    assert r.as_of == days[-2]
    assert r.features.loc["AAA.NS", "close"] == pytest.approx(100.0)


def test_an_as_of_not_present_in_the_history_is_refused(store):
    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    h = store.history(as_of=days[-1], lookback=40)
    with pytest.raises(ValueError, match="not in this history"):
        run_stage1(h, as_of=date(2030, 1, 1))


# ===========================================================================
# refusing to invent a number
# ===========================================================================

def test_a_window_longer_than_the_history_is_nan_not_a_shorter_average(store):
    """dist_sma200 over 60 bars must be NaN. A 60-bar mean reported as a
    200-day distance is the kind of number that looks right on a dashboard
    and is wrong."""
    days = _sessions(60)
    _build(store, days, {"AAA.NS": [100.0] * 60})
    r = run_stage1(store.history(as_of=days[-1], lookback=60), min_bars=10)
    row = r.features.loc["AAA.NS"]
    assert np.isnan(row["dist_sma200_pct"])
    assert not np.isnan(row["dist_sma20_pct"])
    assert row["bars"] == 60


def test_thin_symbols_are_dropped_rather_than_reported_as_rows_of_nan(store):
    """A row of NaNs invites a downstream .fillna(0), which turns 'unknown'
    into 'calm'. Dropping with a counted reason cannot be misread."""
    days = _sessions(60)
    _build(store, days, {
        "OLD.NS": [100.0] * 60,
        "NEW.NS": [50.0] * 5,               # listed five sessions ago
    })
    r = run_stage1(store.history(as_of=days[-1], lookback=60), min_bars=30)
    assert r.features.index.tolist() == ["OLD.NS"]
    assert r.excluded["fewer than 30 bars"] == 1
    assert r.universe_in == 2 and r.universe_out == 1


def test_a_feature_that_could_not_be_computed_never_raises_a_flag(store):
    """Comparisons against NaN are False. An unknown is not an opportunity -
    if this ever inverts, the scanner starts recommending the names it knows
    least about."""
    days = _sessions(35)
    _build(store, days, {"AAA.NS": [100.0] * 35})
    h = store.history(as_of=days[-1], lookback=35, columns=["close"])
    r = run_stage1(h, min_bars=10)
    row = r.features.loc["AAA.NS"]
    assert np.isnan(row["rel_volume"]) and np.isnan(row["atr_pct"])
    assert row["unusual_volume"] == False          # noqa: E712 - numpy bool
    assert row["unusual_move"] == False            # noqa: E712
    assert row["flag_count"] == 0


def test_missing_columns_leave_features_nan_rather_than_substituting(store):
    """History loaded without high/low must not fall back to close-to-close
    as if it were a true range."""
    days = _sessions(35)
    _build(store, days, {"AAA.NS": [100.0] * 35})
    r = run_stage1(store.history(as_of=days[-1], lookback=35,
                                 columns=["close"]), min_bars=10)
    assert np.isnan(r.features.loc["AAA.NS", "atr_pct"])
    assert np.isnan(r.features.loc["AAA.NS", "gap_pct"])


# ===========================================================================
# corporate actions
# ===========================================================================

def test_a_split_inside_the_window_drops_the_symbol_instead_of_reporting_it(store):
    """Raw bhavcopy closes halve across a 1:2 split. Left in, that is a -50%
    single-day return and a colossal ATR - the strongest 'signal' in the
    scan, and entirely fictional."""
    days = _sessions(60)
    split_day = days[30]
    px = [100.0] * 30 + [50.0] * 30
    _build(store, days, {"AAA.NS": px, "BBB.NS": [200.0] * 60})

    action = CorporateAction.split(symbol="AAA.NS", ex_date=split_day,
                                   old_shares=1, new_shares=2)
    r = run_stage1(store.history(as_of=days[-1], lookback=60),
                   actions=[action], min_bars=10)
    assert "AAA.NS" not in r.features.index
    assert r.unadjusted_actions == {"AAA.NS": 1}
    assert r.excluded["unadjusted corporate action in window"] == 1
    assert "BBB.NS" in r.features.index          # unaffected name survives


def test_an_action_outside_the_window_does_not_drop_the_symbol(store):
    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    old = CorporateAction.split(symbol="AAA.NS",
                                ex_date=days[0] - timedelta(days=400),
                                old_shares=1, new_shares=2)
    r = run_stage1(store.history(as_of=days[-1], lookback=40),
                   actions=[old], min_bars=10)
    assert "AAA.NS" in r.features.index
    assert r.unadjusted_actions == {}


def test_a_dividend_does_not_drop_the_symbol(store):
    """A dividend moves the price by its own size, not by a factor. Treating
    every action as disqualifying would drop most of the large caps every
    year for no reason."""
    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    div = CorporateAction.dividend(symbol="AAA.NS", ex_date=days[20],
                                   rupees_per_share=5.0)
    r = run_stage1(store.history(as_of=days[-1], lookback=40),
                   actions=[div], min_bars=10)
    assert "AAA.NS" in r.features.index


# ===========================================================================
# flags and reporting
# ===========================================================================

def test_unusual_volume_and_near_52w_high_flag_the_right_names(store):
    days = _sessions(60)
    rising = list(np.linspace(100, 200, 60))         # ends at its own high
    flat = [100.0] * 60
    vols = {"RISE.NS": [1000.0] * 60,
            "SPIKE.NS": [1000.0] * 59 + [9000.0]}
    _build(store, days, {"RISE.NS": rising, "SPIKE.NS": flat}, vols)
    r = run_stage1(store.history(as_of=days[-1], lookback=60), min_bars=10)

    assert bool(r.features.loc["RISE.NS", "near_52w_high"])
    assert not bool(r.features.loc["SPIKE.NS", "near_52w_high"])
    assert bool(r.features.loc["SPIKE.NS", "unusual_volume"])
    assert not bool(r.features.loc["RISE.NS", "unusual_volume"])


def test_pos_52w_is_0_at_the_low_and_100_at_the_high(store):
    days = _sessions(60)
    _build(store, days, {"UP.NS": list(np.linspace(100, 200, 60)),
                         "DOWN.NS": list(np.linspace(200, 100, 60))})
    r = run_stage1(store.history(as_of=days[-1], lookback=60), min_bars=10)
    # The fixture's high is close*1.01 and low is close*0.99, so the extremes
    # sit just inside 0 and 100 rather than exactly on them.
    assert r.features.loc["UP.NS", "pos_52w_pct"] > 98
    assert r.features.loc["DOWN.NS", "pos_52w_pct"] < 2


def test_every_run_declares_what_it_could_not_check(store):
    """Sector strength, news flags and relative strength are Stage 1 filters
    in the master plan. Their absence must appear on every result, not in a
    docstring nobody reads at 9am."""
    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    r = run_stage1(store.history(as_of=days[-1], lookback=40), min_bars=10)
    joined = " ".join(r.unavailable)
    assert "Sector strength" in joined
    assert "News and event flags" in joined
    assert "Relative strength" in joined
    assert np.isnan(r.features.loc["AAA.NS", "rs_rank"])


def test_supplying_a_benchmark_computes_relative_strength(store):
    days = _sessions(80)
    _build(store, days, {"STRONG.NS": list(np.linspace(100, 200, 80)),
                         "WEAK.NS": [100.0] * 80})
    bench = pd.Series(np.linspace(100, 150, 80), index=days)
    r = run_stage1(store.history(as_of=days[-1], lookback=80),
                   benchmark=bench, min_bars=10)
    assert "Relative strength" not in " ".join(r.unavailable)
    # STRONG outpaces the benchmark, WEAK loses to it.
    assert r.features.loc["STRONG.NS", "rs_rank"] > \
           r.features.loc["WEAK.NS", "rs_rank"]


def test_the_feature_table_shape_is_pinned(store):
    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    r = run_stage1(store.history(as_of=days[-1], lookback=40), min_bars=10)
    assert list(r.features.columns) == list(FEATURE_COLUMNS)
    assert r.features.index.name == "symbol"


def test_coverage_is_carried_through_so_gaps_stay_visible(store):
    """A Stage 1 feature table whose history had holes must not look like one
    computed over clean data."""
    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    h = store.history(as_of=days[-1], lookback=40)
    r = run_stage1(h, min_bars=10)
    assert r.coverage is h.coverage
    assert "UNCHECKED" in r.summary()


def test_flagged_returns_only_names_with_a_flag(store):
    days = _sessions(60)
    _build(store, days, {"QUIET.NS": [100.0] * 60,
                         "SPIKE.NS": [100.0] * 60},
           {"QUIET.NS": [1000.0] * 60, "SPIKE.NS": [1000.0] * 59 + [9000.0]})
    r = run_stage1(store.history(as_of=days[-1], lookback=60), min_bars=10)
    assert "SPIKE.NS" in r.flagged.index
    assert (r.flagged["flag_count"] > 0).all()


def test_an_empty_history_is_refused_with_the_command_to_fix_it(store):
    from desk.store import History
    empty = store.history(as_of=date(2026, 9, 11), lookback=5)
    with pytest.raises(ValueError, match="refresh bhavcopy"):
        run_stage1(empty)
    assert isinstance(empty, History)


# ===========================================================================
# REGRESSION: the unguarded-division bug found by review on 2026-09-15
# ===========================================================================

def test_a_near_zero_historical_close_cannot_fabricate_a_top_momentum_signal(store):
    """REGRESSION, measured on the real code before the fix: a single bad tick
    recorded as 0.0001 twenty sessions back produced ret_20d_pct of 1.39e8 and
    scored 94.4/100 in Stage 2 - the best momentum in the universe, entirely
    fabricated. Stage 0's min_price only checks TODAY's close, so a glitched
    historical print passes every upstream filter, and it is not a corporate
    action so unadjusted_actions cannot catch it either."""
    from desk.contracts.enums import Regime
    from desk.scanner.stage2 import run_stage2

    days = _sessions(60)
    px = [100.0] * 60
    px[-21] = 0.0001                       # one glitched print, 20 back
    _build(store, days, {"GLITCH.NS": px, "NORMAL.NS": [100.0] * 60})

    r = run_stage1(store.history(as_of=days[-1], lookback=60), min_bars=10)
    ret = r.features.loc["GLITCH.NS", "ret_20d_pct"]
    assert np.isnan(ret), f"expected NaN, got {ret}"

    # And it must not then win the ranking on the strength of that NaN.
    r2 = run_stage2(r, regime=Regime.TRENDING_UP, min_factors=1,
                    flagged_only=False)
    assert pd.isna(r2.ranked.loc["GLITCH.NS", "rank_momentum"])


def test_an_exact_zero_denominator_yields_nan_not_inf(store):
    """inf is WORSE than NaN here: pd.isna(inf) is False, so it is invisible
    to every NaN-based guard in the project, and Series.rank(pct=True) ranks
    it as the maximum. It would win the scan outright."""
    days = _sessions(40)
    px = [100.0] * 40
    px[-2] = 0.0
    _build(store, days, {"AAA.NS": px})
    r = run_stage1(store.history(as_of=days[-1], lookback=40), min_bars=10)
    row = r.features.loc["AAA.NS"]
    assert not np.isinf(row["ret_1d_pct"])
    assert np.isnan(row["ret_1d_pct"])


def test_no_feature_column_ever_contains_an_infinity(store):
    """A blanket guard. Every division in this module masks a non-positive
    denominator, and inf is swept once more at the boundary - so a future
    feature added without that care fails here rather than in production."""
    days = _sessions(60)
    rng = np.random.RandomState(3)
    px = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 60)))
    px[10] = 0.0
    px[25] = 1e-9
    vols = [1000.0] * 60
    vols[30] = 0.0
    _build(store, days, {"AAA.NS": px, "BBB.NS": [50.0] * 60},
           {"AAA.NS": vols, "BBB.NS": [1000.0] * 60})
    r = run_stage1(store.history(as_of=days[-1], lookback=60), min_bars=10)
    numeric = r.features.select_dtypes(include="number")
    assert not np.isinf(numeric.to_numpy(dtype="float64")).any()


def test_a_zero_factor_action_is_not_waved_through_as_a_dividend(store):
    """`factor or 1.0` would turn 0.0 into 1.0, because 0.0 is falsy - and a
    price-affecting action would slip through unadjusted. load_actions
    rejects factor <= 0 today, so this is a landmine rather than a live bug,
    but a CorporateAction built directly would arm it."""
    from desk.marketdata.corporate_actions import ActionType, CorporateAction

    days = _sessions(40)
    _build(store, days, {"AAA.NS": [100.0] * 40})
    weird = CorporateAction(symbol="AAA.NS", ex_date=days[20],
                            type=ActionType.SPLIT, factor=0.0)
    r = run_stage1(store.history(as_of=days[-1], lookback=40),
                   actions=[weird], min_bars=10)
    assert r.unadjusted_actions == {"AAA.NS": 1}
    assert "AAA.NS" not in r.features.index
