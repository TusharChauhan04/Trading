"""Scanner Stage 2 - cross-sectional scoring and ranking.

The failure mode this stage invites is a plausible ranking built on missing
data: a name scored on one lucky factor sitting above a name scored honestly
on five. Most of these tests exist to pin the arithmetic that prevents that.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from desk.contracts.enums import Regime
from desk.scanner.stage1 import FEATURE_COLUMNS, Stage1Result
from desk.scanner.stage2 import (
    DEFAULT_FACTORS,
    FactorSpec,
    run_stage2,
)
from desk.store import Coverage

AS_OF = date(2026, 9, 11)


def _features(rows: dict[str, dict]) -> pd.DataFrame:
    """Build a Stage 1-shaped feature table from partial per-symbol dicts.
    Anything unspecified is NaN, which is the point - Stage 2 must cope."""
    flags = ("compressed", "unusual_volume", "unusual_move",
             "near_52w_high", "extended")
    df = pd.DataFrame(index=pd.Index(list(rows), name="symbol"),
                      columns=list(FEATURE_COLUMNS), dtype="float64")
    # Booleans are assigned column-wise, never through .loc into a float64
    # column - pandas warns about that and will raise in a future version.
    for flag in flags:
        df[flag] = [bool(rows[s].get(flag, False)) for s in df.index]
    for sym, vals in rows.items():
        for k, v in vals.items():
            if k not in flags:
                df.loc[sym, k] = v
    df["flag_count"] = df["flag_count"].fillna(1).astype("int64")
    df["bars"] = df["bars"].fillna(250).astype("int64")
    return df


def _stage1(rows: dict[str, dict]) -> Stage1Result:
    feats = _features(rows)
    return Stage1Result(
        as_of=AS_OF, features=feats,
        coverage=Coverage(start=None, end=AS_OF, days_loaded=250,
                          sessions_expected=250, calendar_checked=True),
        universe_in=len(feats), universe_out=len(feats),
    )


def _basic(n: int = 10) -> Stage1Result:
    """n names with monotonically improving factors, so the expected ranking
    is unambiguous: SYM0 worst, SYM{n-1} best."""
    return _stage1({
        f"SYM{i}.NS": {
            "ret_20d_pct": float(i), "dist_sma50_pct": float(i),
            "pos_52w_pct": float(i * 10), "rel_volume": 1.0 + i * 0.1,
            "atr_pct_rank": float(100 - i * 10),      # lower is better
        }
        for i in range(n)
    })


# ===========================================================================
# ranking arithmetic
# ===========================================================================

def test_the_best_name_on_every_factor_ranks_first(_=None):
    r = run_stage2(_basic(), regime=Regime.TRENDING_UP)
    assert r.ranked.index[0] == "SYM9.NS"
    assert r.ranked.index[-1] == "SYM0.NS"
    assert r.ranked["score"].is_monotonic_decreasing


def test_direction_minus_one_inverts_the_factor(_=None):
    """`calmness` uses atr_pct_rank with direction=-1: the LOWEST volatility
    percentile must earn the HIGHEST factor rank. Getting this sign backwards
    would silently rank the most dangerous names first."""
    s1 = _stage1({
        "CALM.NS": {"atr_pct_rank": 5.0, "ret_20d_pct": 1.0,
                    "dist_sma50_pct": 1.0, "pos_52w_pct": 50.0,
                    "rel_volume": 1.0},
        "WILD.NS": {"atr_pct_rank": 95.0, "ret_20d_pct": 1.0,
                    "dist_sma50_pct": 1.0, "pos_52w_pct": 50.0,
                    "rel_volume": 1.0},
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP)
    assert r.ranked.loc["CALM.NS", "rank_calmness"] > \
           r.ranked.loc["WILD.NS", "rank_calmness"]
    assert r.ranked.index[0] == "CALM.NS"


def test_scores_are_a_weight_respecting_mean_of_the_percentile_ranks(_=None):
    """Hand-verified: two names, so ranks are 50 and 100 on every factor."""
    s1 = _stage1({
        "HI.NS": {"ret_20d_pct": 10.0, "dist_sma50_pct": 10.0,
                  "pos_52w_pct": 90.0, "rel_volume": 3.0,
                  "atr_pct_rank": 10.0},
        "LO.NS": {"ret_20d_pct": 1.0, "dist_sma50_pct": 1.0,
                  "pos_52w_pct": 10.0, "rel_volume": 1.0,
                  "atr_pct_rank": 90.0},
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP)
    # HI wins every factor -> every rank is 100 -> weighted mean is 100.
    assert r.ranked.loc["HI.NS", "score"] == pytest.approx(100.0)
    assert r.ranked.loc["LO.NS", "score"] == pytest.approx(50.0)


def test_a_single_extreme_outlier_cannot_distort_the_ranking(_=None):
    """The reason ranks are used instead of z-scores. One 10,000% print must
    cost every other name exactly one rank place, not compress the whole
    distribution toward zero."""
    rows = {f"SYM{i}.NS": {"ret_20d_pct": float(i), "dist_sma50_pct": float(i),
                           "pos_52w_pct": float(i * 5), "rel_volume": 1.0,
                           "atr_pct_rank": 50.0}
            for i in range(20)}
    baseline = run_stage2(_stage1(rows), regime=Regime.TRENDING_UP)
    rows["MOON.NS"] = {"ret_20d_pct": 10_000.0, "dist_sma50_pct": 10_000.0,
                       "pos_52w_pct": 100.0, "rel_volume": 1.0,
                       "atr_pct_rank": 50.0}
    withb = run_stage2(_stage1(rows), regime=Regime.TRENDING_UP)

    assert withb.ranked.index[0] == "MOON.NS"
    # Every original name keeps its relative order.
    before = [s for s in baseline.ranked.index]
    after = [s for s in withb.ranked.index if s != "MOON.NS"]
    assert before == after


# ===========================================================================
# refusing to score on missing data
# ===========================================================================

def test_a_missing_factor_is_not_treated_as_the_median(_=None):
    """Handing a NaN factor the 50th percentile would be inventing data. The
    composite must average over AVAILABLE factors only and say how many."""
    s1 = _stage1({
        "FULL.NS": {"ret_20d_pct": 5.0, "dist_sma50_pct": 5.0,
                    "pos_52w_pct": 50.0, "rel_volume": 2.0,
                    "atr_pct_rank": 50.0},
        "PART.NS": {"ret_20d_pct": 9.0, "dist_sma50_pct": 9.0,
                    "pos_52w_pct": 90.0},      # no volume, no volatility
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP)
    assert r.ranked.loc["FULL.NS", "factors_used"] == 5
    assert r.ranked.loc["PART.NS", "factors_used"] == 3
    assert pd.isna(r.ranked.loc["PART.NS", "rank_participation"])


def test_names_below_min_factors_are_dropped_with_a_counted_reason(_=None):
    s1 = _stage1({
        "GOOD.NS": {"ret_20d_pct": 5.0, "dist_sma50_pct": 5.0,
                    "pos_52w_pct": 50.0, "rel_volume": 2.0,
                    "atr_pct_rank": 50.0},
        "THIN.NS": {"ret_20d_pct": 9.0},        # one factor only
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP, min_factors=3)
    assert r.ranked.index.tolist() == ["GOOD.NS"]
    assert r.excluded["fewer than 3 usable factors"] == 1
    assert r.universe_in == 2 and r.universe_out == 1


def test_a_partly_scored_name_is_not_penalised_by_the_full_weight_total(_=None):
    """The denominator must be the weights that actually contributed. Divide
    by the full total instead and every name with missing data is pushed to
    the bottom regardless of how well it scored on what it has."""
    s1 = _stage1({
        "A.NS": {"ret_20d_pct": 9.0, "dist_sma50_pct": 9.0,
                 "pos_52w_pct": 90.0},
        "B.NS": {"ret_20d_pct": 1.0, "dist_sma50_pct": 1.0,
                 "pos_52w_pct": 10.0},
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP, min_factors=3)
    # A beats B on all three available factors -> a perfect 100, not 100
    # scaled down by the two factors neither name had.
    assert r.ranked.loc["A.NS", "score"] == pytest.approx(100.0)


def test_an_entirely_nan_factor_is_silenced_with_a_reason(_=None):
    s1 = _stage1({
        f"SYM{i}.NS": {"ret_20d_pct": float(i), "dist_sma50_pct": float(i),
                       "pos_52w_pct": float(i * 10), "atr_pct_rank": 50.0}
        for i in range(5)
    })                                            # rel_volume never set
    r = run_stage2(s1, regime=Regime.TRENDING_UP)
    assert "participation" in r.silenced
    assert "entirely NaN" in r.silenced["participation"]
    assert all(f.name != "participation" for f in r.factors)


# ===========================================================================
# regime gating
# ===========================================================================

def test_a_hostile_regime_silences_a_factor_rather_than_down_weighting_it(_=None):
    """Same rule as desk.strategies.catalog.eligible(): momentum in a crisis
    is not a weak signal, it is a wrong one."""
    r = run_stage2(_basic(20), regime=Regime.CRISIS, min_factors=1)
    assert "momentum" in r.silenced
    assert "hostile" in r.silenced["momentum"]
    assert "rank_momentum" not in r.ranked.columns
    assert {f.name for f in r.factors} == {"participation", "calmness"}


def test_a_crisis_that_silences_most_factors_produces_no_ranking_at_all(_=None):
    """A first-class NO TRADE. With the default set, a crisis leaves two
    factors - below the three needed to rank - so the scan returns nothing
    rather than ranking the universe on participation and calmness alone.
    The excluded reason must name the CAUSE, or whoever reads the log goes
    looking for a data problem that does not exist."""
    r = run_stage2(_basic(20), regime=Regime.CRISIS, min_factors=3)
    assert r.universe_out == 0
    assert r.ranked.empty
    reason = next(iter(r.excluded))
    assert "survived regime crisis" in reason
    assert "crisis" in r.summary()


def test_participation_is_never_silenced(_=None):
    """Unusual volume is informative in every regime including a crisis -
    the one factor whose meaning does not depend on the weather."""
    for regime in Regime:
        r = run_stage2(_basic(20), regime=regime, min_factors=1)
        assert "participation" not in r.silenced


# ===========================================================================
# penalty
# ===========================================================================

def test_over_extension_is_a_penalty_not_a_filter(_=None):
    """Being extended makes the ENTRY worse, not the thesis wrong. The name
    must survive with a lower score so a human can still see it - the risk
    engine's stop-distance gate is what actually refuses a bad entry."""
    s1 = _stage1({
        "CLEAN.NS": {"ret_20d_pct": 5.0, "dist_sma50_pct": 5.0,
                     "pos_52w_pct": 50.0, "rel_volume": 2.0,
                     "atr_pct_rank": 50.0},
        "STRETCH.NS": {"ret_20d_pct": 9.0, "dist_sma50_pct": 9.0,
                       "pos_52w_pct": 90.0, "rel_volume": 3.0,
                       "atr_pct_rank": 40.0, "extended": True},
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP, extended_penalty=15.0)
    assert "STRETCH.NS" in r.ranked.index              # not filtered out
    assert r.ranked.loc["STRETCH.NS", "penalty"] == pytest.approx(15.0)
    assert r.ranked.loc["STRETCH.NS", "score"] == pytest.approx(85.0)


def test_scores_stay_inside_0_and_100(_=None):
    s1 = _stage1({
        "A.NS": {"ret_20d_pct": 1.0, "dist_sma50_pct": 1.0,
                 "pos_52w_pct": 10.0, "rel_volume": 1.0,
                 "atr_pct_rank": 90.0, "extended": True},
        "B.NS": {"ret_20d_pct": 9.0, "dist_sma50_pct": 9.0,
                 "pos_52w_pct": 90.0, "rel_volume": 3.0,
                 "atr_pct_rank": 10.0},
    })
    r = run_stage2(s1, regime=Regime.TRENDING_UP, extended_penalty=99.0)
    assert (r.ranked["score"] >= 0).all() and (r.ranked["score"] <= 100).all()


# ===========================================================================
# explainability and honesty
# ===========================================================================

def test_every_score_can_be_read_back_to_its_inputs(_=None):
    """Section 32: a recommendation must be traceable. The per-factor rank
    columns and explain() are that trace, produced at the cheapest stage
    rather than reconstructed later."""
    r = run_stage2(_basic(), regime=Regime.TRENDING_UP)
    top = r.ranked.index[0]
    text = r.explain(top)
    assert top in text
    for f in r.factors:
        assert f.name in text
        assert f"rank_{f.name}" in r.ranked.columns


def test_explain_says_so_for_a_name_that_is_not_ranked(_=None):
    r = run_stage2(_basic(), regime=Regime.TRENDING_UP)
    assert "not in the Stage 2 ranking" in r.explain("NOPE.NS")


def test_explain_names_the_silenced_factors(_=None):
    r = run_stage2(_basic(20), regime=Regime.CRISIS, min_factors=1)
    text = r.explain(r.ranked.index[0])
    assert "SILENCED momentum" in text


def test_untrusted_strategies_are_declared_as_not_voting(_=None):
    """The maturity ladder is the point: even once the six strategy scripts
    are back on disk, a strategy below walk_forward may not influence a live
    shortlist. This must be stated on every run, not assumed."""
    r = run_stage2(_basic(), regime=Regime.TRENDING_UP)
    joined = " ".join(r.unavailable)
    assert "maturity ladder" in joined
    assert "Fundamental filters" in joined
    assert "Sentiment" in joined


def test_an_empty_stage1_result_ranks_nothing_without_raising(_=None):
    """NO TRADE is a successful outcome, not an exception."""
    s1 = _stage1({})
    s1.features = s1.features.iloc[0:0]
    r = run_stage2(s1, regime=Regime.TRENDING_UP)
    assert r.ranked.empty
    assert r.universe_out == 0


def test_flagged_only_true_scores_only_what_stage1_flagged(_=None):
    s1 = _stage1({
        "FLAGGED.NS": {"ret_20d_pct": 5.0, "dist_sma50_pct": 5.0,
                       "pos_52w_pct": 50.0, "rel_volume": 2.0,
                       "atr_pct_rank": 50.0, "flag_count": 2},
        "QUIET.NS": {"ret_20d_pct": 9.0, "dist_sma50_pct": 9.0,
                     "pos_52w_pct": 90.0, "rel_volume": 3.0,
                     "atr_pct_rank": 10.0, "flag_count": 0},
    })
    assert run_stage2(s1, regime=Regime.TRENDING_UP,
                      flagged_only=True).ranked.index.tolist() == ["FLAGGED.NS"]
    assert len(run_stage2(s1, regime=Regime.TRENDING_UP,
                          flagged_only=False).ranked) == 2


# ===========================================================================
# factor definitions
# ===========================================================================

def test_a_factor_with_a_bad_direction_or_weight_is_refused(_=None):
    with pytest.raises(ValueError, match="direction"):
        FactorSpec("x", "ret_20d_pct", 0, 1.0)
    with pytest.raises(ValueError, match="weight"):
        FactorSpec("x", "ret_20d_pct", 1, 0.0)


def test_the_default_factor_set_avoids_near_duplicate_columns(_=None):
    """Section 37's ensemble-bias rule applied to factors: six correlated
    momentum variants would be one voice counted six times. In particular
    dist_sma20_pct must NOT appear alongside dist_sma50_pct - they are
    near-duplicates on daily bars, and dist_sma20 is already used as the
    over-extension penalty."""
    cols = [f.column for f in DEFAULT_FACTORS]
    assert len(cols) == len(set(cols))
    assert "dist_sma20_pct" not in cols


def test_a_custom_factor_set_is_honoured(_=None):
    custom = (FactorSpec("only_momentum", "ret_20d_pct", +1, 1.0),)
    r = run_stage2(_basic(), regime=Regime.TRENDING_UP, factors=custom,
                   min_factors=1)
    assert [f.name for f in r.factors] == ["only_momentum"]
    assert list(r.ranked.columns) == ["score", "factors_used", "penalty",
                                      "rank_only_momentum"]


def test_a_factor_naming_a_column_stage1_does_not_produce_is_silenced(_=None):
    custom = DEFAULT_FACTORS + (
        FactorSpec("imaginary", "does_not_exist", +1, 1.0),)
    r = run_stage2(_basic(), regime=Regime.TRENDING_UP, factors=custom)
    assert "imaginary" in r.silenced
    assert "not in the Stage 1 table" in r.silenced["imaginary"]


# ===========================================================================
# REVERSAL_FACTORS - the measured alternative, not yet adopted
# ===========================================================================

def test_the_default_factor_set_is_still_what_production_ranks_on():
    """REVERSAL_FACTORS exists to be MEASURED against the default, not to
    replace it quietly. Adopting a factor set because its information
    coefficients look good, without a net-of-costs backtest saying it earns
    more, is how a measurement becomes a belief."""
    from desk.scanner.stage2 import DEFAULT_FACTORS, run_stage2
    import inspect
    assert inspect.signature(run_stage2).parameters["factors"].default \
        is DEFAULT_FACTORS
    assert [f.name for f in DEFAULT_FACTORS] == [
        "momentum", "trend", "range_position", "participation", "calmness"]


def test_every_reversal_factor_points_the_way_the_measurement_said():
    """The whole content of this factor set is its SIGNS. Every measured
    information coefficient was negative, so every direction here is -1;
    a +1 appearing in this tuple would mean someone reverted a sign
    without redoing the study."""
    from desk.scanner.stage2 import REVERSAL_FACTORS
    assert REVERSAL_FACTORS, "the set must not be empty"
    for f in REVERSAL_FACTORS:
        assert f.direction == -1, (
            f"{f.name} is +1, but every IC measured for these columns was "
            f"negative - see the table above REVERSAL_FACTORS")


def test_the_reversal_set_avoids_stacking_correlated_price_moves():
    """The failure the default set's own docstring warns about: several
    measures of "this went up recently" is one voice counted many times.
    ret_1d_pct is the strongest single column measured and is deliberately
    absent - one-day reversal is the bid-ask-bounce artifact, and at a
    0.42% round trip it is the first thing costs eat."""
    from desk.scanner.stage2 import REVERSAL_FACTORS
    cols = {f.column for f in REVERSAL_FACTORS}
    price_move = cols & {"ret_1d_pct", "ret_5d_pct", "ret_20d_pct",
                         "dist_sma20_pct", "dist_sma50_pct"}
    assert len(price_move) == 1, (
        f"{len(price_move)} correlated price-move factors: {price_move}")
    assert "ret_1d_pct" not in cols
    assert "pos_52w_pct" not in cols, "measured IC +0.007, t 0.66 - no signal"


def test_the_reversal_set_actually_ranks_and_inverts_the_default_order():
    """Wired enough to run: it must produce a real ranking through the same
    run_stage2, and it must disagree with the default - a corrected sign
    that changes nothing would mean the signs are not reaching the score."""
    from desk.scanner.stage2 import DEFAULT_FACTORS, REVERSAL_FACTORS
    rows = {
        # WINNER on the default set: big 5d gain, heavy volume, calm.
        "MOMO.NS": {"ret_5d_pct": 12.0, "rel_volume": 3.0,
                    "atr_pct_rank": 20.0, "ret_20d_pct": 30.0,
                    "dist_sma50_pct": 15.0, "pos_52w_pct": 95.0},
        # LOSER on the default set: fell, quiet, calm.
        "QUIET.NS": {"ret_5d_pct": -8.0, "rel_volume": 0.4,
                     "atr_pct_rank": 20.0, "ret_20d_pct": -12.0,
                     "dist_sma50_pct": -9.0, "pos_52w_pct": 20.0},
    }
    s1 = _stage1(rows)
    d = run_stage2(s1, factors=DEFAULT_FACTORS, flagged_only=False,
                   min_factors=1)
    r = run_stage2(s1, factors=REVERSAL_FACTORS, flagged_only=False,
                   min_factors=1)
    assert d.ranked.index[0] == "MOMO.NS", "default should favour the mover"
    assert r.ranked.index[0] == "QUIET.NS", "reversal should favour the loser"


def test_flagged_only_still_defaults_true_despite_the_measurement():
    """The D1 finding is recorded in run_stage2's docstring and has NOT
    been acted on, deliberately.

    Stage 1's flag gate admits a median 34.9% of the universe and those
    names underperform the ones it discards by 0.149% per 5 sessions
    (t -2.80), driven by unusual_move (-0.807%, t -4.75) and
    unusual_volume (-0.318%, t -4.24). That is a measurement about
    SELECTION. Whether a different pool earns more money is a different
    question, and this project has already been caught once promoting the
    first into the second.

    So the default stands until a net-of-costs backtest moves it, and this
    test is the tripwire: changing the default without the measurement is
    exactly the shortcut the docstring argues against."""
    import inspect
    from desk.scanner.stage2 import run_stage2
    assert inspect.signature(run_stage2).parameters["flagged_only"].default \
        is True
    doc = run_stage2.__doc__
    assert "MEASURED 2026-09-22" in doc, "the finding must travel with the code"
    assert "NOT CHANGED HERE" in doc
