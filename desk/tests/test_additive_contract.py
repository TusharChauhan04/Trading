"""This session's extraction was additive. These tests keep it that way.

A regression audit ran the whole funnel on the current tree and on dd21486 - the
commit before the extraction began - against the same bhavcopy and the same date.
Result: 995 symbols both, index identical, and ALL 32 pre-existing Stage 1
feature columns bit-identical at atol=0. The daily plan's decisions were
identical. The fundamentals table gained 7 columns and every pre-existing
column's checksum matched.

That audit needed a git worktree of an old commit, so it cannot live in the test
suite. What CAN live here is the guarantee it established: the original columns
are still present, still in their original order, and nothing downstream reads a
column by position. A future change that reorders or drops one fails here
immediately rather than at the point where a strategy silently reads the wrong
number.

ORDER MATTERS AND IS NOT PEDANTRY. FEATURE_COLUMNS is documented as "in report
order" and Stage 1's own docstring says a caller may "assert on the shape".
Anything constructing a frame positionally from that tuple - a parquet round
trip, a numpy view, a dashboard column index - breaks silently if the order
moves.
"""

from __future__ import annotations

from desk.research.fundamentals import COLUMNS as FUND_COLUMNS
from desk.scanner.stage1 import FEATURE_COLUMNS

#: Stage 1's feature columns exactly as they stood at dd21486, in order.
#: Captured from the regression audit, not retyped from memory.
BASELINE_FEATURES = (
    "bars", "close", "volume",
    "ret_1d_pct", "ret_5d_pct", "ret_20d_pct",
    "rel_volume", "gap_pct",
    "atr_pct", "atr_pct_rank",
    "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "pos_52w_pct", "high_52w", "low_52w",
    "swing_low", "swing_low_age", "low_20",
    "prior_high_20", "prior_low_10",
    "bb_width_pct", "bb_mid", "bb_lower", "rsi_14", "compressed",
    "rs_rank",
    "unusual_volume", "unusual_move", "near_52w_high", "extended",
    "flag_count",
)

#: The fundamentals table's columns at dd21486, in order.
BASELINE_FUNDAMENTALS = (
    "nature", "period_end", "disclosed_at", "days_since_filing",
    "revenue", "profit_after_tax", "eps_basic", "net_margin_pct",
    "revenue_growth_yoy_pct", "profit_growth_yoy_pct", "margin_change_pp",
)

#: What this session added. Named so the diff is a decision rather than drift -
#: adding a column means editing this tuple on purpose.
ADDED_FEATURES = ("adx_14", "plus_di_14", "minus_di_14")
ADDED_FUNDAMENTALS = ("shares_outstanding", "ebit", "ebitda",
                      "effective_tax_rate_pct", "nopat", "op_margin_pct",
                      "interest_cover", "industry")


def test_no_stage1_feature_column_was_removed() -> None:
    """Every column a downstream stage might read is still there."""
    missing = [c for c in BASELINE_FEATURES if c not in FEATURE_COLUMNS]
    assert not missing, f"Stage 1 lost columns: {missing}"


def test_the_original_feature_order_is_preserved() -> None:
    """The pre-existing columns appear in their original relative order.

    FEATURE_COLUMNS is documented as "in report order", and anything building a
    frame positionally from it breaks silently if that order moves.
    """
    kept = [c for c in FEATURE_COLUMNS if c in BASELINE_FEATURES]
    assert kept == list(BASELINE_FEATURES), (
        "the pre-existing feature columns were reordered:\n"
        f"  was {list(BASELINE_FEATURES)}\n  now {kept}")


def test_stage1_additions_are_exactly_what_was_declared() -> None:
    """A new column must be added here on purpose, not discovered later."""
    extra = [c for c in FEATURE_COLUMNS
             if c not in BASELINE_FEATURES and c not in ADDED_FEATURES]
    assert not extra, (
        f"undeclared Stage 1 columns: {extra}. Add them to ADDED_FEATURES "
        f"deliberately, and say what measured them.")


def test_no_fundamentals_column_was_removed() -> None:
    """Stage 2's fundamental filters read these by name."""
    missing = [c for c in BASELINE_FUNDAMENTALS if c not in FUND_COLUMNS]
    assert not missing, f"the fundamentals table lost columns: {missing}"


def test_the_original_fundamentals_order_is_preserved() -> None:
    kept = [c for c in FUND_COLUMNS if c in BASELINE_FUNDAMENTALS]
    assert kept == list(BASELINE_FUNDAMENTALS), (
        "pre-existing fundamentals columns were reordered:\n"
        f"  was {list(BASELINE_FUNDAMENTALS)}\n  now {kept}")


def test_fundamentals_additions_are_exactly_what_was_declared() -> None:
    extra = [c for c in FUND_COLUMNS
             if c not in BASELINE_FUNDAMENTALS
             and c not in ADDED_FUNDAMENTALS]
    assert not extra, f"undeclared fundamentals columns: {extra}"


def test_the_additions_come_after_the_originals_in_both_tables() -> None:
    """Appending is safe for a positional reader; inserting is not.

    A column spliced into the middle shifts every index after it, which is the
    failure a positional reader cannot detect.
    """
    for name, cols, base in (("Stage 1", FEATURE_COLUMNS, BASELINE_FEATURES),
                             ("fundamentals", FUND_COLUMNS,
                              BASELINE_FUNDAMENTALS)):
        first_new = next((i for i, c in enumerate(cols) if c not in base), None)
        if first_new is None:
            continue
        after = [c for c in cols[first_new:] if c in base]
        assert not after, (
            f"{name}: pre-existing columns {after} appear AFTER a new one, so "
            f"a new column was inserted rather than appended")


def test_the_cost_model_defaults_are_untouched() -> None:
    """Every net-R figure in this project assumes these three numbers.

    The regression audit confirmed them identical across the session; this keeps
    them that way. `sized_for` was ADDED as an alternative constructor and must
    not have changed what `CostModel()` means.
    """
    from desk.backtest.costs import CostModel

    c = CostModel()
    assert c.slippage_bps == 15.0, "the slippage guess moved"
    assert c.stt_sell_pct == 0.100
    assert c.brokerage_pct == 0.0
    assert round(c.round_trip_pct(), 6) == 0.422245, (
        "the round-trip cost changed - every measured net R in this project "
        "and in the roadmap assumes 0.422%")


def test_the_risk_config_defaults_are_untouched() -> None:
    """The config's defaults are a contract, so a change to one must be
    deliberate and attributable rather than a side effect.

    risk_pct IS NOW 0.5, NOT 1.0, and that is the one default this project has
    ever changed. The drawdown work had argued 1.0 was too high for the
    measured edge - P(20% drawdown) of 32.6% against 2.5% at 0.5% - and this
    test deliberately held the line at 1.0 on the grounds that changing
    someone's sizing silently is not the code's decision. The OPERATOR chose
    0.5 on 2026-10-06. The guard now records the new value, which is what it
    is for: it did its job by failing when the default moved."""
    from desk.risk.engine import RiskConfig

    c = RiskConfig(capital=100_000)
    assert c.risk_pct == 0.5
    assert c.min_risk_reward == 1.5
    assert c.max_open_positions == 5
