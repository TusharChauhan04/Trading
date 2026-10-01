"""Screens must not silently skip, and Pine must not emit a rule that can't fire.

Two failure modes, both of which produce output that looks fine:

  - A blocked screen returning an empty frame instead of raising. "Nothing
    passed" and "this cannot run on our data" are different facts, and only the
    second is true of the value and quality screens.
  - Pine whose channel includes the current bar. `close` can never exceed the
    20-bar high that already contains today's high, so the strategy tester
    reports zero trades and it reads as a quiet market. This project has now met
    that off-by-one three times: its own Donchian adapter, the preset
    translation, and upstream's Pine exporter.
"""

from __future__ import annotations

import pandas as pd
import pytest

from desk.research.screens import (
    FIELD_MAP, SCREEN_STATUS, SCREENS, run_screen, screen_report,
    verify_against_upstream,
)
from desk.strategies.pine import MEASURED, to_pine


def _table() -> pd.DataFrame:
    """Four symbols: one clear pass, three failing one rule each."""
    return pd.DataFrame(
        {
            "revenue_growth_yoy_pct": [25.0, 2.0, 30.0, 40.0],
            "profit_growth_yoy_pct": [30.0, 40.0, 1.0, 50.0],
            "net_margin_pct": [15.0, 20.0, 18.0, 2.0],
            "op_margin_pct": [22.0, 25.0, 20.0, 5.0],
        },
        index=["PASS", "SLOWREV", "SLOWEPS", "THINMARGIN"],
    )


# -- screens: three, not six, and only one runs -----------------------------

def test_there_are_three_screens_not_six() -> None:
    """The manifest and the salvage map both said six. The file has three.

    Same class of miscount as "19 strategy templates" for a dict of 6, so it is
    pinned here.
    """
    assert len(SCREENS) == 3
    assert set(SCREENS) == {"value", "quality", "growth"}


def test_only_growth_can_run_on_our_data() -> None:
    assert SCREENS["growth"].runnable
    assert not SCREENS["value"].runnable
    assert not SCREENS["quality"].runnable


def test_a_blocked_screen_raises_and_names_the_missing_fields() -> None:
    """Raising, not returning empty. An empty frame reads as "nothing passed"."""
    for name in ("value", "quality"):
        with pytest.raises(NotImplementedError, match="roe_pct"):
            run_screen(name, _table())


def test_quality_is_one_field_short_not_two() -> None:
    """op_margin_pct is now derived from EBIT and revenue, so if a
    balance-sheet source ever appears this screen costs one more derivation
    rather than two. Worth recording because it changes the priority."""
    assert set(SCREENS["quality"].missing_fields) == {
        "roe_pct", "debt_to_market_cap"}
    assert "op_margin_pct" in FIELD_MAP


def test_the_status_map_explains_every_screen() -> None:
    assert set(SCREEN_STATUS) == set(SCREENS)
    assert SCREEN_STATUS["growth"] == "runnable"
    assert "roe_pct" in SCREEN_STATUS["value"]


def test_growth_applies_every_rule_as_an_and() -> None:
    hits = run_screen("growth", _table())
    assert list(hits.index) == ["PASS"]


def test_a_nan_fails_rather_than_passes() -> None:
    """A company that did not report a margin has not demonstrated an 8% one.

    This is the direction that matters: treating NaN as passing would admit
    every symbol with missing fundamentals into a quality screen.
    """
    t = _table()
    t.loc["PASS", "net_margin_pct"] = float("nan")
    assert run_screen("growth", t).empty


def test_an_unknown_screen_raises() -> None:
    with pytest.raises(KeyError, match="unknown screen"):
        run_screen("momentum", _table())


def test_a_missing_column_raises_rather_than_dropping_the_rule() -> None:
    """Silently ignoring a rule whose column is absent would run a DIFFERENT,
    looser screen under the same name."""
    with pytest.raises(KeyError, match="net_margin_pct"):
        run_screen("growth", _table().drop(columns=["net_margin_pct"]))


def test_the_transcription_still_matches_upstream() -> None:
    """The rules are transcribed rather than YAML-parsed, so drift is the risk.

    An absent upstream returns {} - it is gitignored, so that is the normal
    state on a fresh clone and is not evidence of drift.
    """
    assert verify_against_upstream() == {}


def test_the_report_lists_blocked_screens_too() -> None:
    out = screen_report(_table())
    assert "NOT RUN" in out
    assert "roe_pct" in out
    assert "growth" in out


# -- Pine: the shift is the whole correctness -------------------------------

@pytest.mark.parametrize("key", sorted(MEASURED))
def test_pine_compares_against_the_prior_bars_channel(key: str) -> None:
    """THE test. Without the [1], ta.highest(high, n) contains today's high,
    close can never exceed it, and the strategy tester shows zero trades."""
    code = to_pine(MEASURED[key])
    assert "ta.highest(high, channelLen)[1]" in code, \
        "the channel includes the current bar - the rule can never fire"


@pytest.mark.parametrize("key", sorted(MEASURED))
def test_pine_is_long_only(key: str) -> None:
    """Indian retail cannot short cash equity beyond intraday, so a short side
    would be signals nobody here can act on."""
    code = to_pine(MEASURED[key])
    assert "strategy.short" not in code
    assert "strategy.long" in code


@pytest.mark.parametrize("key", sorted(MEASURED))
def test_pine_exports_the_measured_risk_not_the_catalogued_one(key: str) -> None:
    """The catalogue says 5%/10%, which measured -0.027R. What survived was
    12%/24%. Exporting the losing configuration to a chart someone will believe
    is worse than exporting nothing."""
    s = MEASURED[key]
    assert (s.stop_pct, s.target_pct) == (12.0, 24.0)
    assert s.risk_reward == 2.0
    assert "12.0" in to_pine(s) and "24.0" in to_pine(s)


@pytest.mark.parametrize("key", sorted(MEASURED))
def test_pine_carries_its_own_evidence_and_its_own_warning(key: str) -> None:
    """A chart opened in six months must still say what the evidence was, and
    that it is DRAFT."""
    s = MEASURED[key]
    code = to_pine(s)
    assert f"{s.permutation_p:.4f}" in code
    assert "NOT VALIDATED" in code
    assert "DRAFT" in code
    # And that TradingView's own costs will not match ours.
    assert "commission" in code


def test_pine_declares_the_hold_limit_as_part_of_the_rule() -> None:
    """The 20-bar hold is not a safety net, it is the measured rule - the
    60-bar version is the one the permutation test rejected."""
    code = to_pine(MEASURED["donchian_20d"])
    assert "holdBars" in code
    assert MEASURED["donchian_20d"].hold_bars == 20


def test_both_measured_configurations_agree_on_the_cell() -> None:
    """Independent agreement between two rules on the same stop, target and
    hold is the main reason to take either seriously."""
    d, b = MEASURED["donchian_20d"], MEASURED["bollinger_20d"]
    assert (d.stop_pct, d.target_pct, d.hold_bars) == \
           (b.stop_pct, b.target_pct, b.hold_bars)
    assert abs(d.measured_net_r - b.measured_net_r) < 0.01
