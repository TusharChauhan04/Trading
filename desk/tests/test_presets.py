"""Translated presets: the rules must fire, and must fire for the right reason.

Two failure modes are specific to translating someone else's declarative spec,
and both produce output that looks like a finding:

  1. A rule that can NEVER fire. Donchian declares `close cross_above
     highest(high, 20)`. Evaluated over a window containing today, the highest
     high is >= today's high >= today's close, so the condition is
     arithmetically impossible - and the strategy reports zero setups, which
     is indistinguishable from a quiet market. The desk already shipped this
     bug once in its own Donchian adapter.
  2. A rule tested at a risk profile nobody declared. rsi_reversion is 1:1,
     below the desk's 1:2 floor. Silently widening its target to make it
     testable would measure a strategy that does not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.strategies.presets import (
    ALREADY_OURS, MIN_RR, PRESET_IDS, available, preset_rules,
)

requires_upstream = pytest.mark.skipif(
    not available(),
    reason="agents/openterminal_ui/upstream absent - gitignored by design",
)


def _panel(n: int = 160, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Three symbols with enough shape that every rule has something to find:
    a trender, a chopper, and one that breaks out late."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    trend = 100 * np.cumprod(1 + rng.normal(0.004, 0.013, n))
    chop = 250 + np.cumsum(rng.normal(0, 1.5, n))
    late = np.concatenate([
        300 + np.cumsum(rng.normal(0, 0.8, n // 2)),
        300 + np.linspace(0, 60, n - n // 2),
    ])
    close = pd.DataFrame({"TREND": trend, "CHOP": chop, "LATE": late},
                         index=idx)
    span = close * rng.uniform(0.005, 0.02, close.shape)
    return {"close": close, "open": close.shift(1).fillna(close),
            "high": close + span, "low": close - span,
            "volume": pd.DataFrame(1e6, index=idx, columns=close.columns)}


@requires_upstream
def test_all_six_presets_load() -> None:
    rules = preset_rules()
    assert set(rules) == set(PRESET_IDS)
    assert len(rules) == 6, "six, not the nineteen first reported"


@requires_upstream
def test_five_of_six_are_natively_one_to_two() -> None:
    """The reason these were worth translating at all.

    If upstream changes a risk block this fails, and the roadmap's claim that
    the catalogue already targets 1:2 needs re-checking rather than patching.
    """
    rules = preset_rules()
    at_target = [p for p, r in rules.items() if r.rr == 2.0]
    assert len(at_target) == 5, f"expected 5 at 1:2, got {at_target}"
    assert rules["rsi_reversion"].rr == 1.0


@requires_upstream
def test_donchian_declares_the_cell_the_hourly_scan_converged_on() -> None:
    """5% stop, 10% target - the same configuration an unconstrained factor
    scan on hourly bars optimised its way to. Recorded as a test because the
    convergence is what makes step 4 and roadmap H5 one measurement."""
    d = preset_rules()["donchian_breakout"]
    assert (d.stop_pct, d.take_pct) == (5.0, 10.0)


@requires_upstream
def test_a_sub_one_to_two_preset_is_refused_not_rescued() -> None:
    """Refused, RETURNED, and the reason stated - three distinct requirements.

    Dropping it would make the catalogue look smaller than it is; widening its
    target would measure something nobody declared.
    """
    r = preset_rules()["rsi_reversion"]
    assert r.unsupported is not None
    assert "1:1" in r.unsupported
    assert str(int(MIN_RR)) in r.unsupported
    with pytest.raises(NotImplementedError, match="as written"):
        r.entries(_panel())


@requires_upstream
def test_the_overlap_with_our_own_adapters_is_declared() -> None:
    rules = preset_rules()
    assert set(ALREADY_OURS) <= set(rules)
    for pid in ALREADY_OURS:
        assert rules[pid].already_ours
    assert not rules["sma_cross"].already_ours


@requires_upstream
@pytest.mark.parametrize("pid", [p for p in PRESET_IDS if p != "rsi_reversion"])
def test_every_testable_rule_actually_fires(pid: str) -> None:
    """A rule that never fires is the bug this file exists for.

    Zero signals and "no setups in a quiet market" look identical in a report,
    so the distinction is asserted here where the data is known to contain
    trends, chop and a late breakout.
    """
    sig = preset_rules()[pid].entries(_panel())
    assert sig.to_numpy().sum() > 0, f"{pid} never fired - check the operands"
    assert sig.dtypes.eq(bool).all(), "entries must be a boolean frame"
    assert sig.shape == _panel()["close"].shape


@requires_upstream
def test_the_donchian_channel_excludes_today() -> None:
    """THE correctness test for a breakout rule.

    Constructed so the answer is known: a flat base long enough to establish a
    20-bar channel, then a step up through it. That is a crossing event with a
    known bar.

    NOT a monotonic ramp, which was the first attempt and was wrong for an
    instructive reason: on a series making a new high every bar, close is
    ALREADY above the prior channel on every bar, so `cross_above` - an event,
    needing a bar below followed by a bar above - has nothing to cross from.
    Its one real crossing sits inside the 20-bar warmup where the channel is
    still NaN. A monotonic series is the one shape that cannot test this.

    If the shift(1) is ever lost, the channel includes today, today's high is
    already the maximum, close can never exceed it, and the rule fires zero
    times while looking like a quiet market.
    """
    n, base = 60, 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    px = np.concatenate([np.full(base, 100.0),
                         np.linspace(101.0, 130.0, n - base)])
    frame = pd.DataFrame({"UP": px}, index=idx)
    panel = {"close": frame, "open": frame, "high": frame * 1.001,
             "low": frame * 0.999,
             "volume": pd.DataFrame(1e6, index=idx, columns=["UP"])}
    sig = preset_rules()["donchian_breakout"].entries(panel)
    assert sig.to_numpy().sum() > 0, \
        "a base-then-breakout broke no channel - the shift(1) was lost"
    # And it must fire AT the breakout, not during the flat base.
    first = int(np.argmax(sig.to_numpy().ravel()))
    assert first >= base, f"fired at bar {first}, inside the flat base"


@requires_upstream
def test_cost_in_r_is_inverse_to_stop_width() -> None:
    """cost_R = round_trip% / stop%, so a TIGHTER stop is proportionally more
    expensive. This is the identity that explains why the 2.5% preset starts
    0.085R behind the 5% one before either is right about anything."""
    rules = preset_rules()
    assert rules["ema_cross"].cost_r > rules["donchian_breakout"].cost_r
    assert rules["donchian_breakout"].cost_r == pytest.approx(0.422 / 5.0)


@requires_upstream
def test_rsi_matches_stage_ones_wilder_convention() -> None:
    """Same indicator, same bars, same answer - including the awkward case.

    A name with no down bar has zero average loss, and the reference divides by
    `loss.replace(0, nan)` so its RSI is NaN rather than 100. Reproducing
    everything EXCEPT that case is how two implementations quietly diverge on
    exactly the rows a mean-reversion rule reads.
    """
    from desk.strategies.presets import _rsi

    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    # NO DOWN BARS AT ALL.
    only_up = pd.DataFrame({"UP": np.linspace(100, 150, n)}, index=idx)
    got = _rsi(only_up, 14)
    assert got["UP"].iloc[-1] != 100.0
    assert pd.isna(got["UP"].iloc[-1]), \
        "a no-down-bar series must be NaN, not 100 - see Stage 1"


@requires_upstream
def test_a_cross_needs_the_previous_bar() -> None:
    """cross_above is an EVENT, not a state. If it degrades into `a > b` the
    rule fires on every bar of a trend instead of once at the crossing, and
    the trade count inflates by an order of magnitude."""
    from desk.strategies.presets import _apply_op

    a = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]})
    b = pd.DataFrame({"X": [2.5, 2.5, 2.5, 2.5]})
    out = _apply_op("cross_above", a, b)
    assert list(out["X"]) == [False, False, True, False], \
        "a cross must fire once, at the crossing"


def test_a_missing_upstream_yields_nothing_rather_than_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """agents/*/upstream/ is gitignored and OneDrive has dehydrated one
    mid-session already, so this path is load-bearing."""
    import desk.strategies.presets as mod

    monkeypatch.setattr(mod, "_load", lambda: None)
    assert mod.load_presets() == {}
    assert mod.preset_rules() == {}
