"""Correlation clusters, and the two concentration caps they switch on.

THREE DEFECTS ARE PINNED HERE, in order of how much damage they did:

  1. `sector` defaulted to "UNKNOWN" for every Stage 4 candidate, so
     `sector_exposure("UNKNOWN")` summed the WHOLE BOOK and the 30% sector cap
     silently became a cap on total gross exposure. Not inert - mislabelled.
  2. `_with_position` added approvals to the running portfolio WITHOUT their
     sector or cluster, so even a correctly wired `size_position` would see an
     empty book for both caps and neither could bind within one scan.
  3. Upstream clusters with `ward` on a correlation distance, which is not
     Euclidean here, and fills unknown correlations with 0 - the most
     favourable diversification assumption available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desk.risk.clusters import (
    DEFAULT_THRESHOLD, MIN_NAMES, available, cluster_map,
)
from desk.risk.engine import Portfolio, Position, RiskConfig, size_position

requires_scipy = pytest.mark.skipif(
    not available(), reason="scipy not installed")


def _panel(n_days: int = 300, seed: int = 11) -> pd.DataFrame:
    """Two tight blocks plus independents, so the right answer is known.

    A and B share a factor at high loading; C and D share a different one;
    E, F, G are independent. Any correct clustering puts {A,B} together,
    {C,D} together, and leaves the rest alone.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    f1 = rng.normal(0, 0.012, n_days)
    f2 = rng.normal(0, 0.012, n_days)
    cols = {}
    for name, f, load in (("AAA", f1, 0.95), ("BBB", f1, 0.95),
                          ("CCC", f2, 0.95), ("DDD", f2, 0.95)):
        r = load * f + np.sqrt(1 - load ** 2) * rng.normal(0, 0.012, n_days)
        cols[name] = 100 * np.exp(np.cumsum(r))
    for name in ("EEE", "FFF", "GGG"):
        cols[name] = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n_days)))
    return pd.DataFrame(cols, index=idx)


# -- the clustering itself --------------------------------------------------

@requires_scipy
def test_names_sharing_a_factor_land_in_one_cluster() -> None:
    cm = cluster_map(_panel())
    assert cm.group_for("AAA") == cm.group_for("BBB") != None
    assert cm.group_for("CCC") == cm.group_for("DDD") != None
    assert cm.group_for("AAA") != cm.group_for("CCC"), (
        "two independent factors were merged into one cluster")


@requires_scipy
def test_independent_names_are_not_clustered_together() -> None:
    cm = cluster_map(_panel())
    singles = [cm.group_for(s) for s in ("EEE", "FFF", "GGG")]
    assert len(set(singles)) == 3, (
        f"independent names were grouped: {singles}")


@requires_scipy
def test_a_higher_threshold_never_makes_bigger_clusters() -> None:
    """Monotonicity. If this fails the distance/threshold conversion is
    inverted, which would silently flip the gate's meaning."""
    panel = _panel()
    prev = None
    for thr in (0.3, 0.5, 0.7, 0.9):
        cm = cluster_map(panel, threshold=thr)
        if prev is not None:
            assert cm.largest <= prev, (
                f"threshold {thr} gave a larger cluster than the looser cut")
        prev = cm.largest


@requires_scipy
def test_the_label_names_a_real_member() -> None:
    """A rejection saying "cluster 7" is unactionable. The label must point at
    something a human can look up."""
    cm = cluster_map(_panel())
    for sym, lab in cm.labels.items():
        assert lab.startswith("CORR:")
        head = lab[len("CORR:"):].split("+")[0]
        assert head in cm.labels, f"label {lab} names a non-member"


# -- refusals, which are where upstream invents answers ---------------------

@requires_scipy
def test_a_gap_is_never_padded_into_a_zero_return() -> None:
    """pandas' pct_change defaults to fill_method="pad", which reports a
    return of exactly 0.0 across a gap. Names with gaps on the same dates then
    correlate for no reason but both being untraded - manufacturing exactly
    the clusters this module is meant to detect honestly.
    """
    panel = _panel(300)
    # Punch the SAME 40 holes in two otherwise independent names.
    holes = panel.index[50:90]
    panel.loc[holes, "EEE"] = np.nan
    panel.loc[holes, "FFF"] = np.nan
    cm = cluster_map(panel, min_completeness=0.80)
    if cm.group_for("EEE") and cm.group_for("FFF"):
        assert cm.group_for("EEE") != cm.group_for("FFF"), (
            "two independent names with matching gaps were clustered - the "
            "gaps were padded into correlated zero returns")


@requires_scipy
def test_a_name_with_too_little_history_is_unclustered_not_uncorrelated() -> None:
    """Upstream's corr().fillna(0) calls an unmeasurable pair UNCORRELATED,
    the most favourable diversification assumption there is."""
    panel = _panel(300)
    panel.iloc[:250, panel.columns.get_loc("GGG")] = np.nan
    cm = cluster_map(panel)
    assert cm.group_for("GGG") is None
    assert "GGG" in cm.unclustered


@requires_scipy
def test_a_frozen_name_is_unclustered() -> None:
    panel = _panel(300)
    panel["FFF"] = 100.0
    cm = cluster_map(panel)
    assert cm.group_for("FFF") is None


def test_too_few_names_refuses_rather_than_grouping() -> None:
    panel = _panel(300).iloc[:, :MIN_NAMES - 1]
    cm = cluster_map(panel)
    assert not cm.labels
    assert cm.why


@requires_scipy
def test_the_map_uses_only_data_up_to_as_of() -> None:
    """Point-in-time. A cluster that knows next month's correlation would
    gate today's trade on the future."""
    panel = _panel(600)
    cut = panel.index[400]
    a = cluster_map(panel, as_of=cut)
    b = cluster_map(panel.loc[:cut], as_of=cut)
    assert a.labels == b.labels
    assert a.n_obs == b.n_obs


# -- the caps these switch on ----------------------------------------------

def test_the_sector_cap_no_longer_sums_the_whole_book() -> None:
    """DEFECT 1. With every position defaulting to sector "UNKNOWN",
    sector_exposure("UNKNOWN") summed everything, so the 30% sector cap acted
    as a total gross exposure cap and said "UNKNOWN would reach 45.0%".
    """
    cfg = RiskConfig(capital=1_000_000.0, max_sector_pct=30.0)
    # Two held names in DIFFERENT real sectors, 15% of capital each.
    held = Portfolio(positions=[
        Position(symbol="A", qty=1500, entry=100.0, stop=90.0, sector="IT"),
        Position(symbol="B", qty=1500, entry=100.0, stop=90.0, sector="Metals"),
    ])
    # A third name in a THIRD sector must pass: no single sector is near 30%.
    ok = size_position(symbol="C", entry=100.0, stop=90.0, target=120.0,
                       cfg=cfg, portfolio=held, sector="Pharma")
    assert ok.approved, (
        f"a third sector was rejected, so the cap is still bucketing "
        f"everything together: {ok.reasons} {ok.notes}")
    # And the SAME name inside an already-30% sector must be rejected.
    heavy = Portfolio(positions=[
        Position(symbol="A", qty=1500, entry=100.0, stop=90.0, sector="IT"),
        Position(symbol="B", qty=1500, entry=100.0, stop=90.0, sector="IT"),
    ])
    bad = size_position(symbol="C", entry=100.0, stop=90.0, target=120.0,
                        cfg=cfg, portfolio=heavy, sector="IT")
    assert not bad.approved
    assert any("IT" in n for n in bad.notes), bad.notes


def test_the_correlated_cap_fires_once_a_cluster_is_supplied() -> None:
    """The gate has never fired in production because nothing supplied a
    cluster. Given one, it must bind - and bind BEFORE the sector cap, since
    25% is tighter than 30% and the buckets are now genuinely different.
    """
    cfg = RiskConfig(capital=1_000_000.0, max_correlated_pct=25.0,
                     max_sector_pct=30.0)
    # Two names in DIFFERENT sectors that the market says move together -
    # the case the sector cap cannot see. 12.5% each = 25%, at the cap.
    held = Portfolio(positions=[
        Position(symbol="PFC", qty=1250, entry=100.0, stop=90.0,
                 sector="Financial Services", corr_group="CORR:PSU_CAPEX"),
        Position(symbol="RVNL", qty=1250, entry=100.0, stop=90.0,
                 sector="Construction", corr_group="CORR:PSU_CAPEX"),
    ])
    out = size_position(symbol="IRFC", entry=100.0, stop=90.0, target=120.0,
                        cfg=cfg, portfolio=held, sector="Financial Services",
                        corr_group="CORR:PSU_CAPEX")
    assert not out.approved, "the correlated cap did not bind"
    assert any("CORR:PSU_CAPEX" in n for n in out.notes), out.notes


def test_an_unclustered_name_skips_the_gate_rather_than_passing_it() -> None:
    """None must mean "not checked", never "checked and safe"."""
    cfg = RiskConfig(capital=1_000_000.0)
    out = size_position(symbol="X", entry=100.0, stop=90.0, target=120.0,
                        cfg=cfg, portfolio=Portfolio(), corr_group=None)
    assert out.approved
    assert any("correlation" in c.lower() for c in out.checks_skipped), (
        f"an unchecked correlation was not recorded as skipped: "
        f"{out.checks_skipped}")
