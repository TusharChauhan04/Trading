"""Measuring the regime, and the sector map that feeds part of it.

The property that matters: UNKNOWN IS A REAL ANSWER. Every dimension has a
minimum history and returns UNKNOWN below it rather than computing a
50-day average over 12 days and labelling it. A regime silences factors
and gates strategies, so a half-measured one presented as a finding is
worse than none at all.

The second property is that BREADTH AND PRICE CAN DISAGREE, and the engine
must not resolve that with a casting vote. A market near its highs on
collapsing breadth is the late stage of a narrow rally - not risk-on,
however the index looks.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from desk.contracts.enums import Breadth, Regime, RiskAppetite, Trend, VolState
from desk.marketdata.sectors import SectorMap, parse_constituents
from desk.regime.engine import MIN_SESSIONS, MIN_VOL_HISTORY, compute_regime
from desk.store.bars import Coverage, History

AS_OF = date(2026, 9, 17)


def _history(paths: dict[str, list[float]], start=date(2026, 1, 1)) -> History:
    """paths: symbol -> list of closes, one per session."""
    n = max(len(v) for v in paths.values())
    days = [start + timedelta(days=i) for i in range(n)]
    rows = []
    for sym, closes in paths.items():
        for d, c in zip(days, closes):
            rows.append({"symbol": sym, "date": d, "open": c, "high": c,
                         "low": c, "close": c, "volume": 1000})
    frame = pd.DataFrame(rows)
    return History(frame=frame,
                   coverage=Coverage(start=days[0], end=days[-1],
                                     days_loaded=n, sessions_expected=n))


def _trending(n=200, slope=0.004, names=20, noise=0.008):
    rng = np.random.default_rng(7)
    out = {}
    for i in range(names):
        base = 100.0
        path = []
        for t in range(n):
            base *= (1.0 + slope + (rng.normal(0, noise) if noise else 0.0))
            path.append(base)
        out[f"S{i}.NS"] = path
    return out


# --- unknown is a real answer --------------------------------------------

def test_too_little_history_measures_nothing():
    st = compute_regime(_history(_trending(n=MIN_SESSIONS - 1)), as_of=AS_OF)
    assert st.trend is Trend.UNKNOWN
    assert st.breadth is Breadth.UNKNOWN
    assert st.volatility is VolState.UNKNOWN
    assert not st.is_measured
    assert "not measured" in st.sources["all"]
    assert st.label is Regime.UNKNOWN


def test_the_shortfall_is_named_with_numbers():
    st = compute_regime(_history(_trending(n=30)), as_of=AS_OF)
    assert "30 sessions on file" in st.sources["all"]
    assert str(MIN_SESSIONS) in st.sources["all"]


def test_volatility_stays_unknown_without_enough_to_rank_against():
    """A percentile over 30 observations is noise, so it is refused even
    though the 20-day vol itself is computable."""
    st = compute_regime(_history(_trending(n=MIN_VOL_HISTORY - 20)),
                        as_of=AS_OF)
    assert st.volatility is VolState.UNKNOWN
    assert "needed to rank a percentile" in st.sources["volatility"]
    assert st.trend is not Trend.UNKNOWN, "trend should still be measured"
    assert not st.is_measured


def test_an_empty_history_does_not_crash():
    empty = History(frame=pd.DataFrame(),
                    coverage=Coverage(start=None, end=None, days_loaded=0,
                                      sessions_expected=None))
    st = compute_regime(empty, as_of=AS_OF)
    assert st.label is Regime.UNKNOWN
    assert not st.is_measured


def test_every_measured_dimension_records_its_source():
    """A regime is an input to every downstream decision, so it has to be
    arguable."""
    st = compute_regime(_history(_trending(n=250)), as_of=AS_OF)
    for key in ("breadth", "trend", "volatility", "risk_appetite"):
        assert key in st.sources
        assert st.sources[key], f"{key} has no stated source"


# --- trend ---------------------------------------------------------------

def test_a_steadily_rising_market_is_an_uptrend():
    st = compute_regime(_history(_trending(n=250, slope=0.004)), as_of=AS_OF)
    assert st.trend in (Trend.UP, Trend.STRONG_UP)
    assert st.label in (Regime.TRENDING_UP, Regime.HIGH_VOL, Regime.CRISIS)


def test_a_steadily_falling_market_is_a_downtrend():
    st = compute_regime(_history(_trending(n=250, slope=-0.004)), as_of=AS_OF)
    assert st.trend in (Trend.DOWN, Trend.STRONG_DOWN)


def test_a_flat_market_is_flat():
    """The RECENT window has to be flat, not merely zero-drift overall: a
    zero-drift random walk can and does move 4% in 20 sessions, which is a
    real trend reading rather than a fault."""
    paths = _trending(n=250, slope=0.0)
    for sym, path in paths.items():
        # Pin the last 21 sessions to a flat line around the prior level.
        level = path[-21]
        paths[sym] = path[:-21] + [level * (1.0 + 0.0005 * (i % 3 - 1))
                                   for i in range(21)]
    st = compute_regime(_history(paths), as_of=AS_OF)
    assert st.trend is Trend.FLAT, st.sources["trend"]


def test_the_trend_is_measured_in_sigma_not_percent():
    """So the same cut-offs mean the same thing in a calm market and a
    violent one."""
    st = compute_regime(_history(_trending(n=250)), as_of=AS_OF)
    assert "sigma" in st.sources["trend"]


# --- breadth, and the composite it is not ---------------------------------

def test_breadth_counts_names_above_their_own_average():
    up = _trending(n=250, slope=0.004, names=8)
    down = _trending(n=250, slope=-0.004, names=2)
    paths = {**up, **{f"D{i}.NS": v for i, v in enumerate(down.values())}}
    st = compute_regime(_history(paths), as_of=AS_OF)
    assert st.breadth is Breadth.BROAD
    assert "% of 10 liquid names" in st.sources["breadth"]


def test_a_mostly_falling_market_is_narrow():
    up = _trending(n=250, slope=0.004, names=2)
    down = _trending(n=250, slope=-0.004, names=8)
    paths = {**{f"U{i}.NS": v for i, v in enumerate(up.values())}, **down}
    st = compute_regime(_history(paths), as_of=AS_OF)
    assert st.breadth is Breadth.NARROW


def test_the_composite_is_equal_weighted_not_price_weighted():
    """A 40,000-rupee stock must not outvote a 40-rupee one. Nine names
    fall hard and one expensive name rises; the composite must fall."""
    paths = {f"CHEAP{i}.NS": [40.0 * (0.99 ** t) for t in range(250)]
             for i in range(9)}
    paths["DEAR.NS"] = [40_000.0 * (1.01 ** t) for t in range(250)]
    st = compute_regime(_history(paths), as_of=AS_OF)
    assert st.trend in (Trend.DOWN, Trend.STRONG_DOWN), \
        "one expensive name outvoted nine cheap ones"


def test_an_absurd_single_day_move_does_not_move_the_composite():
    """An unadjusted split is a -50% print, not a market event, and one of
    them visibly moves an equal-weighted mean."""
    clean = _trending(n=250, slope=0.0, names=9)
    broken = [100.0] * 249 + [1.0]          # a 99% "drop" on the last bar
    with_split = compute_regime(_history({**clean, "BAD.NS": broken}),
                                as_of=AS_OF)
    without = compute_regime(_history(clean), as_of=AS_OF)
    assert with_split.trend == without.trend


# --- risk appetite: the two must agree -----------------------------------

def test_strong_price_and_broad_breadth_is_risk_on():
    st = compute_regime(_history(_trending(n=250, slope=0.004)), as_of=AS_OF)
    assert st.risk_appetite is RiskAppetite.RISK_ON


def test_a_narrow_rally_is_not_risk_on():
    """Near the highs on collapsing breadth is the late stage of a narrow
    rally. Either signal alone would call this wrong."""
    rising = {"BIG.NS": [100.0 * (1.02 ** t) for t in range(250)]}
    # Nine names drifting just under their averages, near flat overall.
    for i in range(9):
        rising[f"W{i}.NS"] = [100.0 - 0.02 * t for t in range(250)]
    st = compute_regime(_history(rising), as_of=AS_OF)
    assert st.breadth is Breadth.NARROW
    assert st.risk_appetite is not RiskAppetite.RISK_ON


def test_a_deep_drawdown_with_narrow_breadth_is_risk_off():
    st = compute_regime(_history(_trending(n=250, slope=-0.004)), as_of=AS_OF)
    assert st.risk_appetite is RiskAppetite.RISK_OFF
    assert st.risk_off is True
    assert st.label is Regime.CRISIS


def test_risk_appetite_reports_both_inputs():
    st = compute_regime(_history(_trending(n=250)), as_of=AS_OF)
    note = st.sources["risk_appetite"]
    assert "from its high" in note and "breadth" in note


# --- sectors -------------------------------------------------------------

_CSV = (b"Company Name,Industry,Symbol,Series,ISIN Code\n"
        b"Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018\n"
        b"Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029\n"
        b"Infosys Ltd.,Information Technology,INFY,EQ,INE009A01021\n")


def test_constituents_parse():
    m = parse_constituents(_CSV, index="NIFTY 50", as_of=AS_OF)
    assert len(m) == 3
    assert m.sector_for("RELIANCE.NS") == "Oil Gas & Consumable Fuels"
    assert m.indices_for("TCS") == ("NIFTY 50",)
    assert m.in_index("INFY", "NIFTY 50")
    assert m.as_of == AS_OF


def test_a_byte_order_mark_does_not_hide_the_first_column():
    """NSE ships these with a BOM, which turns the first header into
    '\\ufeffCompany Name' and makes a plain DictReader miss it."""
    m = parse_constituents(b"\xef\xbb\xbf" + _CSV, index="NIFTY 50")
    assert len(m) == 3


def test_a_file_missing_the_industry_column_is_refused():
    with pytest.raises(ValueError, match="Industry"):
        parse_constituents(b"Company Name,Symbol\nX,Y\n", index="NIFTY 50")


def test_an_empty_file_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        parse_constituents(b"Company Name,Industry,Symbol\n", index="NIFTY 50")


def test_maps_merge_and_accumulate_index_membership():
    a = parse_constituents(_CSV, index="NIFTY 50", as_of=AS_OF)
    b = parse_constituents(_CSV, index="NIFTY 500", as_of=AS_OF)
    both = a.merged_with(b)
    assert set(both.indices_for("RELIANCE")) == {"NIFTY 50", "NIFTY 500"}
    assert len(both) == 3


def test_coverage_is_reported_honestly():
    """A sector cap that could only see a third of the book is not a
    sector cap, and the caller has to be able to say so."""
    m = parse_constituents(_CSV, index="NIFTY 50")
    assert m.coverage(["RELIANCE.NS", "TCS.NS", "UNKNOWN.NS"]) == (2, 3)


def test_an_unclassified_symbol_returns_none_not_a_guess():
    m = parse_constituents(_CSV, index="NIFTY 50")
    assert m.sector_for("NOSUCH.NS") is None
    assert m.indices_for("NOSUCH.NS") == ()


def test_the_map_round_trips(tmp_path):
    m = parse_constituents(_CSV, index="NIFTY 50", as_of=AS_OF)
    p = tmp_path / "sectors.json"
    m.save(p)
    back = SectorMap.load(p)
    assert back.sector_for("RELIANCE") == "Oil Gas & Consumable Fuels"
    assert back.as_of == AS_OF
    assert not list(tmp_path.glob("*.tmp"))


def test_a_missing_map_is_none_not_a_crash(tmp_path):
    assert SectorMap.load(tmp_path / "nope.json") is None


def test_leading_sectors_need_at_least_three_names():
    """Two names is not a sector; reporting one as 'leading' would put a
    single stock's move forward as an industry finding."""
    paths = {"A.NS": [100.0 * (1.01 ** t) for t in range(250)],
             "B.NS": [100.0 * (0.99 ** t) for t in range(250)]}
    thin = SectorMap(industry={"A": "Tiny Sector", "B": "Other"})
    st = compute_regime(_history(paths), as_of=AS_OF, sectors=thin)
    assert st.leading_sectors == []
    assert "3+" in st.sources["sectors"]


def test_leading_and_lagging_sectors_are_ranked():
    paths = {}
    industry = {}
    for i in range(3):
        paths[f"UP{i}.NS"] = [100.0 * (1.01 ** t) for t in range(250)]
        industry[f"UP{i}"] = "Winners"
        paths[f"DN{i}.NS"] = [100.0 * (0.99 ** t) for t in range(250)]
        industry[f"DN{i}"] = "Losers"
    st = compute_regime(_history(paths), as_of=AS_OF,
                        sectors=SectorMap(industry=industry))
    assert st.leading_sectors[0] == "Winners"
    assert st.lagging_sectors[0] == "Losers"


# --- the label -----------------------------------------------------------

def test_the_label_is_derived_not_stored():
    st = compute_regime(_history(_trending(n=250, slope=-0.004)), as_of=AS_OF)
    assert st.label is Regime.CRISIS, "risk-off outranks trend"


def test_explain_says_when_it_is_not_fully_measured():
    st = compute_regime(_history(_trending(n=MIN_VOL_HISTORY - 20)),
                        as_of=AS_OF)
    assert any("NOT FULLY MEASURED" in line for line in st.explain())


# --- the api wiring ------------------------------------------------------

def test_a_partly_measured_regime_becomes_a_plan_caveat():
    from desk.api.main import _regime_caveats
    st = compute_regime(_history(_trending(n=MIN_VOL_HISTORY - 20)),
                        as_of=AS_OF)
    caveats = _regime_caveats(st)
    assert any("PARTLY measured" in c for c in caveats)
    assert any("volatility" in c for c in caveats)


def test_no_measured_regime_adds_no_caveats():
    """A caller who passed an explicit regime is not told about a
    measurement that never happened."""
    from desk.api.main import _regime_caveats
    assert _regime_caveats(None) == []
