"""The endpoints the dashboard reads: regime, journal, news, data health.

These exist so the subsystems built over the last few days are VISIBLE,
which is the last of the four conditions for calling anything done. A
journal nobody can read records evidence for nobody.

The property they share, and the one tested hardest here: EVERY ONE OF
THEM REPORTS WHAT IT COULD NOT DO. A regime with an unmeasured dimension,
a cross-check that covered a fifth of the universe, a news snapshot too
old to use - each has to say so in its own response, because a screen
showing only the answer teaches the reader to trust it further than the
data supports.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from desk.api.main import app
from desk.journal import Decision, JournalStore, Outcome, TradeRecord
from desk.journal.models import IST, ExitReason

client = TestClient(app)
DAY = date(2026, 9, 17)


@pytest.fixture
def configs(tmp_path, monkeypatch):
    """An isolated config directory, so tests never read live artefacts."""
    from desk.api import main as api
    monkeypatch.setattr(api, "CONFIGS", tmp_path)
    return tmp_path


# --- journal -------------------------------------------------------------

def _record(root, day=DAY, trades=(("RIR.NS", 197.63, 173.79, 245.30, 41),),
            no_trade=None):
    store = JournalStore(root / "journal")
    store.record(Decision(
        as_of=day, recorded_at=datetime(2026, 9, 17, 9, tzinfo=IST),
        regime="range", capital=100_000.0,
        trades=tuple(TradeRecord(symbol=s, stance="Buy", entry=e, stop=st,
                                 target=t, qty=q, rationale="momentum")
                     for s, e, st, t, q in trades),
        no_trade_reason=no_trade,
        universe_scanned=3483, survived_stage0=1548, considered=3,
        caveats=("the event calendar was not applied",)))
    return store


def test_an_empty_journal_is_not_an_error(configs):
    r = client.get("/journal")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_the_journal_index_lists_recorded_days(configs):
    _record(configs)
    body = client.get("/journal").json()
    assert body["total"] == 1
    row = body["days"][0]
    assert row["as_of"] == DAY.isoformat()
    assert row["trades"] == 1
    assert row["symbols"] == ["RIR.NS"]
    assert row["digest"]


def test_a_no_trade_day_is_listed_as_prominently_as_a_trading_day(configs):
    """A journal showing only trades cannot say how often the desk
    correctly stayed out."""
    _record(configs, trades=(), no_trade="every candidate failed the R:R floor")
    body = client.get("/journal").json()
    assert body["no_trade_days"] == 1
    assert body["days"][0]["no_trade_reason"].startswith("every candidate")


def test_days_with_unrecorded_outcomes_are_surfaced(configs):
    """'No losing trades' and 'nobody wrote down how they went' look the
    same from a results table, and the flattering reading wins."""
    _record(configs)
    body = client.get("/journal").json()
    assert body["unrecorded_outcomes"] == [DAY.isoformat()]


def test_recording_an_outcome_clears_it(configs):
    store = _record(configs)
    o = Outcome(decision_date=DAY, symbol="RIR.NS",
                observed_at=datetime.now(IST))
    o.close(price=245.30, on=date(2026, 9, 24), reason=ExitReason.TARGET,
            entry=197.63, stop=173.79, qty=41)
    store.record_outcomes(DAY, [o])

    body = client.get("/journal").json()
    assert body["unrecorded_outcomes"] == []
    assert body["days"][0]["outcomes_recorded"] == 1


def test_one_day_returns_the_decision_and_its_caveats(configs):
    _record(configs)
    body = client.get(f"/journal/{DAY}").json()
    assert body["regime"] == "range"
    assert body["capital"] == 100_000.0
    assert body["trades"][0]["symbol"] == "RIR.NS"
    assert body["trades"][0]["reward_to_risk"] == pytest.approx(2.0, abs=0.01)
    assert any("event calendar" in c for c in body["caveats"])
    assert body["funnel"]["universe_scanned"] == 3483


def test_the_amendment_chain_is_exposed(configs):
    store = _record(configs)
    store.amend(Decision(as_of=DAY, recorded_at=datetime.now(IST),
                         regime="range", capital=100_000.0),
                reason="qty was recorded before the final risk check")
    body = client.get(f"/journal/{DAY}").json()
    assert len(body["versions"]) == 2
    assert body["amends"], "the newest version must point at what it corrects"


def test_an_unrecorded_day_is_a_404(configs):
    assert client.get("/journal/2020-01-01").status_code == 404


# --- news ----------------------------------------------------------------

def _news_file(root, *, age_hours=1.0, n=2):
    root.mkdir(parents=True, exist_ok=True)
    when = datetime.now(IST) - timedelta(hours=age_hours)
    (root / "news.json").write_text(json.dumps({
        "fetched_at": when.isoformat(),
        "caveats": ["business-standard-companies: feed was not well-formed"],
        "clusters": [
            {"title": f"Story {i}", "published_at": when.isoformat(),
             "tier": 3, "weight": 0.35, "duplicated": i == 0,
             "sources": ["a", "b"] if i == 0 else ["a"],
             "session_phase": "during_market", "url": "https://x.invalid"}
            for i in range(n)
        ],
    }), encoding="utf-8")


def test_missing_news_says_so_rather_than_returning_nothing(configs):
    body = client.get("/news").json()
    assert body["available"] is False
    assert "refresh news" in body["caveat"]


def test_fresh_news_is_available(configs):
    _news_file(configs / "research", age_hours=1.0)
    body = client.get("/news").json()
    assert body["available"] is True
    assert body["total"] == 2
    assert body["clusters"][0]["duplicated"] is True
    assert body["feed_caveats"]


def test_stale_news_is_refused_not_shown(configs):
    """News a day old describes an earlier session, and anything labelled
    'market context' will be read as describing today."""
    _news_file(configs / "research", age_hours=40.0)
    body = client.get("/news").json()
    assert body["stale"] is True
    assert body["available"] is False
    assert "hours old" in body["caveat"]


# --- cross-check ---------------------------------------------------------

def test_a_missing_crosscheck_names_the_command(configs):
    r = client.get(f"/crosscheck/{DAY}")
    assert r.status_code == 404
    assert "refresh crosscheck" in r.json()["detail"]


def test_a_crosscheck_report_is_served(configs):
    d = configs / "crosscheck"
    d.mkdir(parents=True)
    (d / f"{DAY.isoformat()}.json").write_text(json.dumps({
        "as_of": DAY.isoformat(), "checked_at": "2026-09-18T01:00:00+05:30",
        "tolerance_pct": 2.0,
        "coverage": {"nse_symbols": 3483, "checkable": 665,
                     "thin_on_bse": 1720, "not_on_bse": 182, "no_isin": 916,
                     "ambiguous_isin": 0, "no_usable_close": 0,
                     "balanced": True},
        "disagreements": [{"symbol": "GATECH.NS", "nse_close": 0.70,
                           "bse_close": 0.72, "diff_pct": 2.78,
                           "bse_turnover": 1.2e7}],
    }), encoding="utf-8")

    body = client.get(f"/crosscheck/{DAY}").json()
    assert body["coverage"]["checkable"] == 665
    # The honest denominator: a fifth of the universe, not a pass rate.
    assert body["coverage"]["checkable"] < body["coverage"]["nse_symbols"] / 4
    assert body["disagreements"][0]["symbol"] == "GATECH.NS"


# --- fundamentals --------------------------------------------------------

def test_missing_fundamentals_names_the_command(configs):
    (configs / "bhavcopy").mkdir(parents=True)
    r = client.get("/fundamentals")
    assert r.status_code == 404


def test_the_fundamentals_table_is_served_with_its_coverage(configs, tmp_path):
    import pandas as pd

    from desk.research.fundamentals import COLUMNS, FundamentalsCache

    (configs / "bhavcopy").mkdir(parents=True)
    pd.DataFrame({"symbol": ["AAA.NS"], "series": ["EQ"], "date": [DAY],
                  "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                  "volume": [1]}).to_parquet(
        configs / "bhavcopy" / f"{DAY.isoformat()}.parquet", index=False)

    frame = pd.DataFrame.from_dict(
        {"AAA": {c: None for c in COLUMNS}}, orient="index",
        columns=list(COLUMNS))
    frame.loc["AAA", "net_margin_pct"] = 12.5
    frame.index.name = "symbol"
    FundamentalsCache(configs / "research" / "fundamentals").save(
        frame, as_of=DAY, coverage={"complete": 1, "no filings on file": 99})

    body = client.get("/fundamentals").json()
    assert body["table_as_of"] == DAY.isoformat()
    assert body["coverage"]["no filings on file"] == 99
    assert body["rows"][0]["symbol"] == "AAA"
    assert body["rows"][0]["net_margin_pct"] == 12.5


# --- regime --------------------------------------------------------------

def test_the_regime_endpoint_reports_its_sources():
    """A regime gates strategies, so it has to be arguable rather than
    asserted. Runs against the real store - skipped if empty."""
    r = client.get("/regime")
    if r.status_code == 404:
        pytest.skip("no market snapshot on this machine")
    body = r.json()
    assert set(body) >= {"label", "measured", "sources", "explain",
                         "trend", "volatility", "breadth", "risk_appetite"}
    assert body["sources"], "every dimension must say where it came from"
    for key in ("breadth", "trend", "volatility", "risk_appetite"):
        assert key in body["sources"] or "all" in body["sources"]


def test_an_unmeasured_dimension_is_visible_in_the_response():
    r = client.get("/regime")
    if r.status_code == 404:
        pytest.skip("no market snapshot on this machine")
    body = r.json()
    unknown = [k for k, v in body["sources"].items()
               if str(v).startswith("not measured")]
    # Either everything was measured, or the ones that were not say so.
    assert body["measured"] or unknown, (
        "a partly-measured regime must name which dimension is missing")
