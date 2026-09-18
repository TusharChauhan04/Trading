"""Nothing is built and then left disconnected.

THIS FILE EXISTS BECAUSE THE TEST SUITE KEPT BEING GREEN ABOUT IT.

Six times in this project something was written, covered by passing tests,
and reachable from nothing:

  1. `refresh_bhavcopy()` with its CLI subcommand missing from argparse
  2. `refresh_research()` with no CLI at all
  3. `StubSession` standing in for NseSession with a smaller surface
  4. The fundamentals join that never matched a row
  5. R5 (news) - written, 35 tests, imported by nothing
  6. R6 (BSE cross-check) - written, 38 tests, imported by nothing

Every one of those had a green suite. A passing test proves the code works
WHEN CALLED; it says nothing about whether anything calls it. So this file
tests the other half: that every module is reachable from a real entry
point, and that the specific subsystems the desk depends on are actually
on the live path.

A module deliberately not wired belongs in EXPECTED_UNWIRED with the
reason written down. That list is the whole point - it turns "we forgot"
into "we decided", and it has to be edited on purpose.
"""

from __future__ import annotations

import pathlib
import re
from datetime import date

import pytest

from desk.contracts.enums import Regime

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "desk"

#: Everything that can start execution. A module reachable from none of
#: these cannot run in production, whatever its tests say.
ENTRY_POINTS = (
    "desk.api.main",              # the live plan
    "desk.marketdata.refresh",    # the market-data CLI
    "desk.research.refresh",      # the research CLI
    "desk.backtest.engine",       # the backtest harness
    "desk.backtest.baseline",     # the random-baseline harness
)

#: Modules that are legitimately not reachable, each with its reason.
#: EDIT THIS ON PURPOSE. An entry added to silence a failure, without a
#: reason that survives being read aloud, defeats the file.
EXPECTED_UNWIRED = {
    "desk.contracts.envelope":
        "The agent-integration boundary. Unused because NO AGENT IS WIRED "
        "yet - 0 of 6 strategies are on disk. Pending integration, not "
        "forgotten code. Delete this entry the day an agent lands.",
    "desk.indicators.trend":
        "Reference implementation. Stage 1 computes features vectorised; "
        "test_stage1 imports these as ref_* to prove parity. Wiring them "
        "into production would replace the fast path with the slow one.",
    "desk.indicators.volatility": "Reference implementation - see indicators.trend.",
    "desk.indicators.volume": "Reference implementation - see indicators.trend.",
    "desk.indicators.structure": "Reference implementation - see indicators.trend.",
    "desk.indicators.relative_strength":
        "Reference implementation - see indicators.trend. Stage 1 reports "
        "rs_rank as NaN and declares it in `unavailable`.",
}


def _modules() -> dict[str, pathlib.Path]:
    return {
        str(p.relative_to(ROOT).with_suffix("")).replace("\\", "/").replace("/", "."): p
        for p in PKG.rglob("*.py")
        if "tests" not in p.parts and p.name != "__init__.py"
    }


def _reachable(entry: str, mods: dict[str, pathlib.Path]) -> set[str]:
    seen: set[str] = set()
    stack = [entry]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        path = mods.get(cur)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"from (desk[\w.]*) import|import (desk[\w.]*)", text):
            target = m.group(1) or m.group(2)
            stack.extend(c for c in mods
                         if c == target or c.startswith(target + "."))
    return seen


@pytest.fixture(scope="module")
def wiring():
    mods = _modules()
    reached: set[str] = set()
    for entry in ENTRY_POINTS:
        assert entry in mods, f"entry point {entry} does not exist"
        reached |= _reachable(entry, mods)
    return mods, reached


def test_every_module_is_reachable_from_an_entry_point(wiring):
    """The guard. A module nothing imports cannot run, however well tested."""
    mods, reached = wiring
    orphans = sorted(set(mods) - reached - set(EXPECTED_UNWIRED))
    assert not orphans, (
        "These modules are reachable from NO entry point, so they cannot run "
        "in production no matter how many tests pass:\n  "
        + "\n  ".join(orphans)
        + "\n\nWire them, or add them to EXPECTED_UNWIRED with a reason."
    )


def test_the_exemption_list_has_no_stale_entries(wiring):
    """An exemption for something now wired is a lie left in the code."""
    mods, reached = wiring
    stale = sorted(m for m in EXPECTED_UNWIRED if m in reached)
    assert not stale, (
        "These are listed as deliberately unwired but ARE now reachable. "
        "Remove them from EXPECTED_UNWIRED:\n  " + "\n  ".join(stale))


def test_the_exemption_list_has_no_deleted_entries(wiring):
    mods, _ = wiring
    gone = sorted(m for m in EXPECTED_UNWIRED if m not in mods)
    assert not gone, f"EXPECTED_UNWIRED names modules that no longer exist: {gone}"


def test_every_exemption_states_a_reason():
    for name, reason in EXPECTED_UNWIRED.items():
        assert len(reason) > 40, (
            f"{name} is exempted without a real reason. The list exists to "
            f"turn 'we forgot' into 'we decided'.")


# --- the subsystems the desk actually depends on -------------------------
#
# Reachability alone is not enough: a module can be imported and never
# called. These name the specific things whose absence would silently
# weaken the daily plan, and assert they are on the LIVE path.

@pytest.mark.parametrize("module,why", [
    ("desk.research.fundamentals",
     "R3 - without it no candidate is screened on revenue, profit or margin"),
    ("desk.research.events",
     "R4 - without it a name reporting inside the holding window is not caught"),
    ("desk.research.news",
     "R5 - without it Stage 3 judges a shortlist with no market context"),
    ("desk.marketdata.crosscheck",
     "R6 - without it the plan rests on a single price source"),
    ("desk.marketdata.isin", "R6 - the NSE<->BSE join key"),
    ("desk.scanner.stage3", "R7 - the narrative pass"),
    ("desk.llm.budget", "R7 - no paid call may happen unmetered"),
    ("desk.journal.store",
     "every unrecorded day is evidence about the desk that cannot be "
     "recovered later"),
    ("desk.marketdata.quality",
     "the bar-level screen on names about to be sized"),
    ("desk.regime.engine", "otherwise the regime is an opinion, not a measurement"),
    ("desk.marketdata.sectors", "sector strength and index membership"),
    ("desk.settings", "the API key, the spend ceiling and the risk-reward"),
])
def test_the_live_plan_reaches(module, why, wiring):
    mods, _ = wiring
    live = _reachable("desk.api.main", mods)
    assert module in live, f"{module} is NOT on the live plan path - {why}"


def test_the_backtest_never_builds_a_live_llm_client(monkeypatch):
    """REGRESSION, caught by this very file on its first run.

    Module reachability is the wrong assertion here: the backtest reaches
    Stage 3 through desk.api.main, so it legitimately imports everything
    the API imports, provider included. What must be guaranteed is
    BEHAVIOURAL - that no client is ever constructed for a replay.

    It was not. run_backtest -> _run_funnel -> _llm_client(), and
    _llm_client builds a real provider from OPENAI_API_KEY. A backtest run
    on a machine with a key would have made hundreds of live paid calls,
    each asking a model about a date it may partly remember. stage3.py's
    "never call this inside a backtest loop" was true of run_stage3 and
    false of the only path that reaches it.
    """
    import inspect

    from desk.api import main as api
    from desk.backtest import engine

    # 1. The funnel must offer the switch...
    assert "use_llm" in inspect.signature(api._run_funnel).parameters

    # 2. ...and the backtest must actually use it.
    code = [ln.split("#")[0]
            for ln in inspect.getsource(engine.run_backtest).splitlines()]
    assert any("use_llm=False" in ln for ln in code), (
        "run_backtest must pass use_llm=False into the funnel")

    # 3. And with a key present, Stage 3 must still receive nothing.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-would-have-been-charged")
    assert api._llm_client() is not None, "the key should build a client"

    seen = {}
    real = api.run_stage3

    def _spy(stage2, *, client, **kw):
        seen["client"] = client
        return real(stage2, client=client, **kw)

    monkeypatch.setattr(api, "run_stage3", _spy)
    api._run_funnel(date(2026, 9, 17), regime=Regime.UNKNOWN, capital=100_000,
                    max_trades=3, lookback=60, portfolio=None,
                    today=date(2026, 9, 17), use_llm=False)
    assert seen.get("client", "unset") is None, (
        "a replay was handed a live LLM client")
