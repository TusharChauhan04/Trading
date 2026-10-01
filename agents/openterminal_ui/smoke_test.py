"""Verify the OpenTerminalUI modules we depend on still load and still work.

Run in the DESK venv, not an agent venv - this agent has none, because we
import ~16 of its files rather than installing its application:

    .venv/Scripts/python.exe agents/openterminal_ui/smoke_test.py

WHY THIS EXISTS. Two failure modes have already happened in this project
and both are silent:

  1. OneDrive dehydrated agents/vibe_trading/upstream/src/factors/base.py in
     September and the tree looked complete while an import failed. Every
     agents/*/upstream/ is gitignored, so git cannot tell us either.
  2. This repo is an APPLICATION, not a library. Its module boundaries were
     never designed for outside import. A future upstream commit could move
     a helper and break our loads without that being a bug on their side.

So this asserts BEHAVIOUR, not just importability. An import that succeeds
while adx() returns the wrong columns is the kind of pass that teaches
nothing. Every number checked here was measured on 2026-10-01.
"""

from __future__ import annotations

import importlib.util as iu
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UP = HERE / "upstream"
PINNED = "8c46cbc"

# The 16 modules verified usable with no install. 4 of the 20 probed need a
# dependency the desk does not have and are listed as SKIP below, so that
# "not checked" never reads as "checked and passed".
USABLE = [
    "backend/robustness/scorecard.py",
    "backend/services/indicators.py",
    "backend/core/valuation.py",
    "backend/core/ratios.py",
    "backend/core/monte_carlo.py",
    "backend/core/execution_model.py",
    "backend/core/backtest_robustness.py",
    "backend/core/fundamental_scores.py",
    "backend/core/formula_engine.py",
    "backend/core/bond_analytics.py",
    "backend/strategy_export/presets.py",
    "backend/strategy_export/pine.py",
    "backend/nlp/sentiment.py",
    "backend/scanner_engine/indicators.py",
    "backend/scanner_engine/detectors.py",
    "backend/scanner_engine/ranking.py",
]

SKIP = {
    "backend/adapters/yahoo.py": "httpx not installed - not needed, our own fetch works",
    "backend/model_lab/montecarlo.py": "PyYAML not installed",
    "backend/nlp/filing_parser.py": "httpx not installed",
    "backend/core/factor_analysis.py": "SQLAlchemy not installed",
}

DATA = {"backend/config/screeners.yaml": "3 declarative screens"}


def load(rel: str):
    """Load one upstream module by path, with the repo root importable."""
    p = UP / rel
    if not p.is_file():
        raise FileNotFoundError(rel)
    root = str(UP)
    if root not in sys.path:
        sys.path.insert(0, root)
    name = "otui_" + rel.replace("/", "_").removesuffix(".py")
    spec = iu.spec_from_file_location(name, p)
    mod = iu.module_from_spec(spec)
    # REGISTER BEFORE EXECUTING. @dataclass resolves its field types via
    # sys.modules[cls.__module__].__dict__, so a module that is exec'd
    # while absent from sys.modules dies with
    #     'NoneType' object has no attribute '__dict__'
    # the moment it defines a dataclass. That is a bug in the loader, not
    # in the module - valuation, execution_model and bond_analytics all
    # failed this way until the line below was added.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def main() -> int:
    print("openterminal_ui smoke test\n" + "-" * 62)
    failures: list[str] = []

    if not UP.is_dir():
        print(f"FAIL  {UP} is missing entirely - re-clone at {PINNED}")
        return 1

    # ---- every module we claim to depend on must load -----------------
    loaded = {}
    for rel in USABLE:
        try:
            loaded[rel] = load(rel)
            print(f"ok    {rel}")
        except Exception as exc:
            print(f"FAIL  {rel}: {type(exc).__name__}: {exc}")
            failures.append(rel)

    for rel, why in SKIP.items():
        print(f"skip  {rel}  ({why})")

    for rel, why in DATA.items():
        present = (UP / rel).is_file()
        print(f"{'ok  ' if present else 'FAIL'}  {rel}  ({why})")
        if not present:
            failures.append(rel)

    # THREE screens, not six. The manifest and the salvage map both said six
    # until the top-level keys were actually counted - the same class of
    # miscount that produced "19 strategy templates" for a dict of 6. Asserted
    # here so the wrong number cannot come back, and so an upstream change to
    # the file is noticed rather than assumed away.
    yml = UP / "backend/config/screeners.yaml"
    if yml.is_file():
        import re as _re
        keys = _re.findall(r"^([a-z_]+):\s*$", yml.read_text(encoding="utf-8"),
                           _re.MULTILINE)
        if len(keys) != 3:
            failures.append(f"screeners.yaml has {len(keys)} screens, "
                            f"expected 3: {keys}")
            print(f"FAIL  screeners.yaml has {len(keys)} screens, expected 3")
        else:
            print(f"ok    screeners.yaml: {len(keys)} screens {keys}")

    # ---- and the ones we rely on must still BEHAVE --------------------
    print("-" * 62)

    sc = loaded.get("backend/robustness/scorecard.py")
    if sc:
        # A deliberately flat series: PSR should be high, and DSR must
        # COLLAPSE once we admit how many trials were tried. If DSR ever
        # stops falling with num_trials, the deflation is broken and every
        # verdict built on it is worthless.
        # Seeded noise with a small positive mean, NOT a repeating pattern:
        # a cycle of 12 values repeated six times has almost no variance
        # between cycles, which flatters the Sharpe so much that DSR barely
        # moves and the assertion below stops testing anything.
        import random as _r
        _rng = _r.Random(7)
        series = [_rng.gauss(0.004, 0.02) for _ in range(72)]
        one = sc.compute_robustness(series, num_trials=1, seed=7)
        many = sc.compute_robustness(series, num_trials=615, seed=7)
        if not (one["dsr"] > many["dsr"]):
            failures.append("scorecard: DSR did not fall as num_trials rose")
            print(f"FAIL  scorecard DSR {one['dsr']:.4f} -> {many['dsr']:.4f}")
        else:
            print(f"ok    scorecard: PSR {one['psr']:.4f}, DSR "
                  f"{one['dsr']:.4f} (1 trial) -> {many['dsr']:.4f} (615)")
        for key in ("psr", "dsr", "min_track_record_length", "verdict"):
            if key not in one:
                failures.append(f"scorecard: '{key}' gone from the result")
                print(f"FAIL  scorecard result lost '{key}'")

    ind = loaded.get("backend/services/indicators.py")
    if ind:
        names = ind.list_indicators()
        n = len(names)
        if n < 13:
            failures.append(f"indicators: {n} registered, expected >= 13")
            print(f"FAIL  indicator registry has {n}, expected >= 13")
        else:
            print(f"ok    indicators: {n} registered")

    pre = loaded.get("backend/strategy_export/presets.py")
    if pre:
        n = len(pre.PRESETS)
        # SIX, not nineteen. An earlier count of 19 came from grepping
        # '"id"', which matches once per preset AND once per indicator
        # inside it. The assertion is here so the wrong number cannot
        # come back.
        if n != 6:
            failures.append(f"presets: {n} templates, expected 6")
            print(f"FAIL  PRESETS has {n} templates, expected 6")
        else:
            print(f"ok    presets: {n} strategy templates")

    val = loaded.get("backend/core/valuation.py")
    if val and not hasattr(val, "DcfInputs"):
        failures.append("valuation: DcfInputs is gone")
        print("FAIL  valuation lost DcfInputs")
    elif val:
        print("ok    valuation: DcfInputs present")

    print("-" * 62)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        print("\nIf a file is 'missing', check OneDrive dehydration before")
        print("assuming the clone is bad: `attrib` on the file will show O.")
        return 1
    print(f"all {len(USABLE)} modules load and {len(DATA)} data asset present")
    print(f"upstream pinned at {PINNED}; 4 modules skipped by design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
