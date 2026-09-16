"""Verify the Vibe-Trading install without needing any API key.

Doubles as the agent's health check. Everything here is offline: CLI probes,
module imports, and a fast subset of the repo's own factor tests.

The LLM agent itself (`vibe-trading run "..."`) DOES need a provider key in
~/.vibe-trading/.env - that path is deliberately not exercised here, so this
test stays free and deterministic.

    agents/vibe_trading/.venv/Scripts/python.exe agents/vibe_trading/smoke_test.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv" / "Scripts"
CLI = VENV / "vibe-trading.exe"
PY = VENV / "python.exe"
AGENT = HERE / "upstream" / "agent"

# Fast, fully offline subset of the repo's 624 test files.
TEST_SUBSET = [
    "tests/factors/test_alpha101_samples.py",
    "tests/factors/test_academic_samples.py",
    "tests/factors/test_factor_analysis_core.py",
]


def sh(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(
        args, capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=cwd,
    )
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    print("vibe-trading smoke test\n" + "-" * 62)
    failures: list[str] = []

    # --- 1. CLI ------------------------------------------------------------
    rc, out = sh([str(CLI), "--version"])
    version = out.strip().splitlines()[0] if out.strip() else "?"
    print(f"  cli version   : {version}")
    if rc != 0:
        failures.append("CLI --version")

    # --- 2. Skills registry ------------------------------------------------
    rc, out = sh([str(CLI), "--skills"])
    # Rich table: count the leading-pipe rows that carry a skill name.
    skills = len(re.findall(r"^│ [a-z0-9][\w-]+\s+│", out, re.M))
    print(f"  skills loaded : {skills}")
    if skills < 50:
        failures.append(f"skills registry returned only {skills}")

    # --- 3. Backtest engines + alpha zoos ----------------------------------
    probe = (
        "import importlib,sys;"
        "mods=['backtest.engines.global_equity','backtest.engines.india_equity',"
        "'backtest.engines.crypto','backtest.engines.china_a','backtest.metrics'];"
        "ok=[];bad=[]\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m); ok.append(m)\n"
        "    except Exception as e: bad.append(f'{m}: {type(e).__name__}')\n"
        "print('OK', len(ok));print('BAD', bad)"
    )
    rc, out = sh([str(PY), "-c", probe], cwd=AGENT)
    m = re.search(r"OK (\d+)", out)
    n_ok = int(m.group(1)) if m else 0
    print(f"  bt engines    : {n_ok}/5 importable")
    if n_ok < 5:
        failures.append(f"backtest engines: {out.strip().splitlines()[-1] if out else 'probe failed'}")

    zoos = sorted(
        p.name for p in (AGENT / "src" / "factors" / "zoo").iterdir()
        if p.is_dir() and not p.name.startswith("__")
    )
    print(f"  alpha zoos    : {', '.join(zoos)}")
    if len(zoos) < 4:
        failures.append("alpha zoos missing")

    # --- 4. Real factor tests ----------------------------------------------
    print("  running offline factor tests ...", flush=True)
    rc, out = sh(
        [str(PY), "-m", "pytest", *TEST_SUBSET, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=AGENT,
    )
    m = re.search(r"(\d+) passed", out)
    passed = int(m.group(1)) if m else 0
    print(f"  tests passed  : {passed}")
    if rc != 0 or passed == 0:
        failures.append("factor tests")
        print("\n".join(out.splitlines()[-10:]))

    print("-" * 62)
    if failures:
        print("\nRESULT: FAIL -", "; ".join(failures))
        return 1
    print("\nRESULT: PASS - CLI, skills, engines, alpha zoos and factor tests all OK")
    print("NOTE  : the LLM agent path needs a provider key in ~/.vibe-trading/.env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
