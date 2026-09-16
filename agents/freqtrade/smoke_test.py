"""Verify the Freqtrade install by running a real backtest and a real hyperopt.

Doubles as the agent's health check. Fully offline: uses the bundled test data
staged into user_data/data/binance, dry_run only, no API keys, no network.

Deliberately shells out to the freqtrade CLI rather than importing freqtrade.
Two reasons:
  1. Freqtrade is GPL-3.0. Importing it into platform code would make the
     platform a derivative work. Subprocess + JSON keeps it at arm's length.
  2. The CLI is freqtrade's actual supported interface; its internals are not.

    agents/freqtrade/.venv/Scripts/python.exe agents/freqtrade/smoke_test.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREQTRADE = HERE / ".venv" / "Scripts" / "freqtrade.exe"
CONFIG = HERE / "config.smoke.json"
USERDIR = HERE / "user_data"
DATADIR = USERDIR / "data" / "binance"

COMMON = [
    "--config", str(CONFIG),
    "--datadir", str(DATADIR),
    "--userdir", str(USERDIR),
]


def run(args: list[str], label: str) -> str:
    print(f"  running {label} ...", flush=True)
    proc = subprocess.run(
        [str(FREQTRADE), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=HERE,
    )
    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode})")
        print("\n".join(proc.stdout.splitlines()[-15:]))
        print("\n".join(proc.stderr.splitlines()[-15:]))
        raise SystemExit(1)
    return proc.stdout + proc.stderr


def check_preconditions() -> None:
    missing = [p for p in (FREQTRADE, CONFIG, DATADIR) if not p.exists()]
    if missing:
        print("MISSING:", *[str(m) for m in missing], sep="\n  ")
        raise SystemExit(1)


def main() -> int:
    print("freqtrade smoke test\n" + "-" * 62)
    check_preconditions()

    # --- 1. Backtest -------------------------------------------------------
    out = run(
        ["backtesting", *COMMON, "--strategy", "SampleStrategy"],
        "backtest (SampleStrategy)",
    )
    row = re.search(
        r"\|\s*SampleStrategy\s*\|\s*(\d+)\s*\|.*?\|\s*([\d.-]+)\s*\|",
        out,
    )
    trades = int(row.group(1)) if row else 0
    win = re.search(r"\|\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*\|", out)
    print(f"    trades   : {trades}")
    if win:
        print(f"    win rate : {win.group(4)}%")

    # --- 2. Hyperopt -------------------------------------------------------
    out = run(
        [
            "hyperopt", *COMMON,
            "--strategy", "HyperoptableStrategy",
            "--hyperopt-loss", "SharpeHyperOptLoss",
            "--spaces", "buy",
            "--epochs", "5",
            "--job-workers", "1",
        ],
        "hyperopt (5 epochs, optuna)",
    )
    saved = re.search(r"(\d+) epochs saved", out)
    best = re.search(r"Best result:\s*\n?\s*\*?\s*(\d+)/(\d+)", out)
    print(f"    epochs   : {saved.group(1) if saved else '?'}")
    if best:
        print(f"    best at  : epoch {best.group(1)}/{best.group(2)}")

    # --- 3. Capability probes ---------------------------------------------
    print("\n  capabilities:")
    for label, mod in [
        ("TA-Lib", "talib"),
        ("ccxt", "ccxt"),
        ("hyperopt", "freqtrade.optimize.hyperopt"),
        ("FreqAI", "freqtrade.freqai"),
        ("plotting", "plotly"),
    ]:
        try:
            __import__(mod)
            print(f"    {label:<10} OK")
        except Exception as exc:  # noqa: BLE001 — reporting, not handling
            print(f"    {label:<10} MISSING ({type(exc).__name__})")

    print("-" * 62)
    ok = trades > 0 and saved is not None
    print("\nRESULT:", "PASS - backtest traded and hyperopt optimised" if ok
          else "FAIL - see output above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
