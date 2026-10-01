"""Declarative fundamental screens, and an honest account of which can run.

WHAT THIS IS
------------
OpenTerminalUI ships `backend/config/screeners.yaml` - fundamental screens
declared as data rather than code, which is the right shape: a screen is a
policy, and policy belongs in a file a human edits, not in a function someone
has to re-deploy.

THREE SCREENS, NOT SIX. The salvage map and this project's manifest both said
six; the file has three (`value`, `quality`, `growth`) in 38 lines. Counted by
reading the top-level keys, after the same kind of miscount produced "19
strategy templates" for a dict holding 6.

AND ONLY ONE OF THE THREE CAN RUN, for the same reason the DCF cannot: the
fields come from `core/ratios.py`, which is shaped for a Yahoo Finance `.info`
dict, and NSE quarterly XBRL is a profit-and-loss statement.

    screen     needs                                              status
    growth     rev_growth_pct, eps_growth_pct, net_margin_pct      RUNNABLE
    quality    roe_pct, op_margin_pct, debt_to_market_cap          blocked
    value      pe, roe_pct, debt_to_market_cap                     blocked

`roe_pct` needs shareholders' equity and `debt_to_market_cap` needs borrowings.
Neither is in a quarterly results filing. `op_margin_pct` WAS missing and is now
derived from EBIT and revenue, so `quality` is one field short rather than two -
which is worth recording, because if a balance-sheet source ever appears, that
screen becomes available for the cost of one more derivation.

A BLOCKED SCREEN IS RETURNED, NOT HIDDEN. `run_screen` on `value` raises with
the missing fields named, and `SCREEN_STATUS` reports all three. Dropping the
two would make the catalogue look smaller than it is and lose the information
about what a future data source would unlock.

THE RULES ARE TRANSCRIBED, NOT PARSED, and that is a deliberate trade. Reading
the YAML needs PyYAML, which is not installed, and installing a dependency to
read 38 lines where two of three screens cannot run is not yet worth the risk to
the venv. `SCREENS` below reproduces the file exactly; `verify_against_upstream`
re-reads the raw text and asserts they still agree, so a drift is caught rather
than assumed away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

__all__ = ["Rule", "SCREENS", "SCREEN_STATUS", "Screen", "run_screen",
           "verify_against_upstream"]

_UPSTREAM_YAML = (Path(__file__).resolve().parents[2] / "agents"
                  / "openterminal_ui" / "upstream" / "backend" / "config"
                  / "screeners.yaml")

#: Columns the desk's fundamentals table actually provides, mapped from the
#: names screeners.yaml uses. Anything not here is why a screen is blocked.
FIELD_MAP = {
    "rev_growth_pct": "revenue_growth_yoy_pct",
    "eps_growth_pct": "profit_growth_yoy_pct",
    "net_margin_pct": "net_margin_pct",
    "op_margin_pct": "op_margin_pct",
}

_OPS = {
    "<=": lambda s, v: s <= v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    ">": lambda s, v: s > v,
    "==": lambda s, v: s == v,
}


@dataclass(frozen=True, slots=True)
class Rule:
    field: str
    op: str
    value: float

    def __str__(self) -> str:
        return f"{self.field} {self.op} {self.value:g}"


@dataclass(frozen=True, slots=True)
class Screen:
    name: str
    description: str
    rules: tuple[Rule, ...]

    @property
    def missing_fields(self) -> tuple[str, ...]:
        """Fields this screen needs that the desk cannot supply."""
        return tuple(r.field for r in self.rules if r.field not in FIELD_MAP)

    @property
    def runnable(self) -> bool:
        return not self.missing_fields


#: Transcribed from screeners.yaml. `verify_against_upstream` checks the match.
SCREENS: dict[str, Screen] = {
    "value": Screen(
        "value", "Low valuation with baseline quality",
        (Rule("pe", "<=", 25), Rule("roe_pct", ">=", 12),
         Rule("debt_to_market_cap", "<=", 0.8)),
    ),
    "quality": Screen(
        "quality", "High quality business profile",
        (Rule("roe_pct", ">=", 15), Rule("op_margin_pct", ">=", 12),
         Rule("debt_to_market_cap", "<=", 0.5)),
    ),
    "growth": Screen(
        "growth", "Growth with positive profitability",
        (Rule("rev_growth_pct", ">=", 10), Rule("eps_growth_pct", ">=", 10),
         Rule("net_margin_pct", ">=", 8)),
    ),
}

#: Which screens can run, and precisely what blocks the others. Reported rather
#: than inferred, so the answer to "why is there no value screen" is a fact
#: about the data instead of a shrug.
SCREEN_STATUS: dict[str, str] = {
    name: ("runnable" if s.runnable
           else "blocked: " + ", ".join(s.missing_fields))
    for name, s in SCREENS.items()
}


def run_screen(name: str, table: pd.DataFrame) -> pd.DataFrame:
    """Rows of `table` passing every rule in the named screen.

    `table` is the fundamentals frame from `desk.research.fundamentals`,
    indexed by symbol. Rules are ANDed, and a row with a NaN in ANY tested
    column is EXCLUDED rather than treated as passing - a company that did not
    report a margin has not demonstrated an 8% one.

    Raises for a blocked screen, naming the fields, so that "this screen cannot
    run here" can never be mistaken for "this screen matched nothing".
    """
    if name not in SCREENS:
        raise KeyError(f"unknown screen {name!r}; have {sorted(SCREENS)}")
    screen = SCREENS[name]
    if not screen.runnable:
        raise NotImplementedError(
            f"screen {name!r} needs {', '.join(screen.missing_fields)}, which "
            f"NSE quarterly results XBRL does not carry - the same "
            f"balance-sheet gap that blocks the DCF. See "
            f"desk.research.derived.DCF_MISSING_INPUTS.")

    mask = pd.Series(True, index=table.index)
    for rule in screen.rules:
        col = FIELD_MAP[rule.field]
        if col not in table.columns:
            raise KeyError(
                f"screen {name!r} wants column {col!r}, absent from the "
                f"table (have: {sorted(table.columns)})")
        series = pd.to_numeric(table[col], errors="coerce")
        mask &= series.notna() & _OPS[rule.op](series, rule.value)
    return table.loc[mask]


def verify_against_upstream(path: Path | None = None) -> dict[str, list[str]]:
    """Do the transcribed rules still match the file they were copied from?

    Returns `{screen: [differences]}`, empty when everything agrees. A missing
    upstream returns `{}` rather than raising: `agents/*/upstream/` is
    gitignored, so its absence is the normal state on a fresh clone and is not
    evidence of drift.

    Parsed with a regex over the two-level structure the file actually has
    rather than with a YAML library, for the reason in the module docstring.
    """
    p = path or _UPSTREAM_YAML
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")

    found: dict[str, list[tuple[str, str, float]]] = {}
    current: str | None = None
    pending: dict[str, str] = {}
    for line in text.splitlines():
        top = re.match(r"^([a-z_]+):\s*$", line)
        if top:
            current = top.group(1)
            found[current] = []
            pending = {}
            continue
        if current is None:
            continue
        for key in ("field", "op", "value"):
            m = re.search(rf"\b{key}:\s*\"?([^\"\s]+)\"?\s*$", line)
            if m:
                pending[key] = m.group(1)
        if {"field", "op", "value"} <= pending.keys():
            found[current].append((pending["field"], pending["op"],
                                   float(pending["value"])))
            pending = {}

    diffs: dict[str, list[str]] = {}
    for name in set(found) | set(SCREENS):
        ours = [(r.field, r.op, float(r.value))
                for r in SCREENS[name].rules] if name in SCREENS else None
        theirs = found.get(name)
        if ours is None:
            diffs[name] = [f"upstream has {name!r}, we do not"]
        elif theirs is None:
            diffs[name] = [f"we have {name!r}, upstream does not"]
        elif sorted(ours) != sorted(theirs):
            diffs[name] = [f"upstream {theirs} != ours {ours}"]
    return diffs


def screen_report(table: pd.DataFrame,
                  names: Sequence[str] | None = None) -> str:
    """Every screen's outcome, including the ones that could not run."""
    lines = []
    for name in (names or sorted(SCREENS)):
        s = SCREENS[name]
        if not s.runnable:
            lines.append(f"  {name:<10} NOT RUN - needs "
                         f"{', '.join(s.missing_fields)}")
            continue
        hits = run_screen(name, table)
        lines.append(f"  {name:<10} {len(hits):>4} of {len(table)} pass  "
                     f"({'; '.join(str(r) for r in s.rules)})")
    return "\n".join(lines)
