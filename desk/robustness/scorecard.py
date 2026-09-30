"""PSR, DSR and MinTRL, borrowed from openterminal_ui and made honest.

THE ONE DESIGN DECISION HERE: `num_trials` is a required argument with no
default. The upstream function defaults it to 1, and a default of 1 is the
precise lie that makes a Deflated Sharpe Ratio worthless - it says "I tried
exactly one thing", which is almost never true of anything worth deflating.
Forcing the caller to state the number makes the search cost visible at the
call site, where the person who knows it is standing.

Measured on this project's own data, that decision is not academic:

    a series with PSR 0.879  ->  DSR 0.026  once 615 trials are declared

Same numbers, same code. The only change is admitting what was tried.

UNAVAILABLE IS A THIRD STATE, not a failure and not a pass. Every
agents/*/upstream/ is gitignored and OneDrive has already dehydrated one of
them mid-session. So a missing upstream returns a report with
`unavailable` set and every statistic None, rather than raising or - far
worse - quietly returning a number that was never computed. A caller that
prints "DSR: None (upstream missing)" is telling the truth; one that prints
0.0 is not.
"""

from __future__ import annotations

import importlib.util as iu
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

_UPSTREAM = (Path(__file__).resolve().parents[2]
             / "agents" / "openterminal_ui" / "upstream")
_MODULE = _UPSTREAM / "backend" / "robustness" / "scorecard.py"

# Below this, Bailey & Lopez de Prado's estimators are too unstable to
# report - it is upstream's own threshold, surfaced here so callers can see
# why a verdict came back "insufficient" rather than guessing.
MIN_OBSERVATIONS = 60


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """What survived the search. None everywhere means it was not computed."""

    n: int
    num_trials: int
    psr: float | None = None
    dsr: float | None = None
    annual_sharpe: float | None = None
    skew: float | None = None
    excess_kurtosis: float | None = None
    min_track_record_length: float | None = None
    verdict: str = "unavailable"
    reasons: tuple[str, ...] = ()
    unavailable: str | None = None
    """Why nothing was computed. None when the statistics are real."""

    @property
    def underpowered(self) -> bool:
        """Too few observations for the estimators to mean anything.

        Distinct from a bad result: this says the measurement cannot answer
        the question, not that the answer is no.
        """
        return self.n < MIN_OBSERVATIONS

    def __str__(self) -> str:
        if self.unavailable:
            return f"robustness unavailable: {self.unavailable}"
        bits = [f"n={self.n}", f"trials={self.num_trials}"]
        if self.psr is not None:
            bits.append(f"PSR {self.psr:.4f}")
        if self.dsr is not None:
            bits.append(f"DSR {self.dsr:.4f}")
        if self.annual_sharpe is not None:
            bits.append(f"annual Sharpe {self.annual_sharpe:+.3f}")
        bits.append(f"verdict {self.verdict}")
        if self.underpowered:
            bits.append(f"UNDERPOWERED (< {MIN_OBSERVATIONS} observations)")
        return ", ".join(bits)


@lru_cache(maxsize=1)
def _load() -> Any | None:
    """The upstream module, or None if it is not on disk.

    Registered in sys.modules BEFORE exec_module: @dataclass resolves its
    field types through sys.modules[cls.__module__].__dict__, so a module
    exec'd while absent from sys.modules dies with "'NoneType' object has
    no attribute '__dict__'" the moment it defines a dataclass. Three
    upstream modules failed exactly that way before this was fixed.
    """
    if not _MODULE.is_file():
        return None
    root = str(_UPSTREAM)
    if root not in sys.path:
        sys.path.insert(0, root)
    name = "otui_robustness_scorecard"
    if name in sys.modules:
        return sys.modules[name]
    spec = iu.spec_from_file_location(name, _MODULE)
    if spec is None or spec.loader is None:
        return None
    mod = iu.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        return None
    return mod


def available() -> bool:
    """Whether the deflated statistics can be computed at all."""
    return _load() is not None


def deflated_sharpe(returns: Sequence[float], *, num_trials: int,
                    periods_per_year: int = 252,
                    benchmark: float = 0.0,
                    seed: int | None = 7) -> RobustnessReport:
    """PSR, DSR and MinTRL for one series, deflated for the search.

    `returns` is one number per period - R-multiples per trade, or per
    cohort where trades overlap. OVERLAPPING RETURNS BREAK THIS: two trades
    open at once are not two observations, and feeding them in inflates
    every statistic here. Make the series non-overlapping before calling.

    `num_trials` is how many variants were tried before this one was
    picked - strategies times parameter cells times regimes. It has no
    default on purpose; see the module docstring.

    `periods_per_year` must match what one element of `returns` spans, not
    the bar size the data came in. A 20-session holding period is about 13
    periods a year; passing 252 for it annualises a monthly return as if it
    were daily and inflates the Sharpe by sqrt(20).
    """
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")

    vals = [float(r) for r in returns]
    mod = _load()
    if mod is None:
        return RobustnessReport(
            n=len(vals), num_trials=num_trials,
            unavailable=f"agents/openterminal_ui/upstream missing at "
                        f"{_MODULE.relative_to(_MODULE.parents[4])} - "
                        f"re-clone at 8c46cbc",
        )
    if not vals:
        return RobustnessReport(n=0, num_trials=num_trials,
                                unavailable="no returns given")

    raw = mod.compute_robustness(
        vals, num_trials=num_trials, periods_per_year=periods_per_year,
        benchmark_sharpe=benchmark, seed=seed,
    )
    return RobustnessReport(
        n=raw.get("n_periods") or len(vals),
        num_trials=num_trials,
        psr=raw.get("psr"),
        dsr=raw.get("dsr"),
        annual_sharpe=raw.get("annual_sharpe"),
        skew=raw.get("skew"),
        excess_kurtosis=raw.get("excess_kurtosis"),
        min_track_record_length=raw.get("min_track_record_length"),
        verdict=raw.get("verdict") or "unknown",
        reasons=tuple(raw.get("verdict_reasons") or ()),
    )
