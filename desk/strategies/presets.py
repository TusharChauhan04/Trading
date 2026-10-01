"""OpenTerminalUI's declarative strategy presets, translated to desk signals.

WHAT THESE ARE, AND WHY THEY ARE WORTH THE TROUBLE
--------------------------------------------------
`agents/openterminal_ui/upstream/backend/strategy_export/presets.py` holds six
strategies declared as DATA - an indicator list, entry and exit rules written
as `cross_above` / `cross_below` comparisons, and a risk block. Six, not the
nineteen first reported; that count came from grepping `"id"`, which matches
once per preset and once per indicator inside it.

Unlike every other module borrowed from that repository, THIS ONE IS IMPORTED
rather than reimplemented. A dict of numbers and operator names cannot backfill
a warmup or smuggle a look-ahead; there is no arithmetic in it to get wrong.
What needed writing is the evaluation, and that is here, using the desk's own
indicators so the RSI and MACD are the same ones Stage 1 computes.

THE REASON THIS MATTERS MORE THAN IT LOOKS
------------------------------------------
Every preset carries `stop_pct` and `take_pct`, and five of the six are
natively 1:2 - the desk's stated goal:

    sma_cross            3.0% / 6.0%    1:2
    ema_cross            2.5% / 5.0%    1:2
    rsi_reversion        4.0% / 4.0%    1:1   <- refused, see below
    macd_signal          3.0% / 6.0%    1:2
    bollinger_breakout   4.0% / 8.0%    1:2
    donchian_breakout    5.0% / 10.0%   1:2

That last line is the striking one. The hourly factor scan, left free to
optimise, converged on a 5% stop with a 2R target - the same cell this
catalogue specifies for Donchian. Two unrelated routes arriving at one
configuration is weak evidence on its own, but it makes the daily re-test
(roadmap H5) and this extraction step the SAME measurement rather than two.

FIXED PERCENTAGE STOPS, NOT STRUCTURAL ONES, and that is a real difference
from how the desk normally works. Stage 4 derives a stop from where the trade
would be proven wrong - a swing low, a channel edge - and the measured median
came out near 12% of price. A flat 3% is a different claim: it says "I will
accept being wrong at 3%" regardless of where the structure sits. Neither is
obviously right, but they are not interchangeable, and the earlier daily
results used the structural stops. So nothing measured here can be compared
to those numbers line by line.

ONLY THE LONG SIDE IS TRANSLATED. Every preset declares `entry_short` too, and
Indian retail cannot short cash equity beyond intraday - the roadmap's P4
constraint. Translating shorts would produce signals the desk is not permitted
to act on, which is worse than not having them.
"""

from __future__ import annotations

import importlib.util as iu
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["PRESET_IDS", "PresetRule", "available", "load_presets",
           "preset_rules"]

_UPSTREAM = (Path(__file__).resolve().parents[2]
             / "agents" / "openterminal_ui" / "upstream")
_MODULE = _UPSTREAM / "backend" / "strategy_export" / "presets.py"

#: The desk's own floor. A trade whose target is not at least twice its risk
#: is refused by the risk engine, so a preset below this cannot be tested as
#: written - and silently widening its target to comply would be measuring a
#: strategy nobody declared.
MIN_RR = 2.0

PRESET_IDS = ("sma_cross", "ema_cross", "rsi_reversion", "macd_signal",
              "bollinger_breakout", "donchian_breakout")

#: Already implemented natively in desk/strategies/adapters.py and measured in
#: the three-year walk-forward. Kept here so the overlap is explicit rather
#: than discovered twice.
ALREADY_OURS = ("donchian_breakout", "bollinger_breakout")


@lru_cache(maxsize=1)
def _load() -> Any | None:
    """Upstream's PRESETS module, or None when the clone is absent.

    Registered in sys.modules before exec_module: @dataclass and pydantic both
    resolve field types through sys.modules[cls.__module__], and a module
    exec'd while absent from it dies with "'NoneType' object has no attribute
    '__dict__'".
    """
    if not _MODULE.is_file():
        return None
    root = str(_UPSTREAM)
    if root not in sys.path:
        sys.path.insert(0, root)
    name = "otui_strategy_presets"
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
    return _load() is not None


# --------------------------------------------------------------------------
# indicator evaluation, matrix-native
# --------------------------------------------------------------------------

def _sma(close: pd.DataFrame, length: int) -> pd.DataFrame:
    return close.rolling(length, min_periods=length).mean()


def _ema(close: pd.DataFrame, length: int) -> pd.DataFrame:
    return close.ewm(span=length, adjust=False).mean()


def _rsi(close: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Wilder RSI, reproducing Stage 1 and strategy_lib EXACTLY.

    Including the part that looks like an omission: a name with no down bar in
    the window has zero average loss, and dividing by `loss.replace(0, nan)`
    makes its RSI NaN rather than 100. NaN is right - 100 would read as
    maximally overbought and feed an overbought rule, when the truth is that
    the indicator is undefined there.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.DataFrame, fast: int, slow: int, signal: int,
          component: str) -> pd.DataFrame:
    line = _ema(close, fast) - _ema(close, slow)
    if component == "line":
        return line
    if component == "signal":
        return line.ewm(span=signal, adjust=False).mean()
    if component == "hist":
        return line - line.ewm(span=signal, adjust=False).mean()
    raise ValueError(f"unknown macd component {component!r}")


def _bb(close: pd.DataFrame, length: int, mult: float,
        component: str) -> pd.DataFrame:
    mid = close.rolling(length, min_periods=length).mean()
    # ddof=0: the population standard deviation, which is what TradingView and
    # every retail chart draw. ddof=1 shifts the bands slightly and therefore
    # moves which bars qualify as a breakout.
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    if component == "upper":
        return mid + mult * sd
    if component == "lower":
        return mid - mult * sd
    if component in ("basis", "middle", "mid"):
        return mid
    raise ValueError(f"unknown bollinger component {component!r}")


def _donchian(high: pd.DataFrame, low: pd.DataFrame, length: int,
              component: str) -> pd.DataFrame:
    """Donchian channel EXCLUDING the current bar.

    This is the whole correctness of a breakout rule rather than a detail of
    it. A channel computed over a window containing the current bar can never
    be broken - the bar's own high is already the maximum - so the rule fires
    never while looking like a quiet market. The desk hit exactly this when
    building its own Donchian adapter.
    """
    if component == "upper":
        return high.shift(1).rolling(length, min_periods=length).max()
    if component == "lower":
        return low.shift(1).rolling(length, min_periods=length).min()
    raise ValueError(f"unknown donchian component {component!r}")


def _extreme(series: pd.DataFrame, length: int, want: str) -> pd.DataFrame:
    """Rolling highest/lowest over a window that EXCLUDES the current bar.

    THE SHIFT IS THE WHOLE CORRECTNESS OF A BREAKOUT RULE, not a detail of it.
    Donchian declares `close cross_above highest(high, 20)`. If that window
    contains today, then highest >= today's high >= today's close, so the
    condition is arithmetically impossible and the rule fires NEVER - while
    looking exactly like a strategy that found no setups in a quiet market.

    The desk hit precisely this building its own Donchian adapter, which is
    why `prior_high_20` in Stage 1 is defined with `.shift(1)` too. Upstream's
    spec does not say to shift, because on a charting platform the breakout is
    evaluated against the previous bar's channel by convention. Translating it
    literally reproduces the silence, not the strategy.
    """
    shifted = series.shift(1).rolling(length, min_periods=length)
    return shifted.max() if want == "highest" else shifted.min()


def _evaluate_indicator(ind: Any, panel: dict[str, pd.DataFrame]
                        ) -> pd.DataFrame:
    """One declared indicator -> a date x symbol frame."""
    kind = ind.type if hasattr(ind, "type") else ind["type"]
    params = dict(ind.params if hasattr(ind, "params") else ind["params"])
    close = panel["close"]
    # `source` names which price series feeds the indicator. Defaulting to
    # close is right for the moving averages and wrong for a channel, where
    # the high and low are the point - so it is read rather than assumed.
    src_name = str(getattr(ind, "source", None)
                   or (ind.get("source") if isinstance(ind, dict) else None)
                   or "close")
    src = panel.get(src_name)
    if src is None:
        raise NotImplementedError(
            f"indicator {kind!r} wants source {src_name!r}, which is not in "
            f"the panel (have: {sorted(panel)})")

    if kind in ("highest", "lowest"):
        return _extreme(src, int(params.get("length", 20)), kind)
    if kind == "sma":
        return _sma(close, int(params.get("length", 20)))
    if kind == "ema":
        return _ema(close, int(params.get("length", 20)))
    if kind == "rsi":
        return _rsi(close, int(params.get("length", 14)))
    if kind == "macd":
        return _macd(close, int(params.get("fast", 12)),
                     int(params.get("slow", 26)),
                     int(params.get("signal", 9)),
                     str(params.get("component", "line")))
    if kind in ("bbands", "bollinger"):
        return _bb(close, int(params.get("length", 20)),
                   float(params.get("mult", params.get("stddev", 2.0))),
                   str(params.get("component", "upper")))
    if kind in ("donchian", "donchian_channel"):
        return _donchian(panel["high"], panel["low"],
                         int(params.get("length", 20)),
                         str(params.get("component", "upper")))
    raise NotImplementedError(f"indicator type {kind!r} is not translated")


def _side(operand: Any, inds: dict[str, pd.DataFrame],
          panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    kind = operand.kind if hasattr(operand, "kind") else operand["kind"]
    if kind == "indicator":
        ref = operand.ref if hasattr(operand, "ref") else operand["ref"]
        return inds[ref]
    if kind == "const":
        v = operand.value if hasattr(operand, "value") else operand["value"]
        close = panel["close"]
        return pd.DataFrame(float(v), index=close.index, columns=close.columns)
    if kind in ("price", "source", "series"):
        ref = (operand.ref if hasattr(operand, "ref")
               else operand.get("ref", "close"))
        return panel[str(ref)]
    raise NotImplementedError(f"operand kind {kind!r} is not translated")


def _apply_op(op: str, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """A comparison, as a boolean frame.

    The two cross operators need the PREVIOUS bar, so a cross is only true
    where both bars are known. NaN comparisons are False in pandas, which is
    the behaviour wanted here - an unknown indicator has not crossed anything.
    """
    if op == "cross_above":
        return (a > b) & (a.shift(1) <= b.shift(1))
    if op == "cross_below":
        return (a < b) & (a.shift(1) >= b.shift(1))
    if op in ("gt", ">"):
        return a > b
    if op in ("lt", "<"):
        return a < b
    if op in ("gte", ">="):
        return a >= b
    if op in ("lte", "<="):
        return a <= b
    raise NotImplementedError(f"operator {op!r} is not translated")


@dataclass(frozen=True, slots=True)
class PresetRule:
    """One preset, evaluated against the desk's own data.

    `unsupported` is set when the preset cannot be tested AS WRITTEN. It is a
    third state alongside "fires" and "stays silent", because a strategy the
    risk gate refuses is a different fact from one that found no setups, and
    conflating them is how an untested rule comes to look like a quiet one.
    """

    id: str
    name: str
    stop_pct: float
    take_pct: float
    entry_long: tuple
    indicators: tuple
    already_ours: bool = False
    unsupported: str | None = None

    @property
    def rr(self) -> float:
        """Reward-to-risk as declared. 2.0 means the desk's 1:2 target."""
        return self.take_pct / self.stop_pct if self.stop_pct else float("nan")

    @property
    def cost_r(self) -> float:
        """Round-trip cost as a share of risk, at the desk's 0.422% model.

        cost_R = round_trip% / stop%, so a tighter stop is proportionally MORE
        expensive - the quantity cancels and only stop width matters. This is
        why the 5% presets start 0.085R ahead of the 2.5% ones before either
        is right about anything.
        """
        return 0.422 / self.stop_pct if self.stop_pct else float("nan")

    def entries(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Long entry signals as a date x symbol boolean frame.

        Every declared condition must hold on the same bar; the upstream
        schema is a list and lists are ANDed.
        """
        if self.unsupported:
            raise NotImplementedError(
                f"{self.id} cannot be evaluated as written: {self.unsupported}")
        inds = {(i.id if hasattr(i, "id") else i["id"]):
                _evaluate_indicator(i, panel) for i in self.indicators}
        out: pd.DataFrame | None = None
        for cond in self.entry_long:
            left = cond.left if hasattr(cond, "left") else cond["left"]
            right = cond.right if hasattr(cond, "right") else cond["right"]
            op = cond.op if hasattr(cond, "op") else cond["op"]
            got = _apply_op(op, _side(left, inds, panel),
                            _side(right, inds, panel))
            out = got if out is None else (out & got)
        if out is None:
            raise NotImplementedError(f"{self.id} declares no long entry")
        return out.fillna(False)


def load_presets() -> dict[str, Any]:
    """Upstream's raw PRESETS dict, or {} when the clone is absent."""
    mod = _load()
    return dict(getattr(mod, "PRESETS", {})) if mod else {}


def preset_rules() -> dict[str, PresetRule]:
    """Every preset as a PresetRule, including the ones we cannot test.

    A preset that fails the 1:2 gate is RETURNED with `unsupported` set rather
    than dropped. Dropping it would make the catalogue look smaller than it is
    and hide the reason.
    """
    out: dict[str, PresetRule] = {}
    for pid, p in load_presets().items():
        spec = p.spec if hasattr(p, "spec") else p["spec"]
        risk = spec.risk if hasattr(spec, "risk") else spec["risk"]
        stop = float(risk.stop_pct if hasattr(risk, "stop_pct")
                     else risk["stop_pct"])
        take = float(risk.take_pct if hasattr(risk, "take_pct")
                     else risk["take_pct"])
        entry = tuple(spec.entry_long if hasattr(spec, "entry_long")
                      else spec["entry_long"])
        inds = tuple(spec.indicators if hasattr(spec, "indicators")
                     else spec["indicators"])

        unsupported = None
        if stop <= 0:
            unsupported = f"stop_pct is {stop}, which cannot define risk"
        elif take / stop < MIN_RR:
            unsupported = (
                f"declares 1:{take / stop:g} ({stop:g}% stop, {take:g}% "
                f"target), below the desk's 1:{MIN_RR:g} floor. Widening the "
                f"target to comply would measure a strategy nobody declared.")

        out[pid] = PresetRule(
            id=pid,
            name=str(p.name if hasattr(p, "name") else p["name"]),
            stop_pct=stop, take_pct=take, entry_long=entry, indicators=inds,
            already_ours=pid in ALREADY_OURS, unsupported=unsupported,
        )
    return out
