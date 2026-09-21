"""Turning a catalogued strategy into proposals the desk can read.

THE GAP THIS CLOSES. `desk.strategies.catalog` describes six strategies -
family, parameters, regime gates, maturity, known defects - and describes is
all it does. There is no entry logic in a `StrategySpec`, and `eligible()`
was reachable from exactly one place: the `/strategies/eligible/{regime}`
endpoint. Nothing turned a spec into a trade idea, so the desk has only ever
ranked on one thing, the Stage 2 factor funnel, and the other five
strategies in the catalog were documentation.

WHY THIS READS STAGE 1 RATHER THAN `strategies/india/strategy_lib.py`. The
shared library works on one symbol's DataFrame at a time - `rsi(close)`,
`supertrend(df)` - which is right for studying a single name and wrong for
a pre-open scan: ~1,500 survivors would be ~1,500 trips through the Python
interpreter for arithmetic Stage 1 already does on a date x symbol matrix
in one pass. So an adapter consumes the feature table that has already been
computed, and any window it needs is added to Stage 1 beside the others.
The library stays the reference; this is the scan path.

WHAT AN ADAPTER MAY NOT DO. It proposes. It does not size, it does not
decide, and it cannot relax anything: every proposal still has to pass
`size_position`, which owns every gate and does not know which strategy
produced the setup. An adapter also does not decide whether it is allowed
to speak - `eligible()` does, from the regime, and a strategy that names
today's regime as hostile is silenced rather than down-weighted.

MATURITY IS THE OTHER HALF OF THAT. Nothing here makes a strategy
tradeable. `StrategySpec.trusted` is False below WALK_FORWARD and all six
entries are AUDITED today, so these proposals are for display and for
measurement until a walk-forward run says otherwise. `propose()` reports
the maturity on every result it returns so a caller cannot forget.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from desk.contracts.enums import Regime, Stance
from desk.contracts.envelope import AnalysisResult, PriceZone, Target
from desk.scanner.stage1 import Stage1Result
from desk.strategies.catalog import BY_KEY, StrategySpec, eligible

__all__ = ["propose", "propose_all", "ADAPTERS", "donchian_breakout"]


def _num(row: pd.Series, name: str) -> float | None:
    """One feature as a real number, or None.

    None and NaN are the same answer here - "this could not be computed" -
    and every caller below must branch on it rather than let NaN through.
    NaN survives every comparison as False, so an unguarded one does not
    raise; it quietly makes a rule stop firing, which looks exactly like a
    market with no setups in it.
    """
    v = row.get(name)
    if v is None or pd.isna(v):
        return None
    v = float(v)
    return v if v == v else None


def donchian_breakout(symbol: str, row: pd.Series, spec: StrategySpec,
                      as_of: date) -> AnalysisResult | None:
    """Close above the prior 20-session high; exit on the prior 10-day low.

    The catalog calls this "cheapest of the six to validate, because the
    rule has almost no parameters to overfit", which is why it is the first
    adapter. Its thesis is "new highs beget new highs; cut the ones that
    fail immediately", and its stated weakness is a low win rate by
    construction - it needs a large trade count before the number means
    anything, so it is the one strategy where a short backtest is
    especially misleading.

    Returns None rather than a neutral result when the rule does not fire.
    A HOLD from a breakout scanner would be noise on every non-breakout
    name in the universe - roughly 1,500 of them - and `Stance.HOLD` is a
    tradeable neutral, not an absence of opinion.

    THE STOP IS THE STRATEGY'S OWN, not Stage 4's structural one: this spec
    carries `atr_stop_mult`, and honouring the catalogued parameter is the
    point of running the catalogued strategy. Stage 4 will still refuse the
    setup if that stop is too wide, which is the correct division - the
    strategy says where it is wrong, the risk engine says whether that is
    affordable.
    """
    close = _num(row, "close")
    channel = _num(row, "prior_high_20")
    atr_pct = _num(row, "atr_pct")
    if close is None or channel is None or atr_pct is None:
        return None
    if close <= 0 or channel <= 0 or atr_pct <= 0:
        return None

    if close <= channel:
        return None                      # no breakout, nothing to say

    atr = close * atr_pct / 100.0
    stop = close - float(spec.params["atr_stop_mult"]) * atr
    if stop <= 0:
        # An ATR wide enough to put the stop under zero is a broken
        # volatility estimate for this name, not a merely risky trade.
        return None

    risk = close - stop
    exit_level = _num(row, "prior_low_10")

    invalidations = [
        f"Close back inside the channel, under {channel:.2f} - a breakout "
        f"that does not hold is the failure mode this rule exists to cut.",
    ]
    if exit_level is not None:
        invalidations.append(
            f"Close below the {int(spec.params['exit_lookback'])}-session "
            f"low at {exit_level:.2f} is the strategy's own exit, and it "
            f"can trigger before the {stop:.2f} stop is reached.")

    return AnalysisResult(
        agent=spec.key,
        symbol=symbol,
        as_of=as_of,
        stance=Stance.BUY,
        # NOT calibrated, and deliberately flat. Nothing has measured
        # whether a breakout 3% above the channel outperforms one 0.5%
        # above it, so scaling confidence by the size of the break would
        # invent a gradient from nothing. One number until a walk-forward
        # run earns the right to vary it.
        confidence=0.5,
        horizon=spec.horizon,
        setup_type="donchian channel breakout",
        entry=close,
        # The zone runs from the channel to today's close: anywhere in
        # here the breakout is intact. Below the channel it is not a
        # breakout at all, which is why the zone stops there rather than
        # extending some invented percentage below the level.
        entry_zone=PriceZone(low=channel, high=close),
        trigger=(f"Only on a close above {channel:.2f}, the highest high of "
                 f"the previous {int(spec.params['entry_lookback'])} "
                 f"sessions. A touch is not a break."),
        stop=stop,
        targets=[Target(price=close + 2.0 * risk, fraction=1.0,
                        rationale=f"2R against the {risk:.2f} risk this "
                                  f"stop accepts.")],
        expected_holding_days=None,
        invalidations=invalidations,
        narrative=(f"{symbol} closed at {close:.2f}, above its "
                   f"{int(spec.params['entry_lookback'])}-session high of "
                   f"{channel:.2f}. {spec.thesis} Stop is "
                   f"{spec.params['atr_stop_mult']}x ATR ({atr_pct:.1f}% of "
                   f"price) below the close."),
        metadata={
            "strategy": spec.key,
            "family": spec.family.value,
            # Carried on every single result, because it is the field that
            # decides whether this may be sized, and a caller that has to
            # look it up somewhere else is a caller that will forget to.
            "maturity": spec.maturity.value,
            "trusted": spec.trusted,
            "channel_high": channel,
            "break_pct": 100.0 * (close - channel) / channel,
            "atr_pct": atr_pct,
        },
    )


#: Strategy key -> the function that turns one feature row into a proposal.
#: A catalogued strategy WITHOUT an entry here is not broken, it is
#: unimplemented, and `propose_all` reports it as such rather than skipping
#: it silently. Five of the six are in that state today: the shared library
#: has supertrend/adx/rsi/bollinger but no Donchian channel at all, and
#: pairs_trading and ml_classifier carry open look-ahead defects (BUG-03,
#: BUG-04) that must be fixed before their output could mean anything.
ADAPTERS = {
    "donchian_breakout": donchian_breakout,
}


def propose(key: str, stage1: Stage1Result, *,
            limit: int | None = None) -> list[AnalysisResult]:
    """Run one strategy across the whole Stage 1 table.

    Raises KeyError for an unknown key and NotImplementedError for a
    catalogued strategy with no adapter - both are programming errors, and
    returning an empty list for either would be indistinguishable from "the
    market offered nothing today", which is the failure this codebase keeps
    finding in other forms.
    """
    spec = BY_KEY.get(key)
    if spec is None:
        raise KeyError(f"{key!r} is not in the strategy catalog")
    fn = ADAPTERS.get(key)
    if fn is None:
        raise NotImplementedError(
            f"{key!r} is catalogued but has no adapter - implemented: "
            f"{sorted(ADAPTERS)}")

    out: list[AnalysisResult] = []
    for symbol, row in stage1.features.iterrows():
        r = fn(str(symbol), row, spec, stage1.as_of)
        # A result carrying validation errors is DROPPED, not returned with
        # a warning. AnalysisResult appends to `errors` instead of raising,
        # so an incoherent setup (stop above entry, a NaN that reached a
        # level) is a well-formed object that reads as a real proposal.
        if r is not None and not r.errors:
            out.append(r)
    out.sort(key=lambda r: r.metadata.get("break_pct", 0.0), reverse=True)
    return out[:limit] if limit else out


def propose_all(stage1: Stage1Result, *, regime: Regime,
                trusted_only: bool = False,
                limit_per_strategy: int | None = None) -> dict:
    """Every eligible strategy's proposals for one day, plus who stayed quiet.

    `trusted_only` defaults to FALSE here and True in `eligible()`, and the
    difference is deliberate. All six strategies are AUDITED, so the trusted
    filter returns an empty list and this endpoint would report "no
    strategies eligible" every day - true, useless, and indistinguishable
    from a bug. Showing untrusted proposals labelled with their maturity is
    how a strategy earns trust in view. NOTHING here may be sized: that is
    `StrategySpec.trusted`'s job at the point of sizing, not this function's.

    WHO STAYED SILENT IS RETURNED, not omitted. "bollinger_rsi proposed
    nothing" and "bollinger_rsi was silenced because today is trending_up,
    which it declares hostile" are different facts, and a plan that cannot
    tell them apart is the same class of mistake as a skipped check that
    reads like a passed one.
    """
    allowed = {s.key for s in eligible(regime, trusted_only=trusted_only)}
    proposals: dict[str, list[AnalysisResult]] = {}
    silenced: dict[str, str] = {}
    unimplemented: dict[str, str] = {}

    for key, spec in BY_KEY.items():
        if key not in allowed:
            why = (f"{regime.value} is a hostile regime for this strategy"
                   if regime in spec.hostile_regimes
                   else f"maturity is {spec.maturity.value}, below the "
                        f"trusted threshold")
            silenced[key] = why
            continue
        if key not in ADAPTERS:
            unimplemented[key] = (
                f"catalogued at {spec.maturity.value} but no adapter is "
                f"written, so it could not be asked")
            continue
        proposals[key] = propose(key, stage1, limit=limit_per_strategy)

    return {
        "as_of": stage1.as_of,
        "regime": regime.value,
        "trusted_only": trusted_only,
        "proposals": proposals,
        "silenced": silenced,
        "unimplemented": unimplemented,
        "tradeable_now": sorted(
            k for k in proposals if BY_KEY[k].trusted),
    }
