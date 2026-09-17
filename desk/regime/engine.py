"""Measuring the regime from the market itself.

`RegimeState` has existed as a shape since the regime vector was designed,
and its docstring said the engine "needs India VIX, Nifty breadth and a
sectoral-index feed" - so nothing ever computed it, and every caller passed
a hand-chosen `Regime` enum. A hand-chosen regime is worse than none: it
silences factors and gates strategies on an opinion, and the plan prints it
as though it were a measurement.

None of those three feeds turned out to be necessary. THE BHAVCOPY IS A
CROSS-SECTION OF THE WHOLE MARKET, and breadth is a cross-sectional fact.
What percentage of names are above their 50-day average, how many advanced
against how many declined, how wide the spread between the best and worst
decile - all of that is already on disk, for every session, with no new
provider.

WHAT IS MEASURED HERE, AND FROM WHAT
-------------------------------------
  breadth        % of the liquid universe above its own 50-day SMA
  trend          an EQUAL-WEIGHTED market composite and its slope
  volatility     realised 20-day vol of that composite, PERCENTILE-RANKED
                 against its own trailing history
  risk_appetite  breadth and drawdown together

Every dimension writes into `RegimeState.sources` saying what it came from
and over what window, because a regime is an input to every downstream
decision and therefore has to be arguable.

EQUAL-WEIGHTED, NOT CAP-WEIGHTED, and it matters
-------------------------------------------------
We have no market caps, so a cap-weighted composite is not available
anyway - but equal weight is also the better measure for this purpose. A
cap-weighted index can rise on four heavyweights while most of the market
falls, which is exactly the narrow-breadth regime the desk most needs to
detect. An equal-weighted composite cannot hide that, because every name
votes once. The composite is therefore a BREADTH-SENSITIVE market proxy,
not an attempt to track the Nifty, and it should not be compared to one.

PERCENTILES, NOT THRESHOLDS, FOR VOLATILITY
--------------------------------------------
"Vol above 20%" means different things in different decades and different
markets. Ranking today's realised vol against its own trailing history
answers the question actually being asked - is this market unusually
volatile FOR ITSELF - and it is the same reasoning Stage 1 already uses for
`atr_pct_rank`.

UNKNOWN IS A REAL ANSWER
------------------------
Every dimension needs a minimum history and returns UNKNOWN below it,
rather than computing a 50-day average over 12 days and labelling it. The
plan then reports `is_measured = False` and says which dimensions have
nothing behind them - the same convention as `checks_skipped`.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from desk.contracts.enums import Breadth, RiskAppetite, Trend, VolState
from desk.regime.state import RegimeState

__all__ = ["MIN_SESSIONS", "compute_regime"]

#: Sessions needed before any dimension is reported. The 50-day SMA is the
#: longest window used, and a 50-day average over 40 sessions is not a
#: 50-day average - it is a shorter one wearing the wrong name.
MIN_SESSIONS = 60

#: And for the volatility PERCENTILE, which needs enough history to rank
#: against. Below this, volatility stays UNKNOWN even when the composite
#: itself is computable: a percentile over 30 observations is noise.
MIN_VOL_HISTORY = 120

# Breadth cut-offs, as a percentage of the universe above its 50-day SMA.
# Deliberately wide in the middle: most days are unremarkable, and a regime
# that flips label on a 2% move in breadth would gate strategies on noise.
_BROAD_ABOVE = 60.0
_NARROW_BELOW = 40.0

# Trend, from the composite's 20-day change measured in its own volatility.
# Expressed in standard deviations rather than percent so the same numbers
# work in a calm market and a violent one.
_STRONG = 1.5
_MILD = 0.4

#: Below this daily standard deviation the sigma normalisation is unusable
#: and _trend falls back to a raw percentage move. A real market composite
#: of 1,500 names never gets close; a synthetic one does.
_MIN_SIGMA = 1e-6

#: The 20-day move that counts as "one unit" when sigma is unusable.
_FALLBACK_MOVE = 0.05

_VOL_ELEVATED_PCTILE = 80.0
_VOL_EXTREME_PCTILE = 95.0
_VOL_LOW_PCTILE = 20.0


def compute_regime(history, *, as_of: date, sectors=None,
                   events=None) -> RegimeState:
    """Measure the regime from one History of the liquid universe.

    `history` is a desk.store.bars.History - long-format bars for many
    symbols. It should already be the LIQUID slice (Stage 0's survivors):
    breadth computed over every listed shell company measures the wrong
    market, and those names do not move on the same information.

    Returns a RegimeState with UNKNOWN in any dimension that could not be
    measured, and `sources` naming what each one came from.
    """
    state_kwargs: dict = {"as_of": as_of, "sources": {}}
    if events is not None:
        state_kwargs["events"] = events

    closes = _close_matrix(history)
    if closes is None or closes.shape[0] < MIN_SESSIONS:
        have = 0 if closes is None else closes.shape[0]
        state_kwargs["sources"]["all"] = (
            f"not measured: {have} sessions on file, {MIN_SESSIONS} needed "
            f"for a 50-day average")
        return RegimeState(**state_kwargs)

    sessions, names = closes.shape
    breadth_pct, breadth = _breadth(closes)
    state_kwargs["breadth"] = breadth
    state_kwargs["sources"]["breadth"] = (
        f"{breadth_pct:.0f}% of {names} liquid names above their own 50-day "
        f"SMA (broad >{_BROAD_ABOVE:.0f}%, narrow <{_NARROW_BELOW:.0f}%)")

    composite = _composite(closes)
    trend, trend_note = _trend(composite)
    state_kwargs["trend"] = trend
    state_kwargs["sources"]["trend"] = trend_note

    vol, vol_note = _volatility(composite)
    state_kwargs["volatility"] = vol
    state_kwargs["sources"]["volatility"] = vol_note

    appetite, appetite_note = _risk_appetite(composite, breadth_pct)
    state_kwargs["risk_appetite"] = appetite
    state_kwargs["sources"]["risk_appetite"] = appetite_note

    if sectors is not None:
        lead, lag, note = _sectors(closes, sectors)
        state_kwargs["leading_sectors"] = lead
        state_kwargs["lagging_sectors"] = lag
        state_kwargs["sources"]["sectors"] = note

    return RegimeState(**state_kwargs)


# ---------------------------------------------------------------------------

def _close_matrix(history) -> pd.DataFrame | None:
    """(date x symbol) closes, with symbols that never traded dropped."""
    if history is None or getattr(history, "frame", None) is None:
        return None
    if history.frame.empty:
        return None
    try:
        wide = history.wide_many(["close"])["close"]
    except (KeyError, ValueError):
        return None
    if wide.empty:
        return None
    # A symbol with no closes at all contributes nothing and would count
    # against breadth as a silent "not above its average".
    return wide.dropna(axis=1, how="all").sort_index()


def _breadth(closes: pd.DataFrame) -> tuple[float, Breadth]:
    sma50 = closes.rolling(50, min_periods=50).mean()
    last_close = closes.iloc[-1]
    last_sma = sma50.iloc[-1]
    comparable = last_close.notna() & last_sma.notna()
    if not comparable.any():
        return 0.0, Breadth.UNKNOWN

    above = (last_close[comparable] > last_sma[comparable])
    pct = 100.0 * float(above.sum()) / float(comparable.sum())
    if pct > _BROAD_ABOVE:
        return pct, Breadth.BROAD
    if pct < _NARROW_BELOW:
        return pct, Breadth.NARROW
    return pct, Breadth.NEUTRAL


def _composite(closes: pd.DataFrame) -> pd.Series:
    """An equal-weighted market index built from daily returns.

    Averaging RETURNS rather than prices, then compounding. Averaging
    prices would let a 40,000-rupee stock outvote a 40-rupee one by three
    orders of magnitude, which is a cap-weighted index by accident and the
    opposite of what this is for.
    """
    # fill_method=None, explicitly. pandas' default pads NaNs forward,
    # which turns "this name did not trade that day" into a manufactured
    # 0.0% return - and across 1,548 names those fake zeros pull the mean
    # toward flat, damping exactly the moves the composite exists to show.
    rets = closes.pct_change(fill_method=None)
    # Refuse absurd single-day moves before they reach the mean: an
    # unadjusted split is a -50% print that is not a market event, and one
    # of them moves an equal-weighted composite of 1,500 names visibly.
    rets = rets.mask(rets.abs() > 0.5)
    mean_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + mean_ret).cumprod()


def _trend(composite: pd.Series) -> tuple[Trend, str]:
    if len(composite) < 21:
        return Trend.UNKNOWN, "not measured: fewer than 21 composite points"

    change = float(composite.iloc[-1] / composite.iloc[-21] - 1.0)
    daily = composite.pct_change(fill_method=None).dropna()
    sigma = float(daily.tail(60).std())

    if np.isfinite(sigma) and sigma >= _MIN_SIGMA:
        # The 20-day move expressed in 20-day standard deviations, so the
        # same cut-offs mean the same thing in a calm market and a violent
        # one.
        z = change / (sigma * np.sqrt(20))
        note = (f"equal-weighted composite moved {100 * change:+.2f}% over 20 "
                f"sessions = {z:+.2f} sigma (strong >|{_STRONG}|, "
                f"mild >|{_MILD}|)")
    else:
        # A composite with essentially no variance would divide by ~zero
        # and produce a meaningless z - or, worse, an enormous one that
        # reads as a violent trend. Falling back to the raw move and SAYING
        # so beats returning UNKNOWN for a market that is plainly moving.
        # Real markets never sit here; a synthetic or single-name universe
        # does, and so would a genuinely frozen one.
        z = change / _FALLBACK_MOVE
        note = (f"equal-weighted composite moved {100 * change:+.2f}% over 20 "
                f"sessions; volatility too low to normalise against "
                f"(sigma {sigma:.2e}), so measured against a fixed "
                f"{100 * _FALLBACK_MOVE:.0f}% move instead")
    if z >= _STRONG:
        return Trend.STRONG_UP, note
    if z >= _MILD:
        return Trend.UP, note
    if z <= -_STRONG:
        return Trend.STRONG_DOWN, note
    if z <= -_MILD:
        return Trend.DOWN, note
    return Trend.FLAT, note


def _volatility(composite: pd.Series) -> tuple[VolState, str]:
    daily = composite.pct_change(fill_method=None).dropna()
    if len(daily) < MIN_VOL_HISTORY:
        return VolState.UNKNOWN, (
            f"not measured: {len(daily)} return observations, "
            f"{MIN_VOL_HISTORY} needed to rank a percentile against")

    rolling = daily.rolling(20, min_periods=20).std().dropna()
    if rolling.empty:
        return VolState.UNKNOWN, "not measured: no 20-day vol available"

    today = float(rolling.iloc[-1])
    pctile = 100.0 * float((rolling <= today).sum()) / float(len(rolling))
    annual = today * np.sqrt(252) * 100.0
    note = (f"20-day realised vol {annual:.1f}% annualised, at the "
            f"{pctile:.0f}th percentile of its own {len(rolling)}-session "
            f"history")
    if pctile >= _VOL_EXTREME_PCTILE:
        return VolState.EXTREME, note
    if pctile >= _VOL_ELEVATED_PCTILE:
        return VolState.ELEVATED, note
    if pctile <= _VOL_LOW_PCTILE:
        return VolState.LOW, note
    return VolState.NORMAL, note


def _risk_appetite(composite: pd.Series, breadth_pct: float
                   ) -> tuple[RiskAppetite, str]:
    """Breadth and drawdown together, because either alone misleads.

    A market can sit near its highs while breadth collapses - the late
    stage of a narrow rally, which is not risk-on however the index looks.
    It can also be deep in a drawdown with breadth recovering, which is
    where risk appetite actually returns first. Requiring the two to agree
    means this dimension says NEUTRAL when they disagree, which is the
    honest answer rather than a casting vote.
    """
    peak = float(composite.max())
    now = float(composite.iloc[-1])
    drawdown = 100.0 * (now / peak - 1.0) if peak else 0.0
    note = (f"composite {drawdown:+.1f}% from its high over "
            f"{len(composite)} sessions, breadth {breadth_pct:.0f}%")

    if drawdown <= -10.0 and breadth_pct < _NARROW_BELOW:
        return RiskAppetite.RISK_OFF, note + " - both weak"
    if drawdown >= -3.0 and breadth_pct > _BROAD_ABOVE:
        return RiskAppetite.RISK_ON, note + " - both strong"
    return RiskAppetite.NEUTRAL, note + " - mixed"


def _sectors(closes: pd.DataFrame, sectors, top: int = 3
             ) -> tuple[list[str], list[str], str]:
    """Best and worst industries by median 20-session return."""
    if len(closes) < 21:
        return [], [], "not measured: fewer than 21 sessions"

    change = (closes.iloc[-1] / closes.iloc[-21] - 1.0).dropna()
    by_sector: dict[str, list[float]] = {}
    for symbol, value in change.items():
        name = sectors.sector_for(str(symbol))
        if name:
            by_sector.setdefault(name, []).append(float(value))

    classified = sum(len(v) for v in by_sector.values())
    if not by_sector:
        return [], [], (f"not measured: none of {len(change)} names are in "
                        f"the sector map")

    # Two names is not a sector. Reporting one as "leading" would put a
    # single stock's move forward as an industry finding.
    ranked = sorted(((s, float(np.median(v))) for s, v in by_sector.items()
                     if len(v) >= 3), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return [], [], "not measured: no industry has 3+ classified names"

    note = (f"{classified} of {len(change)} names classified across "
            f"{len(ranked)} industries, median 20-session return")
    return ([s for s, _ in ranked[:top]],
            [s for s, _ in ranked[-top:]][::-1], note)
