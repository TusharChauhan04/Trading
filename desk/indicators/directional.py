"""ADX and the directional indicators, across the whole universe at once.

WHY THIS IS WRITTEN RATHER THAN IMPORTED
----------------------------------------
OpenTerminalUI ships `backend/services/indicators.py` with 13 indicators the
desk lacks, ADX among them, and the salvage map marked it "take". Reading it
before importing it changed the decision twice over, and both reasons are
recorded here because they are the kind of thing that is invisible once the
numbers look plausible:

1. IT BACKFILLS THE WARMUP PERIOD. Its `_fillna_local` is
   `series.ffill().bfill()`, and `bfill` fills EARLIER NaNs from LATER
   values. On a 14-period ADX, bars 0-13 come back carrying the value
   computed at bar 14 - a number that did not exist yet on those bars. Nine
   of its thirteen indicators do this. Reading only the final bar would
   dodge it, but a backtest that slices history and walks forward would not,
   and "safe as long as nobody uses it the obvious way" is not a property
   worth importing. Here the warmup is NaN, which is what "not enough data
   yet" actually looks like.

2. ITS ADX IS NOT WILDER'S ADX. It smooths DM and TR with
   `.rolling(period).mean()`, a simple moving average. Wilder's ADX - the one
   TradingView, Chartink and every retail screen draw - smooths with
   alpha = 1/period. The two put crossings on different bars, which is the
   whole point of this package's existing note about SMA-smoothing an
   ADX-style indicator. The desk already had `wilder()`, so the correct
   version was cheaper to write than the incorrect one was to adapt.

So upstream's file served as a specification and a warning. That makes it the
second of two "take" decisions about this repository that close reading turned
into "reference, don't import" - the first being lm_studio_client.py.

WHY MATRICES. Stage 1 holds a date x symbol frame per field and computes
every feature across all ~1,500 names in one pass, so a per-symbol function
would turn one vectorised operation into a 1,500-iteration Python loop.
Everything here takes and returns wide frames for that reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_ADX_PERIOD", "adx_wide", "directional_movement",
           "true_range_wide"]

#: Wilder's own choice and the default on every retail chart. Changing it
#: changes which bars cross, so it is a parameter rather than a constant -
#: but a different value is a different indicator, not a tuning.
DEFAULT_ADX_PERIOD = 14


def true_range_wide(high: pd.DataFrame, low: pd.DataFrame,
                    close: pd.DataFrame) -> pd.DataFrame:
    """True range across every symbol at once.

    np.fmax rather than pd.concat(...).max(axis=1): concat on a wide frame
    builds a 3x-wide intermediate and then reduces it, allocating three
    copies of the universe.

    The first bar has no previous close, so the two gap terms are NaN there
    and the standard convention falls back to high-low. `np.fmax` gives that
    for free, because it prefers a defined operand over NaN.
    """
    prev = close.shift(1)
    return np.fmax(np.fmax(high - low, (high - prev).abs()),
                   (low - prev).abs())


def directional_movement(high: pd.DataFrame, low: pd.DataFrame
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wilder's +DM and -DM.

    THE RULE THAT IS EASY TO GET WRONG: on any given bar at most one of the
    two can be non-zero. An outside bar - higher high AND lower low - moved
    in both directions, and Wilder's answer is that it therefore signals
    neither, so whichever move was smaller is zeroed. Implementations that
    simply take `max(up, 0)` and `max(down, 0)` independently report both
    directions at once on exactly the bars where direction is least clear.
    """
    up = high.diff()
    down = -low.diff()

    # TWO DIFFERENT ZEROES, AND THEY MUST NOT BE CONFLATED.
    #   "the bar moved, but not in this direction"  -> 0.0, a real reading
    #   "there is no previous bar to compare to"    -> NaN, no reading
    # `.where(cond, 0.0)` alone collapses both into 0.0, because a NaN
    # comparison is False and therefore takes the 0.0 branch. That would put
    # a manufactured zero on the first bar of every symbol and feed it into
    # the smoothing below as though it were measured.
    known = up.notna() & down.notna()
    plus = up.where((up > down) & (up > 0), 0.0).where(known)
    minus = down.where((down > up) & (down > 0), 0.0).where(known)
    return plus, minus


def adx_wide(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, *,
             period: int = DEFAULT_ADX_PERIOD) -> dict[str, pd.DataFrame]:
    """ADX, +DI and -DI for every symbol at once, Wilder-smoothed.

    Returns `{"adx": ..., "plus_di": ..., "minus_di": ...}`, each a date x
    symbol frame aligned to the inputs.

    NO BACKFILL AND NO FORWARD FILL. The first `period` rows of a Wilder
    series are warming up and are left as they come out; a caller that needs
    "is this reading trustworthy" should check bar count, not look for a
    non-NaN value. Inventing early values is how a backtest comes to trade on
    a number that did not exist on the day it claims to.

    ADX IS DIRECTIONLESS ON PURPOSE. It measures how strongly price is
    trending, not which way - a hard down-trend and a hard up-trend both
    print a high ADX. The sign lives in +DI vs -DI, which is why all three
    are returned together and why using ADX alone to pick a side is a
    category error.

    IT IS ALSO DEGENERATE ON NEAR-FLAT INSTRUMENTS, AND THAT IS NOT A BUG
    HERE - it is a trap at the point of use. ADX is a ratio, so it does not
    care how large the move was, only how consistent. A money-market ETF that
    grinds up a paisa a day with almost no range is therefore a PERFECT
    trend, and pegs near 100.
    Measured on the real universe, 2026-09-29, 995 names through Stage 0:

        LIQUIDCASE.NS   adx 100.0   atr 0.029%
        LIQUIDETF.NS    adx  80.1   atr 0.005%      universe median atr 3.36%

    Only 12 of 995 names sat under a 1% ATR floor, but those 12 occupied the
    entire top of the ADX ranking - the one place contamination does maximum
    damage. Above the floor the ranking is real equities at adx 56-72 with atr
    4-7%, split 10 up and 10 down in the top 20.

    SO ANY GATE ON ADX MUST ALSO FLOOR VOLATILITY. `atr_pct` already exists in
    Stage 1 for exactly this. A screen written on adx_14 alone does not select
    trending stocks; it selects liquid ETFs while looking like it selected
    trends. This is asserted by a tripwire in test_directional.py so that the
    behaviour cannot be "fixed" into silence.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    tr = true_range_wide(high, low, close)
    plus_dm, minus_dm = directional_movement(high, low)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_sm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_sm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # A zero ATR means a genuinely flat bar - typically a circuit-locked or
    # untraded name - and dividing by it would yield inf, which survives
    # pd.isna() and sorts to the top of any percentile rank. NaN is the
    # honest answer: direction is undefined when nothing moved.
    safe_atr = atr.where(atr > 0)
    plus_di = 100.0 * plus_sm / safe_atr
    minus_di = 100.0 * minus_sm / safe_atr

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum > 0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    # ewm propagates a leading NaN but then treats later NaNs as gaps rather
    # than resetting, so inf can only arrive via the divisions above. Swept
    # once at the boundary anyway - inf is the one bad value that looks
    # finite to isna() and still ranks first.
    clean = lambda f: f.replace([np.inf, -np.inf], np.nan)   # noqa: E731
    return {"adx": clean(adx), "plus_di": clean(plus_di),
            "minus_di": clean(minus_di)}
