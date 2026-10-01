"""Slippage as a function of ORDER SIZE, instead of one flat guess.

WHY THIS MATTERS MORE THAN IT SOUNDS
------------------------------------
`costs.py` charges a flat 15bps a side and says so plainly: "the 15bps default
is a guess, not a measurement, and it is labelled as one." That guess is 0.30%
of the 0.422% round trip - 71% of the total - and it decides the sign of every
marginal result this desk has produced. The catalogued preset measurement came
in at 0 of 10 positive, and the break-even slippage for the three best was
7-9bps a side. The whole conclusion rests on a number nobody checked.

A flat number is also wrong in a specific, knowable direction: slippage depends
on how much of the day's volume you are trying to take. An institution moving
2% of a stock's turnover and a retail account moving 0.002% of it do not get the
same fill, and charging them the same bps flatters the institution while
punishing the retail account this desk is actually built for.

WHAT IS BORROWED AND WHAT IS NOT
--------------------------------
The functional form comes from OpenTerminalUI's `core/execution_model.py`:

    slippage_bps = base + coefficient * sqrt(participation)       impact curve
    slippage_bps = base + coefficient * participation             linear

The square-root form is the standard market-impact law (Almgren, Chriss et al.)
and is the right shape - impact grows sublinearly, so doubling the order less
than doubles the cost. THE COEFFICIENTS ARE THEIRS AND ARE NOT VALIDATED ON
NSE. 35bps at full participation is a plausible number from somebody's
equities desk; it is not an Indian measurement, and this module does not
pretend otherwise. What the model gives us is a defensible way to say "impact
at retail size is far below 15bps", which is a bound, not a calibration.

WHAT THIS DELIBERATELY DOES NOT MODEL, because pretending otherwise is how a
cost model becomes optimistic:

  THE BID-ASK SPREAD. Crossing it costs roughly half the spread per side and is
  the FLOOR on slippage regardless of order size. Bhavcopy carries no quotes,
  so it cannot be measured from the data this desk holds. `base_bps` exists to
  carry it and defaults to a deliberately non-zero value.

  TIMING AND ADVERSE SELECTION. A breakout entry is bought into strength, which
  is systematically the worst moment. Daily bars cannot see it.

  THE DAY'S OWN VOLUME IS NOT KNOWN AT ENTRY. Using it is mild look-ahead; see
  `participation` for why the previous session's turnover is used instead.

So this REPLACES the impact half of a guess with a model, and leaves the spread
half as an explicit parameter. It should narrow the uncertainty on marginal
results, not eliminate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["SlippageModel", "participation", "retail_bound"]


@dataclass(frozen=True, slots=True)
class SlippageModel:
    """Per-side slippage in basis points, as a function of participation.

    `base_bps` is the size-independent floor - spread crossing, tick rounding,
    the fact that a market order never gets the midpoint. It is NOT zero by
    default: a model that charges nothing for a one-share order would make
    every strategy look profitable at small size, which is exactly the
    direction a cost model must not err in.
    """

    base_bps: float = 2.0
    """Half-spread plus tick effects, per side. For NSE names clearing Stage
    0's turnover floor, quoted spreads are typically 1-5bps, so 2bps a side is
    the middle of that and still a judgement rather than a measurement -
    bhavcopy has no quotes."""

    impact_coefficient_bps: float = 35.0
    """Slippage at 100% participation, from OpenTerminalUI's defaults. THEIR
    number, not an NSE measurement. It only bites at institutional size: at
    1% participation the sqrt form charges 3.5bps, at 0.01% it charges
    0.35bps."""

    form: str = "sqrt"
    """`sqrt` for the standard sublinear impact law, `linear` for the harsher
    proportional one. `linear` is kept because it is strictly more pessimistic
    and a marginal result should be checked against the pessimistic form."""

    atr_multiplier: float = 0.0
    """Slippage as a fraction of the instrument's own daily ATR, per side.
    DEFAULT ZERO, and that is a declaration of ignorance rather than a claim
    that the term is absent.

    The idea is borrowed from OpenTerminalUI's `execution_sim/simulator.py`,
    which carries an `atr_slippage_mult`, and it is a real effect our
    participation model misses entirely: a stock that moves 7% a day has a
    wider quoted spread and worse fills than one that moves 1.5%, at ANY order
    size. Participation says nothing about it.

    It defaults to zero because the right value for NSE has not been measured
    and a guessed default would silently re-price every result in the project.
    Set it to run a sensitivity: at 0.02, a 3.5% ATR name costs 7bps a side.

    MEASURED AND WORTH KNOWING: donchian's breakout signals fire on names with
    a 3.53% median ATR against the universe's 3.61% - ratio 0.978. So the
    wide-stop result is NOT quietly selecting volatile names, which was the
    obvious way this term could have invalidated it."""

    min_bps: float = 0.0
    max_bps: float = 500.0

    def __post_init__(self) -> None:
        if self.form not in ("sqrt", "linear"):
            raise ValueError(f"form must be sqrt or linear, got {self.form!r}")
        if self.base_bps < 0 or self.impact_coefficient_bps < 0 \
                or self.atr_multiplier < 0:
            raise ValueError("slippage coefficients cannot be negative")

    def bps(self, participation_rate: float, atr_pct: float = 0.0) -> float:
        """Per-side slippage for one order at this participation and volatility.

        `participation_rate` is order value / the session's traded value, as a
        FRACTION (0.01 = taking 1% of the day's turnover). `atr_pct` is the
        instrument's ATR as a PERCENT of price, and only bites when
        `atr_multiplier` is set.
        """
        # The volatility term does not depend on order size, so it applies even
        # where participation is unknown.
        vol_bps = 0.0
        if self.atr_multiplier > 0 and math.isfinite(atr_pct) and atr_pct > 0:
            # atr_pct is a percent; 1% = 100bps.
            vol_bps = self.atr_multiplier * atr_pct * 100.0

        p = float(participation_rate)
        if not math.isfinite(p) or p <= 0:
            # No order, or an unmeasurable session. The base cost still applies
            # - returning 0 here would let a symbol with missing volume data
            # look free to trade, which is the cheapest possible wrong answer.
            return min(self.max_bps,
                       max(self.min_bps, self.base_bps + vol_bps))
        p = min(p, 1.0)
        shaped = math.sqrt(p) if self.form == "sqrt" else p
        raw = self.base_bps + self.impact_coefficient_bps * shaped + vol_bps
        return min(self.max_bps, max(self.min_bps, raw))

    def round_trip_pct(self, participation_rate: float,
                       atr_pct: float = 0.0) -> float:
        """Both sides, as a PERCENT of notional - the unit costs.py uses."""
        return 2.0 * self.bps(participation_rate, atr_pct) / 100.0

    @staticmethod
    def breakeven_atr_multiplier(gross_r: float, stop_pct: float,
                                 atr_pct: float,
                                 statutory_pct: float = 0.1222) -> float:
        """The ATR multiplier at which a result reaches exactly zero.

        Answers the sensitivity question directly instead of leaving a
        parameter dangling: how bad would volatility-driven slippage have to be
        before this edge disappears? A result that only survives an
        implausibly small multiplier is not robust.

        Returns NaN when the gross edge does not even cover the statutory
        charges, since no slippage assumption can rescue that.
        """
        budget = gross_r * stop_pct - statutory_pct
        if budget <= 0 or atr_pct <= 0:
            return float("nan")
        return budget / (2.0 * atr_pct)


def participation(position_value: float, turnover_value: float) -> float:
    """Order value as a fraction of the session's traded value.

    PASS THE PREVIOUS SESSION'S TURNOVER, not the current one. The day's own
    volume is not knowable when the order is placed, and sizing against it is
    a mild look-ahead that systematically flatters exactly the days a breakout
    fires - breakouts come with volume, so using same-day turnover would
    understate participation precisely when it matters.
    """
    if turnover_value is None or not math.isfinite(turnover_value) \
            or turnover_value <= 0:
        return float("nan")
    return float(position_value) / float(turnover_value)


def retail_bound(position_value: float, turnover: pd.Series | np.ndarray,
                 model: SlippageModel | None = None) -> dict[str, float]:
    """What slippage this model implies across a real distribution of turnover.

    Returns the median and the pessimistic tail, because the mean of a
    participation distribution is dominated by the thinnest names and is not
    what a trader experiences on a typical fill.
    """
    model = model or SlippageModel()
    t = pd.Series(turnover).astype(float)
    t = t[t > 0]
    if t.empty:
        return {}
    part = position_value / t
    bps = part.map(model.bps)
    return {
        "median_participation_pct": float(100.0 * part.median()),
        "p95_participation_pct": float(100.0 * part.quantile(0.95)),
        "median_bps": float(bps.median()),
        "p95_bps": float(bps.quantile(0.95)),
        "median_round_trip_pct": float(2.0 * bps.median() / 100.0),
    }
