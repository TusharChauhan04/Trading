"""What a thin edge actually feels like: drawdown, ruin, and time-to-know.

THE QUESTION THIS ANSWERS
-------------------------
The surviving configuration earns about +0.043R per trade with t +0.84. That is
a real number and a small one, and "positive expectancy" says nothing about
whether a person can sit through the path. Three things decide that, and none of
them is the expectancy:

  - how deep the worst drawdown gets
  - how likely it is to be deep enough to make you stop
  - how many trades before the edge is distinguishable from zero

A strategy that earns +0.043R and spends eighteen months 30% underwater is not
tradeable by a human with one account, whatever the arithmetic says.

WHY THE BORROWED VERSION WAS NOT USED
-------------------------------------
OpenTerminalUI's `core/monte_carlo.py` resamples with
`rng.choice(returns, replace=True)` - an IID bootstrap, which assumes every
return is independent of the one before it.

THAT UNDERSTATES DRAWDOWN, which is the one direction a risk model must never
err in. Losing trades cluster: when a regime turns, a breakout book takes
consecutive stops, and it is precisely those runs that produce the drawdown a
trader actually quits during. An IID bootstrap shuffles those runs apart and
reports a shallower worst case than the strategy really had. It is the same
shape of error as the sign-flip permutation null - a model that flatters the
result in the direction of action.

So this uses a MOVING BLOCK BOOTSTRAP: contiguous runs of trades are resampled
whole, so a losing streak stays a losing streak. `compare_iid` exists to print
the gap between the two, because the gap is the argument.

WHAT IS STILL OPTIMISTIC HERE, stated so the output is not over-read:
  - Block bootstrap preserves clustering WITHIN a block and breaks it at the
    seams, so even this understates a very long regime.
  - Each trade risks a fixed fraction, so there is no position-sizing error,
    no missed fill, and no day when the plan was not followed.
  - The R-multiples come from a backtest whose costs include a modelled
    slippage figure, and adverse selection on breakout entries is not in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["RuinReport", "compare_iid", "trades_to_significance",
           "simulate_paths"]


@dataclass(frozen=True, slots=True)
class RuinReport:
    """The distribution of outcomes, not a single projected equity curve."""

    n_paths: int
    n_trades: int
    risk_per_trade_pct: float

    median_return_pct: float
    p5_return_pct: float
    p95_return_pct: float

    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    """The 95th percentile of the WORST drawdown - i.e. the bad-but-not-absurd
    case. Reported because the median drawdown is what a backtest shows and the
    p95 is what makes people stop."""

    prob_loss: float
    """Share of paths finishing below the starting equity."""

    prob_drawdown_over_20: float
    prob_drawdown_over_35: float
    """Two thresholds chosen for human reasons rather than statistical ones: 20%
    is roughly where a retail trader starts changing the rules, and 35% is
    where most stop entirely. Ruin is behavioural before it is arithmetic."""

    def __str__(self) -> str:
        return (
            f"{self.n_paths} paths x {self.n_trades} trades at "
            f"{self.risk_per_trade_pct:g}% risk\n"
            f"  return     median {self.median_return_pct:+.1f}%   "
            f"p5 {self.p5_return_pct:+.1f}%   p95 {self.p95_return_pct:+.1f}%\n"
            f"  worst DD   median {self.median_max_drawdown_pct:.1f}%   "
            f"p95 {self.p95_max_drawdown_pct:.1f}%\n"
            f"  P(lose money) {100 * self.prob_loss:.1f}%   "
            f"P(DD>20%) {100 * self.prob_drawdown_over_20:.1f}%   "
            f"P(DD>35%) {100 * self.prob_drawdown_over_35:.1f}%"
        )


def _max_drawdown(path: np.ndarray) -> float:
    """Worst peak-to-trough as a POSITIVE fraction."""
    peaks = np.maximum.accumulate(path)
    return float(-np.min(path / np.where(peaks <= 0, 1.0, peaks) - 1.0))


def _block_sample(r: np.ndarray, n: int, block: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Resample contiguous blocks, so a losing streak survives resampling."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, max(1, r.size - block + 1)))
        chunk = r[start:start + block]
        take = min(chunk.size, n - filled)
        out[filled:filled + take] = chunk[:take]
        filled += take
    return out


def simulate_paths(r_multiples: Sequence[float], *, n_trades: int = 200,
                   n_paths: int = 2000, risk_per_trade_pct: float = 1.0,
                   block: int = 20, seed: int = 7,
                   iid: bool = False) -> RuinReport:
    """Bootstrap equity paths from a real sequence of per-trade R multiples.

    `risk_per_trade_pct` is the fraction of CURRENT equity risked per trade, so
    the path compounds and a drawdown reduces subsequent position sizes - which
    is what a fixed-fractional desk actually does, and is gentler than risking a
    fixed rupee amount.

    `block` is the bootstrap block length in trades. 20 is about a month of a
    breakout book's activity, which is the horizon over which regime clustering
    actually bites.

    `iid=True` switches to the independent resampling the borrowed module used.
    It is kept only so the two can be compared - see `compare_iid`.
    """
    r = np.asarray([float(x) for x in r_multiples
                    if x is not None and math.isfinite(float(x))])
    if r.size < 2:
        raise ValueError(f"need at least 2 R multiples, got {r.size}")
    if n_trades < 1 or n_paths < 1:
        raise ValueError("n_trades and n_paths must be >= 1")
    if not 0 < risk_per_trade_pct <= 100:
        raise ValueError(f"risk_per_trade_pct out of range: {risk_per_trade_pct}")
    block = max(1, min(int(block), r.size))

    rng = np.random.default_rng(seed)
    f = risk_per_trade_pct / 100.0
    finals = np.empty(n_paths)
    dds = np.empty(n_paths)

    for p in range(n_paths):
        sample = (rng.choice(r, size=n_trades, replace=True) if iid
                  else _block_sample(r, n_trades, block, rng))
        # Fixed-fractional compounding: equity multiplies by (1 + f*R) per
        # trade. Clipped at a 99% loss per trade so a pathological R cannot
        # drive equity negative, which would make the drawdown meaningless.
        steps = np.maximum(1.0 + f * sample, 0.01)
        path = np.cumprod(steps)
        finals[p] = path[-1]
        dds[p] = _max_drawdown(np.concatenate(([1.0], path)))

    return RuinReport(
        n_paths=n_paths, n_trades=n_trades,
        risk_per_trade_pct=risk_per_trade_pct,
        median_return_pct=float(100 * (np.median(finals) - 1)),
        p5_return_pct=float(100 * (np.percentile(finals, 5) - 1)),
        p95_return_pct=float(100 * (np.percentile(finals, 95) - 1)),
        median_max_drawdown_pct=float(100 * np.median(dds)),
        p95_max_drawdown_pct=float(100 * np.percentile(dds, 95)),
        prob_loss=float(np.mean(finals < 1.0)),
        prob_drawdown_over_20=float(np.mean(dds > 0.20)),
        prob_drawdown_over_35=float(np.mean(dds > 0.35)),
    )


def compare_iid(r_multiples: Sequence[float], **kwargs) -> dict[str, float]:
    """How much shallower an IID bootstrap claims the drawdown is.

    The argument for the block bootstrap, as a number rather than an assertion.
    A positive `understatement_pp` means the IID model - the one the borrowed
    module implements - reports a p95 drawdown that many percentage points
    kinder than the clustering-preserving one.
    """
    blocked = simulate_paths(r_multiples, iid=False, **kwargs)
    independent = simulate_paths(r_multiples, iid=True, **kwargs)
    return {
        "block_p95_dd_pct": blocked.p95_max_drawdown_pct,
        "iid_p95_dd_pct": independent.p95_max_drawdown_pct,
        "understatement_pp": (blocked.p95_max_drawdown_pct
                              - independent.p95_max_drawdown_pct),
        "block_prob_dd_over_20": blocked.prob_drawdown_over_20,
        "iid_prob_dd_over_20": independent.prob_drawdown_over_20,
    }


def trades_to_significance(mean_r: float, std_r: float, *,
                           t_target: float = 2.0) -> float:
    """How many trades before this edge is distinguishable from zero.

    n = (t * sd / mean)^2, inverted from t = mean / (sd / sqrt(n)).

    The number that decides whether a thin edge is a plan or a wish. Measured on
    the real donchian 12%/20-session sequence - mean +0.082R, sd 0.938R -
    that is 523 trades to reach t=2.

    AND 523 TRADES IS NOT 523 OPPORTUNITIES. The backtest took 12,429 trades in
    three years because it bought every signal across 1,100 symbols at once. One
    account holding a handful of positions at a time takes perhaps 100-200 a
    year, so 523 is three to five years of live trading, not one. Dividing the
    backtest's trade count by its years and calling that a trade rate is the
    error to avoid here.

    WHICH MEAN MATTERS: on the more conservative month-equal-weighted figure
    (+0.043R) the same arithmetic gives about 1,900 trades. Both are defensible
    and they differ by nearly 4x, because the answer is quadratic in the mean -
    which is itself the point. A thin edge is not slightly harder to confirm
    than a fat one, it is quadratically harder.

    So this is not an argument against trading the edge; it is an argument
    against ever expecting live results to confirm the backtest in a useful
    timeframe, and therefore against sizing up on a good run.

    Returns inf when the mean is non-positive - no amount of data makes a
    non-edge significant.
    """
    if mean_r <= 0 or std_r <= 0:
        return float("inf")
    return (t_target * std_r / mean_r) ** 2
