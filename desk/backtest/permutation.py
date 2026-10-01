"""Did the signal pick good MOMENTS, or just good stocks in a good window?

WHY THIS EXISTS ALONGSIDE THE DEFLATED SHARPE RATIO
---------------------------------------------------
`desk/robustness` answers "would the best of N tries still look good if I
admitted N". It answers it through Bailey & Lopez de Prado's asymptotics, and
asymptotics need observations - the donchian result came back
`verdict="insufficient"` on 36 monthly cohorts against the 60 those estimators
want. That is a real limitation and no amount of care in the implementation
fixes it.

A permutation test has no such requirement. It builds the null distribution
from this data, by this strategy's own trade mechanics, so its p-value is
exact at any sample size. The two tests are complements: DSR punishes the
search, permutation punishes the timing. A result should survive both.

WHAT WAS BORROWED AND WHAT WAS REJECTED
---------------------------------------
OpenTerminalUI's `core/backtest_robustness.py` has a permutation test, and its
own comment carries the insight worth taking:

    "simply REORDERING returns leaves mean/std - and therefore Sharpe and
     total return - unchanged, so it tests nothing."

That is correct and easy to get wrong. Their fix is to randomly FLIP THE SIGN
of each return, building a null centred on zero.

THE SIGN-FLIP NULL IS WRONG FOR THIS DESK, and in the dangerous direction. Our
returns are R-multiples from a stop-and-target system, so they are asymmetric
BY CONSTRUCTION rather than by edge: a stop is exactly -1R and a target is
exactly +2R. Flipping the sign of a -1R loss produces +1R, and flipping a +2R
win produces -2R - an outcome the stop makes impossible. The null therefore
contains trade sequences the strategy could never generate, with a wider spread
than the real one, and a wider null makes the observed result look MORE
significant than it is. A test that errs toward finding edges is worse than no
test.

THE RIGHT NULL HERE: keep the price process, keep the trade mechanics, keep how
many times each symbol fired, and destroy ONLY the timing. Each symbol's signal
column is shuffled within itself. That leaves the question in its sharpest form -
given these stocks and this market, did the rule enter on better days than
chance would have? It also controls for the two things the cross-sectional
random arm could not: a rising window, and any tendency to pick symbols that
simply went up.

WHY IT IS CHEAP. Re-walking every trade for every permutation would be
hundreds of millions of bar steps. Instead the outcome of entering ANY symbol
on ANY bar is computed once into a lattice, and each permutation becomes a
lookup. One up-front pass buys an unlimited number of permutations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PermutationResult", "outcome_lattice", "signal_timing_test"]


@dataclass(frozen=True, slots=True)
class PermutationResult:
    """How the real entry timing compared with the same signals shuffled."""

    observed: float
    """Mean R per trade at the real signal dates, net of whatever cost was
    passed in."""

    null_mean: float
    null_std: float
    p_value: float
    """Fraction of permutations that matched or beat the observed result, with
    the standard +1/+1 correction so a p-value can never be exactly zero -
    500 permutations cannot evidence p < 1/501."""

    n_permutations: int
    n_trades: int
    percentile: float

    @property
    def significant(self) -> bool:
        """p < 0.05. Says the timing carried information; says NOTHING about
        whether the edge survives costs or the search - that is DSR's job."""
        return self.p_value < 0.05

    def __str__(self) -> str:
        return (f"observed {self.observed:+.4f}R vs null "
                f"{self.null_mean:+.4f}+/-{self.null_std:.4f}, "
                f"p={self.p_value:.4f} ({self.percentile:.1f}th pct), "
                f"{self.n_permutations} permutations, {self.n_trades} trades"
                f"{' SIGNIFICANT' if self.significant else ''}")


def outcome_lattice(high: np.ndarray, low: np.ndarray, close: np.ndarray, *,
                    stop_pct: float, rr: float, hold: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """R and exit row for entering EVERY symbol on EVERY bar. Computed once.

    Returns `(r, exit_row)`, both bars x symbols. `r` is NaN where an entry was
    impossible (no price, or not enough bars left to resolve the trade).

    The stop wins a bar that touches both levels - a daily bar cannot say which
    came first, and assuming the favourable one is how a backtest flatters
    itself. Same tie-break as simulate.py.
    """
    if stop_pct <= 0:
        raise ValueError(f"stop_pct must be > 0, got {stop_pct}")
    if hold < 1:
        raise ValueError(f"hold must be >= 1, got {hold}")

    n, m = close.shape
    r = np.full((n, m), np.nan)
    exit_row = np.full((n, m), -1, dtype=np.int32)

    for c in range(m):
        cl, hc, lc = close[:, c], high[:, c], low[:, c]
        for i in range(n - 1):
            e = cl[i]
            if not np.isfinite(e) or e <= 0:
                continue
            stop_px = e * (1 - stop_pct / 100.0)
            take_px = e * (1 + rr * stop_pct / 100.0)
            last = min(i + hold, n - 1)
            out = None
            j = i
            for j in range(i + 1, last + 1):
                bh, bl = hc[j], lc[j]
                if not (np.isfinite(bh) and np.isfinite(bl)):
                    continue
                if bl <= stop_px:
                    out = -1.0
                    break
                if bh >= take_px:
                    out = rr
                    break
            if out is None:
                px = cl[last]
                if not np.isfinite(px):
                    continue
                out = (px - e) / (e * stop_pct / 100.0)
                j = last
            r[i, c] = out
            exit_row[i, c] = j
    return r, exit_row


def _sequential_mean(sig: np.ndarray, r: np.ndarray, exit_row: np.ndarray
                     ) -> tuple[float, int]:
    """Mean R over non-overlapping trades, one position per symbol at a time.

    The non-overlap rule is applied to the PERMUTED signals too, not just the
    real ones. Skipping it for the null would let the null take more trades
    than the strategy ever could and compare two different things.
    """
    total, count = 0.0, 0
    n, m = sig.shape
    for c in range(m):
        busy = -1
        col = sig[:, c]
        for i in range(n - 1):
            if col[i] and i > busy:
                v = r[i, c]
                if np.isfinite(v):
                    total += v
                    count += 1
                    busy = exit_row[i, c]
    return (total / count if count else float("nan")), count


def signal_timing_test(sig: np.ndarray, r: np.ndarray, exit_row: np.ndarray, *,
                       cost_r: float = 0.0, n_permutations: int = 500,
                       seed: int = 7) -> PermutationResult:
    """Is the real entry timing better than the same signals shuffled in time?

    `sig` is a bars x symbols boolean array of entry signals. Each COLUMN is
    permuted independently, which preserves how many times each symbol fired
    and preserves the price process entirely. Only the timing is destroyed.

    `cost_r` is subtracted from both the observed and the null, so it shifts
    both equally and does not affect the p-value. Passed in anyway so the
    reported numbers are the ones a trader would keep.
    """
    if sig.shape != r.shape:
        raise ValueError(f"signal shape {sig.shape} != lattice {r.shape}")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    observed, n_trades = _sequential_mean(sig, r, exit_row)
    if not math.isfinite(observed):
        raise ValueError("the real signals produced no resolvable trades")
    observed -= cost_r

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    work = np.empty_like(sig)
    for k in range(n_permutations):
        # Shuffle WITHIN each column. np.random.Generator.permuted with axis=0
        # does exactly this and does not mix symbols, which shuffling the flat
        # array would - and mixing symbols would change how many times each
        # name traded, testing selection rather than timing.
        work[:] = rng.permuted(sig, axis=0)
        m, _ = _sequential_mean(work, r, exit_row)
        null[k] = (m - cost_r) if math.isfinite(m) else np.nan

    null = null[np.isfinite(null)]
    if null.size == 0:
        raise ValueError("every permutation failed to produce trades")

    # +1/+1 so p can never be 0: N permutations cannot evidence p < 1/(N+1).
    p = float((np.sum(null >= observed) + 1) / (null.size + 1))
    return PermutationResult(
        observed=float(observed), null_mean=float(null.mean()),
        null_std=float(null.std()), p_value=p, n_permutations=int(null.size),
        n_trades=int(n_trades),
        percentile=float(100.0 * np.mean(null < observed)),
    )
