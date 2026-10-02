"""Correlation clusters, measured from prices. What D7 has been waiting for.

WHY THIS EXISTS
---------------
`desk/risk/engine.py` has a correlated-exposure cap - `max_correlated_pct`,
default 25% - and it has never once fired. The gate runs only on an explicitly
supplied `corr_group`, nothing supplies one, and so every sized trade carries
the note "correlation: no cluster supplied, so only the sector cap applied".
The field's own docstring says a measured correlation matrix replaces the label
"once there is enough price history to compute one". There are 1,241 sessions
on disk. This computes it.

The gate deliberately refuses to fall back to `sector`, and that refusal is
right for a reason this module's measurements confirm: on the 138 most liquid
names with full history, same-industry pairs average 0.310 correlation against
0.220 for different-industry pairs. INDUSTRY EXPLAINS ONLY 0.089 OF
CO-MOVEMENT. Bucketing by sector and calling it a correlation cap would be the
sector gate with a tighter number on it.

WHAT UPSTREAM GOT WRONG, both measured rather than asserted
-----------------------------------------------------------
`core/riskfolio/clustering.py` and `hrp.py` cluster on `sqrt((1 - rho) / 2)`
with `linkage_method="ward"`.

1. WARD IS INVALID ON THIS DISTANCE AND THE DAMAGE IS VISIBLE. Ward's update
   formula assumes Euclidean distances. A correlation distance matrix built
   from pairwise-deleted correlations need not be Euclidean, and on this
   desk's data it is not: 7 of 120 eigenvalues of the centred Gram matrix are
   negative, the most negative being -1.161. The consequence is not subtle -
   ward merges clusters at a height of 1.5570 when the MAXIMUM POSSIBLE
   DISTANCE IS 1.0, so the dendrogram contains merge heights that correspond
   to no distance in the input, and a threshold cut on it means nothing.
   Cophenetic correlation, which measures how faithfully a tree reproduces the
   distances it was built from, ranks the methods:

       average   0.6998      <- used here
       ward      0.4591      <- upstream's default, and it inverts
       complete  0.3735
       single    0.3479      <- the canonical HRP choice, poor fit here

   `average` is used on that measurement. Note that single linkage - what
   Lopez de Prado's HRP specifies - is the WORST fit on this data; it chains.
   The canonical choice is not automatically the right one.

2. `corr().fillna(0)` TURNS "UNKNOWN" INTO "UNCORRELATED", which is the most
   favourable diversification assumption available and therefore the worst
   possible default for a risk control. 239 of 14,400 correlation cells were
   NaN on a 120-name panel (1.66%), and after the fill the matrix had a
   smallest eigenvalue of -4.689 - not positive semi-definite, so not a
   correlation matrix at all. No set of random variables has those
   correlations. Here a name without enough overlapping history is reported
   UNCLUSTERED and the gate skips it, which the engine already models as a
   first-class third state.

Two smaller ones, not fixed because this module does not do their job:
`n_clusters = round(sqrt(n))` is arbitrary, and `hrp_weights` accepts a
`risk_measure` argument it never reads, so a caller asking for CVaR silently
gets mean-variance.

THE THRESHOLD IS A CORRELATION, NOT A COUNT, because the risk question is "do
these move together enough to be one bet?" and that has an answer on a scale a
human can argue with. Measured on the liquid universe, pairwise correlation
runs mean 0.219, median 0.211, p95 0.469, p99 0.602, max 0.888 - so
DEFAULT_THRESHOLD = 0.5 sits around the 98th percentile of pairs and flags
genuinely unusual co-movement. What that cut produces on 138 names:

    rho >= 0.3    37 clusters, largest 65   <- half the universe is one bet
    rho >= 0.4    62 clusters, largest 32
    rho >= 0.5    89 clusters, largest 16   <- default
    rho >= 0.6   106 clusters, largest  7
    rho >= 0.7   122 clusters, largest  4

0.3 is too coarse to be useful: a cluster holding 65 of 138 names would reject
nearly every second trade and the cap would be switched off within a week,
which is how a safety check dies.

WHAT THIS IS NOT. It is not a view on whether the cap's 25% is the right
number, and it is not evidence that clustering improves returns. It makes an
existing, tested, inert gate able to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_LOOKBACK", "DEFAULT_THRESHOLD", "MIN_NAMES",
           "MIN_OVERLAP", "ClusterMap", "available", "cluster_map"]

#: Trailing sessions used for the correlation estimate. A year balances a
#: stable estimate against a correlation structure that genuinely changes -
#: rate-sensitive names move together in a rate cycle and apart outside one.
DEFAULT_LOOKBACK = 250

#: Pairwise correlation at or above which two names count as one bet. See the
#: module docstring for the distribution this was cut from.
DEFAULT_THRESHOLD = 0.5

#: A name needs this many overlapping return observations with the panel to be
#: clustered at all. Below it the correlation is too noisy to act on and the
#: name is reported UNCLUSTERED rather than filled with a zero.
MIN_OVERLAP = 200

#: Fraction of the window a name must have priced before it joins the matrix.
#:
#: THIS IS WHAT MAKES LISTWISE DELETION SURVIVABLE, and getting it wrong looks
#: like the module failing rather than the filter being wrong. A common date
#: index is needed for a positive semi-definite correlation matrix, so the
#: panel is reduced to rows where every included name traded - and across
#: 1,400 columns, ONE name missing ONE day removes that day for everybody.
#: Admitting names at 80% completeness left a common index of 167 rows out of
#: 250. Measured on the real survivor set:
#:
#:     completeness   names kept   common index
#:            >=0.80         1438      167 rows
#:            >=0.90         1413      193
#:            >=0.95         1405      227
#:            >=0.98         1398      235   <- default
#:            >=1.00         1382      249
#:
#: The distribution is sharply bimodal - the median name has 249 of 250 - so
#: demanding near-complete history costs almost no coverage and buys back the
#: whole window. Names below the bar are reported UNCLUSTERED.
MIN_COMPLETENESS = 0.98

#: Fewer names than this and there is no structure to find; everything comes
#: back unclustered rather than forced into groups.
MIN_NAMES = 3


def available() -> bool:
    """Whether scipy is installed. False is an answer, not an error - the gate
    then skips, which is the same honest state as supplying no cluster."""
    try:
        import scipy.cluster.hierarchy  # noqa: F401
    except Exception:                                   # noqa: BLE001
        return False
    return True


@dataclass(frozen=True, slots=True)
class ClusterMap:
    """Which names move together, as of one day."""

    labels: dict[str, str] = field(default_factory=dict)
    """symbol -> cluster label, for names that could be clustered.

    The label names its largest member and its size, e.g. "CORR:HDFCBANK+3",
    so a rejection message says something a human can check rather than
    "cluster 7".
    """

    unclustered: tuple[str, ...] = ()
    """Names with too little overlapping history to place. NOT a cluster of
    their own and NOT correlated with nothing - simply unknown, so the gate
    must skip them."""

    threshold: float = DEFAULT_THRESHOLD
    n_obs: int = 0
    why: str = ""

    def group_for(self, symbol: str) -> str | None:
        """The cluster label, or None when the name could not be placed.

        None is what the risk engine needs: it runs the correlated cap only on
        a supplied cluster and records `checks_skipped` otherwise, which is the
        correct outcome for a name whose co-movement is unknown.
        """
        return self.labels.get(symbol)

    @property
    def sizes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for lab in self.labels.values():
            out[lab] = out.get(lab, 0) + 1
        return out

    @property
    def largest(self) -> int:
        s = self.sizes
        return max(s.values()) if s else 0

    @property
    def multi_name_clusters(self) -> dict[str, int]:
        """Only the clusters that actually bind. A singleton cluster cannot
        trip a correlated cap, so these are the ones worth reporting."""
        return {k: v for k, v in self.sizes.items() if v > 1}


def cluster_map(closes: pd.DataFrame, *, as_of=None,
                lookback: int = DEFAULT_LOOKBACK,
                threshold: float = DEFAULT_THRESHOLD,
                min_overlap: int = MIN_OVERLAP,
                min_completeness: float = MIN_COMPLETENESS) -> ClusterMap:
    """Group `closes`' columns by return correlation, using only past data.

    Everything comes from the `lookback` sessions ending at `as_of`; no
    estimate uses a bar after the day being assessed.

    Returns an empty map with a reason rather than guessing when it cannot
    answer - scipy missing, too few names, too little history. An empty map
    makes the correlated cap skip, which is exactly where it has been all
    along, so a failure here cannot make the risk gate WRONG, only inert.
    """
    if not available():
        return ClusterMap(threshold=threshold,
                          why="scipy is not installed, so no correlation "
                              "clustering is available; the correlated cap "
                              "will skip as before")

    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd

    px = closes.copy()
    if as_of is not None:
        px = (px.loc[:pd.Timestamp(as_of)]
              if isinstance(px.index, pd.DatetimeIndex) else px.loc[:as_of])
    px = px.tail(lookback)
    # fill_method=None IS LOAD-BEARING. pandas' default is fill_method="pad",
    # which forward-fills a missing price and therefore reports a return of
    # EXACTLY ZERO for every gap. Names with gaps on the same dates then show
    # matching zeros and correlate for no reason other than both being
    # untraded, which would manufacture exactly the clusters this module
    # exists to detect honestly. A gap stays NaN and the pair-overlap filter
    # below decides whether there is enough real data.
    rets = px.pct_change(fill_method=None)

    # Keep only names with enough history to correlate HONESTLY. This is the
    # fillna(0) fix: a name that cannot be measured is excluded and named,
    # never assumed uncorrelated. The bar is a FRACTION of the window rather
    # than an absolute count - see MIN_COMPLETENESS for why that distinction
    # decides whether anything can be clustered at all.
    counts = rets.notna().sum()
    need = max(int(round(min_completeness * (len(rets) - 1))), MIN_NAMES)
    usable = [c for c in rets.columns if int(counts[c]) >= need]
    dropped = tuple(c for c in rets.columns if c not in usable)

    if len(usable) < MIN_NAMES:
        return ClusterMap(
            unclustered=tuple(rets.columns), threshold=threshold,
            n_obs=len(rets),
            why=f"only {len(usable)} of {len(rets.columns)} names priced "
                f"{need}+ of the last {len(rets)} sessions "
                f"({min_completeness:.0%} completeness); too few to cluster")

    block = rets[usable].dropna()
    if len(block) < min_overlap:
        return ClusterMap(
            unclustered=tuple(rets.columns), threshold=threshold,
            n_obs=len(block),
            why=f"only {len(block)} sessions where all {len(usable)} usable "
                f"names traded together; below the {min_overlap} floor")

    # A constant-price name has no correlation with anything. Excluded rather
    # than left to produce a NaN row that a fill would turn into zeros.
    sd = block.std()
    live = [c for c in block.columns if float(sd[c]) > 0]
    frozen = tuple(c for c in block.columns if c not in live)
    if len(live) < MIN_NAMES:
        return ClusterMap(
            unclustered=tuple(rets.columns), threshold=threshold,
            n_obs=len(block),
            why=f"only {len(live)} names had any price variation")
    block = block[live]

    corr = block.corr()
    if bool(corr.isna().to_numpy().any()):
        # Should be impossible after the filters above. If it ever happens,
        # refuse rather than fill - the fill is the upstream bug.
        return ClusterMap(
            unclustered=tuple(rets.columns), threshold=threshold,
            n_obs=len(block),
            why="the correlation matrix still held NaNs after filtering; "
                "refusing rather than filling them with zeros")

    dist = np.sqrt(np.clip((1.0 - corr.to_numpy(float)) / 2.0, 0.0, 1.0))
    dist = (dist + dist.T) / 2.0           # enforce exact symmetry
    np.fill_diagonal(dist, 0.0)

    # `average`, on the cophenetic measurement in the module docstring. NOT
    # ward: it is invalid on a non-Euclidean distance and produces merge
    # heights above the maximum possible distance of 1.0.
    link = sch.linkage(ssd.squareform(dist, checks=False), method="average")
    cut = float(np.sqrt((1.0 - threshold) / 2.0))
    ids = sch.fcluster(link, t=cut, criterion="distance")

    members: dict[int, list[str]] = {}
    for sym, cid in zip(block.columns, ids):
        members.setdefault(int(cid), []).append(str(sym))

    # Label by the cluster's most volatile-weighted... no: by its
    # alphabetically first member plus size. Deterministic and checkable, and
    # it does not silently change when a price moves.
    labels: dict[str, str] = {}
    for syms in members.values():
        head = sorted(syms)[0]
        lab = (f"CORR:{head}" if len(syms) == 1
               else f"CORR:{head}+{len(syms) - 1}")
        for s in syms:
            labels[s] = lab

    return ClusterMap(labels=labels, unclustered=dropped + frozen,
                      threshold=threshold, n_obs=len(block))
