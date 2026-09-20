"""Stage 2 of the daily scanner funnel: cross-sectional scoring and ranking.

Deterministic. No AI. Takes Stage 1's feature table and produces a RANKED,
EXPLAINED shortlist - the handful of names Stage 3 is allowed to spend LLM
money on. This is the last stage before anything costs money per call, so it
is also the last stage that can afford to look at the whole universe.

WHY CROSS-SECTIONAL RANKS AND NOT Z-SCORES. Every factor is converted to its
percentile rank WITHIN THE DAY'S UNIVERSE before being combined. A z-score is
destroyed by one 300% mover: that single name compresses every other z toward
zero and the day's ranking becomes noise. A percentile rank cannot be moved
more than one place by any single outlier, which is the property that matters
when the input is a few thousand Indian smallcaps with real limit-up prints
in it.

WHY RANKS AND NOT ABSOLUTE THRESHOLDS. "20-day return above 15%" means
something different in a bull quarter and a flat one. Ranking asks the only
question a daily scan can answer honestly: compared with everything else
tradeable TODAY, how does this name look.

WHAT IT REFUSES TO DO:
  - It will not score a name on a factor it does not have. The composite is
    the weighted mean over AVAILABLE factors only, `factors_used` is reported
    per name, and a name below `min_factors` is dropped with a counted reason
    rather than being handed the 50th percentile as a stand-in for "unknown".
  - It SILENCES a factor whose regime is hostile rather than down-weighting
    it, exactly as desk.strategies.catalog.eligible() silences a strategy.
    Momentum in a crisis is not a weak signal; it is a wrong one.
  - It does not let untrusted strategies vote. See `unavailable`.

STRATEGY VOTES ARE NOT IMPLEMENTED, AND THAT IS PARTLY DELIBERATE. The master
plan calls for registered strategies to vote at this stage. Two things block
it, and only one is a missing feature: the six strategy scripts are not on
disk, AND 0 of 6 are `trusted` on the maturity ladder. Even with the files
present, a strategy below `walk_forward` may not influence a live shortlist -
that is what the ladder is for. Wiring votes from untrusted strategies would
be a maturity-ladder violation wearing the costume of a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from desk.contracts.enums import Regime
from desk.scanner.stage1 import Stage1Result

__all__ = ["FactorSpec", "DEFAULT_FACTORS", "REVERSAL_FACTORS",
           "Stage2Result", "run_stage2"]


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """One ranked input to the composite score."""

    name: str
    column: str
    """A column of Stage1Result.features."""
    direction: int
    """+1 when a higher raw value is better, -1 when lower is better."""
    weight: float
    hostile_regimes: tuple[Regime, ...] = ()
    """Regimes in which this factor is SILENCED entirely, not down-weighted."""
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"{self.name}: direction must be +1 or -1")
        if self.weight <= 0:
            raise ValueError(f"{self.name}: weight must be positive")


#: The default factor set. Small and defensible on purpose - six correlated
#: momentum variants would be the ensemble-bias failure the master plan warns
#: about (section 37), five voices carrying one voice's information.
#:
#: Note what is NOT here: `dist_sma20_pct` is deliberately excluded even
#: though it is a fine trend measure, because `dist_sma50_pct` already carries
#: trend and the two are near-duplicates on daily bars. Over-extension is
#: handled as an explicit penalty instead, so it cannot be double-counted as
#: both a virtue and a vice.
DEFAULT_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        "momentum", "ret_20d_pct", +1, 1.0,
        hostile_regimes=(Regime.CRISIS,),
        rationale="20-session return. Silenced in a crisis, where the "
                  "strongest recent movers are the most crowded exits.",
    ),
    FactorSpec(
        "trend", "dist_sma50_pct", +1, 1.0,
        hostile_regimes=(Regime.CRISIS,),
        rationale="Distance above the 50-day average - trend participation "
                  "measured on a slower clock than momentum.",
    ),
    FactorSpec(
        "range_position", "pos_52w_pct", +1, 0.75,
        hostile_regimes=(Regime.CRISIS,),
        rationale="Where the price sits in its own 52-week range.",
    ),
    FactorSpec(
        "participation", "rel_volume", +1, 0.5,
        rationale="Volume against the name's own 20-day baseline. Never "
                  "silenced: unusual participation is informative in every "
                  "regime, including a crisis.",
    ),
    FactorSpec(
        "calmness", "atr_pct_rank", -1, 0.5,
        rationale="Prefer the quieter of two names showing the same move - "
                  "the same rupee target costs a wider stop on the noisier "
                  "one, which the risk engine will charge for later.",
    ),
)

UNAVAILABLE = (
    "Strategy votes not counted - the six strategy scripts are not on disk, "
    "and 0 of 6 are `trusted` on the maturity ladder. An untrusted strategy "
    "may not influence a live shortlist even when its code is present.",
    "Sentiment and per-symbol news filters not applied - ranking uses no "
    "news input. Market-wide headlines reach Stage 3 as context, but "
    "nothing here filters or scores a name on its own announcements.",
    "Pattern and market-structure rules not applied - swing-point structure "
    "is computed per symbol and has not been wired into this stage yet.",
)

#: Said only when no table was supplied. It used to sit in UNAVAILABLE
#: unconditionally, which meant that once the fundamentals cache WAS wired
#: the plan kept reporting "no fundamentals source is wired" while happily
#: using one. A caveat that contradicts what the code did is worse than no
#: caveat: it tells the reader a check was skipped when it ran.
UNAVAILABLE_NO_FUNDAMENTALS = (
    "Fundamental filters not applied - no fundamentals table was supplied.",
)

#: And this is said when a table IS present but no hard threshold is set.
#: "Described but not screened" is a real third state, distinct from both
#: "no data" and "filtered": the figures reach Stage 3's prompt and the
#: ranking, and nothing is excluded on them.
UNAVAILABLE_NO_THRESHOLDS = (
    "Fundamentals were loaded and reported, but NO hard filter is "
    "configured - neither max_filing_age_days nor min_net_margin_pct. Names "
    "were described on their fundamentals, not screened by them.",
)


@dataclass(slots=True)
class Stage2Result:
    as_of: date
    regime: Regime
    ranked: pd.DataFrame
    """Indexed by symbol, descending by `score`. Columns: score,
    factors_used, penalty, plus one `rank_<factor>` column per active factor
    so every score can be read back to its inputs - section 32's
    explainability requirement applied at the cheapest possible stage."""
    factors: tuple[FactorSpec, ...]
    """The factors that ACTUALLY ran, after regime silencing."""
    silenced: dict[str, str] = field(default_factory=dict)
    """factor name -> why it did not run."""
    universe_in: int = 0
    universe_out: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=lambda: list(UNAVAILABLE))

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.ranked.head(n)

    def explain(self, symbol: str) -> str:
        """Why this name scored what it scored, in words. The audit trail has
        to survive into Stage 3's prompt, or the LLM is reasoning about a
        shortlist it cannot see the derivation of."""
        if symbol not in self.ranked.index:
            return f"{symbol} is not in the Stage 2 ranking for {self.as_of}."
        row = self.ranked.loc[symbol]
        lines = [f"{symbol}  score {row['score']:.1f}/100  "
                 f"(rank {self.ranked.index.get_loc(symbol) + 1} of "
                 f"{len(self.ranked)}, {int(row['factors_used'])} factors)"]
        for f in self.factors:
            col = f"rank_{f.name}"
            v = row.get(col)
            if v is None or pd.isna(v):
                lines.append(f"  {f.name:<16} n/a")
            else:
                lines.append(f"  {f.name:<16} {v:5.1f} pctile  "
                             f"(weight {f.weight}, "
                             f"{'higher' if f.direction > 0 else 'lower'} is better)")
        if row.get("penalty"):
            lines.append(f"  PENALTY          -{row['penalty']:.1f} "
                         f"(over-extended from the 20-day average)")
        for name, why in self.silenced.items():
            lines.append(f"  SILENCED {name}: {why}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [f"Stage 2: {self.universe_in} -> {self.universe_out} "
                 f"({self.as_of}, regime={self.regime.value})"]
        lines.append(f"  factors: {', '.join(f.name for f in self.factors) or 'NONE'}")
        for name, why in self.silenced.items():
            lines.append(f"  SILENCED {name}: {why}")
        for reason, n in sorted(self.excluded.items(), key=lambda kv: -kv[1]):
            if n:
                lines.append(f"  -{n:5} {reason}")
        for c in self.unavailable:
            lines.append(f"  UNAVAILABLE: {c}")
        return "\n".join(lines)


#: Fundamental factors. NOT in DEFAULT_FACTORS: they only work when a
#: fundamentals table is supplied, and a factor that silently scores NaN for
#: the whole universe is worse than one that is absent - it dilutes every
#: other factor's weight while contributing nothing.
FUNDAMENTAL_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        "revenue_growth", "revenue_growth_yoy_pct", +1, 1.0,
        rationale="Year-on-year revenue growth for the latest quarter, "
                  "compared like-for-like against the same quarter a year "
                  "earlier and the same reporting nature.",
    ),
    FactorSpec(
        "profit_growth", "profit_growth_yoy_pct", +1, 1.0,
        rationale="Year-on-year profit growth, same comparison basis.",
    ),
    FactorSpec(
        "margin_trend", "margin_change_pp", +1, 0.75,
        rationale="Change in net margin in percentage points. Direction "
                  "matters more than level - a 4% margin improving beats an "
                  "18% margin eroding, and the level differs by industry in "
                  "ways a cross-sectional rank cannot see.",
    ),
)


def run_stage2(
    stage1: Stage1Result,
    *,
    regime: Regime = Regime.UNKNOWN,
    factors: tuple[FactorSpec, ...] = DEFAULT_FACTORS,
    min_factors: int = 3,
    extended_penalty: float = 15.0,
    flagged_only: bool = True,
    fundamentals: "pd.DataFrame | None" = None,
    max_filing_age_days: int | None = None,
    min_net_margin_pct: float | None = None,
) -> Stage2Result:
    """Rank Stage 1's output into an explained shortlist.

    `flagged_only` (default True) scores only the names Stage 1 flagged as
    having something unusual happening. That is the funnel working as
    designed - but pass False to rank the full surviving universe when you
    want to see where an ordinary name sits rather than only the outliers.

    `min_factors` (default 3) is the floor below which a name is DROPPED. A
    name scored on one factor is not comparable with one scored on five, and
    averaging over whatever happened to be available quietly rewards names
    with missing data whenever the surviving factor is a favourable one.

    `extended_penalty` subtracts from the composite for names Stage 1 marked
    `extended` (far above the 20-day average relative to their own ATR).
    Deliberately a PENALTY rather than a filter: over-extension makes an
    entry worse, not the thesis wrong, and the risk engine's stop-distance
    gate is the thing that actually refuses a bad entry later.
    """
    src = stage1.flagged if flagged_only else stage1.features
    universe_in = len(src)
    excluded: dict[str, int] = {}
    fundamental_notes: list[str] = []

    if fundamentals is not None and not src.empty:
        src, excluded, fundamental_notes = _apply_fundamentals(
            src, fundamentals, excluded,
            max_filing_age_days=max_filing_age_days,
            min_net_margin_pct=min_net_margin_pct,
        )

    if universe_in == 0:
        empty = Stage2Result(
            as_of=stage1.as_of, regime=regime, ranked=_empty_ranked(()),
            factors=(), universe_in=0, universe_out=0,
            excluded={"nothing flagged by Stage 1": 0},
        )
        _note_fundamentals(empty, fundamentals, max_filing_age_days,
                           min_net_margin_pct, [])
        return empty

    # --- regime silencing, and the honest reasons a factor cannot run ------
    active: list[FactorSpec] = []
    silenced: dict[str, str] = {}
    for f in factors:
        if regime in f.hostile_regimes:
            silenced[f.name] = f"regime {regime.value} is hostile to it"
        elif f.column not in src.columns:
            silenced[f.name] = f"column {f.column!r} is not in the Stage 1 table"
        elif src[f.column].notna().sum() == 0:
            silenced[f.name] = f"{f.column} is entirely NaN for this universe"
        else:
            active.append(f)

    if not active:
        # MERGE, never replace. This path used to build a fresh dict, which
        # threw away every exclusion counted before it - so when a
        # fundamental filter removed the whole shortlist, the empty frame
        # silenced every factor, this branch fired, and the plan reported
        # "no factor could run in this regime". The regime was blameless;
        # the filter had done it. A wrong cause is worse than no cause,
        # because it sends the reader to fix the wrong thing.
        reached = len(src)
        blocked = dict(excluded)
        if reached:
            blocked["no factor could run in this regime"] = reached
        stalled = Stage2Result(
            as_of=stage1.as_of, regime=regime, ranked=_empty_ranked(()),
            factors=(), silenced=silenced, universe_in=universe_in,
            universe_out=0, excluded=blocked,
        )
        _note_fundamentals(stalled, fundamentals, max_filing_age_days,
                           min_net_margin_pct, fundamental_notes)
        return stalled

    # --- cross-sectional percentile ranks ---------------------------------
    ranks = pd.DataFrame(index=src.index)
    for f in active:
        col = src[f.column].astype("float64")
        # pct=True gives 0-1 and, crucially, LEAVES NaN AS NaN rather than
        # ranking missing data as if it were the minimum.
        r = 100.0 * col.rank(pct=True, ascending=(f.direction > 0))
        ranks[f"rank_{f.name}"] = r

    available = ranks.notna()
    factors_used = available.sum(axis=1)

    thin = factors_used[factors_used < min_factors].index
    if len(thin):
        # Name the CAUSE, not just the symptom. When regime silencing has
        # taken the active factor set below the floor, every name fails for
        # the same structural reason and reporting it as a per-name data
        # problem sends whoever reads the log looking in the wrong place.
        if len(active) < min_factors and silenced:
            reason = (
                f"only {len(active)} factor(s) survived regime "
                f"{regime.value} - below the {min_factors} needed to rank"
            )
        else:
            reason = f"fewer than {min_factors} usable factors"
        excluded[reason] = len(thin)
        ranks = ranks.drop(index=thin)
        available = available.drop(index=thin)
        factors_used = factors_used.drop(index=thin)
        src = src.drop(index=thin)

    if ranks.empty:
        return Stage2Result(
            as_of=stage1.as_of, regime=regime, ranked=_empty_ranked(active),
            factors=tuple(active), silenced=silenced, universe_in=universe_in,
            universe_out=0, excluded=excluded,
        )

    # Weighted mean over AVAILABLE factors only. The denominator is the sum of
    # the weights that actually contributed, so a name missing one factor is
    # not silently penalised by dividing through the full weight total.
    weights = pd.Series({f"rank_{f.name}": f.weight for f in active})
    weighted = ranks.mul(weights, axis=1)
    denom = available.mul(weights, axis=1).sum(axis=1)
    score = weighted.sum(axis=1) / denom.replace(0, np.nan)

    penalty = pd.Series(0.0, index=ranks.index)
    if "extended" in src.columns:
        penalty = src["extended"].fillna(False).astype(bool) * extended_penalty

    out = ranks.copy()
    out["factors_used"] = factors_used.astype("int64")
    out["penalty"] = penalty
    out["score"] = (score - penalty).clip(lower=0.0, upper=100.0)

    lead = ["score", "factors_used", "penalty"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out = out.sort_values("score", ascending=False)

    result = Stage2Result(
        as_of=stage1.as_of, regime=regime, ranked=out, factors=tuple(active),
        silenced=silenced, universe_in=universe_in, universe_out=len(out),
        excluded=excluded,
    )
    _note_fundamentals(result, fundamentals, max_filing_age_days,
                       min_net_margin_pct, fundamental_notes)
    return result


def _note_fundamentals(result, fundamentals, max_filing_age_days,
                       min_net_margin_pct, notes) -> None:
    """Say which of the three fundamental states this run was in.

    Called from EVERY exit, because a run that stopped early still has a
    true answer to "were fundamentals checked" - and an exit that skips this
    reports whichever state the default happened to be.
    """
    if fundamentals is None:
        result.unavailable.extend(UNAVAILABLE_NO_FUNDAMENTALS)
    elif max_filing_age_days is None and min_net_margin_pct is None:
        result.unavailable.extend(UNAVAILABLE_NO_THRESHOLDS)
    result.unavailable.extend(notes)


def _apply_fundamentals(src, fundamentals, excluded, *,
                        max_filing_age_days, min_net_margin_pct):
    """Join the fundamentals table on, then apply the HARD filters.

    Ranked fundamentals and hard filters are deliberately different
    mechanisms. A FactorSpec ranks and weights into a composite score; a
    filter removes a name from consideration. "Its margin ranks in the bottom
    decile" and "it is loss-making" are not the same statement, and collapsing
    them into one score lets a strong technical setup outvote a company that
    should not be on the list at all.

    "No data" and "fails the filter" are counted SEPARATELY. That is R3's
    stated requirement and it falls out of Stage 2's existing `excluded`
    convention for free - two distinct reason strings, two distinct counts.
    """
    notes: list[str] = []

    # NORMALISE THE JOIN KEY. Stage 1 indexes by the full NSE symbol
    # ("RELIANCE.NS"); build_table keys by the base symbol ("RELIANCE"),
    # because that is how the filings store is laid out. A plain join
    # matched NOTHING - not one row, ever.
    #
    # It hid for as long as it did because the failure was indistinguishable
    # from the normal state: with no filings on disk the join legitimately
    # matched nothing, and the caveat said "438 of 438 candidates have no
    # filing on file", which was TRUE OF THE JOIN and false of reality. An
    # honest message about the wrong thing is the hardest kind of bug to
    # see.
    right = fundamentals.copy()
    right.index = [str(i).split(".")[0].upper() for i in right.index]
    right = right[~right.index.duplicated(keep="first")]
    keys = pd.Index([str(i).split(".")[0].upper() for i in src.index],
                    name="base")
    joined = src.join(right.reindex(keys).set_axis(src.index), how="left")

    have = joined["period_end"].notna() if "period_end" in joined else None
    if have is not None:
        missing = int((~have).sum())
        if missing:
            notes.append(
                f"{missing} of {len(joined)} candidates have no filing on "
                f"file, so every fundamental filter below was SKIPPED for "
                f"them - they were not checked, not cleared."
            )

    if max_filing_age_days is not None and "days_since_filing" in joined:
        # A stale filing is not a bad filing, but a company that has not
        # reported in eight months is a different risk from one that reported
        # last week, and the numbers behind any filter are that old too.
        stale = joined["days_since_filing"] > max_filing_age_days
        n = int(stale.fillna(False).sum())
        if n:
            excluded[f"last filing older than {max_filing_age_days} days"] = n
            joined = joined[~stale.fillna(False)]

    if min_net_margin_pct is not None and "net_margin_pct" in joined:
        thin = joined["net_margin_pct"] < min_net_margin_pct
        n = int(thin.fillna(False).sum())
        if n:
            excluded[f"net margin below {min_net_margin_pct}%"] = n
            joined = joined[~thin.fillna(False)]

    return joined, excluded, notes


#: THE MEASURED ALTERNATIVE. Not adopted - `DEFAULT_FACTORS` is still what
#: production ranks on - and it stays that way until the backtest says this
#: is better NET OF COSTS. It exists because the information coefficients
#: below are the first direct evidence this project has about whether any
#: of its factors predict anything, and they say four of the five default
#: factors are pointed the wrong way.
#:
#: Measured 2026-09-20 over 2023-09-01 to 2026-09-17: 120 cross-sections
#: sampled every 5th session, ~1,500 names each, Spearman rank correlation
#: against the forward 5-session return, averaged across sessions. The
#: figures quoted are from the 500 most-traded names per session, because a
#: reversal effect that only exists in microcaps is not tradeable:
#:
#:     column            IC       t     DEFAULT_FACTORS says   verdict
#:     rel_volume      -0.0280   -4.30  +1 (participation)     BACKWARDS
#:     ret_1d_pct      -0.0360   -3.75  not used
#:     ret_5d_pct      -0.0307   -3.11  not used
#:     atr_pct_rank    -0.0203   -2.74  -1 (calmness)          correct
#:     dist_sma20_pct  -0.0246   -2.42  not used
#:     dist_sma50_pct  -0.0149   -1.27  +1 (trend)             wrong, weak
#:     ret_20d_pct     -0.0137   -1.30  +1 (momentum)          wrong, weak
#:     pos_52w_pct     +0.0073   +0.66  +1 (range_position)    NO SIGNAL
#:
#: Robustness: the ICs are unchanged when every forward window containing a
#: single-session move beyond +/-20% is dropped (desk.marketdata.quality's
#: circuit-band threshold), so unadjusted splits - which would bias exactly
#: this way, since companies that split are companies whose price rose - are
#: not producing them.
#:
#: WHY ONLY THREE FACTORS. ret_1d_pct is the strongest single number here
#: and it is deliberately NOT used: one-day reversal is the classic
#: bid-ask-bounce artifact, and at a 0.42% round trip it is precisely the
#: signal costs eat first. ret_5d_pct carries the same effect on a horizon
#: the desk can actually hold. dist_sma20_pct is left out for the reason
#: the default set already gives for excluding it - it is a near-duplicate
#: of the other price-move factors on daily bars, and three correlated
#: measures of "this went up recently" is one voice counted three times.
#: rel_volume is kept because volume is not price and carries its own
#: information; atr_pct_rank because it was already right.
REVERSAL_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        "reversal", "ret_5d_pct", -1, 1.0,
        hostile_regimes=(Regime.CRISIS,),
        rationale="5-session return, INVERTED: recent losers outperform on "
                  "this horizon (IC -0.031, t -3.11). Silenced in a crisis "
                  "for the same reason momentum is - buying the worst "
                  "performers in a falling market is catching knives, and "
                  "the effect measured here was not measured in one.",
    ),
    FactorSpec(
        "quiet_participation", "rel_volume", -1, 1.0,
        rationale="Volume against the name's own 20-day baseline, INVERTED. "
                  "The single most reliable factor measured (t -4.30) and "
                  "the one the default set had exactly backwards: unusual "
                  "volume marks a move that has already happened, not one "
                  "about to.",
    ),
    FactorSpec(
        "calmness", "atr_pct_rank", -1, 0.75,
        rationale="Prefer the quieter name. Carried over unchanged from "
                  "DEFAULT_FACTORS - it is the one factor the measurement "
                  "confirmed rather than contradicted (IC -0.020, t -2.74).",
    ),
)


def _empty_ranked(factors) -> pd.DataFrame:
    cols = ["score", "factors_used", "penalty"] + [f"rank_{f.name}"
                                                   for f in factors]
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    df.index.name = "symbol"
    return df
