"""Stage 1 of the daily scanner funnel: per-symbol features and opportunity
flags.

Deterministic. No AI. Runs pre-open on the Stage 0 survivors plus the history
`desk.store` accumulates, and answers one question per name: is anything
unusual happening here? It does NOT rank, score or form an opinion - alpha
scoring, strategy votes and pattern rules are Stage 2, and the LLM does not
appear until Stage 3.

VECTORISED ACROSS SYMBOLS, not looped. Every feature is computed on a
date x symbol matrix (`History.wide`), so 1,600 names cost roughly what one
name costs in Python overhead. A per-symbol loop calling the indicator
functions individually would be ~1,600 x the interpreter overhead for
identical arithmetic; at this universe size that is the difference between a
scan that fits in the pre-open window and one that does not.

WHAT IT CANNOT DO YET, declared on every run in `Stage1Result.unavailable`
rather than silently omitted - the master plan names these as Stage 1 filters
and their absence is a real hole in the screen, not a design choice:
  - SECTOR STRENGTH: bhavcopy carries no sector classification and no sector
    map exists in the project.
  - NEWS AND EVENT FLAGS: no announcements feed is wired.
  - RELATIVE STRENGTH vs Nifty, unless a `benchmark` series is supplied.
    Bhavcopy holds equities only; there is no index series in the store, and
    synthesising one from the universe's own mean would be a different
    measure wearing the same name.

PRICES ARE RAW. `desk.store` returns unadjusted bhavcopy closes, so a split
inside the lookback window shows up as a ~50% single-day "return" and poisons
every rolling statistic over it. Pass `actions` and this module will refuse
to compute features for an affected symbol rather than reporting a fictional
100% move as the day's strongest momentum. See `unadjusted_actions`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from desk.marketdata.corporate_actions import CorporateAction
from desk.store import Coverage, History

__all__ = ["Stage1Result", "run_stage1", "FEATURE_COLUMNS", "REQUIRED_BARS"]

#: The only snapshot columns this stage reads. Pass these as `columns=` to
#: BarStore.history: a bhavcopy row carries prev_close, last, turnover_lacs,
#: trades, delivery_qty, delivery_pct and series as well, none of which are
#: used here, and reading them across several hundred daily files is roughly
#: half the bytes and half the resident memory for nothing.
REQUIRED_BARS = ("open", "high", "low", "close", "volume")

UNAVAILABLE = (
    "Sector strength not computed AT THIS STAGE - no sector classification "
    "in bhavcopy, so no feature here is relative to a peer group. A sector "
    "map DOES now exist (configs/sectors.json, 501 symbols from NSE's index "
    "constituent files) and the regime engine uses it for leading and "
    "lagging industries; it is not wired into per-symbol features.",
    "PER-SYMBOL news and event flags not computed - NSE corporate "
    "announcements (Tier 1, the company's own words, timestamped) are "
    "parsed but not wired into feature generation. Market-wide headlines "
    "ARE available to Stage 3; this is the per-symbol layer, which is a "
    "different and stronger signal.",
    "Bar-level data corruption is only screened by a minimum-tick floor AT "
    "THIS STAGE. A corrupt print that is still a plausible price (1.00 on a "
    "stock trading at 1000) is not caught here, so every FEATURE on this "
    "table is computed from unvetted bars. The discontinuity screen in "
    "desk.marketdata.quality now runs, but on the Stage 3 SHORTLIST only - "
    "it costs 3.6s across the full universe and milliseconds across eight "
    "names, and it is the eight that get sized.",
)

#: Every feature column produced, in report order. Named explicitly so a
#: caller can assert on the shape and a downstream stage cannot silently
#: depend on a column that a future refactor renames.
FEATURE_COLUMNS = (
    "bars", "close", "volume",
    "ret_1d_pct", "ret_5d_pct", "ret_20d_pct",
    "rel_volume", "gap_pct",
    "atr_pct", "atr_pct_rank",
    "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "pos_52w_pct", "high_52w", "low_52w",
    "bb_width_pct", "compressed",
    "rs_rank",
    "unusual_volume", "unusual_move", "near_52w_high", "extended",
    "flag_count",
)


@dataclass(slots=True)
class Stage1Result:
    as_of: date
    features: pd.DataFrame
    """One row per symbol, indexed by symbol, columns = FEATURE_COLUMNS.
    A feature whose window exceeds the symbol's available history is NaN -
    never a shorter-window value wearing the longer window's name."""
    coverage: Coverage
    """Inherited from the history that fed this. Read it before trusting any
    long-window column: `sma200` over 40 sessions is NaN by design, but
    `sma20` over a window with three missing fetches is a real number
    computed from the wrong bars."""
    universe_in: int
    universe_out: int
    excluded: dict[str, int] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=lambda: list(UNAVAILABLE))
    unadjusted_actions: dict[str, int] = field(default_factory=dict)
    """symbol -> number of price-affecting corporate actions found inside the
    lookback window. These symbols are DROPPED, because their rolling
    statistics are computed over a price series with an artificial step in
    it."""

    @property
    def flagged(self) -> pd.DataFrame:
        """Only the names with at least one opportunity flag - what Stage 2
        actually receives."""
        return self.features[self.features["flag_count"] > 0]

    def summary(self) -> str:
        lines = [f"Stage 1: {self.universe_in} -> {self.universe_out} "
                 f"({self.as_of}); {len(self.flagged)} flagged"]
        lines.append(f"  history: {self.coverage.describe()}")
        for reason, n in sorted(self.excluded.items(), key=lambda kv: -kv[1]):
            if n:
                lines.append(f"  -{n:5} {reason}")
        for c in self.unavailable:
            lines.append(f"  UNAVAILABLE: {c}")
        return "\n".join(lines)


def run_stage1(
    history: History,
    *,
    as_of: date | None = None,
    symbols: list[str] | None = None,
    benchmark: pd.Series | None = None,
    actions: list[CorporateAction] | None = None,
    min_bars: int = 30,
    rel_volume_threshold: float = 2.0,
    unusual_move_atrs: float = 1.5,
    near_high_pct: float = 95.0,
    extended_atrs: float = 4.0,
) -> Stage1Result:
    """Compute Stage 1 features for every symbol in `history`.

    `min_bars` (default 30) is the floor below which a name is DROPPED rather
    than reported with mostly-NaN features - a freshly listed scrip has no
    meaningful relative volume or ATR, and letting it through as a row of
    NaNs invites a downstream `.fillna(0)` that turns "unknown" into "calm".

    The four threshold arguments are YOUR numbers, same philosophy as
    RiskConfig: 2x average volume, a move of 1.5 ATR, within 5% of the
    52-week high, and 4 ATR above the 20-DMA are conventional starting
    points, not recommendations.
    """
    if history.frame.empty:
        raise ValueError(
            "history is empty - fetch snapshots with "
            "`python -m desk.marketdata.refresh bhavcopy --date ...` first"
        )

    # One index factorisation for every field, not one per field - see
    # History.wide_many. Fields absent from this history come back as None
    # rather than being substituted for.
    present = [f for f in ("close", *REQUIRED_BARS)
               if f in history.frame.columns]
    matrices = history.wide_many(list(dict.fromkeys(present)))

    if "close" not in matrices:
        raise ValueError("history has no close column - nothing to compute")
    close = matrices["close"]
    if symbols is not None:
        keep = [s for s in close.columns if s in set(symbols)]
        close = close[keep]
    if close.empty or not len(close.columns):
        raise ValueError("no symbols in history after filtering")

    as_of = as_of or close.index.max()
    if as_of not in close.index:
        raise ValueError(
            f"{as_of} is not in this history (last session: {close.index.max()})"
        )
    close = close.loc[:as_of]                 # point-in-time, again, locally

    universe_in = len(close.columns)
    excluded: dict[str, int] = {}

    def take(name: str) -> pd.DataFrame | None:
        return _slice(matrices.get(name), close.columns, as_of)

    high, low, open_, volume = (take("high"), take("low"), take("open"),
                                take("volume"))

    # --- corporate actions inside the window make raw prices unusable ------
    unadjusted: dict[str, int] = {}
    if actions:
        window_start = close.index.min()
        for a in actions:
            ex = getattr(a, "ex_date", None)
            if ex is None or not (window_start <= ex <= as_of):
                continue
            factor = getattr(a, "factor", 1.0)
            factor = 1.0 if factor is None else float(factor)
            # NOT `factor or 1.0`: 0.0 is falsy, so that idiom would turn a
            # zero factor into 1.0 and wave a price-affecting action through
            # as a harmless dividend. load_actions rejects factor <= 0 today,
            # so this is a landmine rather than a live bug - but a
            # CorporateAction built directly in a test or a future importer
            # would arm it.
            if factor == 1.0:
                continue                       # dividend-only, no price step
            if a.symbol in close.columns:
                unadjusted[a.symbol] = unadjusted.get(a.symbol, 0) + 1
        if unadjusted:
            excluded["unadjusted corporate action in window"] = len(unadjusted)
            drop = list(unadjusted)
            close = close.drop(columns=drop)
            for m in (high, low, open_, volume):
                if m is not None:
                    m.drop(columns=[c for c in drop if c in m.columns],
                           inplace=True)

    # --- bars available per symbol ----------------------------------------
    bars = close.notna().sum()
    thin = bars[bars < min_bars].index.tolist()
    if thin:
        excluded[f"fewer than {min_bars} bars"] = len(thin)
        close = close.drop(columns=thin)
        for m in (high, low, open_, volume):
            if m is not None:
                m.drop(columns=[c for c in thin if c in m.columns], inplace=True)
        bars = bars.drop(index=thin)

    if not len(close.columns):
        return Stage1Result(
            as_of=as_of, features=_empty_features(), coverage=history.coverage,
            universe_in=universe_in, universe_out=0, excluded=excluded,
            unadjusted_actions=unadjusted,
        )

    out = pd.DataFrame(index=close.columns)
    out.index.name = "symbol"
    out["bars"] = bars.astype("int64")
    out["close"] = close.iloc[-1]
    out["volume"] = volume.iloc[-1] if volume is not None else np.nan

    # --- returns -----------------------------------------------------------
    for n in (1, 5, 20):
        out[f"ret_{n}d_pct"] = _pct_change(close, n)

    # --- volume ------------------------------------------------------------
    if volume is not None:
        baseline = volume.shift(1).rolling(20).mean()
        rel = volume / baseline.replace(0, np.nan)
        out["rel_volume"] = _finite(rel.iloc[-1])
    else:
        out["rel_volume"] = np.nan

    # --- gap ---------------------------------------------------------------
    if open_ is not None:
        prev_close = close.shift(1)
        out["gap_pct"] = _finite(
            (100.0 * (open_ / _price(prev_close) - 1.0)).iloc[-1])
    else:
        out["gap_pct"] = np.nan

    # --- volatility --------------------------------------------------------
    if high is not None and low is not None:
        tr = _true_range_wide(high, low, close)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()     # Wilder
        atr_pct = 100.0 * atr / _price(close)
        out["atr_pct"] = _finite(atr_pct.iloc[-1])
        # Where today's volatility sits inside this stock's OWN trailing year.
        # An absolute ATR% threshold means something different for a large-cap
        # and a smallcap; a percentile does not.
        out["atr_pct_rank"] = (
            100.0 * atr_pct.rolling(252, min_periods=60).rank(pct=True).iloc[-1]
        )
    else:
        atr_pct = None
        out["atr_pct"] = np.nan
        out["atr_pct_rank"] = np.nan

    # --- trend -------------------------------------------------------------
    sma20 = close.rolling(20).mean()
    for period in (20, 50, 200):
        ma = sma20 if period == 20 else close.rolling(period).mean()
        out[f"dist_sma{period}_pct"] = _finite(
            (100.0 * (close / _price(ma) - 1.0)).iloc[-1])

    # --- 52-week position --------------------------------------------------
    # 252 sessions, and min_periods=1 deliberately: the 52-week high of a
    # stock with 60 bars is the highest of those 60, which is a true statement
    # about the data we have. `bars` is reported alongside so a caller can
    # tell a real annual high from a two-month one.
    win = 252
    hi = (high if high is not None else close).rolling(win, min_periods=1).max()
    lo = (low if low is not None else close).rolling(win, min_periods=1).min()
    out["high_52w"] = hi.iloc[-1]
    out["low_52w"] = lo.iloc[-1]
    span = (hi - lo).replace(0, np.nan)
    out["pos_52w_pct"] = _finite((100.0 * (close - lo) / span).iloc[-1])

    # --- compression -------------------------------------------------------
    mid = sma20
    std = close.rolling(20).std()
    width = 100.0 * (4.0 * std) / _price(mid)   # 2 std either side of the mid
    out["bb_width_pct"] = _finite(width.iloc[-1])
    squeeze = width.rolling(120, min_periods=40).quantile(0.20)
    out["compressed"] = (width <= squeeze).iloc[-1].fillna(False).astype(bool)

    # --- relative strength -------------------------------------------------
    unavailable = list(UNAVAILABLE)
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.reindex(close.index).ffill()
        rs = close.div(_price(bench), axis=0)
        out["rs_rank"] = (
            100.0 * rs.rolling(55, min_periods=20).rank(pct=True).iloc[-1]
        )
    else:
        out["rs_rank"] = np.nan
        unavailable.append(
            "Relative strength vs Nifty not computed - no benchmark series "
            "supplied, and the store holds equities only."
        )

    # --- opportunity flags -------------------------------------------------
    # Comparisons against NaN are False, so a feature that could not be
    # computed never raises a flag. That is the intended behaviour: an
    # unknown is not an opportunity.
    out["unusual_volume"] = out["rel_volume"] >= rel_volume_threshold
    out["unusual_move"] = (
        out["ret_1d_pct"].abs() >= unusual_move_atrs * out["atr_pct"]
    )
    out["near_52w_high"] = out["pos_52w_pct"] >= near_high_pct
    out["extended"] = out["dist_sma20_pct"] >= extended_atrs * out["atr_pct"]
    out["flag_count"] = (
        out[["unusual_volume", "unusual_move", "near_52w_high", "compressed"]]
        .sum(axis=1).astype("int64")
    )

    out = out[list(FEATURE_COLUMNS)].sort_values("flag_count", ascending=False)
    return Stage1Result(
        as_of=as_of, features=out, coverage=history.coverage,
        universe_in=universe_in, universe_out=len(out), excluded=excluded,
        unavailable=unavailable, unadjusted_actions=unadjusted,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _slice(matrix: pd.DataFrame | None, columns, as_of: date
           ) -> pd.DataFrame | None:
    """Align one field's matrix to the close matrix, or pass None through.

    None means "this column was not fetched", which is different from "it was
    fetched and is NaN" - features that need it come back NaN and their flags
    stay False, rather than being computed from a substitute.
    """
    if matrix is None:
        return None
    keep = [c for c in columns if c in matrix.columns]
    return matrix[keep].loc[:as_of].copy()


def _true_range_wide(high: pd.DataFrame, low: pd.DataFrame,
                     close: pd.DataFrame) -> pd.DataFrame:
    """True range across every symbol at once.

    np.maximum rather than pd.concat(...).max(axis=1): concat on a wide frame
    would build a 3x-wide intermediate and then reduce it, which is both
    slower and allocates three copies of the universe.
    """
    prev = close.shift(1)
    a = high - low
    b = (high - prev).abs()
    c = (low - prev).abs()
    # The first bar has no previous close, so b and c are NaN there and the
    # standard convention falls back to high-low - which np.fmax gives us,
    # since it prefers the defined operand over NaN.
    return np.fmax(np.fmax(a, b), c)


#: NSE's minimum tick size. A "price" below this cannot be a real traded
#: print on this exchange, so it is data corruption by definition rather than
#: a judgement call - which is what makes it a defensible floor instead of an
#: arbitrary epsilon.
MIN_VALID_PRICE = 0.01


def _positive(denominator, floor: float = 0.0):
    """A denominator with every value at or below `floor` masked to NaN.

    WHY THIS IS NOT PARANOIA. Stage 0's min_price filter only looks at the
    CURRENT day's close - it says nothing about the 251 historical closes
    Stage 1 reads behind it. One glitched bhavcopy print in that window
    (a bad tick recorded as 0.0001, a zero) is not a corporate action, so
    `unadjusted_actions` cannot catch it, and it will eventually land exactly
    1, 5 or 20 rows before some future as_of.

    Measured on the real code before this guard existed: a single 0.0001
    close 20 sessions back produced ret_20d_pct of 1.39e8 and scored 94.4/100
    in Stage 2 - the best momentum in the universe, entirely fabricated.
    An exact zero produced `inf`, which is WORSE than NaN here: pd.isna(inf)
    is False, so it is invisible to every NaN-based guard in the project, and
    Series.rank(pct=True) ranks it as the maximum. It would win outright.
    """
    return denominator.where(denominator > floor)


def _price(denominator):
    """A price denominator, floored at NSE's minimum tick.

    Plain `> 0` is not enough: a bad tick recorded as 0.0001 is positive, and
    dividing by it produced a MEASURED ret_20d_pct of 99,999,900% in the
    regression test below. Below one paisa is not a price.

    WHAT THIS STILL DOES NOT CATCH: a corrupt print that is a PLAUSIBLE price
    - 1.00 on a stock that trades at 1000 - is indistinguishable from a real
    crash by any per-value rule, and passes this floor. Detecting it needs the
    discontinuity screen in desk.marketdata.quality, which compares a bar
    against its own neighbours and is NOT yet wired into the scanner. That gap
    is declared in Stage1Result.unavailable rather than left implicit.
    """
    return _positive(denominator, MIN_VALID_PRICE)


def _finite(series: pd.Series) -> pd.Series:
    """Defence in depth: inf must never leave this module.

    Every division here is already guarded, but `inf` is the one bad value
    that survives `pd.isna()` and still sorts to the top of a percentile
    rank, so it is swept once more at the boundary rather than trusted not
    to reappear.
    """
    return series.replace([np.inf, -np.inf], np.nan)


def _pct_change(close: pd.DataFrame, n: int) -> pd.Series:
    prior = _price(close.shift(n))
    return _finite((100.0 * (close / prior - 1.0)).iloc[-1])


def _empty_features() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in FEATURE_COLUMNS})
    df.index.name = "symbol"
    return df
