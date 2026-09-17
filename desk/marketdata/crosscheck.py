"""Reconciling NSE against BSE - the second source quality.py never had.

`quality.py` has carried a CROSS_SOURCE check since it was written and it has
never run once: with no second source it reports "no second source supplied"
and skips. This module supplies one.

THE CENTRAL FINDING, AND WHY THE GATE EXISTS
--------------------------------------------
Two exchanges quoting the same security do NOT close at the same price, and
the gap is much larger than intuition suggests. Measured across the six
sessions held locally (2026-09-04 to 2026-09-11, ~2,397 matched names a day):

    median   0.14%      p90   1.11%      p99   4.60%      max  61.22%

It is not bias - mean signed difference is -0.129%, median -0.006%, and NSE
is the higher of the two on 45.6% of names. Neither exchange leads.

It is LIQUIDITY. A closing price struck on almost no volume is one trade, not
an independent measurement. SANWARIA, the 61% case, traded 549,342 shares on
NSE that day and 6,504 on BSE. Divergence falls away almost monotonically as
BSE's traded value rises:

    BSE turnover        n     median      p99        max
    < 1 lakh          362      0.954     17.258     61.224
    1L - 10L          501      0.338      3.218      6.402
    10L - 1Cr         775      0.114      1.770      4.166
    1Cr - 10Cr        579      0.061      0.753      0.966
    10Cr - 100Cr      171      0.077      0.561      0.752
    > 100Cr             9      0.071      1.321      1.401

So a symbol below the gate is reported as NOT CHECKED, never as checked-and-
passed. This is the `checks_skipped` convention the quality module already
uses, and here it is load-bearing rather than tidy: a thin BSE print that
agrees with NSE is not corroboration, and a thin BSE print that disagrees is
not evidence NSE is wrong. Neither outcome carries information, so asserting
either would be worse than saying we cannot tell.

WHAT THE GATE COSTS
-------------------
Reach. Only about a third of matched names clear it - 759 of 2,397 on
2026-09-11. `CrossCoverage` reports that plainly so a daily plan can say "the
cross-source check covered 759 of 3,485 names" instead of implying the other
2,726 were checked and were fine.

CHOOSING THE TOLERANCE
----------------------
Pooled over the six sessions, above the gate (4,569 observations):

    over 0.5%   175   (3.830%)
    over 1.0%    28   (0.613%)
    over 1.5%    11   (0.241%)
    over 2.0%     9   (0.197%)
    over 3.0%     2   (0.044%)
    over 5.0%     0   (0.000%)
    p99 0.865%   p99.9 2.164%   max 4.393%

DEFAULT_TOLERANCE_PCT is 2.0 - just above the pooled 99.9th percentile. That
fires on roughly 1.5 names per session out of ~760, which is a reviewable
number rather than a wall of noise, and it sits far below the signature of
the defects this check exists to catch: an unapplied split or bonus moves a
close by 20-50%, a decimal slip by 900%. A tolerance tight enough to catch
routine microstructure would fire hundreds of times a day and be switched
off within a week, which is the real failure mode for a check like this.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from desk.marketdata.isin import IsinMap

__all__ = [
    "CrossCoverage", "DEFAULT_TOLERANCE_PCT", "MIN_BSE_TURNOVER",
    "bse_reference", "reconcile",
]

#: Rupees of BSE traded value below which a BSE close is not independent
#: evidence. 1 crore. See the table in the module docstring: above it the
#: worst observed disagreement over six sessions was 4.39%, below it 61%.
MIN_BSE_TURNOVER = 1e7

#: Percent. Just above the pooled p99.9 of 2.164%.
DEFAULT_TOLERANCE_PCT = 2.0


@dataclass(frozen=True, slots=True)
class CrossCoverage:
    """Why each NSE symbol could or could not be cross-checked.

    Every symbol lands in exactly one bucket, and `checkable + no_isin +
    not_on_bse + thin_on_bse + no_usable_close == nse_symbols`. A coverage
    report that does not add up is hiding a case.
    """

    nse_symbols: int = 0
    checkable: int = 0
    no_isin: int = 0
    """Not in NSE's own equity master - typically SME, or a series the
    master does not list."""
    not_on_bse: int = 0
    """Has an ISIN, but BSE did not trade it that day."""
    thin_on_bse: int = 0
    """Traded on BSE below MIN_BSE_TURNOVER. NOT checked - see the module
    docstring. This is usually the largest bucket and that is expected."""
    no_usable_close: int = 0
    """Matched and liquid, but the BSE close was missing or non-positive."""
    ambiguous_isin: int = 0
    """Symbol's ISIN is claimed by more than one NSE symbol, so the join
    could attach the wrong security. Excluded deliberately."""

    @property
    def balanced(self) -> bool:
        return (self.checkable + self.no_isin + self.not_on_bse
                + self.thin_on_bse + self.no_usable_close
                + self.ambiguous_isin) == self.nse_symbols

    def report(self) -> str:
        pct = (100.0 * self.checkable / self.nse_symbols) if self.nse_symbols else 0.0
        return (
            f"cross-source: {self.checkable} of {self.nse_symbols} NSE symbols "
            f"checkable against BSE ({pct:.1f}%)\n"
            f"  not in NSE's equity master : {self.no_isin}\n"
            f"  ambiguous ISIN             : {self.ambiguous_isin}\n"
            f"  not traded on BSE          : {self.not_on_bse}\n"
            f"  below the liquidity gate   : {self.thin_on_bse}   "
            f"(BSE turnover < Rs {MIN_BSE_TURNOVER:,.0f})\n"
            f"  no usable BSE close        : {self.no_usable_close}"
        )

    def __str__(self) -> str:
        return self.report()


def bse_reference(nse: pd.DataFrame, bse: pd.DataFrame, isin_map: IsinMap, *,
                  min_turnover: float = MIN_BSE_TURNOVER
                  ) -> tuple[pd.DataFrame, CrossCoverage]:
    """Build the `reference` frame quality.check expects, plus its coverage.

    `nse` needs `symbol`, `date` and `close`; `bse` is the output of
    desk.marketdata.sources.bse.parse_bhavcopy. Both may span several days.

    The returned frame is indexed by date with `symbol` and `close` columns,
    which is what `quality.check_panel` splits per symbol. Only rows that
    CLEARED the liquidity gate are in it - a caller must not be able to
    reconcile against a thin print by forgetting to filter.
    """
    for col in ("symbol", "date", "close"):
        if col not in nse.columns:
            raise KeyError(f"nse frame has no '{col}' column")
    for col in ("isin", "date", "close", "turnover"):
        if col not in bse.columns:
            raise KeyError(f"bse frame has no '{col}' column")

    left = nse[["symbol", "date", "close"]].copy()
    left["base"] = left["symbol"].astype(str).str.split(".").str[0].str.upper()
    left["isin"] = left["base"].map(isin_map.by_symbol)

    total = len(left)
    ambiguous = set(isin_map.ambiguous)
    amb = int(left["isin"].isin(ambiguous).sum())
    no_isin = int(left["isin"].isna().sum())

    keyed = left[left["isin"].notna() & ~left["isin"].isin(ambiguous)]

    right = bse[["isin", "date", "close", "turnover"]].rename(
        columns={"close": "bse_close", "turnover": "bse_turnover"})

    if keyed.empty or right.empty:
        # An empty side is a normal outcome, not an error: a holiday, a
        # universe of SME names with no ISIN, or a BSE fetch that legitimately
        # returned nothing. Merging through it would compare an all-NaN
        # float64 key against an object one and raise a pandas dtype error -
        # a confusing failure for the case that should simply report zero
        # coverage.
        merged = keyed.assign(bse_close=pd.Series(dtype="float64"),
                              bse_turnover=pd.Series(dtype="float64"))
    else:
        # Both keys as plain strings. NSE's mapped column can arrive as
        # object-with-NaN and BSE's as object; an implicit mismatch here
        # raises rather than silently failing to match, but only sometimes,
        # which is worse than either.
        merged = keyed.assign(isin=keyed["isin"].astype(str)).merge(
            right.assign(isin=right["isin"].astype(str)),
            on=["isin", "date"], how="left")

    # A merge can only ADD rows if the right side has duplicate keys. That
    # would mean BSE listed one ISIN twice on one day, which would make every
    # count below wrong - so it is caught rather than assumed away.
    if len(merged) != len(keyed):
        dupes = right[right.duplicated(["isin", "date"], keep=False)]
        raise ValueError(
            f"BSE frame has {dupes['isin'].nunique()} ISIN(s) appearing more "
            f"than once on the same date, which makes the join ambiguous: "
            f"{sorted(dupes['isin'].unique())[:5]}")

    not_on_bse = int(merged["bse_close"].isna().sum())
    matched = merged[merged["bse_close"].notna()]

    liquid = matched[matched["bse_turnover"].fillna(0.0) >= min_turnover]
    thin = len(matched) - len(liquid)

    usable = liquid[liquid["bse_close"] > 0]
    no_close = len(liquid) - len(usable)

    frame = pd.DataFrame({
        "symbol": usable["symbol"].to_numpy(),
        "close": usable["bse_close"].to_numpy(),
        "bse_turnover": usable["bse_turnover"].to_numpy(),
    }, index=pd.DatetimeIndex(pd.to_datetime(usable["date"]), name="date")
    ).sort_index()

    coverage = CrossCoverage(
        nse_symbols=total, checkable=len(usable), no_isin=no_isin,
        not_on_bse=not_on_bse, thin_on_bse=thin, no_usable_close=no_close,
        ambiguous_isin=amb)
    return frame, coverage


def reconcile(nse: pd.DataFrame, bse: pd.DataFrame, isin_map: IsinMap, *,
              tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
              min_turnover: float = MIN_BSE_TURNOVER
              ) -> tuple[pd.DataFrame, CrossCoverage]:
    """Every checkable (symbol, date) where the two exchanges disagree.

    The standalone counterpart to wiring `bse_reference` into quality.py:
    one frame a human can read, worst first. Empty is the expected result -
    and an empty frame with a coverage report saying only 759 of 3,485 names
    were checkable is a very different statement from "everything agrees".
    """
    ref, coverage = bse_reference(nse, bse, isin_map,
                                  min_turnover=min_turnover)
    if ref.empty:
        return (pd.DataFrame(columns=["symbol", "date", "nse_close",
                                      "bse_close", "diff_pct",
                                      "bse_turnover"]), coverage)

    left = nse[["symbol", "date", "close"]].copy()
    left["date"] = pd.to_datetime(left["date"])
    right = ref.reset_index().rename(columns={"close": "bse_close"})

    j = left.merge(right, on=["symbol", "date"], how="inner")
    j["diff_pct"] = (j["close"] - j["bse_close"]).abs() / j["bse_close"] * 100.0
    out = (j[j["diff_pct"] > tolerance_pct]
           .rename(columns={"close": "nse_close"})
           .sort_values("diff_pct", ascending=False)
           .reset_index(drop=True))
    return out[["symbol", "date", "nse_close", "bse_close", "diff_pct",
                "bse_turnover"]], coverage
