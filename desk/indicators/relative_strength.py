"""Relative strength vs a benchmark (e.g. Nifty 50) or a sector index.

Named explicitly in the master prompt's Stage 1 scanner filters
("relative strength vs Nifty", "sector-relative performance") and had no
implementation anywhere in the project before this module.
"""

from __future__ import annotations

import pandas as pd


def rs_line(close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    """The classic Mansfield-style RS line: price / benchmark, aligned on
    the shared index. A RISING line means outperforming the benchmark
    regardless of whether the stock itself is up or down that day - the
    point of relative strength is separating "up" from "up more than the
    market", which an absolute price chart cannot show.
    """
    price, bench = close.align(benchmark_close, join="inner")
    return price / bench


def rs_rank(rs: pd.Series, period: int = 63) -> pd.Series:
    """Percentile rank (0-100) of TODAY'S rs_line value within its own
    trailing `period` window (63 sessions ~ one quarter). An RS ratio has no
    natural absolute scale - "1.2" means nothing on its own - so it is only
    interpretable relative to where it has itself recently been.
    """
    return rs.rolling(period).apply(
        lambda w: 100.0 * (w <= w.iloc[-1]).mean(), raw=False)
