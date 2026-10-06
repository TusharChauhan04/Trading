"""The F&O segment: parsing it, and the one filter without which it is fiction.

WHY THIS EXISTS. The salvage map reopened a verdict this project closed too
early. F&O was ruled out at "about 12 lakh of capital", and that number was
right FOR FUTURES MARGIN - the cheapest single lot needed Rs 59,616 against a
Rs 15,000 position cap. Options are a different instrument: a defined-risk
spread costs the PREMIUM, not the margin.

And it matters more now than when the map was written, because of what this
project measured in the meantime. `cost_R = round_trip% / stop%` forces any
viable cash-equity structure toward a stop wider than about 8.4%, which forces
a multi-week hold. An option spread caps the loss at the premium WITHOUT
needing a wide stop, so it is the one structure the cost arithmetic does not
automatically kill.

WHAT MAKES IT TESTABLE AT ALL. NSE's option-chain API serves only today, so for
a while options looked un-backtestable. The F&O BHAVCOPY solves that: one
request per day returns every contract's OHLC, settlement price, open interest,
underlying price and lot size. 37,922 rows for 2026-09-29, 216 underlyings,
weekly and monthly expiries. Free, historical, daily - the same archive the
equity bhavcopy comes from.

THE FILTER WITHOUT WHICH ALL OF IT IS FICTION
---------------------------------------------
MOST OPTION CONTRACTS DO NOT TRADE ON A GIVEN DAY, and the file does not say so
in any obvious way - it reports a `ClsPric` for every contract regardless.
Measured on 2026-09-29 across 37,275 option rows:

    with TtlTradgVol > 0     20,054   53.8%
    with OpnIntrst  > 0      24,189   64.9%

So 46% of contracts have NO VOLUME, and for those `ClsPric` is a theoretical
settlement price NSE computes - not a price anyone paid. ABCAPITAL26OCT415PE on
that date has open, high and low all 0.00, zero volume, zero open interest, and
a close of 30.00. A backtest that reads `ClsPric` without checking volume would
buy and sell at prices that never existed, which is the same class of error as
the quarantined module that returns a random walk as OHLCV - except here the
data is real and the TRADEABILITY is invented.

`tradeable_only=True` is therefore the default, and `parse_fo_bhavcopy` reports
how many rows it dropped so the loss is visible rather than silent.

TWO MORE TRAPS IN THIS FILE, both stated because they are not obvious:

  - `ClsPric` IS NOT THE LAST TRADE. It is NSE's close; `LastPric` is the last
    traded price and `SttlmPric` is the settlement. For an option they can
    differ materially on a thin contract. This module keeps all three rather
    than choosing for the caller.
  - `NewBrdLotQty` IS THE LOT SIZE and it is NOT constant across underlyings or
    over time - NSE revises lots periodically. Every capital calculation must
    read it per row and per day. Hard-coding a lot size is how a backtest
    quietly sizes positions it could never have taken.

AND THE PRICES ARE NOT ARBITRAGE-COHERENT, WHICH IS THE SECOND TRAP
-------------------------------------------------------------------
A spread price built from two legs' CLOSING prices is not a price anyone could
have paid, because the two legs traded at different moments of the day. The
feasibility probe that first used this module found ADANIPORTS 1800/1820 calls
priced at 6.70 with the spot at 1821.96 - the spread was already worth its full
20, so that is 13.30 of free money, and free money in a backtest is always a
data artefact.

It is measurable. A call must cost at least its intrinsic value, and a
higher-strike call must never cost more than a lower-strike one. Measured on
4,352 adjacent strike pairs, forward expiries, volume > 0 and OI >= 500:

    price column   below intrinsic   strike-monotonicity violations
    close                   1.86%                            3.24%
    settle                  0.85%                            6.27%
    last                    1.77%                            3.70%

NO COLUMN IS CLEAN. Settlement is the most honest on intrinsic - NSE computes
it - and the WORST on monotonicity, because it is a model price rather than a
market one. So there is no "correct column to use"; there is only filtering.

`coherence_report` measures it and `parse_fo_bhavcopy(drop_incoherent=True)`
removes the offending rows. Any spread study on this data must do one or the
other, and must say which.

WHAT THIS MODULE DOES NOT DO. It does not propose or evaluate a strategy. It
parses the segment and says what was actually tradeable and which prices are
internally consistent.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["FO_COLUMNS", "INSTRUMENT_TYPES", "FoDay", "coherence_report",
           "parse_fo_bhavcopy"]

#: The columns the F&O bhavcopy carries, in order. 34 of them.
FO_COLUMNS = (
    "TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN",
    "TckrSymb", "SctySrs", "XpryDt", "FininstrmActlXpryDt", "StrkPric",
    "OptnTp", "FinInstrmNm", "OpnPric", "HghPric", "LwPric", "ClsPric",
    "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst",
    "ChngInOpnIntrst", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd",
    "SsnId", "NewBrdLotQty", "Rmks", "Rsvd1", "Rsvd2", "Rsvd3", "Rsvd4",
)

#: What `FinInstrmTp` means. Measured counts are for 2026-09-29.
INSTRUMENT_TYPES = {
    "STO": "stock option",      # 31,306
    "IDO": "index option",      #  5,969
    "STF": "stock future",      #    629
    "IDF": "index future",      #     18
}


@dataclass(frozen=True, slots=True)
class FoDay:
    """One day of the F&O segment, parsed and filtered."""

    contracts: pd.DataFrame
    as_of: pd.Timestamp
    rows_in_file: int
    dropped_untraded: int
    dropped_malformed: int
    tradeable_only: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def options(self) -> pd.DataFrame:
        return self.contracts[self.contracts["option_type"].isin(("CE", "PE"))]

    @property
    def futures(self) -> pd.DataFrame:
        return self.contracts[self.contracts["option_type"].isna()]

    @property
    def traded_share(self) -> float | None:
        """What fraction of the file survived the tradeability filter.

        None when nothing was read. Reported rather than assumed because it is
        the number that says how much of an apparent option chain is real: on
        the measured day it is 53.8%.
        """
        if not self.rows_in_file:
            return None
        return len(self.contracts) / self.rows_in_file

    def chain(self, symbol: str, expiry) -> pd.DataFrame:
        """One underlying's chain for one expiry, strikes ascending."""
        exp = pd.Timestamp(expiry)
        sel = self.options
        sel = sel[(sel["symbol"] == symbol.upper()) & (sel["expiry"] == exp)]
        return sel.sort_values(["strike", "option_type"])


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce")


def coherence_report(options: pd.DataFrame, *,
                     price: str = "close",
                     tol: float = 0.05) -> dict[str, float]:
    """How much of an option chain violates basic no-arbitrage bounds.

    Two checks, both of which a real market cannot breach by more than the
    bid-ask spread:
      - a call must cost at least max(spot - strike, 0)
      - a higher-strike call must not cost MORE than a lower-strike one

    Returns the share failing each. A chain with several percent failing is not
    broken data - it is NON-SIMULTANEOUS data, two legs quoted at different
    moments - and the distinction matters because the fix is filtering rather
    than re-fetching.
    """
    ce = options[options["option_type"] == "CE"].dropna(
        subset=["strike", price, "underlying"])
    if ce.empty:
        return {"rows": 0.0, "below_intrinsic": float("nan"),
                "non_monotonic": float("nan")}
    intrinsic = (ce["underlying"] - ce["strike"]).clip(lower=0)
    below = float((ce[price] < intrinsic - tol).mean())
    bad = pairs = 0
    for _, grp in ce.groupby(["symbol", "expiry"]):
        v = grp.sort_values("strike")[price].to_numpy(float)
        pairs += max(len(v) - 1, 0)
        bad += int((np.diff(v) > tol).sum())
    return {"rows": float(len(ce)), "below_intrinsic": below,
            "non_monotonic": bad / pairs if pairs else float("nan")}


def parse_fo_bhavcopy(raw: bytes | str, *, tradeable_only: bool = True,
                      min_open_interest: float = 0.0,
                      drop_incoherent: bool = False,
                      price: str = "close") -> FoDay:
    """Parse one F&O bhavcopy. Drops untraded contracts by default.

    `tradeable_only` keeps only rows with `TtlTradgVol > 0`. THE DEFAULT IS
    TRUE AND SHOULD STAY TRUE: 46% of contracts carry a `ClsPric` with no
    volume behind it, and reading those as prices is the whole trap this module
    exists to prevent. Pass False only to measure the segment's shape, never to
    feed a backtest.

    `min_open_interest` is a second, stricter gate - a contract can print one
    lot and still be untradeable at size. Off by default because the right
    floor depends on the position size being tested, and inventing one here
    would hide the choice.
    """
    data = raw
    if isinstance(raw, bytes):
        if raw[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise ValueError(
                        "the F&O zip contains no .csv member; NSE may have "
                        f"changed the archive layout (members: {z.namelist()})")
                data = z.read(names[0])
        data = bytes(data).decode("utf-8", "replace")

    frame = pd.read_csv(io.StringIO(data), dtype=str)
    rows_in_file = len(frame)
    missing = [c for c in ("TradDt", "TckrSymb", "FinInstrmTp", "ClsPric",
                           "TtlTradgVol", "NewBrdLotQty") if c not in frame]
    if missing:
        raise ValueError(
            f"the F&O bhavcopy is missing {missing}; NSE changed the schema. "
            f"Columns present: {list(frame.columns)[:12]}...")

    out = pd.DataFrame({
        "trade_date": pd.to_datetime(frame["TradDt"], errors="coerce"),
        "symbol": frame["TckrSymb"].str.strip().str.upper(),
        "instrument": frame["FinInstrmTp"].str.strip().str.upper(),
        "expiry": pd.to_datetime(frame["XpryDt"], errors="coerce"),
        "strike": _num(frame, "StrkPric"),
        # NaN for futures, which is correct - a future has no strike, and 0.0
        # would sort it among the deep-in-the-money options.
        "option_type": frame["OptnTp"].str.strip().str.upper().replace(
            {"": None}),
        "open": _num(frame, "OpnPric"),
        "high": _num(frame, "HghPric"),
        "low": _num(frame, "LwPric"),
        "close": _num(frame, "ClsPric"),
        "last": _num(frame, "LastPric"),
        "settle": _num(frame, "SttlmPric"),
        "prev_close": _num(frame, "PrvsClsgPric"),
        "underlying": _num(frame, "UndrlygPric"),
        "open_interest": _num(frame, "OpnIntrst"),
        "oi_change": _num(frame, "ChngInOpnIntrst"),
        "volume": _num(frame, "TtlTradgVol"),
        "turnover": _num(frame, "TtlTrfVal"),
        "trades": _num(frame, "TtlNbOfTxsExctd"),
        "lot_size": _num(frame, "NewBrdLotQty"),
        "contract": frame.get("FinInstrmNm", pd.Series(dtype=str)),
    })

    before = len(out)
    out = out[out["trade_date"].notna() & out["symbol"].str.len().gt(0)]
    dropped_malformed = before - len(out)

    notes: list[str] = []
    dropped_untraded = 0
    if tradeable_only:
        n0 = len(out)
        out = out[out["volume"].fillna(0) > 0]
        dropped_untraded = n0 - len(out)
        notes.append(
            f"dropped {dropped_untraded:,} of {n0:,} contracts with zero "
            f"volume - their ClsPric is a settlement NSE computed, not a "
            f"price anyone paid")
    else:
        notes.append(
            "TRADEABILITY FILTER OFF: rows include contracts with no volume, "
            "whose close is a theoretical settlement. Do not feed these to a "
            "backtest.")
    if min_open_interest > 0:
        n0 = len(out)
        out = out[out["open_interest"].fillna(0) >= min_open_interest]
        notes.append(f"dropped {n0 - len(out):,} contracts below "
                     f"{min_open_interest:,.0f} open interest")

    if drop_incoherent:
        # Only calls can be checked this way; puts need the mirrored bound and
        # the same filter applies, so they are left for the caller rather than
        # half-filtered here.
        ce = out["option_type"] == "CE"
        intrinsic = (out["underlying"] - out["strike"]).clip(lower=0)
        bad = ce & (out[price] < intrinsic - 0.05)
        n0 = len(out)
        out = out[~bad]
        notes.append(
            f"dropped {n0 - len(out):,} calls priced below intrinsic on "
            f"`{price}` - two legs quoted at different moments, not a real "
            f"price. Monotonicity is NOT filtered here; use coherence_report "
            f"and reject the pair at the point of building a spread.")

    # Lot size is per-row and per-day on purpose; a constant would be wrong.
    if bool(out["lot_size"].isna().any()):
        notes.append(
            f"{int(out['lot_size'].isna().sum())} rows carry no lot size, so "
            f"their capital requirement cannot be computed")

    as_of = (out["trade_date"].max() if len(out)
             else pd.to_datetime(frame["TradDt"], errors="coerce").max())
    return FoDay(contracts=out.reset_index(drop=True),
                 as_of=as_of, rows_in_file=rows_in_file,
                 dropped_untraded=dropped_untraded,
                 dropped_malformed=dropped_malformed,
                 tradeable_only=tradeable_only, notes=tuple(notes))
