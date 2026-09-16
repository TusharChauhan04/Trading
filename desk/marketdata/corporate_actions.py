"""NSE/BSE corporate actions and point-in-time price adjustment.

This is the highest-risk module in the project. An unadjusted 1:5 split reads
as a -80% single-day return; a strategy that "buys the dip" will find dozens of
them and report a spectacular, entirely false edge.

THREE RULES THIS MODULE ENFORCES
--------------------------------
1. POINT-IN-TIME. ``adjust(df, actions, as_of=T)`` applies only actions whose
   ex-date is on or before T. An action that has not happened yet cannot be
   used to reshape history you were trading through. This is what makes a
   backtest honest, and it is the single reason this module exists rather
   than a one-line ``yfinance auto_adjust=True``.

2. RAW AND ADJUSTED ARE BOTH KEPT. Adjusted prices are correct for RETURNS and
   wrong for anything absolute - a price floor, a tick size, a share count.
   Consumers must say which they want, so both are returned.

3. INDIAN BONUS RATIOS ARE NOT US RATIOS. In India a "2:1 bonus" means two
   free shares for every one held, so one share becomes THREE. Getting this
   backwards produces a 33% error that looks like a real price move. The
   constructors below refuse to accept a bare ratio for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

PRICE_COLS = ("open", "high", "low", "close")


class ActionType(str, Enum):
    SPLIT = "split"
    BONUS = "bonus"
    DIVIDEND = "dividend"
    RIGHTS = "rights"          # recorded, not auto-adjusted - see note below
    DEMERGER = "demerger"      # recorded, not auto-adjusted


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One action. ``factor`` is the number a share count is MULTIPLIED by.

    split 1 -> 5      factor 5.0
    bonus 1:1         factor 2.0   (1 held + 1 free)
    bonus 2:1         factor 3.0   (1 held + 2 free)   <-- India convention
    dividend          factor 1.0, amount = rupees per share
    """
    symbol: str
    ex_date: date
    type: ActionType
    factor: float = 1.0
    amount: float = 0.0
    note: str = ""

    # ---- named constructors, because bare ratios are how people get this wrong
    @classmethod
    def split(cls, symbol: str, ex_date: date, old_shares: int, new_shares: int, **kw):
        """A 1-for-5 split: old_shares=1, new_shares=5."""
        if old_shares <= 0 or new_shares <= 0:
            raise ValueError("share counts must be positive")
        return cls(symbol, ex_date, ActionType.SPLIT, factor=new_shares / old_shares, **kw)

    @classmethod
    def bonus(cls, symbol: str, ex_date: date, free_shares: int, per_held: int, **kw):
        """Indian convention. A 2:1 bonus is free_shares=2, per_held=1 -> factor 3."""
        if free_shares <= 0 or per_held <= 0:
            raise ValueError("bonus ratio parts must be positive")
        return cls(symbol, ex_date, ActionType.BONUS,
                   factor=(per_held + free_shares) / per_held, **kw)

    @classmethod
    def dividend(cls, symbol: str, ex_date: date, rupees_per_share: float, **kw):
        if rupees_per_share <= 0:
            raise ValueError("dividend must be positive")
        return cls(symbol, ex_date, ActionType.DIVIDEND, amount=rupees_per_share, **kw)

    @property
    def adjusts_price(self) -> bool:
        return self.type in (ActionType.SPLIT, ActionType.BONUS, ActionType.DIVIDEND)


# --------------------------------------------------------------------------
# Adjustment
# --------------------------------------------------------------------------

def adjustment_factors(
    index: pd.DatetimeIndex,
    actions: list[CorporateAction],
    as_of: date | None = None,
    include_dividends: bool = False,
    close: pd.Series | None = None,
) -> pd.Series:
    """Back-adjustment multiplier per bar, in (0, 1].

    Prices BEFORE an ex-date are multiplied by 1/factor so the series is
    continuous across the event. Prices on and after it are untouched, which
    keeps the most recent price equal to the real traded price.

    ``as_of`` is the point-in-time cut. Actions after it are ignored entirely.
    """
    usable = [a for a in actions if a.adjusts_price]
    if as_of is not None:
        usable = [a for a in usable if a.ex_date <= as_of]
    if not usable:
        return pd.Series(1.0, index=index, dtype="float64")

    if include_dividends and close is None:
        raise ValueError("dividend adjustment needs the close series")

    # `index[:pos] < ex` for every position < pos is exactly what a boolean
    # mask `index < ex` selects - but a mask is an O(n) pandas .loc alignment
    # per action, while searchsorted + a positional numpy slice is an O(log n)
    # lookup plus an O(pos) contiguous write. Measured 119x on a 2500-row
    # frame; on a full-universe run (2000 symbols x ~30 actions) the boolean
    # form cost ~38-58s, the positional form ~0.3-0.5s. Requires the index be
    # sorted - it always is by construction upstream (quality.py's own
    # INDEX_ORDER check demands it), but this function had no guard of its
    # own, so a caller skipping that check would get a silently wrong slice
    # rather than a raised error. Fail closed instead.
    if not index.is_monotonic_increasing:
        raise ValueError(
            "adjustment_factors() requires a sorted DatetimeIndex - run "
            "desk.marketdata.quality.check() first, which catches this"
        )

    factors_arr = np.ones(len(index), dtype="float64")
    # Only materialize the numpy copy when a dividend can actually use it -
    # every real caller today passes close= unconditionally (adjust() does),
    # so without this guard the O(n) conversion ran on every call regardless
    # of include_dividends, exactly the unused work this rewrite exists to cut.
    close_arr = (
        close.to_numpy()
        if include_dividends and close is not None
        and any(a.type is ActionType.DIVIDEND for a in usable)
        else None
    )

    for a in sorted(usable, key=lambda x: x.ex_date):
        ex = pd.Timestamp(a.ex_date)
        pos = index.searchsorted(ex, side="left")
        if pos == 0:
            continue

        if a.type in (ActionType.SPLIT, ActionType.BONUS):
            factors_arr[:pos] /= a.factor

        elif a.type is ActionType.DIVIDEND and include_dividends:
            # Standard total-return treatment: scale prior prices by
            # (close_prev - dividend) / close_prev using the last close
            # strictly before the ex-date.
            prior_tail = close_arr[:pos]
            if prior_tail.size == 0 or prior_tail[-1] <= 0:
                continue
            ref = float(prior_tail[-1])
            if a.amount >= ref:
                continue                      # implausible; leave series alone
            factors_arr[:pos] *= (ref - a.amount) / ref

    return pd.Series(factors_arr, index=index, dtype="float64")


def adjust(
    df: pd.DataFrame,
    actions: list[CorporateAction],
    as_of: date | None = None,
    include_dividends: bool = False,
) -> pd.DataFrame:
    """Return a frame carrying BOTH raw and adjusted prices.

    Adds ``adj_open/high/low/close``, ``adj_volume`` and ``adj_factor``.
    Original columns are never overwritten - rule 2 in the module docstring.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("adjust() needs a DatetimeIndex; call set_index('date') first")
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing price columns: {missing}")

    out = df.copy()
    f = adjustment_factors(
        out.index, actions, as_of=as_of,
        include_dividends=include_dividends, close=out["close"],
    )
    out["adj_factor"] = f
    for c in PRICE_COLS:
        out[f"adj_{c}"] = out[c] * f
    if "volume" in out.columns:
        # Share count moves inversely to price on splits and bonuses.
        out["adj_volume"] = out["volume"] / f.replace(0, np.nan)
    return out


# --------------------------------------------------------------------------
# Detection - catches a corporate-action list that is INCOMPLETE
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Discontinuity:
    date: date
    prev_close: float
    open_: float
    gap_pct: float
    implied_factor: float
    nearest_ratio: str

    def __str__(self) -> str:
        return (f"{self.date}  gap {self.gap_pct:+.1f}%  "
                f"implied factor {self.implied_factor:.3f} (~{self.nearest_ratio})")


_COMMON = {
    2.0: "1:1 bonus or 1-for-2 split", 3.0: "2:1 bonus or 1-for-3 split",
    4.0: "3:1 bonus or 1-for-4 split", 5.0: "1-for-5 split",
    10.0: "1-for-10 split", 1.5: "1:2 bonus", 2.5: "3:2 bonus",
}


def find_discontinuities(
    df: pd.DataFrame,
    threshold_pct: float = 20.0,
    known: list[CorporateAction] | None = None,
    symbol: str | None = None,
) -> list[Discontinuity]:
    """Flag overnight gaps large enough to suggest a MISSING corporate action.

    An Indian cash-equity scrip cannot legitimately gap more than its circuit
    band (widest common band is 20%), so anything beyond that with no action on
    file is either an unrecorded split/bonus or bad data. Either way a human
    should look before a backtest runs on it.

    `symbol` scopes `known` to this instrument. Without it, matching is by date
    alone - and one share's split will then "explain" a different share's bad
    print on the same day. Pass it whenever the actions could belong to more
    than one instrument, which is always, once a panel is involved.
    """
    if "close" not in df or "open" not in df:
        raise ValueError("need open and close columns")

    relevant = [a for a in (known or []) if a.adjusts_price]
    if symbol is not None:
        relevant = [a for a in relevant if a.symbol == symbol]
    known_dates = {a.ex_date for a in relevant}
    prev_close = df["close"].shift(1)
    gap = (df["open"] / prev_close - 1.0) * 100.0

    # A Python `for ts, g in gap.items()` row loop costs ~2-3us/row regardless
    # of how few rows actually exceed the threshold - 5.2ms at 2500 rows, and
    # this function also runs from inside quality._check_returns on every
    # check() call, so that cost is paid twice per symbol on a daily scan.
    # A vectorized prefilter finds the (normally handful of) candidates in
    # ~0.01ms; the Python loop then runs only over those, where it belongs -
    # object construction, not a threshold scan, is the part that needs
    # per-row logic. NaN (the first bar, from shift(1)) compares False in
    # numpy just as `pd.isna` skipped it explicitly before, so no bar is
    # newly included or excluded by dropping that check.
    gap_arr = gap.to_numpy()
    candidates = np.flatnonzero(np.abs(gap_arr) >= threshold_pct)

    out: list[Discontinuity] = []
    for i in candidates:
        ts = df.index[i]
        d = ts.date() if hasattr(ts, "date") else ts
        if d in known_dates:
            continue                                  # already explained
        pc, op = float(prev_close.iloc[i]), float(df["open"].iloc[i])
        implied = pc / op if op > 0 else float("nan")
        nearest = min(_COMMON, key=lambda k: abs(k - implied)) if op > 0 else None
        label = _COMMON.get(nearest, "unknown") if nearest and abs(nearest - implied) < 0.15 else "unexplained"
        out.append(Discontinuity(d, pc, op, float(gap_arr[i]), implied, label))
    return out


def assert_clean(df: pd.DataFrame, actions: list[CorporateAction], symbol: str = "") -> None:
    """Hard gate for the backtest harness. Refuses rather than guesses.

    Scopes `known` to `symbol` via find_discontinuities' own symbol filter -
    without it, this exact function would reopen the cross-symbol
    contamination bug (one share's split "explaining" a different share's
    unexplained gap) the moment it is ever called on a multi-symbol panel with
    a shared action list, which is precisely how a backtest harness would use
    a hard gate like this.
    """
    bad = find_discontinuities(df, known=actions, symbol=symbol or None)
    if bad:
        lines = "\n  ".join(str(b) for b in bad[:10])
        raise ValueError(
            f"{symbol or 'series'} has {len(bad)} unexplained price discontinuit"
            f"{'y' if len(bad) == 1 else 'ies'} - refusing to backtest.\n  {lines}\n"
            "Add the missing corporate action(s) or fix the data before continuing."
        )


# --------------------------------------------------------------------------
# loading what `python -m desk.marketdata.refresh actions` wrote
# --------------------------------------------------------------------------

class ActionLoadError(Exception):
    """A corporate-action file exists but cannot be trusted. Raised rather
    than skipped: a file we half-understand is more dangerous than one that
    is simply absent, because the caller will believe the symbol was
    checked."""


def load_actions(
    directory: str | Path,
    symbols: list[str] | set[str] | None = None,
) -> tuple[list["CorporateAction"], list[str]]:
    """Read the per-symbol JSON files the refresh CLI writes.

    Returns (actions, symbols_without_a_file). That second element is the
    whole point of the signature: the scanner needs to know WHICH names it
    could not check for splits, because "no actions on file" and "no file"
    are different facts and only one of them is reassuring. A name with no
    file has not been cleared - it has not been looked at.

    `symbols` may be given in either canonical (`RELIANCE.NS`) or bare
    (`RELIANCE`) form; files are named by the bare symbol.
    """
    import json

    root = Path(directory)
    wanted: dict[str, str] | None = None
    if symbols is not None:
        # bare name -> the caller's spelling, so the returned "missing" list
        # comes back in the form the caller actually uses.
        wanted = {s.split(".")[0].upper(): s for s in symbols}

    actions: list[CorporateAction] = []
    seen: set[str] = set()

    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            bare = path.stem.upper()
            if wanted is not None and bare not in wanted:
                continue
            seen.add(bare)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ActionLoadError(f"{path.name} is unreadable: {exc}") from exc
            if not isinstance(payload, dict):
                raise ActionLoadError(
                    f"{path.name} must contain a JSON object, got "
                    f"{type(payload).__name__}"
                )
            symbol = payload.get("symbol") or f"{path.stem}.NS"
            for raw in payload.get("actions") or []:
                actions.append(_action_from_json(path.name, symbol, raw))

    missing = ([orig for bare, orig in wanted.items() if bare not in seen]
               if wanted is not None else [])
    return actions, sorted(missing)


def _action_from_json(filename: str, symbol: str, raw: dict) -> "CorporateAction":
    if not isinstance(raw, dict):
        raise ActionLoadError(f"{filename}: an action entry is not an object")
    try:
        ex = date.fromisoformat(raw["ex_date"])
        kind = ActionType(raw["type"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionLoadError(
            f"{filename}: unusable action entry {raw!r} ({exc})"
        ) from exc

    factor = raw.get("factor", 1.0)
    amount = raw.get("amount", 0.0)
    try:
        factor = float(factor if factor is not None else 1.0)
        amount = float(amount if amount is not None else 0.0)
    except (TypeError, ValueError) as exc:
        raise ActionLoadError(
            f"{filename}: non-numeric factor/amount in {raw!r}"
        ) from exc
    if not np.isfinite(factor) or factor <= 0:
        raise ActionLoadError(
            f"{filename}: factor {factor!r} is not a positive finite number - "
            f"a share count multiplied by it would be meaningless"
        )
    return CorporateAction(symbol=symbol, ex_date=ex, type=kind,
                           factor=factor, amount=amount,
                           note=str(raw.get("note", "")))
