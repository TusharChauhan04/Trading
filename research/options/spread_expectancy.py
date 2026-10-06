"""Do defined-risk option spreads have any edge? The first options measurement.

WHY THIS IS THE QUESTION. Cash equity is closed at this operator's capital: a
flat Rs 16 DP charge is 3.2% of a Rs 500 trade and the measured edge is worth
about 1.2%. Options have no DP charge, and with a zero-brokerage broker the
round trip costs about 0.15% of premium - cost_R of 0.0015, which is
twenty-six times better than a Rs 10,000 equity position. So the costs stop
mattering. What has never been measured is whether there is an EDGE.

The +0.08R this project quotes was measured on EQUITY BREAKOUTS. Assuming it
transfers to options would be exactly the unexamined assumption this project
exists to avoid.

WHY THE DESIGN IS UNUSUALLY CLEAN. Held to expiry, a bull call spread's payoff
is DETERMINISTIC - it depends only on where the underlying settles:

    payoff = max(0, min(S - K1, K2 - K1)) - premium

So no exit price has to be modelled, and the arbitrage-incoherence that
contaminates option closes (two legs quoted at different moments) touches only
the ENTRY premium, which the coherence filter handles. Compare that to the
equity work, where every exit needed a stop/target lattice.

THE PRE-REGISTERED ACCEPTANCE TEST, fixed before the run: GROSS expectancy in R
must be positive. Costs at a zero-brokerage broker are ~0.0015R and cannot
rescue a negative gross - the same logic that closed intraday, where the
required round-trip cost came out negative. If gross is negative here, no
broker and no sizing helps and options are closed too.

ENTRY IS UNCONDITIONAL - every (symbol, expiry) pair offering a near-1:2 spread
on the entry date. That measures the STRUCTURE rather than a signal, which is
the right first question: a signal could select better spreads, but it would
have to beat whatever the structure gives on its own.

TWO CAVEATS THAT ARE NOT HANDLED AND MUST TRAVEL WITH ANY RESULT:

  PHYSICAL SETTLEMENT. NSE STOCK options have been physically settled since
  2019. A spread held to expiry with only the long leg in the money means
  taking delivery - impossible at Rs 500 of capital. INDEX options are cash
  settled and have no such problem, so index and stock results are reported
  SEPARATELY and the stock figure is a structural measurement rather than
  something tradeable at this account size.

  NO EARLY EXIT. Real traders square off before expiry. Modelling that needs
  exit prices, which reintroduces the incoherence problem. Holding to expiry is
  the conservative, measurable version - and an edge that only appears with
  discretionary early exits is not an edge this desk could act on anyway.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd

STORE = Path("configs/fo")
RR_LOW, RR_HIGH = 1.65, 2.35          # "near 1:2"
MIN_OI = 500
ENTRY_DTE = (21, 35)                  # enter roughly a month before expiry
BROKERAGE_PER_ORDER = 0.0             # zero-brokerage broker; see module doc
STT_PCT, EXCH_PCT, GST_PCT = 0.1, 0.05, 18.0


def load_days() -> dict[pd.Timestamp, pd.DataFrame]:
    out = {}
    for f in sorted(STORE.glob("*.parquet")):
        d = pd.read_parquet(f)
        if d.empty:
            continue
        out[pd.Timestamp(d["trade_date"].iloc[0])] = d
    return out


def settlement(day_frame: pd.DataFrame, symbol: str) -> float | None:
    """The underlying's price on the expiry session, from the chain itself."""
    sel = day_frame[day_frame["symbol"] == symbol]
    if sel.empty:
        return None
    v = pd.to_numeric(sel["underlying"], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else None


def spreads_on(frame: pd.DataFrame, as_of: pd.Timestamp,
               side: str = "CE") -> list[dict]:
    """Every near-1:2 vertical spread available on this session.

    side="CE" is a BULL CALL spread: buy K1, sell K2>K1, profits as the
    underlying rises. side="PE" is the BEAR PUT mirror: buy K2, sell K1<K2,
    profits as it falls.

    THE MIRROR IS NOT OPTIONAL. A call spread is a LONG DIRECTIONAL BET, and
    the equal-weighted index fell 6.5% over the period this is run on - so a
    negative call-spread result is close to tautological, the same trap as
    "a long-only breakout loses when the market falls". Only running both sides
    separates "options do not work" from "buying calls in a down market does
    not work", and those are completely different conclusions.
    """
    o = frame[(frame["option_type"] == side)
              & (frame["open_interest"].fillna(0) >= MIN_OI)
              & (frame["volume"].fillna(0) > 0)].copy()
    if o.empty:
        return []
    o["dte"] = (o["expiry"] - as_of).dt.days
    o = o[o["dte"].between(*ENTRY_DTE)]
    rows = []
    for (sym, exp), g in o.groupby(["symbol", "expiry"]):
        g = g.dropna(subset=["strike", "close", "underlying"]).sort_values("strike")
        if len(g) < 3:
            continue
        spot = float(g["underlying"].iloc[0])
        lot = float(g["lot_size"].iloc[0] or 0)
        if not (spot > 0 and lot > 0):
            continue
        ks = g["strike"].to_numpy(float)
        px = g["close"].to_numpy(float)
        # COHERENCE: an option must cost at least its intrinsic value. Two legs
        # quoted at different moments produce free money otherwise.
        intrinsic = (np.maximum(spot - ks, 0.0) if side == "CE"
                     else np.maximum(ks - spot, 0.0))
        keep = px >= intrinsic - 0.05
        ks, px = ks[keep], px[keep]
        best = None
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                # CE: long the lower strike. PE: long the HIGHER strike, so the
                # premium is the other way round.
                prem = (px[i] - px[j]) if side == "CE" else (px[j] - px[i])
                width = ks[j] - ks[i]
                if prem <= 0 or width <= 0 or prem >= width:
                    continue              # prem >= width is also incoherent
                rr = (width - prem) / prem
                if not (RR_LOW <= rr <= RR_HIGH):
                    continue
                gap = abs(rr - 2.0)
                if best is None or gap < best[0]:
                    best = (gap, ks[i], ks[j], prem, width, rr)
        if best:
            _, k1, k2, prem, width, rr = best
            rows.append({"as_of": as_of, "symbol": sym, "expiry": exp,
                         "side": side,
                         "spot": spot, "K1": k1, "K2": k2, "premium": prem,
                         "width": width, "rr": rr, "lot": lot,
                         "outlay": prem * lot,
                         "index": sym in ("NIFTY", "BANKNIFTY", "FINNIFTY",
                                          "MIDCPNIFTY", "NIFTYNXT50")})
    return rows


def main() -> None:
    days = load_days()
    if not days:
        raise SystemExit(f"no F&O parquets under {STORE}")
    dates = sorted(days)
    print(f"{len(dates)} sessions  {dates[0].date()} -> {dates[-1].date()}",
          flush=True)

    trades = []
    for d in dates:
      for side in ("CE", "PE"):
        for s in spreads_on(days[d], d, side=side):
            exp = s["expiry"]
            if exp not in days:
                continue                      # expiry session not on disk
            S = settlement(days[exp], s["symbol"])
            if S is None:
                continue
            if side == "CE":
                payoff = max(0.0, min(S - s["K1"], s["width"])) - s["premium"]
            else:
                payoff = max(0.0, min(s["K2"] - S, s["width"])) - s["premium"]
            s["settle"] = S
            s["payoff"] = payoff
            s["r"] = payoff / s["premium"]
            fees = (2 * BROKERAGE_PER_ORDER * (1 + GST_PCT / 100) * 2
                    + s["outlay"] * (STT_PCT + EXCH_PCT) / 100)
            s["cost_r"] = fees / s["outlay"]
            trades.append(s)

    if not trades:
        raise SystemExit("no spreads could be matched to an expiry on disk")
    t = pd.DataFrame(trades)
    print(f"{len(t):,} spreads entered and held to expiry  "
          f"{t['symbol'].nunique()} underlyings  "
          f"{t['expiry'].nunique()} expiries\n", flush=True)

    def report(name: str, sub: pd.DataFrame) -> None:
        if sub.empty:
            print(f"{name:<22} no trades")
            return
        gross = sub["r"].mean()
        cost = sub["cost_r"].mean()
        wins = float((sub["r"] > 0).mean())
        maxp = float(np.where(sub["side"] == "CE",
                              sub["settle"] >= sub["K2"],
                              sub["settle"] <= sub["K1"]).mean())
        t_stat = (gross / sub["r"].std() * np.sqrt(len(sub))
                  if sub["r"].std() > 0 else float("nan"))
        print(f"{name:<22} n={len(sub):>6,}  gross {gross:>+7.4f}R  "
              f"cost {cost:>6.4f}R  net {gross - cost:>+7.4f}R  "
              f"win {wins:>5.1%}  maxprofit {maxp:>5.1%}  t {t_stat:>+5.2f}")

    print(f"{'':<22} {'':>8}  {'':>13}  {'':>12}  {'':>13}")
    report("ALL", t)
    report("  BULL call spreads", t[t["side"] == "CE"])
    report("  BEAR put spreads", t[t["side"] == "PE"])
    report("  index options", t[t["index"]])
    report("  stock options", t[~t["index"]])
    print()
    print("affordable at small capital (outlay <= Rs 2,000):")
    report("  all", t[t["outlay"] <= 2000])
    print()
    print("THE PRE-REGISTERED TEST: gross expectancy must be positive.")
    g = t["r"].mean()
    print(f"  gross = {g:+.4f}R  ->  "
          f"{'PASSES, measure further' if g > 0 else 'FAILS - no cost structure or sizing rescues a negative gross'}")
    print()
    print("by expiry, so one month cannot carry the result:")
    for exp, g2 in t.groupby("expiry"):
        print(f"  {exp.date()}  n={len(g2):>5,}  gross {g2['r'].mean():>+7.4f}R"
              f"  win {(g2['r'] > 0).mean():>5.1%}")


if __name__ == "__main__":
    main()
