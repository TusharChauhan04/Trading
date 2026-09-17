"""What a round trip actually costs on an Indian equity delivery trade.

A backtest without costs is not an optimistic backtest, it is a different
question being answered. Measured, with this module's own defaults:

    statutory charges   0.1222%   (buy 0.0186 + sell 0.1036, STT dominates)
    slippage            0.3000%   (15bps a side, and it is a GUESS)
    ------------------------------
    round trip          0.4222%   of turnover

    with a full-service broker at 0.3%/side:   1.1302%

Note which line is larger. The statutory stack is arithmetic and nearly
fixed; SLIPPAGE IS THE DOMINANT TERM AND THE LEAST CERTAIN ONE, so a result
that only just clears zero is really a statement about the slippage
assumption rather than about the strategy. Against the ~2 ATR stops this
desk proposes - call it 4% of price - a 0.42% round trip is about a tenth
of the risk on every trade, every time. A thin but genuine edge is exactly
what that erases, and that is the case where the answer matters most.

THE INDIAN COST STACK, delivery equity, both sides unless stated
-----------------------------------------------------------------
    STT                 0.100%  on SELL only (delivery). The big one.
    Exchange txn        0.00297% NSE cash
    SEBI turnover       0.0001%
    Stamp duty          0.015%  on BUY only
    GST                 18% on (brokerage + exchange txn + SEBI)
    Brokerage           discount brokers are typically zero on delivery;
                        a full-service broker is 0.3-0.5% and would dwarf
                        everything above.

DEFAULTS HERE ASSUME A ZERO-BROKERAGE DISCOUNT BROKER, which is the common
Indian retail setup and the favourable end. Set `brokerage_pct` if that is
not the arrangement - the difference is not marginal.

SLIPPAGE IS SEPARATE AND USUALLY LARGER
---------------------------------------
Charges are arithmetic; slippage is a market fact. These are daily bars, so
fills are modelled at a session's open and the real fill is somewhere
around it. `slippage_bps` is applied AGAINST the trade on both sides - buys
fill higher, sells fill lower - and never in our favour, because a model
that is sometimes generous on fills will find edges that do not exist.

The 15bps default is a guess, not a measurement, and it is labelled as one.
Small caps and the illiquid end of the universe are materially worse; the
Stage 0 turnover floor is what keeps the tested universe on the side of
this being plausible.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CostModel", "TradeCosts", "ZERO_COSTS"]


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """Rupees, for one complete round trip."""

    entry_charges: float = 0.0
    exit_charges: float = 0.0
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0

    @property
    def total(self) -> float:
        return (self.entry_charges + self.exit_charges
                + self.entry_slippage + self.exit_slippage)

    def breakdown(self) -> str:
        return (f"charges Rs {self.entry_charges + self.exit_charges:.2f}, "
                f"slippage Rs {self.entry_slippage + self.exit_slippage:.2f}, "
                f"total Rs {self.total:.2f}")


@dataclass(frozen=True, slots=True)
class CostModel:
    """Percentages, expressed as percent (0.1 means 0.1%)."""

    stt_sell_pct: float = 0.100
    exchange_txn_pct: float = 0.00297
    sebi_turnover_pct: float = 0.0001
    stamp_duty_buy_pct: float = 0.015
    gst_pct: float = 18.0
    brokerage_pct: float = 0.0
    """Zero by default - a discount broker on delivery. A full-service
    broker at 0.3-0.5% per side would dominate every other line here, so
    this is the one worth checking against a real contract note."""

    slippage_bps: float = 15.0
    """Basis points per side, always against us. A GUESS, not a
    measurement - see the module docstring."""

    def charges(self, *, value: float, side: str) -> float:
        """Statutory and broker charges on one side, in rupees."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        value = abs(value)
        brokerage = value * self.brokerage_pct / 100.0
        exch = value * self.exchange_txn_pct / 100.0
        sebi = value * self.sebi_turnover_pct / 100.0
        gst = (brokerage + exch + sebi) * self.gst_pct / 100.0
        stt = value * self.stt_sell_pct / 100.0 if side == "sell" else 0.0
        stamp = value * self.stamp_duty_buy_pct / 100.0 if side == "buy" else 0.0
        return brokerage + exch + sebi + gst + stt + stamp

    def fill_price(self, quoted: float, *, side: str) -> float:
        """The quoted price moved AGAINST us by the slippage allowance.

        Never in our favour. A model that occasionally fills better than
        quoted will discover edges that exist only in the model.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        adj = quoted * self.slippage_bps / 10_000.0
        return quoted + adj if side == "buy" else quoted - adj

    def round_trip(self, *, entry: float, exit_: float, qty: int) -> TradeCosts:
        """Costs for one long round trip, using the QUOTED prices.

        Slippage is reported as its own rupee amount rather than being
        folded into the fills, so a result can say how much of the drag was
        statutory (unavoidable) and how much was execution (improvable).
        """
        if qty <= 0:
            return TradeCosts()
        entry_value = entry * qty
        exit_value = exit_ * qty
        return TradeCosts(
            entry_charges=self.charges(value=entry_value, side="buy"),
            exit_charges=self.charges(value=exit_value, side="sell"),
            entry_slippage=abs(self.fill_price(entry, side="buy") - entry) * qty,
            exit_slippage=abs(exit_ - self.fill_price(exit_, side="sell")) * qty,
        )

    def round_trip_pct(self) -> float:
        """Approximate round-trip drag as a percent of turnover.

        For sanity-checking a result rather than for pricing a trade: it
        assumes entry and exit values are equal, which they are not.
        """
        one_side_value = 100.0
        buy = self.charges(value=one_side_value, side="buy")
        sell = self.charges(value=one_side_value, side="sell")
        slip = 2 * one_side_value * self.slippage_bps / 10_000.0
        return buy + sell + slip


#: For isolating what costs are doing to a result. NOT a default: a
#: zero-cost backtest answers a question nobody is actually asking.
ZERO_COSTS = CostModel(stt_sell_pct=0.0, exchange_txn_pct=0.0,
                       sebi_turnover_pct=0.0, stamp_duty_buy_pct=0.0,
                       gst_pct=0.0, brokerage_pct=0.0, slippage_bps=0.0)
