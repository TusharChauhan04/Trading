"""What a round trip actually costs on an Indian equity delivery trade.

A backtest without costs is not an optimistic backtest, it is a different
question being answered. Measured, with this module's own defaults:

    statutory charges   0.1222%   (buy 0.0186 + sell 0.1036, STT dominates)
    slippage            0.3000%   (15bps a side, and it is a GUESS)
    ------------------------------
    round trip          0.4222%   of turnover

    ...PLUS A FLAT DP CHARGE OF ABOUT Rs 16 ON THE SELL SIDE, which is not a
    percentage and therefore not in that total. At a Rs 100,000 position it
    adds 0.016% and is ignorable; at Rs 500 it adds 3.186% and is the entire
    story. See `dp_charge_rupees` - it is the only line here that does not
    scale, and it decides whether a small account can trade at all.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:            # imported lazily in sized_for to keep this
    from desk.backtest.slippage import SlippageModel   # module dependency-free

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

    dp_charge_rupees: float = 15.93
    """DP charge: a FLAT fee per scrip on the SELL side, independent of size.
    Zerodha's 13.50 plus 18% GST; most discount brokers sit between 15 and 22.

    THE ONLY NON-PERCENTAGE LINE IN THIS MODEL, AND IT CHANGES THE ANSWER FOR
    SMALL ACCOUNTS. Every other charge scales with turnover, so the round trip
    was a constant 0.422% at any size. This one does not, and it dominates
    below a few thousand rupees:

        position    pct costs   DP as %   round trip   cost_R at a 15% stop
        Rs    500      0.422%     3.186%       3.608%                0.2405
        Rs  2,000      0.422%     0.796%       1.219%                0.0812
        Rs  5,000      0.422%     0.319%       0.741%                0.0494
        Rs 10,000      0.422%     0.159%       0.582%                0.0388
        Rs 100,000     0.422%     0.016%       0.438%                0.0292

    The measured gross edge on this desk's best cell is about +0.08R, so the
    BREAK-EVEN POSITION IS ABOUT Rs 2,048: below it this single flat fee eats
    the entire edge before the market does anything. No strategy survives that,
    and a backtest that omits it will report one that does.

    Set to 0.0 only for an account where DP charges genuinely do not apply.
    """

    slippage_bps: float = 15.0
    """Basis points per side, always against us. A GUESS, not a
    measurement - see the module docstring. `sized_for` replaces it with a
    size-aware figure where position value and turnover are known."""

    def cost_r(self, *, stop_pct: float,
               position_value: float | None = None) -> float:
        """Round-trip cost expressed in units of the risk taken.

        cost_R = round_trip% / stop%. The identity every result in this project
        turns on: quantity cancels for the percentage lines, so only stop WIDTH
        matters - EXCEPT for the DP charge, which is where position size
        re-enters and why a small account cannot simply widen its stop out of
        the problem.
        """
        if stop_pct <= 0:
            raise ValueError("stop_pct must be positive to express cost in R")
        return self.round_trip_pct(position_value) / stop_pct

    def breakeven_position(self, *, stop_pct: float,
                           gross_edge_r: float) -> float | None:
        """The smallest position at which `gross_edge_r` survives costs.

        None when the percentage charges alone already exceed the edge, which
        means no position size rescues it and the stop must widen instead.
        """
        pct_only = self.round_trip_pct(None)
        budget = gross_edge_r * stop_pct - pct_only
        if budget <= 0 or not self.dp_charge_rupees:
            return None
        return self.dp_charge_rupees / budget * 100.0

    @classmethod
    def sized_for(cls, *, position_value: float, turnover_value: float,
                  model: "SlippageModel | None" = None,
                  **kwargs: float) -> "CostModel":
        """A CostModel whose slippage reflects how much of the day we take.

        The flat 15bps default is the dominant term in every marginal result
        this desk produces, and it ignores order size entirely. Measured on the
        real post-Stage-0 universe (median turnover Rs 34 crore), a Rs 1 lakh
        position is 0.03% of a session and the modelled slippage is 2.6bps a
        side - not 15.

        THIS IS A MODEL, NOT A CALIBRATION. The impact coefficient is
        OpenTerminalUI's and has never been fitted to NSE, and the bid-ask
        spread cannot be measured from bhavcopy at all. Two things it does not
        capture, both of which push the true figure UP:

          - adverse selection. A breakout entry buys into strength, which is
            systematically the worst moment to be a buyer. Daily bars cannot
            see it, and for a breakout strategy it is the biggest unmodelled
            term.
          - the real spread on the thin end of the universe.

        So use it to show that a result is NOT merely an artefact of an
        inflated slippage guess. Do not use it to claim a result is safe: a
        conclusion that survives only at the modelled figure and dies at 15bps
        is still a conclusion about slippage.

        `turnover_value` should be the PREVIOUS session's traded value in
        rupees - bhavcopy reports `turnover_lacs`, so multiply by 100,000.
        """
        from desk.backtest.slippage import SlippageModel, participation

        model = model or SlippageModel()
        rate = participation(position_value, turnover_value)
        return cls(slippage_bps=model.bps(rate), **kwargs)

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

    def round_trip_pct(self, position_value: float | None = None) -> float:
        """Approximate round-trip drag as a percent of turnover.

        For sanity-checking a result rather than for pricing a trade: it
        assumes entry and exit values are equal, which they are not.

        `position_value` ADDS THE FLAT DP CHARGE, which is the only line here
        that does not scale with size. Omit it and you get the
        percentage-only figure - correct in the limit of a large position and
        badly wrong for a small one. Every walk-forward in this project up to
        2026-10-06 omitted it, so every one of those results is optimistic by
        the DP share at whatever position size was implied.
        """
        one_side_value = 100.0
        buy = self.charges(value=one_side_value, side="buy")
        sell = self.charges(value=one_side_value, side="sell")
        slip = 2 * one_side_value * self.slippage_bps / 10_000.0
        pct = buy + sell + slip
        if position_value and position_value > 0 and self.dp_charge_rupees:
            pct += self.dp_charge_rupees / position_value * 100.0
        return pct


#: For isolating what costs are doing to a result. NOT a default: a
#: zero-cost backtest answers a question nobody is actually asking.
ZERO_COSTS = CostModel(stt_sell_pct=0.0, exchange_txn_pct=0.0,
                       sebi_turnover_pct=0.0, stamp_duty_buy_pct=0.0,
                       gst_pct=0.0, brokerage_pct=0.0, slippage_bps=0.0)
