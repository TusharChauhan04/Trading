"""TradingView Pine for the configuration this desk actually measured.

WHY THIS EXISTS RATHER THAN A DIRECT IMPORT
-------------------------------------------
OpenTerminalUI's `strategy_export/pine.py` works. It emits valid Pine v6 from a
declarative spec, and running it on the Donchian preset produces clean,
readable code. It is also wrong in two ways that matter here, and the first is
not a style complaint:

1. THE EXPORTED STRATEGY CAN NEVER ENTER. The spec says
   `close cross_above highest(high, 20)` and the exporter renders it as
   `ta.crossover(close, ta.highest(high, 20))`. In Pine, `ta.highest(high, 20)`
   INCLUDES the current bar, so that value is at least today's high, which is
   at least today's close. `close > highest` is arithmetically impossible and
   the crossover never fires. Paste that into TradingView and the strategy
   tester reports zero trades - which reads as "no setups in this market"
   rather than as a broken rule.

   This is the THIRD appearance of the same off-by-one: the desk hit it writing
   its own Donchian adapter, the preset translation hit it, and now the Pine
   export. A channel breakout has to compare against the channel as it stood
   BEFORE the current bar, every time, in every language.

2. IT EMITS THE CATALOGUED RISK BLOCK AND BOTH SIDES. The catalogue says a 5%
   stop with a 10% target, which measured -0.027R per trade on three years of
   NSE daily bars. What survived measurement was a 12% stop with a 24% target
   held about 20 sessions. And it emits `strategy.entry(..., strategy.short)`,
   which Indian retail cannot act on in cash equity beyond intraday. Exporting
   a losing configuration with an unusable side, to a chart someone will look at
   and believe, is worse than exporting nothing.

So the structure is borrowed and the content is ours: long only, the measured
stop and target, and the channel shifted so the rule can fire.

WHAT THE CHART WILL AND WILL NOT SHOW. TradingView's strategy tester applies its
own commission and slippage settings, not this project's cost model, so its P&L
will not match any figure in this repository. Set commission to 0.12% and
slippage to a few ticks to get within sight of the desk's 0.422% round trip. And
nothing on a chart can show the permutation result - that the same edge at a
60-session hold is indistinguishable from random entry dates. A chart makes a
strategy look convincing; it does not make it true.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PineStrategy", "MEASURED", "to_pine"]


@dataclass(frozen=True, slots=True)
class PineStrategy:
    """One exportable configuration, with its provenance attached.

    `measured_net_r` and `permutation_p` travel with the code and are written
    into the generated comment header, so a chart someone opens in six months
    carries the evidence rather than just the rules.
    """

    name: str
    channel_len: int
    stop_pct: float
    target_pct: float
    hold_bars: int
    measured_net_r: float
    permutation_p: float
    buffer_pct: float = 1.0
    """Require the close to clear the prior channel high by this much.

    MEASURED, and it is the only filter tried that helped. A 1% buffer raised
    month-weighted net R from +0.0428 to +0.0505 - about 18% - while keeping
    72% of trades, holding the permutation p at 0.0033, and WIDENING the edge
    over shuffled timing from +0.0189 to +0.0293. That last number is the one
    that matters: a filter can raise net R by keeping a luckier subset of the
    same edge, and only a wider timing edge says it added information.

    VOLUME CONFIRMATION WAS TRIED AND IT HURT, which is worth stating because
    it contradicts the oldest piece of breakout folklore. Requiring relative
    volume on the breakout bar made things monotonically worse and destroyed the
    signal outright at moderate thresholds:

        unfiltered    net +0.0428   p 0.0033   timing edge +0.0189
        rvol >= 1.25  net +0.0289   p 0.3422   timing edge +0.0034
        rvol >= 1.50  net +0.0288   p 0.4452   timing edge +0.0016

    So no volume gate is exported. The idea came from OpenTerminalUI's
    scanner_engine/detectors.py, which gates on exactly that - and measuring it
    was the right response to finding it."""

    note: str = ""

    @property
    def risk_reward(self) -> float:
        return self.target_pct / self.stop_pct


#: The two cells that passed BOTH the cost test and the permutation test. Their
#: numbers are the month-equal-weighted net R, which is the conservative of the
#: two poolings this project reports.
MEASURED: dict[str, PineStrategy] = {
    "donchian_20d": PineStrategy(
        name="Donchian 20 - desk measured",
        channel_len=20, stop_pct=12.0, target_pct=24.0, hold_bars=20,
        measured_net_r=0.0505, permutation_p=0.0033, buffer_pct=1.0,
        note="12% stop, 24% target, ~20 session hold, and the close must clear "
             "the prior 20-bar high by 1%. Survives a permutation test on entry "
             "timing; the same rule at a 60-session hold does NOT (p=0.57) and "
             "is market drift rather than signal. Volume confirmation was "
             "tried and made it worse - see buffer_pct.",
    ),
    "bollinger_20d": PineStrategy(
        name="Bollinger breakout - desk measured",
        channel_len=20, stop_pct=12.0, target_pct=24.0, hold_bars=20,
        measured_net_r=0.0407, permutation_p=0.0025, buffer_pct=0.0,
        note="Independent agreement with donchian_20d on the stop, target and "
             "hold, which is the main reason to take either seriously. "
             "buffer_pct is 0 because the buffer sweep was run on donchian "
             "ONLY - carrying donchian's 1% across to this rule would be "
             "claiming a measurement that was never taken.",
    ),
}


def to_pine(strategy: PineStrategy) -> str:
    """Pine v6 source for one measured configuration. Long only.

    The channel is `ta.highest(high, n)[1]` - the previous bar's channel - which
    is the whole correctness of a breakout rule rather than a detail of it. See
    the module docstring for what happens without the `[1]`.
    """
    s = strategy
    return f'''//@version=6
// {s.name}
//
// GENERATED BY desk/strategies/pine.py - do not hand-edit, regenerate.
//
// MEASURED on 3 years of NSE daily bars (2023-09 to 2026-09), 1,100 names
// after the desk's liquidity floor, non-overlapping trades, monthly cohorts:
//
//     net {s.measured_net_r:+.4f}R per trade after a 0.422% round-trip cost
//     permutation test on entry timing: p = {s.permutation_p:.4f}
//
// {s.note}
//
// NOT VALIDATED. t is about +0.95 and the Deflated Sharpe Ratio is 0.280
// against a >0.95 bar, on 36 monthly cohorts against the 60 the estimators
// need. Better than the 0.146 this cell measured before the buffer, and still
// nowhere near passing. A DRAFT configuration on a chart, not a system to trade.
//
// TradingView applies ITS OWN commission and slippage, so the tester's P&L
// will not match the figures above. Set commission ~0.12% and a few ticks of
// slippage to come close.
strategy("{s.name}", overlay=true, margin_long=100, margin_short=100,
     default_qty_type=strategy.percent_of_equity, default_qty_value=10,
     calc_on_every_tick=false, process_orders_on_close=true)

channelLen = input.int({s.channel_len}, "Channel length", minval=2)
bufferPct  = input.float({s.buffer_pct}, "Breakout buffer %", minval=0.0, step=0.25)
stopPct    = input.float({s.stop_pct}, "Stop %", minval=0.1, step=0.5)
targetPct  = input.float({s.target_pct}, "Target %", minval=0.1, step=0.5)
holdBars   = input.int({s.hold_bars}, "Max bars held", minval=1)

// THE [1] IS LOAD-BEARING. ta.highest(high, n) includes the CURRENT bar, so
// close can never exceed it and the crossover would never fire - the rule
// would report zero trades and look like a quiet market. Compare against the
// channel as it stood before this bar.
priorHigh = ta.highest(high, channelLen)[1]
trigger   = priorHigh * (1 + bufferPct / 100)

// THE BUFFER IS MEASURED, NOT COSMETIC. Requiring the close to clear the prior
// high by 1% raised net R about 18% and WIDENED the edge over shuffled entry
// timing, which is what adding information looks like. A close a paisa above
// the high is not a breakout.
longSignal = ta.crossover(close, trigger)

// LONG ONLY, deliberately: Indian retail cannot short cash equity beyond
// intraday, so a short side would be signals nobody here can act on.
if (longSignal and strategy.position_size == 0)
    strategy.entry("Long", strategy.long)

stopPrice   = strategy.position_avg_price * (1 - stopPct / 100)
targetPrice = strategy.position_avg_price * (1 + targetPct / 100)
if (strategy.position_size > 0)
    strategy.exit("Exit", from_entry="Long", stop=stopPrice, limit=targetPrice)

// The hold limit is part of the measured rule, not a safety net. At a 12% stop
// the 24% target needs time: it was hit 10.6% of the time at 20 bars and 26.6%
// at 60 - and the 60-bar version is the one the permutation test rejected.
barsHeld = strategy.position_size > 0 ? bar_index - strategy.opentrades.entry_bar_index(0) : 0
if (barsHeld >= holdBars)
    strategy.close("Long", comment="time")

plot(trigger, "Trigger ({s.buffer_pct}% above prior high)", color=color.new(color.orange, 0))
plotshape(longSignal, "Breakout", shape.triangleup, location.belowbar,
     color=color.new(color.teal, 0), size=size.tiny)
'''
