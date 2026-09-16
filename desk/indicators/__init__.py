"""Technical indicators as their own layer, separate from any one strategy.

Master prompt section 9 lists Indicators as a boundary distinct from
Strategies. Before this package, the only indicators in the project lived
inside strategies/india/strategy_lib.py, sized to what the six shared
strategies happened to need - RSI, ATR, ADX, Bollinger, Supertrend. A scanner
needs indicators none of those six use: VWAP, MACD, gap detection, market
structure, volatility compression, relative strength. That is the gap this
package fills.

This is NOT a refactor of strategy_lib.py. Its RSI/ATR/ADX/Bollinger already
carry audited fixes (Wilder smoothing, the exit-bar slice fix) and are what
the six strategies' eventual rewrite will import - touching that file without
a strategy rewrite driving it would risk regressing something nothing here
yet depends on. Where this package needs the same primitive (Wilder
smoothing, true range), it is reimplemented cleanly rather than copy-pasted,
and the two are expected to converge later, not now.

Every function here is a plain transform over a DataFrame/Series - no I/O, no
network, no strategy opinion. "Is this stock interesting" is Stage 1 of the
scanner's job; these functions only compute the numbers Stage 1 reads.
"""
