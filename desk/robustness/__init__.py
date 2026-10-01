"""Deflated statistics: how much of a result survives admitting the search.

The desk's own arithmetic answers "did this make money". This package
answers the harder question - "would it still look good if I admitted how
many things I tried before it". Those are different questions, and this
project has already been fooled by the gap between them once: an hourly
factor scan produced a best cell of +0.150R at t 6.83, which became t 1.84
once overlapping trades were removed and DSR 0.08 once the 615 tried cells
were declared.

Borrowed from agents/openterminal_ui, which implements Bailey & Lopez de
Prado properly - PSR and MinTRL from "The Sharpe Ratio Efficient Frontier"
(2012), DSR from "The Deflated Sharpe Ratio" (2014). Re-deriving that
arithmetic would add risk for no gain.

TWO TESTS, DIFFERENT QUESTIONS, AND A RESULT SHOULD SURVIVE BOTH
----------------------------------------------------------------
`deflated_sharpe` punishes THE SEARCH: would the best of N tries still look
good if I admitted N? It works through Bailey & Lopez de Prado's asymptotics,
and asymptotics need observations - on 36 monthly cohorts it returns
`verdict="insufficient"`, which is a real limit and not a bug.

`signal_timing_test` punishes THE TIMING: did the rule enter on better days
than the same signals shuffled within the same stocks? It builds its null from
this data using the strategy's own trade mechanics, so its p-value is exact at
any sample size - which is precisely where DSR runs out.

Passing one and failing the other is informative rather than contradictory. A
strategy can time entries well and still be the best of 600 coin flips.
"""

from desk.backtest.permutation import (
    PermutationResult,
    outcome_lattice,
    signal_timing_test,
)
from desk.robustness.scorecard import (
    RobustnessReport,
    available,
    deflated_sharpe,
)

__all__ = ["PermutationResult", "RobustnessReport", "available",
           "deflated_sharpe", "outcome_lattice", "signal_timing_test"]
