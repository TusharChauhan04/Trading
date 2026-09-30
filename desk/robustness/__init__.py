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
"""

from desk.robustness.scorecard import (
    RobustnessReport,
    available,
    deflated_sharpe,
)

__all__ = ["RobustnessReport", "available", "deflated_sharpe"]
