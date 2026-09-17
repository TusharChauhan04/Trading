"""Replaying the funnel over history to find out whether the edge is real."""

from desk.backtest.costs import CostModel, TradeCosts, ZERO_COSTS
from desk.backtest.engine import BacktestResult, DayResult, run_backtest
from desk.backtest.simulate import (
    ExitSimulation, SimulatedTrade, simulate_trade,
)

__all__ = [
    "BacktestResult", "CostModel", "DayResult", "ExitSimulation",
    "SimulatedTrade", "TradeCosts", "ZERO_COSTS", "run_backtest",
    "simulate_trade",
]
