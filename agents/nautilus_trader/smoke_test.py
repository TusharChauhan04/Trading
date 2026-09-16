"""Verify the NautilusTrader install by running a real backtest that trades.

Doubles as the agent's health check: if this passes, the compiled extensions
loaded, the engine runs, orders fill, and the reports generate.

IMPORTANT: run this from ANY directory EXCEPT `upstream/`. Running Python with
the nautilus_trader repo root on sys.path (or as cwd) shadows the compiled
site-packages install with the unbuilt source tree and fails with
`ModuleNotFoundError: No module named 'nautilus_trader.core.data'`.

    agents/nautilus_trader/.venv/Scripts/python.exe agents/nautilus_trader/smoke_test.py
"""

from __future__ import annotations

import sys
from decimal import Decimal

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

N_BARS = 10_000
SEED = 42


def synthetic_bars(n: int = N_BARS) -> pd.DataFrame:
    """A trending random walk, so an EMA crossover actually fires."""
    rng = np.random.default_rng(SEED)
    # Drift alternates every ~1200 bars so fast/slow EMAs cross repeatedly.
    drift = np.where((np.arange(n) // 1200) % 2 == 0, 1.2e-6, -1.2e-6)
    steps = rng.normal(0.0, 4e-5, n) + drift
    close = 1.10 * np.exp(np.cumsum(steps))

    spread = rng.uniform(2e-5, 8e-5, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000, 20_000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
    )


def main() -> int:
    print(f"nautilus_trader smoke test — generating {N_BARS:,} synthetic bars\n")

    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD")
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(synthetic_bars())

    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")),
    )
    venue = Venue("SIM")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, USD)],
        base_currency=USD,
        default_leverage=Decimal(10),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(
        EMACross(
            EMACrossConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                trade_size=Decimal(100_000),
                fast_ema_period=10,
                slow_ema_period=20,
            ),
        ),
    )

    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    account = engine.trader.generate_account_report(venue)

    print("─" * 62)
    print(f"  bars processed : {len(bars):,}")
    print(f"  order fills    : {len(fills):,}")
    print(f"  positions      : {len(positions):,}")

    if not positions.empty and "realized_pnl" in positions:
        pnl = positions["realized_pnl"].map(lambda v: float(str(v).split(" ")[0])).sum()
        wins = (positions["realized_pnl"].map(lambda v: float(str(v).split(" ")[0])) > 0).sum()
        print(f"  realized PnL   : {pnl:,.2f} USD")
        print(f"  win rate       : {wins}/{len(positions)} ({wins / len(positions):.1%})")

    if not account.empty:
        print(f"  final balance  : {account.iloc[-1].get('total', 'n/a')}")
    print("─" * 62)

    engine.dispose()

    ok = len(fills) > 0 and len(positions) > 0
    print("\nRESULT:", "PASS — engine ran, orders filled, positions closed" if ok
          else "FAIL — no trades were generated")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
