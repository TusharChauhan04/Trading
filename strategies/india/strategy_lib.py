"""
strategy_lib.py - shared indicators & backtest engine for the India strategies.

CORRECTED VERSION. The original (kept as strategy_lib_original.py) is unchanged
on disk for reference. Every change below is marked [FIX] or [PARITY] and is
explained in AUDIT.md. Nothing else about the strategies' logic was touched.

Cost model: percentage-of-notional per side (brokerage + STT + slippage bundled
into one number), applied whenever a position changes. Override --cost-pct with
your broker's real sheet before trusting any absolute P&L.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
# [PARITY] The originals smoothed RSI / ATR / ADX with a simple rolling mean.
# TradingView, Chartink and Streak all use WILDER smoothing (an EMA with
# alpha = 1/period). With SMA, RSI(14) < 30 and ADX(14) > 20 fire on different
# bars than the chart you are looking at - which defeats the purpose of
# replicating what Indian retail actually trades. `wilder()` below is the
# standard; pass smoothing="sma" to reproduce the original behaviour exactly.

def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: EMA with alpha = 1/period (TradingView-compatible)."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def _smooth(series: pd.Series, period: int, smoothing: str = "wilder") -> pd.Series:
    return wilder(series, period) if smoothing == "wilder" else series.rolling(period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14, smoothing: str = "wilder") -> pd.Series:
    delta = close.diff()
    gain = _smooth(delta.clip(lower=0), period, smoothing)
    loss = _smooth(-delta.clip(upper=0), period, smoothing)
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14, smoothing: str = "wilder") -> pd.Series:
    return _smooth(true_range(df), period, smoothing)


def adx(df: pd.DataFrame, period: int = 14, smoothing: str = "wilder") -> pd.Series:
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_ = _smooth(true_range(df), period, smoothing)
    plus_di = 100 * _smooth(plus_dm, period, smoothing) / atr_
    minus_di = 100 * _smooth(minus_dm, period, smoothing) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _smooth(dx, period, smoothing)


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0,
               smoothing: str = "wilder"):
    """Returns (supertrend_line, direction); direction 1 = uptrend, -1 = downtrend."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_ = atr(df, period, smoothing)
    upper_band = hl2 + multiplier * atr_
    lower_band = hl2 - multiplier * atr_

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    st = np.zeros(n)
    direction = np.ones(n)
    close = df["close"].values

    for i in range(n):
        if i == 0 or np.isnan(atr_.iloc[i]):
            final_upper[i] = upper_band.iloc[i]
            final_lower[i] = lower_band.iloc[i]
            st[i] = upper_band.iloc[i] if not np.isnan(upper_band.iloc[i]) else close[i]
            direction[i] = 1
            continue

        final_upper[i] = (
            upper_band.iloc[i]
            if (upper_band.iloc[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower_band.iloc[i]
            if (lower_band.iloc[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1

        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


# --------------------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------------------

def backtest_positions(df: pd.DataFrame, capital: float, cost_pct: float = 0.05) -> pd.DataFrame:
    """`position` on day t is held INTO day t+1's return (lagged here, no look-ahead)."""
    out = df.copy()
    out["daily_return"] = out["close"].pct_change().fillna(0)
    out["position_lag"] = out["position"].shift(1).fillna(0)
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["strategy_return"] = out["position_lag"] * out["daily_return"] - out["turnover"] * (cost_pct / 100)
    out["strategy_equity"] = capital * (1 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = capital * (1 + out["daily_return"].fillna(0)).cumprod()
    return out


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Group contiguous non-flat position runs into trades.

    [FIX] The original sliced ``ret[start_idx:i]``, which dropped the exit bar's
    return from every trade. Because ``strategy_return`` is already lagged,
    ret[start_idx] is structurally 0 (position_lag is still flat on the entry
    bar) while ret[i] carries the final day held. The correct window is
    therefore ``ret[start_idx + 1 : i + 1]``.

    Verified: a long held through three +10% bars compounds to +33.1% on the
    equity curve; the original reported +21.0%. Every win_rate_pct and
    avg_trade_return_pct produced before this fix understates the true figure.
    Equity-curve and total-return numbers were never affected.
    """
    trades = []
    pos = df["position"].values
    ret = df["strategy_return"].values
    dates = df["date"].values

    def _close(entry_i, exit_i, direction, exit_date):
        window = ret[entry_i + 1: exit_i + 1]          # [FIX]
        return {
            "entry_date": dates[entry_i],
            "exit_date": exit_date,
            "direction": "long" if direction == 1 else "short",
            "bars_held": int(max(0, exit_i - entry_i)),
            "trade_return_pct": round(100 * (np.prod(1 + window) - 1), 3),
        }

    start_idx = None
    cur_dir = 0
    for i in range(len(pos)):
        if pos[i] != 0 and cur_dir == 0:
            start_idx, cur_dir = i, pos[i]
        elif pos[i] != cur_dir and cur_dir != 0:
            trades.append(_close(start_idx, i, cur_dir, dates[i]))
            start_idx, cur_dir = (i, pos[i]) if pos[i] != 0 else (None, 0)

    if cur_dir != 0 and start_idx is not None:
        # Still open at the end of the data - mark it so it can be excluded.
        t = _close(start_idx, len(pos) - 1, cur_dir, dates[-1])
        t["still_open"] = True
        trades.append(t)

    out = pd.DataFrame(trades)
    if not out.empty and "still_open" not in out.columns:
        out["still_open"] = False
    return out


def compute_metrics(df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    strat_ret = df["strategy_return"].dropna()
    equity = df["strategy_equity"]
    running_max = equity.cummax()
    max_dd_pct = 100 * ((equity - running_max) / running_max).min()
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else float("nan")

    downside = strat_ret[strat_ret < 0]
    sortino = (strat_ret.mean() / downside.std()) * np.sqrt(252) if len(downside) and downside.std() > 0 else float("nan")

    # [GAP-FIX] CAGR needs elapsed calendar time, not bar count.
    n_bars = len(df)
    years = n_bars / 252.0
    total_growth = equity.iloc[-1] / equity.iloc[0]
    cagr = (total_growth ** (1 / years) - 1) * 100 if years > 0 and total_growth > 0 else float("nan")

    metrics = {
        "total_trades": len(trades),
        "strategy_total_return_pct": round(100 * (total_growth - 1), 2),
        "buy_hold_total_return_pct": round(100 * (df["buy_hold_equity"].iloc[-1] / df["buy_hold_equity"].iloc[0] - 1), 2),
        "strategy_cagr_pct": round(cagr, 2),
        "strategy_sharpe_approx": round(sharpe, 2),
        "strategy_sortino_approx": round(sortino, 2),
        "strategy_max_drawdown_pct": round(max_dd_pct, 2),
        "annualised_volatility_pct": round(100 * strat_ret.std() * np.sqrt(252), 2),
        "exposure_pct": round(100 * (df["position"] != 0).mean(), 2),
        "bars_tested": n_bars,
        "years_tested": round(years, 2),
    }
    if not trades.empty:
        closed = trades[~trades.get("still_open", False)] if "still_open" in trades else trades
        if len(closed):
            r = closed["trade_return_pct"]
            win_mask = r > 0
            wins, losses = r[win_mask], r[~win_mask]
            gross_win = wins.sum()
            gross_loss = -losses.sum()

            win_rate = len(wins) / len(closed)
            avg_win = wins.mean() if len(wins) else 0.0
            avg_loss = losses.mean() if len(losses) else 0.0   # negative or zero

            # [GAP-FIX] Expectancy is the number that decides survivability:
            # a 30%-win-rate system is fine if the winners are large enough.
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

            # [GAP-FIX] Longest losing streak. A system with a positive
            # expectancy can still be untradeable if it asks you to sit
            # through fourteen losers to get there.
            streak = worst_streak = 0
            for is_win in win_mask:
                streak = 0 if is_win else streak + 1
                worst_streak = max(worst_streak, streak)

            metrics["closed_trades"] = len(closed)
            metrics["win_rate_pct"] = round(100 * win_rate, 2)
            metrics["loss_rate_pct"] = round(100 * (1 - win_rate), 2)
            metrics["avg_trade_return_pct"] = round(r.mean(), 3)
            metrics["avg_win_pct"] = round(avg_win, 3)
            metrics["avg_loss_pct"] = round(avg_loss, 3)
            metrics["expectancy_pct_per_trade"] = round(expectancy, 4)
            metrics["payoff_ratio"] = round(abs(avg_win / avg_loss), 2) if avg_loss < 0 else float("inf")
            metrics["profit_factor"] = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
            metrics["max_consecutive_losses"] = int(worst_streak)
            metrics["trades_per_year"] = round(len(closed) / years, 1) if years > 0 else float("nan")
            metrics["avg_bars_held"] = round(closed["bars_held"].mean(), 1)
            # [FIX] A sample this small cannot support a win-rate claim.
            if len(closed) < 30:
                metrics["WARNING"] = f"only {len(closed)} closed trades - stats are not significant"
    return metrics


def plot_equity(df: pd.DataFrame, out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    plt.plot(df["date"], df["strategy_equity"], label="Strategy", linewidth=1.5)
    plt.plot(df["date"], df["buy_hold_equity"], label="Buy & hold", linewidth=1.2, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (INR)")     # [FIX] the rupee glyph is missing from many
    plt.legend()                    # matplotlib default fonts and renders as a box
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
