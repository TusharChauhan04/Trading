"""Walking the funnel forward through history.

WHAT MAKES THIS TRUSTWORTHY IS NOT IN THIS FILE
------------------------------------------------
The point-in-time guarantees are already structural, one layer down, and
this module deliberately adds no new ones:

    BarStore          never OPENS a snapshot dated after as_of
    FilingStore       filters by filename before reading a filing
    FundamentalsCache selects the table by filename, never a later one
    EventCalendar     refuses a snapshot fetched after the scan date

So a backtest here is just: pick a date, ask the funnel what it would have
said, write it down, and walk the answer forward through prices it could
not see. If any of those guarantees were merely runtime filters, this loop
would be where look-ahead crept in.

NO LLM. EVER. IN THIS LOOP.
---------------------------
`client=None` is hard-coded, not a default a caller can override. Two
reasons, and the second is the serious one:

  1. Hundreds of paid calls per run.
  2. A model trained on data past the simulated date partly REMEMBERS the
     answer. It is look-ahead that no amount of careful plumbing can audit,
     because it lives in the weights rather than in a file we control.

A backtest of this system is therefore a backtest of its deterministic
~98%, and the result must be read as such - Stage 3 can only ever have
removed names, so the live system takes a SUBSET of these trades. That
makes this an upper bound on trade count, not a prediction of it.

WHAT IS STILL NOT MODELLED, stated because a silent omission reads as a
feature that works:
  - Position-level portfolio interaction beyond max_trades per day. Open
    positions from earlier days do not block later ones.
  - Capital is not compounded; every day sizes against the same figure.
  - No intraday data, so every exit inherits simulate.py's daily-bar
    ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from desk.backtest.costs import CostModel
from desk.backtest.simulate import SimulatedTrade, simulate_trade
from desk.contracts.enums import Regime

__all__ = ["BacktestResult", "DayResult", "run_backtest"]


@dataclass(slots=True)
class DayResult:
    as_of: date
    considered: int = 0
    proposed: int = 0
    no_trade_reason: str | None = None
    error: str | None = None
    """Set when the funnel could not run at all - a missing snapshot, a bad
    read. Counted separately from a genuine NO TRADE, because 'we looked
    and declined' and 'we could not look' are different facts and averaging
    them together understates both."""

    @property
    def is_no_trade(self) -> bool:
        return self.error is None and self.proposed == 0


@dataclass(slots=True)
class BacktestResult:
    start: date
    end: date
    days: list[DayResult] = field(default_factory=list)
    trades: list[SimulatedTrade] = field(default_factory=list)
    costs: CostModel = field(default_factory=CostModel)
    not_taken: dict[str, int] = field(default_factory=dict)
    """Why a proposed trade never became a simulated one - no session
    after the decision, no price history. Kept because a plan whose trades
    cannot be entered is a finding, not a blank."""

    # -- the honest denominators ------------------------------------------

    @property
    def sessions(self) -> int:
        return len(self.days)

    @property
    def scanned(self) -> int:
        return sum(1 for d in self.days if d.error is None)

    @property
    def failed_days(self) -> int:
        return sum(1 for d in self.days if d.error is not None)

    @property
    def no_trade_days(self) -> int:
        return sum(1 for d in self.days if d.is_no_trade)

    @property
    def ambiguous(self) -> int:
        """Trades decided by the both-levels-touched rule rather than by
        the data. THE number to read before any other."""
        return sum(1 for t in self.trades if t.ambiguous)

    @property
    def gapped(self) -> int:
        return sum(1 for t in self.trades if t.gapped)

    # -- results ----------------------------------------------------------

    def net_pnl(self, trade: SimulatedTrade) -> float:
        c = self.costs.round_trip(entry=trade.entry_price,
                                  exit_=trade.exit_price, qty=trade.qty)
        return trade.gross_pnl - c.total

    @property
    def gross_pnl(self) -> float:
        return sum(t.gross_pnl for t in self.trades)

    @property
    def total_costs(self) -> float:
        return sum(self.costs.round_trip(entry=t.entry_price,
                                         exit_=t.exit_price,
                                         qty=t.qty).total
                   for t in self.trades)

    @property
    def net_pnl_total(self) -> float:
        return self.gross_pnl - self.total_costs

    @property
    def r_multiples(self) -> list[float]:
        return [t.r_multiple for t in self.trades if t.r_multiple is not None]

    @property
    def wins(self) -> int:
        return sum(1 for r in self.r_multiples if r > 0)

    @property
    def win_rate(self) -> float | None:
        rs = self.r_multiples
        return (100.0 * self.wins / len(rs)) if rs else None

    @property
    def expectancy_r(self) -> float | None:
        """Average R per trade. The primary number.

        R rather than rupees because it is the only measure comparable
        across names: 2R on a 40-rupee stock and 2R on a 4,000-rupee one
        are the same result and the rupee figures are not.
        """
        rs = self.r_multiples
        return (sum(rs) / len(rs)) if rs else None

    @property
    def total_r(self) -> float:
        return sum(self.r_multiples)

    @property
    def max_drawdown_r(self) -> float:
        """Worst peak-to-trough on the cumulative R curve, in R."""
        peak = running = worst = 0.0
        for t in sorted(self.trades, key=lambda x: x.exit_date):
            r = t.r_multiple
            if r is None:
                continue
            running += r
            peak = max(peak, running)
            worst = min(worst, running - peak)
        return worst

    def by_exit(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.trades:
            out[t.exit_reason] = out.get(t.exit_reason, 0) + 1
        return out

    def report(self) -> str:
        lines = [
            f"Backtest {self.start} to {self.end}",
            f"  sessions            {self.sessions} "
            f"({self.scanned} scanned, {self.failed_days} could not run)",
            f"  NO TRADE days       {self.no_trade_days}",
            f"  trades simulated    {len(self.trades)}",
        ]
        if self.not_taken:
            for why, n in sorted(self.not_taken.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {n} proposed but not taken: {why}")
        if not self.trades:
            lines.append("  no trades to measure")
            return "\n".join(lines)

        exp = self.expectancy_r
        lines += [
            f"  win rate            {self.win_rate:.1f}%  "
            f"({self.wins} of {len(self.r_multiples)})",
            f"  expectancy          {exp:+.3f}R per trade",
            f"  total               {self.total_r:+.1f}R",
            f"  max drawdown        {self.max_drawdown_r:.1f}R",
            f"  gross P&L           Rs {self.gross_pnl:,.0f}",
            f"  costs               Rs {self.total_costs:,.0f}  "
            f"({self.costs.round_trip_pct():.2f}% round trip)",
            f"  NET P&L             Rs {self.net_pnl_total:,.0f}",
            "  exits: " + ", ".join(f"{k} {v}" for k, v in
                                    sorted(self.by_exit().items())),
        ]
        if self.ambiguous:
            pct = 100.0 * self.ambiguous / len(self.trades)
            lines.append(
                f"  AMBIGUOUS           {self.ambiguous} of {len(self.trades)} "
                f"({pct:.0f}%) exits had the stop AND target touched on the "
                f"same bar and were resolved as STOPS. Daily bars cannot say "
                f"which came first; that much of this result is an "
                f"assumption, not a measurement.")
        if self.gapped:
            lines.append(f"  gapped exits        {self.gapped} filled at an "
                         f"open beyond the intended level")
        return "\n".join(lines)


def run_backtest(store, *, start: date, end: date, capital: float = 1_000_000,
                 max_trades: int = 3, lookback: int = 120,
                 holding_days: int = 5, regime: Regime = Regime.UNKNOWN,
                 target_r: float | None = None, stop_atrs: float = 2.0,
                 costs: CostModel | None = None,
                 journal=None, progress=None) -> BacktestResult:
    """Replay the funnel over every session in [start, end].

    `store` is a BarStore. `journal`, if given, receives a Decision per day
    - the same record the live desk writes, so a backtest and a live run
    are audited by identical machinery rather than by two code paths that
    can drift.
    """
    from desk.api.main import _run_funnel

    result = BacktestResult(start=start, end=end,
                            costs=costs or CostModel())
    available = store.available_days()
    sessions = [d for d in available if start <= d <= end]
    if not sessions:
        return result

    # The forward prices, loaded ONCE for the whole run rather than per
    # trade. Two reasons: a read per proposal is hundreds of parquet scans,
    # and - the one that actually bit - `series()` lives on History, not on
    # BarStore. Calling store.series() raised AttributeError, a bare except
    # swallowed it, and all 60 proposals were reported as "no price history
    # for this symbol". A real bug wearing a data problem's clothes.
    #
    # Loading the future here is safe for the same reason simulate.py can
    # see it: every decision is already fixed before this frame is touched.
    forward = store.history(as_of=available[-1], start=sessions[0],
                            columns=["open", "high", "low", "close"])

    for i, day in enumerate(sessions, 1):
        if progress:
            progress(i, len(sessions), day)

        try:
            # today=day, not the wall clock. The risk engine refuses data
            # older than 5 days, so against a real "today" every historical
            # session is stale by definition and the whole replay returns
            # NO TRADE while appearing to work.
            summary = _run_funnel(day, regime=regime, capital=capital,
                                  max_trades=max_trades, lookback=lookback,
                                  portfolio=None, today=day,
                                  target_r=target_r, stop_atrs=stop_atrs,
                                  holding_days=holding_days)
        except Exception as exc:                      # noqa: BLE001
            result.days.append(DayResult(as_of=day, error=str(exc)[:200]))
            continue

        if summary is None:
            result.days.append(
                DayResult(as_of=day, error="no snapshot for this session"))
            continue

        result.days.append(DayResult(
            as_of=day, considered=summary.considered,
            proposed=len(summary.trades),
            no_trade_reason=summary.no_trade_reason))

        if journal is not None:
            _journal_day(journal, day, summary, regime, capital)

        for trade in summary.trades:
            if trade.stop is None or not trade.qty:
                result.not_taken["no stop or no size"] = \
                    result.not_taken.get("no stop or no size", 0) + 1
                continue
            # Deliberately NOT trimmed to `day` - see simulate.py. An
            # unknown symbol yields an empty frame, which simulate_trade
            # reports as "no price history" for real rather than as a
            # disguised exception.
            bars = forward.series(trade.symbol)
            sim = simulate_trade(bars, symbol=trade.symbol, decided_on=day,
                                 planned_entry=trade.entry, stop=trade.stop,
                                 target=trade.target, qty=trade.qty,
                                 horizon_days=holding_days)
            if sim.trade is None:
                why = sim.reason_not_taken or "unknown"
                result.not_taken[why] = result.not_taken.get(why, 0) + 1
            else:
                result.trades.append(sim.trade)

    return result


def _journal_day(journal, day, summary, regime, capital) -> None:
    """Record the day, and never let a journal problem kill the run.

    A JournalError here almost always means this day was already recorded -
    a re-run over the same range - and that refusal is the journal working
    correctly. It must not abort a backtest.
    """
    from desk.journal import Decision, JournalError, TradeRecord
    from desk.journal.models import IST
    from datetime import datetime

    trades = tuple(
        TradeRecord(symbol=t.symbol,
                    stance=getattr(t.stance, "value", str(t.stance)),
                    entry=t.entry, stop=t.stop, target=t.target, qty=t.qty,
                    capital_at_risk=t.capital_at_risk,
                    rationale=t.rationale)
        for t in summary.trades)
    try:
        journal.record(Decision(
            as_of=day, recorded_at=datetime.now(IST),
            regime=getattr(regime, "value", str(regime)), trades=trades,
            no_trade_reason=summary.no_trade_reason,
            universe_scanned=summary.universe_scanned,
            survived_stage0=summary.survived_stage0,
            survived_stage1=summary.survived_stage1,
            survived_stage2=summary.survived_stage2,
            considered=summary.considered,
            caveats=tuple(summary.caveats),
            coverage_note=summary.coverage_note, capital=capital,
            note="recorded by a backtest run, not a live session"))
    except JournalError:
        pass
