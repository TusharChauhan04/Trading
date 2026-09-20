"""Stage 4 of the daily scanner funnel: setup proposal and the risk gate.

The last stage, and the one with veto power. It turns Stage 2's ranked
shortlist into concrete setups (entry, stop, target), runs every one of them
through `desk.risk.engine.size_position`, and returns the 0-3 trades that
survive - or NO TRADE, which is a successful outcome and the expected one on
most days.

THE RISK ENGINE IS NOT RE-IMPLEMENTED HERE. This module proposes prices and
hands them over; `size_position` owns every gate and every rejection reason.
Nothing in this file may loosen a limit, retry a rejected setup with a wider
stop, or convert a refusal into a warning. A high Stage 2 score buys a name
the right to be CONSIDERED, nothing more - and the engine does not know or
care what that score was.

LONG ONLY, and that is a legal constraint rather than a preference: Indian
retail cannot short cash equity beyond intraday. A short thesis needs the
F&O layer, which does not exist yet, so a bearish name belongs on the avoid
list rather than in a trade.

THE SETUP PROPOSER IS DELIBERATELY DUMB. Stop at `stop_atrs` x ATR below the
reference price, target at `target_r` x the stop distance above it - a 1:2
trade by default, meaning risk one rupee to make two.

THAT DEFAULT WAS 2.5 AND IT WAS MEASURED, NOT ARGUED ABOUT. The first real
backtest over 29 sessions took 60 trades and only THREE ever reached a 2.5R
target, while 23 took a full stop and 34 simply timed out at the 5-day
horizon. A target the market rarely reaches inside the holding period is not
a better trade - it is a trade that mostly expires, paying full price for
every loss and collecting almost none of the wins. 2.0 is closer to what a
5-day hold can actually deliver. It is configurable via DESK_RISK_REWARD and
should be re-measured, not assumed. It does
not look for support levels, prior swing lows or round numbers - those are
Stage 3's job once an LLM is reading the chart's context, and inventing them
deterministically here would produce confident-looking prices with nothing
behind them. What this arithmetic IS good for is answering the only question
Stage 4 needs answered: at a sane volatility-scaled stop, does this name pass
the risk gate at all? Most rejections (R:R floor, stop distance, portfolio
heat, liquidity) do not depend on the entry being perfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from desk.contracts.enums import RejectReason, Stance
from desk.settings import DEFAULT_RISK_REWARD
from desk.plan.models import PlanTrade
from desk.risk.engine import Portfolio, RiskConfig, Sizing, size_position
from desk.scanner.stage1 import Stage1Result
from desk.research.events import EventCalendar
from desk.scanner.stage2 import Stage2Result

__all__ = ["Stage4Result", "Setup", "run_stage4", "propose_setup"]

UNAVAILABLE_NO_CALENDAR = (
    "Event proximity not checked - no event calendar supplied, so a name "
    "reporting results inside the holding window was not screened out.",
)

UNAVAILABLE = (
    "Sector caps are not enforced per name - bhavcopy carries no sector "
    "classification, so every setup is sized as sector UNKNOWN and the "
    "sector cap can only ever bind against other UNKNOWN positions.",
    "Correlated-cluster caps are not enforced - no correlation matrix is "
    "computed yet, and the risk engine deliberately refuses to fall back to "
    "sector as a proxy (that would make the sector cap unreachable).",
    "Entry levels are volatility-derived, not structural. Support, prior "
    "swing lows and consolidation edges are Stage 3's job.",
)


@dataclass(frozen=True, slots=True)
class Setup:
    """A proposed trade, before the risk engine has seen it."""

    symbol: str
    stance: Stance
    entry: float
    stop: float
    target: float
    atr_pct: float
    adv_shares: float | None
    score: float
    rationale: str
    stop_basis: str = "atr"
    invalidation: str = ""

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)


@dataclass(slots=True)
class Stage4Result:
    as_of: date
    approved: list[PlanTrade] = field(default_factory=list)
    rejected: list[Sizing] = field(default_factory=list)
    """Every name the engine refused, with its named reason. Kept rather than
    discarded: "we looked at 40 names and every one failed the R:R floor" is
    a materially different day from "nothing was flagged", and a plan that
    cannot tell them apart is not explainable."""
    considered: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    """Names that never reached the engine because a setup could not be
    proposed at all (no ATR, no price)."""
    unavailable: list[str] = field(default_factory=lambda: list(UNAVAILABLE))

    @property
    def is_no_trade(self) -> bool:
        return not self.approved

    def no_trade_reason(self) -> str | None:
        """Why there is nothing to do today, in the plan's own words. None
        when there IS something to do."""
        if self.approved:
            return None
        if self.considered == 0:
            return ("NO TRADE - nothing reached the risk gate today. The "
                    "funnel produced no ranked candidate worth sizing.")
        counts: dict[str, int] = {}
        for s in self.rejected:
            for r in s.reasons:
                counts[r.value] = counts.get(r.value, 0) + 1
        top = ", ".join(f"{k} ({v})" for k, v in
                        sorted(counts.items(), key=lambda kv: -kv[1])[:3])
        return (f"NO TRADE - {self.considered} candidate(s) reached the risk "
                f"gate and every one was refused: {top}.")

    def summary(self) -> str:
        lines = [f"Stage 4: {self.considered} considered -> "
                 f"{len(self.approved)} approved ({self.as_of})"]
        for reason, n in sorted(self.skipped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  -{n:5} {reason}")
        counts: dict[str, int] = {}
        for s in self.rejected:
            for r in s.reasons:
                counts[r.value] = counts.get(r.value, 0) + 1
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  -{n:5} rejected: {reason}")
        for c in self.unavailable:
            lines.append(f"  UNAVAILABLE: {c}")
        return "\n".join(lines)


def propose_setup(
    symbol: str,
    feature_row: pd.Series,
    score: float,
    *,
    stop_atrs: float = 2.0,
    target_r: float = DEFAULT_RISK_REWARD,
    stop_buffer_atrs: float = 0.25,
    min_stop_atrs: float = 1.0,
) -> Setup | None:
    """Turn one Stage 1 feature row into a concrete long setup.

    Returns None - never a guess - when the inputs a stop depends on are
    missing. A setup with an invented stop is worse than no setup, because
    the risk engine will happily size it and every number downstream will
    look legitimate.

    THE STOP GOES WHERE THE TRADE IS WRONG. It is placed below the nearest
    level the setup depends on - the last confirmed swing low, else the
    20-session low - with a `stop_buffer_atrs` cushion so that a TEST of the
    level is not a stop-out. `stop_atrs` is now only the fallback for a name
    with no structure in the loaded history, and `Setup.stop_basis` says
    which of the two happened on every single trade.

    `min_stop_atrs` is the floor and the one place structure is overruled: a
    pivot half an ATR below entry is a real level, but a stop there sits
    inside the range the stock covers on an ordinary day and gets hit by
    drift. When that happens the stop widens to the floor and the setup says
    so in `invalidation` rather than reporting a structural stop that is not
    at the structure.

    NOTHING CAPS THE STOP FROM ABOVE HERE. A structural level 12% away
    produces a 12% stop, and `size_position` refuses it as STOP_TOO_WIDE.
    That refusal belongs to the risk engine: pulling the stop in to make a
    trade fit would put it somewhere the thesis is still intact, which is
    the exact arbitrary-percentage stop this function exists to replace.
    """
    close = feature_row.get("close")
    atr_pct = feature_row.get("atr_pct")
    if close is None or pd.isna(close) or close <= 0:
        return None
    if atr_pct is None or pd.isna(atr_pct) or atr_pct <= 0:
        return None

    atr_rupees = float(close) * float(atr_pct) / 100.0
    entry = float(close)

    level, basis, where = _invalidation_level(feature_row, entry)
    if level is None:
        # No structure in the loaded history. The ATR stop is the FALLBACK,
        # and it is named as one rather than presented as a decision.
        stop = entry - stop_atrs * atr_rupees
        invalidation = (
            f"No confirmed pivot or 20-session low below {entry:.2f} in the "
            f"loaded history, so there is no structural level to fail. Stop "
            f"is {stop_atrs}x ATR - a VOLATILITY stop, not an invalidation "
            f"level: it says how far this stock normally travels, not where "
            f"the setup would be wrong.")
    else:
        # Below the level, not at it. A stop resting exactly on a price the
        # whole market can see is the one place a wick is most likely to
        # reach, and being taken out by the test of a level that then holds
        # is the worst possible outcome - the thesis survives and the
        # position does not.
        stop = level - stop_buffer_atrs * atr_rupees
        invalidation = (
            f"Below {where}. A close under {level:.2f} breaks the level the "
            f"setup rests on; the stop sits {stop_buffer_atrs}x ATR beneath "
            f"it at {stop:.2f} so a test of the level is not a stop-out.")

        floor = entry - min_stop_atrs * atr_rupees
        if stop > floor:
            # The level is nearer than this stock's own daily noise. Honouring
            # it would place the stop inside the range price covers on an
            # ordinary day, so it would be hit by drift rather than by the
            # thesis failing. Widening is the honest move and it is DECLARED,
            # because the stop is no longer where the structure is.
            invalidation = (
                f"{where} sits at {level:.2f}, only "
                f"{(entry - level) / atr_rupees:.2f}x ATR below entry - "
                f"inside this stock's ordinary daily range, so a stop there "
                f"would be hit by noise rather than by the setup failing. "
                f"WIDENED to the {min_stop_atrs}x ATR noise floor at "
                f"{floor:.2f}, which is further than the structure requires.")
            stop = floor
            basis = "noise_floor"

    if stop <= 0:
        # Either an ATR wide enough to put the stop at or below zero, or a
        # structural level that is itself non-positive. Both mean the inputs
        # are nonsense for this name rather than that the trade is risky.
        return None
    target = entry + target_r * (entry - stop)

    vol = feature_row.get("volume")
    adv = None if vol is None or pd.isna(vol) else float(vol)

    bits = [f"Stage 2 score {score:.1f}"]
    for flag, text in (("unusual_volume", "unusual volume"),
                       ("near_52w_high", "near 52-week high"),
                       ("compressed", "volatility compressed"),
                       ("unusual_move", "unusual move")):
        if bool(feature_row.get(flag, False)):
            bits.append(text)
    bits.append(f"stop at {stop:.2f} ({basis}, "
                f"{(entry - stop) / atr_rupees:.2f}x ATR), target {target_r}R")

    return Setup(
        symbol=symbol, stance=Stance.BUY, entry=entry, stop=stop,
        target=target, atr_pct=float(atr_pct), adv_shares=adv, score=score,
        rationale="; ".join(bits), stop_basis=basis, invalidation=invalidation,
    )


def _invalidation_level(feature_row: pd.Series, entry: float
                        ) -> tuple[float | None, str, str]:
    """The nearest price level below `entry` that the setup depends on.

    Two candidates: the last confirmed swing low - where the market most
    recently showed it would defend this name - and the 20-session low, a
    range floor rather than a pivot anything actually turned at.

    THE NEARER ONE WINS, not the stronger one, because the question is where
    the setup FAILS FIRST. Price on its way down reaches the nearer level
    first, and that break is the first evidence available; a stop at the more
    significant level further below would sit through the whole of that
    evidence before acting. Ties go to the swing low, which is the better
    description of the same price.

    MEASURED, because the alternative was the obvious one and I built it
    first: preferring the swing low unconditionally changed the chosen level
    for only 14 of 371 names where both exist, but it produced a stop 152x
    ATR below entry on a name whose last pivot was 111 sessions old and whose
    ATR is a fraction of a percent. Nearest-first caps that at 18x. Neither
    rule changes how many names breach the 15% STOP_TOO_WIDE ceiling (42
    either way), so the pathological tail is the whole of the difference.

    A LEVEL AT OR ABOVE ENTRY IS NOT SUPPORT AND IS SKIPPED. Price below its
    own last pivot means that pivot is now resistance overhead, and a "stop"
    above the entry price would be an instant exit at a guaranteed loss. The
    same guard rejects a zero or negative level from a corrupt bar. That case
    is not exotic: 611 of 1,508 names on 2026-09-17 sat below their own last
    pivot.

    Returns (level, basis token, human phrase). The basis token is what the
    backtest groups by; `None` means no usable structure exists.
    """
    def usable(name: str) -> float | None:
        v = feature_row.get(name)
        if v is None or pd.isna(v):
            return None
        v = float(v)
        return v if 0 < v < entry else None

    swing, low20 = usable("swing_low"), usable("low_20")

    if swing is not None and (low20 is None or swing >= low20):
        age = feature_row.get("swing_low_age")
        when = ("" if age is None or pd.isna(age)
                else f" set {int(age)} sessions ago")
        return (swing, "swing_low",
                f"the last confirmed swing low at {swing:.2f}{when}")

    if low20 is not None:
        return (low20, "low_20", f"the 20-session low at {low20:.2f}")

    return (None, "atr", "")


def run_stage4(
    stage2: Stage2Result,
    stage1: Stage1Result,
    *,
    cfg: RiskConfig,
    portfolio: Portfolio | None = None,
    max_trades: int = 3,
    candidates: int = 20,
    events: "EventCalendar | None" = None,
    holding_days: int = 5,
    stop_atrs: float = 2.0,
    target_r: float = DEFAULT_RISK_REWARD,
    market_risk_off: bool = False,
    today: date | None = None,
) -> Stage4Result:
    """Size the top-ranked candidates and keep the ones that pass.

    `events` is the per-symbol event calendar. It is passed here rather than
    carried on `RegimeState` deliberately: regime is a property of the MARKET,
    and a per-symbol value inside a regime vector would make "the regime" mean
    something different for every candidate. Market-wide event proximity stays
    in `RegimeState.EventProximity`; "does THIS name report on Thursday" is a
    Stage 4 concern, the same way `market_risk_off` and `atr_pct` sit side by
    side on one `size_position` call.

    Omitting it is allowed and is declared in `unavailable` - a scan that did
    not check for earnings must not read like one that checked and found none.

    `candidates` (default 20) bounds how far down the ranking Stage 4 will
    look; `max_trades` (default 3) bounds how many approvals it will return.
    They are separate numbers on purpose: the engine rejects most names, so
    considering only three would routinely return nothing on a day when the
    fourth-ranked name was perfectly tradeable.

    The portfolio is threaded through and UPDATED as trades are approved, so
    the second approval is sized against a portfolio that already contains
    the first. Sizing each candidate against the same empty book would let
    three trades each pass the heat limit individually and breach it
    together - the exact failure a portfolio-level cap exists to prevent.
    """
    pf = portfolio or Portfolio()
    result = Stage4Result(as_of=stage2.as_of)
    if events is None:
        result.unavailable.extend(UNAVAILABLE_NO_CALENDAR)
    elif not stage2.ranked.empty:
        # The gate only sees names NSE has published a forward event for -
        # 41 of the market on a typical day. Saying "the event gate ran" while
        # it could see 3 of 20 candidates would be the same lie as reporting a
        # skipped check as a passed one.
        seen, asked = events.coverage(stage2.ranked.head(candidates).index)
        if seen < asked:
            result.unavailable.append(
                f"Event proximity known for only {seen} of {asked} candidates "
                f"- NSE's forward calendar covers a few weeks, so a name with "
                f"no entry was NOT cleared, it was not checked."
            )

    if stage2.ranked.empty:
        return result

    for symbol in stage2.ranked.head(candidates).index:
        if len(result.approved) >= max_trades:
            break
        if symbol not in stage1.features.index:
            result.skipped["no Stage 1 features"] = \
                result.skipped.get("no Stage 1 features", 0) + 1
            continue

        row = stage1.features.loc[symbol]
        setup = propose_setup(symbol, row, float(stage2.ranked.loc[symbol, "score"]),
                              stop_atrs=stop_atrs, target_r=target_r)
        if setup is None:
            result.skipped["no usable ATR or price"] = \
                result.skipped.get("no usable ATR or price", 0) + 1
            continue

        # THE EVENT GATE. A company announcing results inside the holding
        # window turns the trade into a coin flip on something the chart
        # cannot see, however good the setup is. Checked BEFORE sizing,
        # because there is no quantity that makes this acceptable.
        if events is not None:
            win = events.window(symbol, as_of=stage2.as_of)
            if win.blocks(holding_days):
                result.rejected.append(Sizing(
                    approved=False, symbol=symbol,
                    entry=setup.entry, stop=setup.stop, target=setup.target,
                    reasons=[RejectReason.EVENT_IN_WINDOW],
                    notes=[f"{win.event.purpose or 'scheduled event'} in "
                           f"{win.days_until}d (holding window {holding_days}d)"],
                ))
                result.considered += 1
                continue

        result.considered += 1
        sizing = size_position(
            symbol=symbol, entry=setup.entry, stop=setup.stop,
            target=setup.target, cfg=cfg, portfolio=pf,
            adv_shares=setup.adv_shares, atr_pct=setup.atr_pct,
            market_risk_off=market_risk_off,
            data_as_of=stage2.as_of, today=today,
        )

        if not sizing.approved:
            result.rejected.append(sizing)
            continue

        result.approved.append(PlanTrade(
            symbol=symbol, stance=setup.stance,
            # Confidence is the Stage 2 percentile score expressed 0-1, and
            # it is NOT calibrated - nothing has yet measured whether a 0.8
            # wins 80% of the time. The journal is what will make this number
            # mean something; until then it is a ranking, honestly labelled.
            confidence=round(setup.score / 100.0, 3),
            entry=sizing.entry, stop=sizing.stop, target=sizing.target,
            qty=sizing.qty, capital_at_risk=sizing.capital_at_risk,
            supporting=["scanner-stage2"],
            dissenting=[],
            rationale=_rationale(setup, sizing),
            # Carried, not re-derived. `sizing.stop` IS `setup.stop` - the
            # engine sizes against the stop it was handed and never moves
            # it - so the basis computed with the level is still true of
            # the level that was sized. Recomputing it here from prices
            # alone could not tell a swing low from a 20-session low.
            stop_basis=setup.stop_basis,
            invalidation=setup.invalidation,
        ))
        pf = _with_position(pf, symbol, sizing)

    return result


def _rationale(setup: Setup, sizing: Sizing) -> str:
    bits = [setup.rationale]
    if sizing.risk_reward is not None:
        bits.append(f"R:R {sizing.risk_reward:.2f}")
    if sizing.notes:
        bits.extend(sizing.notes)
    if sizing.checks_skipped:
        # An approval with skipped checks is weaker than one without, and the
        # plan must say so rather than presenting both as equally vetted.
        bits.append("NOT CHECKED: " + "; ".join(sizing.checks_skipped))
    return " | ".join(bits)


def _with_position(pf: Portfolio, symbol: str, sizing: Sizing) -> Portfolio:
    """A copy of the portfolio with this approval added, so the next
    candidate is sized against the book that would actually exist."""
    from desk.risk.engine import Position

    return Portfolio(
        positions=[*pf.positions, Position(
            symbol=symbol, qty=sizing.qty,
            entry=sizing.entry or 0.0, stop=sizing.stop or 0.0,
        )],
        realised_pnl_today=pf.realised_pnl_today,
    )
