"""What the journal records. Decisions and outcomes, kept strictly apart.

WHY THIS SPLIT IS THE WHOLE DESIGN
-----------------------------------
A `Decision` is what the desk believed and chose ON a given morning. An
`Outcome` is what the market later did about it. They are separate records,
written at different times, and a Decision is NEVER modified to carry its
own result.

That sounds fussy until you consider the alternative. A single mutable row
per trade means the file that says "why we took this" is rewritten by the
code that learns how it went - and from then on there is no way to prove
the reasoning was not adjusted to fit the result. Every strategy looks
better in a journal that can be edited after the fact, and the edit does
not even have to be dishonest: "clarifying" a rationale once you know the
answer is the most natural thing in the world.

So: Decisions are append-only and refuse to be overwritten. Outcomes are
written later, linked by (decision date, symbol), and can be updated freely
because they are observations, not claims.

NO TRADE IS A RECORD, NOT AN ABSENCE
------------------------------------
A journal that only stores trades cannot answer the most important question
a cautious system has to answer: how often did it correctly stay out? If
quiet days leave no trace, the sample is every day the desk chose to act,
which is exactly the population that makes a bad strategy look selective.
`Decision` therefore exists for every scanned day, carries
`no_trade_reason`, and `is_no_trade` is a first-class state.

WHAT ELSE GETS STORED, AND WHY IT IS NOT OPTIONAL
--------------------------------------------------
`caveats` and `coverage_note` are part of the decision. Six months from now
"why did it pick that" and "what did it not know at the time" are the same
question, and a record that keeps the answer but not the blind spots is a
record that will be misread. The funnel counts are stored for the same
reason: a shortlist of three out of eight candidates means something very
different from three out of four hundred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone

__all__ = [
    "Decision", "ExitReason", "IST", "Outcome", "TradeRecord",
]

IST = timezone(timedelta(hours=5, minutes=30))


class ExitReason:
    """Why a position ended. Plain strings, because this is written to JSON
    and read by humans far more often than it is compared in code."""

    TARGET = "target"
    STOP = "stop"
    TIME = "time"           # held to the end of the intended horizon
    DISCRETION = "discretion"
    NOT_TAKEN = "not_taken"
    """The plan proposed it and it was never entered. Recorded rather than
    deleted: a plan whose trades are routinely skipped is a fact about the
    system, and silently dropping those makes the journal describe a desk
    that does not exist."""

    ALL = (TARGET, STOP, TIME, DISCRETION, NOT_TAKEN)


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One proposed trade, exactly as the plan stated it.

    Frozen. These are copied out of a PlanTrade at the moment of the
    decision and must read the same forever - including the levels, which
    are the only way to judge later whether the stop was hit before the
    target.
    """

    symbol: str
    stance: str
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    qty: int = 0
    capital_at_risk: float = 0.0
    rationale: str = ""
    supporting: tuple[str, ...] = ()
    dissenting: tuple[str, ...] = ()

    stop_basis: str = ""
    """What the stop was anchored to: "swing_low", "low_20", "noise_floor"
    or "atr". Recorded because the levels alone cannot answer the question
    the journal exists to answer later - not "was the stop hit" but "was
    the stop in a defensible place". Empty on decisions written before
    stops had a basis, which is the truth about those records."""

    invalidation: str = ""
    """The same in a sentence: which price level would prove the setup
    wrong, and why that one. Frozen with the rest, so a review months later
    reads the reasoning the desk actually had rather than the reasoning it
    would have now."""

    @property
    def risk_per_share(self) -> float | None:
        if self.entry is None or self.stop is None:
            return None
        return abs(self.entry - self.stop)

    @property
    def reward_to_risk(self) -> float | None:
        r = self.risk_per_share
        if not r or self.entry is None or self.target is None:
            return None
        return abs(self.target - self.entry) / r


@dataclass(frozen=True, slots=True)
class Decision:
    """One morning's answer, with everything needed to re-read it later."""

    as_of: date
    """The trading day this plan is FOR."""
    recorded_at: datetime
    """When it was written. Distinct from as_of on purpose - a decision
    recorded days later is a reconstruction, and the gap is the evidence."""
    regime: str = "unknown"
    trades: tuple[TradeRecord, ...] = ()
    no_trade_reason: str | None = None

    universe_scanned: int = 0
    survived_stage0: int = 0
    survived_stage1: int = 0
    survived_stage2: int = 0
    considered: int = 0

    caveats: tuple[str, ...] = ()
    """What the system could NOT check that day. Part of the decision, not
    metadata about it."""
    coverage_note: str = ""
    capital: float = 0.0
    note: str = ""
    """Free text from the operator. The only field a human is expected to
    write, and it is never auto-generated."""
    amends: str | None = None
    """The digest of a decision this one supersedes. Set only by an explicit
    amendment - the superseded record stays on disk."""

    @property
    def is_no_trade(self) -> bool:
        return not self.trades

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(t.symbol for t in self.trades)

    def digest(self) -> str:
        """A short content hash, so a record can be referred to and checked.

        Not security - anyone with write access to the directory can rewrite
        both the record and its hash. It is tamper-EVIDENCE for the ordinary
        case: an accidental rewrite, a half-finished edit, a file restored
        from the wrong backup. That is the failure this project has actually
        had, and it is the one worth detecting.
        """
        payload = json.dumps(self.to_json(), sort_keys=True,
                             separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        out = asdict(self)
        out["as_of"] = self.as_of.isoformat()
        out["recorded_at"] = self.recorded_at.astimezone(IST).isoformat()
        out["trades"] = [dict(t) for t in (asdict(x) for x in self.trades)]
        for t in out["trades"]:
            t["supporting"] = list(t["supporting"])
            t["dissenting"] = list(t["dissenting"])
        out["caveats"] = list(self.caveats)
        return out

    @classmethod
    def from_json(cls, raw: dict) -> "Decision":
        trades = tuple(
            TradeRecord(
                symbol=str(t["symbol"]), stance=str(t.get("stance", "")),
                entry=t.get("entry"), stop=t.get("stop"),
                target=t.get("target"), qty=int(t.get("qty", 0) or 0),
                capital_at_risk=float(t.get("capital_at_risk", 0.0) or 0.0),
                rationale=str(t.get("rationale", "")),
                supporting=tuple(t.get("supporting", ())),
                dissenting=tuple(t.get("dissenting", ())),
                # .get with a default, not [..]: every decision recorded
                # before these existed must still load, and it must load
                # as "" - "this record does not say" - rather than as a
                # guess at what the basis probably was.
                stop_basis=str(t.get("stop_basis", "")),
                invalidation=str(t.get("invalidation", "")),
            )
            for t in raw.get("trades", [])
        )
        return cls(
            as_of=date.fromisoformat(raw["as_of"]),
            recorded_at=datetime.fromisoformat(raw["recorded_at"]),
            regime=str(raw.get("regime", "unknown")),
            trades=trades,
            no_trade_reason=raw.get("no_trade_reason"),
            universe_scanned=int(raw.get("universe_scanned", 0)),
            survived_stage0=int(raw.get("survived_stage0", 0)),
            survived_stage1=int(raw.get("survived_stage1", 0)),
            survived_stage2=int(raw.get("survived_stage2", 0)),
            considered=int(raw.get("considered", 0)),
            caveats=tuple(raw.get("caveats", ())),
            coverage_note=str(raw.get("coverage_note", "")),
            capital=float(raw.get("capital", 0.0) or 0.0),
            note=str(raw.get("note", "")),
            amends=raw.get("amends"),
        )

    def summary(self) -> str:
        head = (f"{self.as_of}  {self.regime}  "
                f"{self.universe_scanned} -> {self.considered} considered")
        if self.is_no_trade:
            return f"{head}\n  NO TRADE - {self.no_trade_reason or 'no reason recorded'}"
        lines = [head]
        for t in self.trades:
            rr = t.reward_to_risk
            lines.append(
                f"  {t.symbol:<14} {t.qty:>6} @ {t.entry}  stop {t.stop}  "
                f"target {t.target}" + (f"  R:R {rr:.1f}" if rr else ""))
        return "\n".join(lines)


@dataclass(slots=True)
class Outcome:
    """What happened to one proposed trade. Mutable, unlike a Decision.

    Mutable on purpose and it is not an inconsistency: an outcome is an
    OBSERVATION, and observations get corrected and completed as a position
    runs. What must never change is the claim that was made beforehand, and
    that lives in the Decision.
    """

    decision_date: date
    symbol: str
    observed_at: datetime
    status: str = "open"                 # "open" | "closed"
    exit_price: float | None = None
    exit_date: date | None = None
    exit_reason: str | None = None
    pnl: float | None = None
    r_multiple: float | None = None
    """Profit or loss in units of the risk originally taken. The only
    comparable measure across names: 2R on a 40-rupee stock and 2R on a
    4,000-rupee one are the same result, and rupees are not."""
    note: str = ""

    def close(self, *, price: float, on: date, reason: str,
              entry: float | None, stop: float | None, qty: int) -> None:
        """Record the exit and derive the result from the ORIGINAL levels.

        `entry` and `stop` are passed in from the Decision rather than
        stored here, so the R-multiple is always computed against the risk
        that was actually accepted at the time - not against a stop moved
        afterwards.
        """
        if reason not in ExitReason.ALL:
            raise ValueError(
                f"unknown exit reason {reason!r}; expected one of "
                f"{ExitReason.ALL}")
        self.status = "closed"
        self.exit_price = price
        self.exit_date = on
        self.exit_reason = reason
        if entry is not None and qty:
            self.pnl = (price - entry) * qty
        risk = abs(entry - stop) if (entry is not None and stop is not None) else None
        if risk and entry is not None:
            self.r_multiple = (price - entry) / risk

    def to_json(self) -> dict:
        out = asdict(self)
        out["decision_date"] = self.decision_date.isoformat()
        out["observed_at"] = self.observed_at.astimezone(IST).isoformat()
        out["exit_date"] = self.exit_date.isoformat() if self.exit_date else None
        return out

    @classmethod
    def from_json(cls, raw: dict) -> "Outcome":
        return cls(
            decision_date=date.fromisoformat(raw["decision_date"]),
            symbol=str(raw["symbol"]),
            observed_at=datetime.fromisoformat(raw["observed_at"]),
            status=str(raw.get("status", "open")),
            exit_price=raw.get("exit_price"),
            exit_date=(date.fromisoformat(raw["exit_date"])
                       if raw.get("exit_date") else None),
            exit_reason=raw.get("exit_reason"),
            pnl=raw.get("pnl"),
            r_multiple=raw.get("r_multiple"),
            note=str(raw.get("note", "")),
        )
