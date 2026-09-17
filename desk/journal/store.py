"""The journal on disk. Append-only for decisions, updatable for outcomes.

    <root>/decisions/<YYYY-MM-DD>.json
    <root>/decisions/<YYYY-MM-DD>.amend-<n>.json
    <root>/outcomes/<YYYY-MM-DD>.json

THE ONE RULE: A RECORDED DECISION IS NEVER OVERWRITTEN
-------------------------------------------------------
`record()` refuses if a decision already exists for that day. Not a warning,
not a merge - a refusal, with the existing digest in the message.

This is the entire value of the module. A journal that can be rewritten
proves nothing, and the rewrite is almost never malicious: a scheduled job
runs twice, someone re-runs the morning scan after lunch with more data, a
bug reprocesses last week. Any of those silently replaces what the desk
actually decided with what it would decide now, knowing more. The record
stops being evidence and becomes a second opinion wearing yesterday's date.

AMENDMENTS ADD, THEY DO NOT REPLACE
------------------------------------
Real corrections happen - a typo in a note, a trade recorded against the
wrong symbol. `amend()` writes a NEW file alongside the original and points
at it via `Decision.amends`. Both stay on disk, `latest()` returns the
newest, and `history()` returns the chain. So a correction is visible AS a
correction, which is the opposite of an overwrite.

WHY ONE FILE PER DAY RATHER THAN ONE APPENDED LOG
--------------------------------------------------
Same reasoning as the bhavcopy store: a corrupted write damages one day
instead of the whole history, and the point-in-time question - "what did we
know on the 11th" - is answered by a filename rather than by scanning. The
spend ledger in desk/llm is a single append-only JSONL for the opposite
reason: it is a running total that is only ever summed, never asked about
one particular day.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path

from desk.journal.models import IST, Decision, Outcome

__all__ = ["JournalError", "JournalStore"]

_DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\.amend-(\d+))?$")


class JournalError(Exception):
    """The journal refused. Usually because something already exists."""


class JournalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- layout ------------------------------------------------------------

    @property
    def decisions_dir(self) -> Path:
        return self.root / "decisions"

    @property
    def outcomes_dir(self) -> Path:
        return self.root / "outcomes"

    def _decision_path(self, day: date, amend: int = 0) -> Path:
        stem = day.isoformat() + (f".amend-{amend}" if amend else "")
        return self.decisions_dir / f"{stem}.json"

    def _outcome_path(self, day: date) -> Path:
        return self.outcomes_dir / f"{day.isoformat()}.json"

    # -- decisions ---------------------------------------------------------

    def record(self, decision: Decision) -> Path:
        """Write one day's decision. REFUSES if that day already has one."""
        path = self._decision_path(decision.as_of)
        if path.exists():
            existing = self.latest(decision.as_of)
            raise JournalError(
                f"a decision for {decision.as_of} is already recorded "
                f"(digest {existing.digest() if existing else 'unreadable'}). "
                f"A journal that can be overwritten is not evidence. Use "
                f"amend() to record a correction alongside it.")
        return self._write(path, decision.to_json())

    def amend(self, decision: Decision, *, reason: str) -> Path:
        """Record a correction WITHOUT destroying what it corrects.

        `reason` is required and stored. An amendment with no stated reason
        is indistinguishable from an overwrite six months later, which is
        the thing this module exists to prevent.
        """
        if not reason.strip():
            raise JournalError(
                "an amendment must say why. Without a reason the record is "
                "just a second version with no account of the first.")
        original = self.latest(decision.as_of)
        if original is None:
            raise JournalError(
                f"nothing recorded for {decision.as_of} yet - use record().")

        n = 1 + max([a for _, a in self._versions(decision.as_of)], default=0)
        amended = Decision(
            **{**{k: v for k, v in _fields(decision).items()
                  if k not in ("amends", "note")},
               "amends": original.digest(),
               "note": (decision.note + ("\n" if decision.note else "")
                        + f"AMENDMENT: {reason.strip()}")},
        )
        return self._write(self._decision_path(decision.as_of, n),
                           amended.to_json())

    def latest(self, day: date) -> Decision | None:
        """The newest version for a day, amendments included."""
        versions = self._versions(day)
        if not versions:
            return None
        path, _ = max(versions, key=lambda pa: pa[1])
        return self._read_decision(path)

    def history(self, day: date) -> list[Decision]:
        """Every version for a day, oldest first. The audit trail."""
        out = []
        for path, _ in sorted(self._versions(day), key=lambda pa: pa[1]):
            got = self._read_decision(path)
            if got is not None:
                out.append(got)
        return out

    def days(self) -> list[date]:
        """Every day with a decision on file, oldest first."""
        if not self.decisions_dir.is_dir():
            return []
        found = set()
        for p in self.decisions_dir.glob("*.json"):
            m = _DAY.match(p.stem)
            if m:
                try:
                    found.add(date.fromisoformat(m.group(1)))
                except ValueError:
                    continue
        return sorted(found)

    def _versions(self, day: date) -> list[tuple[Path, int]]:
        if not self.decisions_dir.is_dir():
            return []
        out = []
        for p in self.decisions_dir.glob(f"{day.isoformat()}*.json"):
            m = _DAY.match(p.stem)
            if m and m.group(1) == day.isoformat():
                out.append((p, int(m.group(2) or 0)))
        return out

    def _read_decision(self, path: Path) -> Decision | None:
        try:
            return Decision.from_json(json.loads(
                path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            # A corrupt record is reported as ABSENT rather than guessed at.
            # Half a decision is not a decision, and filling the gaps with
            # defaults would invent a plan nobody made.
            return None

    # -- outcomes ----------------------------------------------------------

    def record_outcomes(self, day: date, outcomes) -> Path:
        """Write or replace the outcomes for one decision day.

        Replaceable, unlike a decision, because an outcome is an observation
        that gets completed as a position runs - see Outcome's docstring.
        """
        rows = [o.to_json() for o in outcomes]
        return self._write(self._outcome_path(day), rows)

    def outcomes(self, day: date) -> list[Outcome]:
        path = self._outcome_path(day)
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for row in raw if isinstance(raw, list) else []:
            try:
                out.append(Outcome.from_json(row))
            except (KeyError, ValueError):
                continue
        return out

    def open_positions(self) -> list[Outcome]:
        """Everything recorded and not yet closed, oldest first."""
        out = []
        for day in self.days():
            out.extend(o for o in self.outcomes(day) if o.status != "closed")
        return out

    def unrecorded(self) -> list[date]:
        """Decision days with trades but no outcomes on file.

        The journal's own to-do list. Without it, "we have no losing trades"
        and "nobody wrote down how the trades went" look identical, and the
        first reading is the flattering one.
        """
        out = []
        for day in self.days():
            dec = self.latest(day)
            if dec is None or dec.is_no_trade:
                continue
            have = {o.symbol for o in self.outcomes(day)}
            if set(dec.symbols) - have:
                out.append(day)
        return out

    # -- writing -----------------------------------------------------------

    def _write(self, path: Path, payload) -> Path:
        """Atomic, and fsynced. A journal entry that a crash can truncate is
        worse than none: it parses as a SHORTER plan - fewer trades, a
        different decision - rather than failing to parse at all."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        return path


def _fields(d: Decision) -> dict:
    return {f: getattr(d, f) for f in d.__slots__}


def decision_from_plan(plan, *, summary=None, capital: float = 0.0,
                       note: str = "", now: datetime | None = None
                       ) -> Decision:
    """Build a Decision from a DailyPlan (and optionally its ScanSummary).

    Lives here rather than in desk.plan so the plan package stays free of
    journal concerns - the dependency runs journal -> plan, never back.
    """
    from desk.journal.models import TradeRecord

    trades = tuple(
        TradeRecord(
            symbol=t.symbol, stance=getattr(t.stance, "value", str(t.stance)),
            entry=t.entry, stop=t.stop, target=t.target, qty=t.qty,
            capital_at_risk=t.capital_at_risk, rationale=t.rationale,
            supporting=tuple(t.supporting), dissenting=tuple(t.dissenting),
        )
        for t in plan.trades
    )
    return Decision(
        as_of=plan.as_of,
        recorded_at=(now or datetime.now(IST)),
        regime=getattr(plan.regime, "value", str(plan.regime)),
        trades=trades,
        no_trade_reason=plan.no_trade_reason,
        universe_scanned=plan.universe_scanned,
        survived_stage0=plan.survived_stage0,
        survived_stage1=plan.survived_stage1,
        survived_stage2=plan.survived_stage2,
        considered=getattr(plan, "analysed", 0),
        caveats=tuple(plan.warnings) + tuple(plan.market_risks),
        coverage_note=(summary.coverage_note if summary else ""),
        capital=capital,
        note=note,
    )
