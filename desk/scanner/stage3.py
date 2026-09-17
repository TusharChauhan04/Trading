"""Stage 3: the narrative pass. The only stage that costs money.

WHERE IT SITS
-------------
    Stage 0  hygiene        3,485 -> 1,598     deterministic, free
    Stage 1  features       per-symbol         deterministic, free
    Stage 2  ranking        1,598 -> ~20       deterministic, free
    Stage 3  narrative      ~8 -> fewer        ONE LLM CALL
    Stage 4  risk gate      sizing             deterministic, free, PURE

THE ONE RULE THAT DEFINES THIS MODULE
--------------------------------------
STAGE 3 MAY ONLY REMOVE NAMES. It cannot add a candidate, cannot raise a
score, cannot widen a stop, cannot increase a size, and cannot reach Stage
4's risk gate at all. `Stage3Result.kept` is asserted to be a SUBSET of the
shortlist it was given, in the order Stage 2 ranked them.

This is not defensive style, it is the project's standing constraint that no
LLM may relax a risk gate, made structural. A language model that can only
narrow cannot turn a bad trade into a permitted one; the worst it can do is
veto a good one, which costs an opportunity rather than money. Stage 4 still
runs afterwards on whatever survives, and it is a pure, unit-tested function
that never sees this module's output except as a shorter list of symbols.

So the asymmetry is deliberate: a WRONG veto is cheap, a wrong approval is
not, and the design makes only the cheap error possible.

WHY ONE CALL, NOT ONE PER NAME
------------------------------
The staged funnel exists so the expensive step sees ~8 names instead of
1,598. Eight separate calls would be eight prompts each re-establishing the
same market context, and it would stop the model from doing the one thing it
is actually better at than the deterministic stages: comparing candidates
against each other. One call per day is also a cost story that can be stated
in a sentence.

WHEN IT CANNOT RUN
------------------
No key, no budget left, provider unreachable, unparseable answer - none of
these are failures of the scan. Stage 3 reports `ran=False`, passes the
shortlist through UNCHANGED, and says so in `unavailable`. That is the
`checks_skipped` convention: a scan that could not consult the model must
not read like one that consulted it and found nothing wrong. The
deterministic 98% of the funnel is unaffected, and the day still produces a
plan - or a NO TRADE, which is a first-class successful outcome.

NEVER CALL THIS INSIDE A BACKTEST LOOP. Pass `client=None`. A backtest over
even one year would be hundreds of paid calls, and every one of them would
be reasoning about a future it can partly remember - the look-ahead problem
the rest of this codebase works hard to prevent, reintroduced through the
one component that cannot be audited for it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pandas as pd

from desk.contracts.enums import Stance
from desk.llm.base import BudgetExceeded, LLMError, LLMUnavailable, Message
from desk.llm.client import MeteredClient
from desk.scanner.stage2 import Stage2Result

__all__ = [
    "KEEP_STANCES", "Stage3Result", "Verdict", "build_prompt", "parse_verdicts",
    "run_stage3",
]

#: Stage 2 ranks for LONG setups - momentum, trend, range position. A model
#: answering SELL on a name in that shortlist is disagreeing with the
#: shortlist, which is a veto, not an instruction to go short. Stage 3 has no
#: mechanism to open a short and must not grow one.
KEEP_STANCES = (Stance.BUY, Stance.OVERWEIGHT)

UNAVAILABLE_NO_CLIENT = (
    "Stage 3 did not run: no LLM client was supplied, so no narrative check "
    "was made. The shortlist passed through unchanged.")

_SYSTEM = """You are a risk-averse equity analyst reviewing a shortlist for \
one trading day on the Indian market (NSE).

The shortlist was produced by a deterministic quantitative funnel that has \
already applied liquidity, price and data-quality filters and ranked what \
survived. You are the last narrative check before a mechanical risk and \
position-sizing engine.

YOUR ONLY POWER IS TO REMOVE NAMES. You cannot add candidates, change \
scores, set stops or influence position size. Those are computed elsewhere \
and your answer cannot reach them.

Judge each name on whether a fresh long position is sensible TODAY given \
what you are shown. Prefer to veto when something looks wrong, unclear or \
unexplained: a wrongly vetoed name costs one missed opportunity, a wrongly \
approved one costs money. Vetoing every name is a valid and often correct \
answer.

Respond with a JSON array and nothing else. One object per name, in the \
order given:

[{"symbol": "...", "stance": "Buy|Overweight|Hold|Underweight|Sell", \
"confidence": 0.0-1.0, "rationale": "one sentence", \
"concerns": ["..."]}]

Use Buy or Overweight ONLY for names you would actually open today. \
Everything else removes the name."""


@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    stance: Stance
    confidence: float
    rationale: str = ""
    concerns: tuple[str, ...] = ()
    parsed: bool = True
    """False when this verdict was manufactured because the model's answer
    could not be read for this symbol. Such a verdict is always REVIEW -
    never HOLD. HOLD is a tradeable neutral the model chose; REVIEW is the
    absence of an answer, and collapsing the two would let a parse failure
    look like a considered judgement. See ADR-001."""

    @property
    def keeps(self) -> bool:
        return self.stance in KEEP_STANCES


@dataclass(slots=True)
class Stage3Result:
    as_of: date
    kept: list[str] = field(default_factory=list)
    """Survivors, in the order Stage 2 ranked them. ALWAYS a subset of the
    input shortlist - asserted in run_stage3, not merely intended."""
    vetoed: dict[str, str] = field(default_factory=dict)
    """symbol -> why it was removed."""
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    needs_review: list[str] = field(default_factory=list)
    """Names whose verdict could not be read. Vetoed for today AND surfaced,
    because a parse failure is a fact about our pipeline, not about the
    stock, and silently dropping it would hide a broken prompt."""
    considered: int = 0
    ran: bool = False
    cost_inr: Decimal = field(default_factory=lambda: Decimal("0"))
    unavailable: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.kept

    def summary(self) -> str:
        head = (f"Stage 3: {self.considered} considered -> {len(self.kept)} "
                f"kept ({self.as_of})")
        if not self.ran:
            return "\n".join([head] + [f"  UNAVAILABLE: {u}"
                                       for u in self.unavailable])
        lines = [head + f", Rs {self.cost_inr:.4f}"]
        for sym in self.kept:
            v = self.verdicts.get(sym)
            if v:
                lines.append(f"  KEEP  {sym:<14} {v.stance.value:<11} "
                             f"conf {v.confidence:.2f}  {v.rationale[:70]}")
        for sym, why in self.vetoed.items():
            lines.append(f"  VETO  {sym:<14} {why[:80]}")
        if self.needs_review:
            lines.append(f"  REVIEW (unreadable verdict): "
                         f"{', '.join(self.needs_review)}")
        return "\n".join(lines)


def run_stage3(
    stage2: Stage2Result,
    *,
    client: MeteredClient | None,
    candidates: int = 8,
    fundamentals: pd.DataFrame | None = None,
    events=None,
    news=None,
    holding_days: int = 5,
    keep_stances: tuple[Stance, ...] = KEEP_STANCES,
) -> Stage3Result:
    """Narrow the Stage 2 shortlist with one narrative pass.

    `client=None` is a first-class, supported mode - see the module
    docstring. It is how backtests and offline runs work, and it is what
    happens automatically when no key is configured.
    """
    shortlist = [str(s) for s in stage2.ranked.index[:candidates]]
    result = Stage3Result(as_of=stage2.as_of, considered=len(shortlist))

    if client is None:
        result.kept = list(shortlist)
        result.unavailable = [UNAVAILABLE_NO_CLIENT]
        return result
    if not shortlist:
        result.ran = True
        return result

    messages = [Message("system", _SYSTEM),
                Message("user", build_prompt(stage2, shortlist,
                                             fundamentals=fundamentals,
                                             events=events, news=news,
                                             holding_days=holding_days))]

    try:
        answer = client.complete(messages, purpose=f"stage3 {stage2.as_of}")
    except BudgetExceeded as exc:
        # NOT retried, and NOT fatal to the day. The shortlist passes
        # through and the plan says the narrative check was skipped.
        result.kept = list(shortlist)
        result.unavailable = [f"Stage 3 did not run - spending ceiling: {exc}"]
        return result
    except LLMUnavailable as exc:
        result.kept = list(shortlist)
        result.unavailable = [f"Stage 3 did not run - {exc}"]
        return result
    except LLMError as exc:
        result.kept = list(shortlist)
        result.unavailable = [f"Stage 3 did not run - the model errored: {exc}"]
        return result

    result.ran = True
    result.cost_inr = answer.cost_inr
    result.verdicts = parse_verdicts(answer.text, shortlist)

    for sym in shortlist:
        v = result.verdicts[sym]
        if not v.parsed:
            result.needs_review.append(sym)
            result.vetoed[sym] = ("verdict could not be read - held back for "
                                  "review rather than treated as neutral")
        elif v.stance in keep_stances:
            result.kept.append(sym)
        else:
            result.vetoed[sym] = (f"{v.stance.value}: "
                                  f"{v.rationale or 'no reason given'}")

    # The invariant, checked rather than trusted. If this ever fires it means
    # a code path invented a candidate, which must never reach Stage 4.
    extra = set(result.kept) - set(shortlist)
    if extra:
        raise AssertionError(
            f"Stage 3 produced names that were not in its shortlist: "
            f"{sorted(extra)}. Stage 3 may only REMOVE candidates.")
    return result


def build_prompt(stage2: Stage2Result, shortlist: list[str], *,
                 fundamentals: pd.DataFrame | None = None,
                 events=None, news=None, holding_days: int = 5) -> str:
    """Everything the model is allowed to see, and nothing it is not.

    Stage 2's own `explain()` output goes in verbatim. Without it the model
    would be judging a ranking it cannot see the derivation of, which is
    section 32's explainability requirement failing at the one stage where a
    human later asks "why did it say that".

    Absences are stated explicitly rather than omitted. A prompt that simply
    leaves out fundamentals reads, to the model, exactly like a company with
    nothing notable in them - so "not available" is written out, for the same
    reason `checks_skipped` exists.
    """
    parts = [
        f"Date: {stage2.as_of.isoformat()}",
        f"Market regime: {stage2.regime.value}",
        f"Intended holding period: about {holding_days} trading days, long only.",
        f"Universe: {stage2.universe_in} names entered the ranking, "
        f"{stage2.universe_out} survived.",
    ]
    if stage2.silenced:
        parts.append("Factors switched off by the current regime: "
                     + "; ".join(f"{k} ({v})" for k, v in stage2.silenced.items()))

    if news is not None:
        parts.append("\nMARKET NEWS CONTEXT (headlines, deduplicated; these "
                     "are market-wide and are NOT mapped to these symbols):")
        for c in list(news)[:12]:
            parts.append(f"  [{c.published_at:%d-%b %H:%M}] {c.title}")
    else:
        parts.append("\nMARKET NEWS CONTEXT: not available for this run.")

    parts.append(f"\nCANDIDATES ({len(shortlist)}), best-ranked first:")
    for i, sym in enumerate(shortlist, 1):
        parts.append(f"\n--- {i}. {sym} ---")
        parts.append(stage2.explain(sym))
        parts.append(_fundamentals_block(fundamentals, sym))
        parts.append(_event_block(events, sym, stage2.as_of, holding_days))

    parts.append(
        f"\nReturn a JSON array of exactly {len(shortlist)} objects, in this "
        f"order: {', '.join(shortlist)}")
    return "\n".join(parts)


def parse_verdicts(text: str, shortlist: list[str]) -> dict[str, Verdict]:
    """Read the model's answer. Every shortlist symbol gets a verdict.

    A symbol the model did not mention, or whose entry cannot be read, gets
    Stance.REVIEW with parsed=False - never HOLD. The distinction is the
    whole point of ADR-001: HOLD is a neutral the model chose, REVIEW is the
    absence of an answer, and a pipeline that cannot tell them apart will
    eventually report a parse failure as a considered judgement.
    """
    missing = {
        s: Verdict(symbol=s, stance=Stance.REVIEW, confidence=0.0,
                   rationale="no readable verdict for this symbol",
                   parsed=False)
        for s in shortlist
    }

    rows = _extract_json_array(text)
    if rows is None:
        return missing

    by_symbol: dict[str, Verdict] = dict(missing)
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).strip().upper()
        match = next((s for s in shortlist if s.upper() == sym
                      or s.split(".")[0].upper() == sym), None)
        if match is None:
            # A name we did not ask about. Ignored rather than added -
            # Stage 3 may only remove.
            continue
        stance = _stance(row.get("stance"))
        if stance is None:
            continue
        by_symbol[match] = Verdict(
            symbol=match, stance=stance,
            confidence=_confidence(row.get("confidence")),
            rationale=str(row.get("rationale", "")).strip()[:400],
            concerns=tuple(str(c)[:200] for c in row.get("concerns", [])
                           if isinstance(c, (str, int, float)))[:6],
            parsed=True)
    return by_symbol


# --------------------------------------------------------------------------

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(text: str) -> list | None:
    """The array, whether or not it arrived wrapped in prose or a code fence.

    Models wrap JSON in ```json fences often enough that refusing those would
    turn a cosmetic difference into a whole day with no narrative check.
    Anything still unreadable returns None and every symbol becomes REVIEW.
    """
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"```\s*$", "", body).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        m = _ARRAY_RE.search(body)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        for key in ("verdicts", "results", "candidates"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        return None
    return parsed if isinstance(parsed, list) else None


def _stance(value) -> Stance | None:
    if not isinstance(value, str):
        return None
    want = value.strip().lower().replace("_", " ")
    for s in Stance:
        if s.value.lower() == want:
            return s
    return {"buy": Stance.BUY, "strong buy": Stance.BUY,
            "overweight": Stance.OVERWEIGHT, "hold": Stance.HOLD,
            "neutral": Stance.HOLD, "underweight": Stance.UNDERWEIGHT,
            "sell": Stance.SELL, "avoid": Stance.SELL,
            "review": Stance.REVIEW}.get(want)


def _confidence(value) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, c))


def _fundamentals_block(table: pd.DataFrame | None, symbol: str) -> str:
    if table is None:
        return "Fundamentals: not available for this run."
    base = symbol.split(".")[0].upper()
    if base not in table.index:
        return "Fundamentals: no filing on file for this symbol."
    row = table.loc[base]

    def num(col, suffix=""):
        v = row.get(col)
        return "n/a" if v is None or pd.isna(v) else f"{float(v):.1f}{suffix}"

    return (
        f"Fundamentals ({row.get('nature', 'unknown')} basis, period ending "
        f"{row.get('period_end')}, filed {row.get('days_since_filing')} days "
        f"ago):\n"
        f"  revenue growth YoY {num('revenue_growth_yoy_pct', '%')}, "
        f"profit growth YoY {num('profit_growth_yoy_pct', '%')}, "
        f"net margin {num('net_margin_pct', '%')} "
        f"(change {num('margin_change_pp', 'pp')})")


def _event_block(events, symbol: str, as_of: date, holding_days: int) -> str:
    if events is None:
        return ("Scheduled events: NOT CHECKED - no calendar was supplied "
                "for this run.")
    win = events.window(symbol, as_of=as_of)
    if not win.known:
        return ("Scheduled events: nothing on file for this symbol. The "
                "calendar covers only part of the market, so this means "
                "unknown, not 'nothing scheduled'.")
    if win.days_until is None:
        return "Scheduled events: none upcoming on the calendar."
    flag = " - INSIDE the intended holding window" \
        if win.days_until <= holding_days else ""
    purpose = getattr(win.event, "purpose", "") or "event"
    return (f"Scheduled events: {purpose} in {win.days_until} calendar "
            f"day(s){flag}.")
