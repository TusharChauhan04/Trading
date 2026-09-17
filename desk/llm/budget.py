"""The cost meter. Nothing calls a paid model without going through this.

WHY THIS EXISTS BEFORE THE FIRST PAID CALL
-------------------------------------------
The staged funnel's entire economic argument is that only Stage 3 costs
money per call, and that it sees ~8 names instead of 1,598. That argument is
UNVERIFIABLE until spend is actually measured, and an unverifiable cost
argument is how a research layer quietly becomes a 40,000-rupee month.

So this was written before any key was configured, and the provider
interface requires usage on every completion specifically so this can work.

THE RULES IT ENFORCES
---------------------
1. NO CALL WITHOUT A KNOWN PRICE. An unpriced model raises rather than being
   treated as free. "We do not know what this costs" and "this costs
   nothing" are opposite statements and only one of them is ever true.

2. THE CEILING IS CHECKED BEFORE THE CALL, against an estimate that rounds
   UP and assumes the full output allowance is used. Checking afterwards
   tells you what you already spent.

3. BudgetExceeded IS NEVER RETRIED. A budget you can retry past is not a
   budget. It is not a transient failure and `fetch_with_retry`-style
   backoff must not be wrapped around it.

4. MONEY IS Decimal, NOT FLOAT. Fractions of a paisa accumulated over
   thousands of calls in binary floating point drift, and a ledger that
   disagrees with the provider's invoice is worse than no ledger.

5. THE LEDGER IS APPEND-ONLY AND FLUSHED PER CALL. A crash must not lose the
   record of money already spent - that is the one number that cannot be
   recomputed from local state.

THE CEILING ITSELF IS NOT YET SET BY THE USER
----------------------------------------------
`DEFAULT_MONTHLY_CEILING_INR` is a PLACEHOLDER, deliberately low. The user
chose OpenAI but has not given a monthly rupee ceiling, and picking a
generous default on their behalf is exactly the wrong way to resolve that -
a low default fails loudly and costs one conversation, a high default fails
silently and costs money. Set it explicitly when the number is known.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from desk.llm.base import BudgetExceeded, Usage

__all__ = [
    "DEFAULT_MONTHLY_CEILING_INR", "DEFAULT_USD_INR", "CostMeter",
    "ModelPrice", "PRICES", "price_for",
]

IST = timezone(timedelta(hours=5, minutes=30))

#: PLACEHOLDER. See the module docstring - the user has not set this yet.
#: Low on purpose: it fails loudly rather than spending quietly.
DEFAULT_MONTHLY_CEILING_INR = Decimal("500.00")

#: Rupees per US dollar. OpenAI bills in USD; the ceiling is in rupees
#: because that is the unit the decision is made in. Deliberately a constant
#: rather than a live FX lookup: a budget that moves on its own is not a
#: budget, and a ceiling should not quietly loosen because the rupee rallied.
#: Rounded UP against us, so the meter over-states rather than under-states.
DEFAULT_USD_INR = Decimal("90.00")


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per 1,000,000 tokens."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal


#: USD per million tokens. THESE ARE NOT AUTHORITATIVE AND THEY CHANGE.
#: Verify against the provider's current pricing page before relying on a
#: spend projection. They are here so that an unpriced model is impossible to
#: call by accident, not so that the number is treated as gospel - and
#: `price_for` raises on anything absent rather than guessing a default.
PRICES: dict[str, ModelPrice] = {
    "gpt-4o-mini":  ModelPrice(Decimal("0.15"), Decimal("0.60")),
    "gpt-4o":       ModelPrice(Decimal("2.50"), Decimal("10.00")),
    "gpt-4.1-mini": ModelPrice(Decimal("0.40"), Decimal("1.60")),
    "gpt-4.1":      ModelPrice(Decimal("2.00"), Decimal("8.00")),
    # Free, local, and exercised by the test suite.
    "scripted-1":   ModelPrice(Decimal("0"), Decimal("0")),
}


def price_for(model: str) -> ModelPrice:
    """The price, or an exception. Never a default.

    An unknown model is a refusal rather than a zero: treating an unpriced
    model as free is how a ledger comes to under-report a month by an order
    of magnitude, and the failure is silent until the invoice arrives.
    """
    try:
        return PRICES[model]
    except KeyError:
        raise BudgetExceeded(
            f"no price on file for model {model!r}, so its cost cannot be "
            f"computed and it will not be called. Add it to "
            f"desk.llm.budget.PRICES with the provider's current published "
            f"rate - do not assume it is cheap."
        ) from None


@dataclass(slots=True)
class CostMeter:
    """Month-to-date spend, a hard ceiling, and an append-only ledger."""

    ledger_path: Path
    monthly_ceiling_inr: Decimal = DEFAULT_MONTHLY_CEILING_INR
    usd_inr: Decimal = DEFAULT_USD_INR

    def __post_init__(self) -> None:
        self.ledger_path = Path(self.ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.monthly_ceiling_inr, Decimal):
            self.monthly_ceiling_inr = Decimal(str(self.monthly_ceiling_inr))
        if not isinstance(self.usd_inr, Decimal):
            self.usd_inr = Decimal(str(self.usd_inr))

    # -- pricing -----------------------------------------------------------

    def cost_inr(self, model: str, usage: Usage) -> Decimal:
        p = price_for(model)
        usd = (Decimal(usage.prompt_tokens) * p.input_per_mtok
               + Decimal(usage.completion_tokens) * p.output_per_mtok
               ) / Decimal(1_000_000)
        return usd * self.usd_inr

    def estimate_inr(self, model: str, prompt_tokens: int,
                     max_output_tokens: int) -> Decimal:
        """What the call could cost AT WORST.

        Assumes the full output allowance is consumed. A pre-flight estimate
        that assumes a short answer is not a ceiling check, it is a hope.
        """
        return self.cost_inr(model, Usage(prompt_tokens=prompt_tokens,
                                          completion_tokens=max_output_tokens))

    # -- the gate ----------------------------------------------------------

    def month_to_date_inr(self, now: datetime | None = None) -> Decimal:
        """Spend so far this calendar month, in IST.

        IST because the person paying thinks in Indian calendar months. On a
        UTC-clock server a naive month boundary would roll over 5.5 hours
        early and carry the last evening of a month into the next one.
        """
        now = (now or datetime.now(IST)).astimezone(IST)
        total = Decimal("0")
        for row in self.entries():
            when = row.get("at")
            if not when:
                continue
            try:
                ts = datetime.fromisoformat(when).astimezone(IST)
            except ValueError:
                continue
            if (ts.year, ts.month) == (now.year, now.month):
                total += Decimal(str(row.get("cost_inr", "0")))
        return total

    def authorise(self, model: str, prompt_tokens: int,
                  max_output_tokens: int, *,
                  now: datetime | None = None) -> Decimal:
        """Check the ceiling and return the worst-case cost. Raises if over.

        Called BEFORE the request leaves. Returns the estimate so a caller
        can log what it was authorised to spend.
        """
        est = self.estimate_inr(model, prompt_tokens, max_output_tokens)
        spent = self.month_to_date_inr(now)
        if spent + est > self.monthly_ceiling_inr:
            raise BudgetExceeded(
                f"this call could cost up to Rs {est:.4f} and Rs {spent:.2f} "
                f"has already been spent this month, which would exceed the "
                f"ceiling of Rs {self.monthly_ceiling_inr:.2f}. Raise "
                f"monthly_ceiling_inr deliberately if that is intended - it "
                f"is not a transient failure and must not be retried.")
        return est

    # -- the ledger --------------------------------------------------------

    def record(self, model: str, usage: Usage, *, purpose: str = "",
               now: datetime | None = None) -> Decimal:
        """Append one call to the ledger and return what it cost."""
        cost = self.cost_inr(model, usage)
        row = {
            "at": (now or datetime.now(IST)).astimezone(IST).isoformat(),
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "estimated_tokens": usage.estimated,
            "cost_inr": str(cost),
            "purpose": purpose,
        }
        # Append and flush per call. The one number that cannot be
        # reconstructed from local state is money already spent.
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return cost

    def entries(self) -> list[dict]:
        """Every ledger row. A malformed line is skipped, not fatal - a
        truncated final write must not make the whole history unreadable."""
        if not self.ledger_path.exists():
            return []
        out = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def report(self, now: datetime | None = None) -> str:
        now = (now or datetime.now(IST)).astimezone(IST)
        rows = [r for r in self.entries()
                if str(r.get("at", "")).startswith(now.strftime("%Y-%m"))]
        spent = self.month_to_date_inr(now)
        left = self.monthly_ceiling_inr - spent
        est = sum(1 for r in rows if r.get("estimated_tokens"))
        lines = [
            f"LLM spend {now:%B %Y}: Rs {spent:.2f} of "
            f"Rs {self.monthly_ceiling_inr:.2f} ceiling "
            f"(Rs {left:.2f} left), {len(rows)} call(s)",
        ]
        if est:
            lines.append(f"  {est} of {len(rows)} used ESTIMATED token counts "
                         f"- actual spend may differ")
        by_model: dict[str, Decimal] = {}
        for r in rows:
            m = str(r.get("model", "?"))
            by_model[m] = by_model.get(m, Decimal("0")) + Decimal(
                str(r.get("cost_inr", "0")))
        for m, c in sorted(by_model.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {m:<16} Rs {c:.2f}")
        return "\n".join(lines)
