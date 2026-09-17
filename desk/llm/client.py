"""The only way to reach a model. Metering is not optional.

A provider can be called directly - it is just an object with a `complete`
method - so this does not pretend to be a security boundary. What it is, is
the one path Stage 3 uses, so that "did we authorise this spend" and "is it
in the ledger" are answered by construction rather than by remembering.

ORDER MATTERS, and it is: AUTHORISE, CALL, RECORD.

Authorising afterwards tells you what you already spent. Recording before
the call books money that may never have been spent. The failure in between
- an exception after the request left - is handled explicitly: if the
provider raises, nothing is recorded, because we have no usage figures to
record and inventing them would corrupt the one number that cannot be
recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from desk.llm.base import (
    BudgetExceeded, Completion, LLMProvider, Message, Usage,
)
from desk.llm.budget import CostMeter

__all__ = ["MeteredClient", "MeteredCompletion"]

#: ~3 characters per token, rounded up. Deliberately pessimistic: the real
#: ratio for English prose is nearer 4, but Stage 3's prompt is mostly
#: numeric tables, which tokenize much less efficiently. The pre-flight
#: check should over-estimate - erring toward refusing a call is recoverable,
#: erring toward allowing one is not.
_CHARS_PER_TOKEN = 3


@dataclass(frozen=True, slots=True)
class MeteredCompletion:
    completion: Completion
    cost_inr: Decimal
    authorised_inr: Decimal
    """What the pre-flight check allowed. Larger than `cost_inr` whenever the
    model answered more briefly than its allowance, which is the normal
    case - the gap is how much headroom the estimate wasted."""

    @property
    def text(self) -> str:
        return self.completion.text

    @property
    def usage(self) -> Usage:
        return self.completion.usage


@dataclass(slots=True)
class MeteredClient:
    provider: LLMProvider
    meter: CostMeter
    max_output_tokens: int = 700
    temperature: float = 0.0
    """Zero by default. Stage 3 is asked to judge a fixed table of numbers
    against fixed rules; sampling variation there is not creativity, it is
    an unreproducible daily plan. The journal has to be able to ask why a
    name was vetoed on Tuesday, and a different answer on a re-run makes
    that question unanswerable."""
    calls: int = field(default=0, init=False)
    spent_inr: Decimal = field(default_factory=lambda: Decimal("0"), init=False)

    def complete(self, messages: list[Message], *, purpose: str = "",
                 max_output_tokens: int | None = None) -> MeteredCompletion:
        cap = max_output_tokens or self.max_output_tokens
        prompt_tokens = self.estimate_prompt_tokens(messages)

        # 1. AUTHORISE. Raises BudgetExceeded before anything leaves.
        authorised = self.meter.authorise(
            self.provider.model, prompt_tokens, cap)

        # 2. CALL.
        completion = self.provider.complete(
            messages, max_output_tokens=cap, temperature=self.temperature)

        # 3. RECORD, with the REAL usage the provider reported.
        cost = self.meter.record(completion.model, completion.usage,
                                 purpose=purpose)
        self.calls += 1
        self.spent_inr += cost
        return MeteredCompletion(completion=completion, cost_inr=cost,
                                 authorised_inr=authorised)

    @staticmethod
    def estimate_prompt_tokens(messages: list[Message]) -> int:
        chars = sum(len(m.content) + len(m.role) for m in messages)
        return -(-chars // _CHARS_PER_TOKEN)

    def affordable(self, messages: list[Message],
                   max_output_tokens: int | None = None) -> bool:
        """Would this call be authorised? Asks without raising.

        For deciding how many candidates to send before sending any of them,
        rather than discovering the ceiling halfway down the shortlist and
        producing a plan that examined the first four names and not the
        last four - a silently truncated analysis that still looks complete.
        """
        try:
            self.meter.authorise(self.provider.model,
                                 self.estimate_prompt_tokens(messages),
                                 max_output_tokens or self.max_output_tokens)
        except BudgetExceeded:
            return False
        return True
