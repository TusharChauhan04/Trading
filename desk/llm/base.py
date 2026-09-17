"""The provider-agnostic seam. Everything above this is provider-blind.

WHY A SEAM AT ALL, WHEN THE ANSWER IS "OPENAI"
----------------------------------------------
The user chose OpenAI. This still does not import the OpenAI SDK anywhere
except desk/llm/providers/openai.py, for three reasons that are not
hypothetical:

1. THE TESTS MUST NOT CALL A PAID API. A test suite that costs money per run
   is a test suite that gets run less often, and 595 tests at even a tenth of
   a paisa each is a reason to skip the slow ones. `ScriptedProvider` below
   makes every Stage 3 test free and deterministic.

2. STAGE 3 IS THE ONLY PART OF THE FUNNEL THAT COSTS MONEY PER CALL. That is
   the entire economic argument for the staged design - Stages 0, 1, 2 and 4
   are deterministic and free, and the LLM sees ~8 names rather than 1,598.
   An argument like that has to be measurable, so the provider interface
   REQUIRES usage to come back with every completion. A provider that cannot
   report tokens cannot be metered, and one that cannot be metered does not
   get to run.

3. Prices and model names change. The seam means that is a one-file change.

WHAT A PROVIDER MAY NOT DO
--------------------------
Return a result with no usage. There is no "unknown cost" path: see
`Usage.unknown`, which exists so an honest provider can SAY the count is
estimated rather than silently reporting zero. Zero-cost is a claim, and a
false one compounds into "Stage 3 is cheap" over thousands of calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "BudgetExceeded", "Completion", "LLMError", "LLMProvider",
    "LLMUnavailable", "Message", "ScriptedProvider", "Usage",
]


class LLMError(Exception):
    """The model could not be consulted, or answered unusably."""


class LLMUnavailable(LLMError):
    """No provider is configured, or it cannot be reached.

    Distinct from LLMError because it is not a failure of the request - it is
    the absence of the capability, and Stage 3 treats it as "this check did
    not run" rather than "this check failed". Same distinction as
    quality.py's checks_skipped.
    """


class BudgetExceeded(LLMError):
    """The call would breach the spending ceiling. Raised BEFORE the call.

    Never caught and retried. A budget that can be retried past is not a
    budget.
    """


@dataclass(frozen=True, slots=True)
class Message:
    role: str           # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens actually consumed. Required on every completion."""

    prompt_tokens: int
    completion_tokens: int
    estimated: bool = False
    """True when the provider could not report real counts and these are
    inferred. The ledger keeps the flag so a month's spend can be reported
    as "measured" or "partly estimated" rather than implying precision it
    does not have."""

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def unknown(cls, prompt_chars: int, completion_chars: int) -> "Usage":
        """A last-resort estimate from character counts.

        ~4 characters per token is a rough English average and is WRONG for
        tables of numbers, which is most of what Stage 3 sends. It rounds UP
        so the meter over-counts rather than under-counts: a budget that
        errs toward stopping early is the safe direction.
        """
        return cls(prompt_tokens=-(-prompt_chars // 3),
                   completion_tokens=-(-completion_chars // 3),
                   estimated=True)


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    usage: Usage
    raw: dict = field(default_factory=dict, repr=False)


@runtime_checkable
class LLMProvider(Protocol):
    """Anything Stage 3 can ask. Deliberately tiny."""

    name: str
    model: str

    def complete(self, messages: list[Message], *, max_output_tokens: int,
                 temperature: float) -> Completion:
        ...


@dataclass(slots=True)
class ScriptedProvider:
    """A provider that returns prepared answers. For tests and dry runs.

    Not a mock in the loose sense - it implements the real protocol including
    usage accounting, so the cost meter, the ledger and the budget refusal
    are all exercised by the same code path production uses. The only thing
    it does not exercise is the HTTP call itself.
    """

    replies: list[str] = field(default_factory=list)
    name: str = "scripted"
    model: str = "scripted-1"
    calls: list[list[Message]] = field(default_factory=list)
    fail_with: Exception | None = None

    def complete(self, messages: list[Message], *, max_output_tokens: int,
                 temperature: float) -> Completion:
        self.calls.append(list(messages))
        if self.fail_with is not None:
            raise self.fail_with
        if not self.replies:
            raise LLMError("ScriptedProvider ran out of prepared replies - "
                           "the code under test made more calls than the "
                           "test expected, which is itself worth knowing")
        text = self.replies.pop(0)
        prompt_chars = sum(len(m.content) for m in messages)
        return Completion(text=text, model=self.model,
                          usage=Usage.unknown(prompt_chars, len(text)))
