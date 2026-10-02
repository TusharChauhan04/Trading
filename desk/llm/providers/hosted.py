"""A hosted OpenAI-compatible endpoint serving FREE models. The third route.

WHY THIS EXISTS, AND THE CORRECTION BEHIND IT
---------------------------------------------
Stage 3 had two routes: pay OpenAI, or install LM Studio and run a model on this
laptop. A third was sitting in OpenTerminalUI's `services/llm/` the whole time
and I did not open that subtree - I reported the extraction complete while
`factory.py`, `model_router.py`, `openai_compatible.py`, `ai_service.py` and
`llm_insights.py` had never been read. About 930 lines, and this is what they
were for:

    openrouter_base_url = "https://openrouter.ai/api/v1"
    DEFAULT_REASONING_MODELS = ["deepseek/deepseek-r1:free", ...]
    SAFETY_MODEL = "openai/gpt-oss-20b:free"

OpenRouter proxies many models behind one OpenAI-compatible API, and the ones
suffixed `:free` cost nothing. So Stage 3 can run on a 70-billion-parameter
model for Rs 0 with no local install and no payment - only a free account.

WHY NOT JUST EXTEND LocalProvider. That provider deliberately sends NO
Authorization header, and its docstring makes that a safety guarantee: if
DESK_LLM_BASE_URL is ever mis-set to a paid endpoint, the request fails with 401
instead of quietly billing someone. Adding a key to it would delete exactly the
property that makes a mis-set URL harmless. So this is a separate class, and the
two cannot be confused.

WHAT IS GENUINELY WORSE HERE THAN WITH LM STUDIO, stated because "free" is doing
a lot of work in that sentence:

  - IT IS AN EXTERNAL SERVICE. LM Studio's "nothing leaves your laptop" property
    is gone. Stage 3 sends a numeric shortlist, not positions or capital, but it
    does leave the machine.
  - FREE TIERS TRAIN ON YOUR DATA by default on some providers, and the policy is
    the provider's to change. For a table of public prices that is a small
    concern; it would not be for anything proprietary.
  - RATE LIMITS ARE REAL and much tighter than a paid tier - typically tens of
    requests a day. Fine for one call per trading session, useless for a
    backtest loop, and the desk already forbids calling an LLM inside one.
  - AVAILABILITY IS BEST-EFFORT. A free model can be withdrawn or saturated
    without notice, which is why upstream keeps an ordered fallback chain and
    a SAFETY_MODEL at the end of it.

WHAT IS NOT MEASURED. Whether any of these judges a shortlist as well as
gpt-4o-mini. Nothing here establishes quality; it establishes that the call can
be made for nothing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from desk.llm.base import (
    Completion, LLMError, LLMUnavailable, Message, Usage,
)
from desk.llm.budget import FREE_SUFFIX

__all__ = ["DEFAULT_FREE_MODEL", "FREE_MODEL_CHAIN", "OPENROUTER_BASE_URL",
           "HostedFreeProvider"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Upstream's own ordered chain, kept in their preference order. The last entry
#: is their SAFETY_MODEL - the one they fall back to when the others are
#: unavailable, which on a free tier happens.
FREE_MODEL_CHAIN = (
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "openai/gpt-oss-20b:free",
)

#: Stage 3 reads a table Stages 0-2 already computed and writes a short
#: structured verdict. It is judging a shortlist, not doing the analysis, so the
#: smallest capable model is the right default and a reasoning model is overkill.
DEFAULT_FREE_MODEL = "openai/gpt-oss-20b:free"

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class HostedFreeProvider:
    """One chat completion per call against a hosted free-tier endpoint.

    Same shape as OpenAIProvider and LocalProvider - `model`, `configured`,
    `complete()` - so MeteredClient cannot tell them apart and the authorise,
    call, record path is identical for all three.
    """

    model: str = DEFAULT_FREE_MODEL
    base_url: str = OPENROUTER_BASE_URL
    api_key: str | None = None
    timeout: float = 90.0
    """Longer than OpenAI's 60s: a free tier queues behind paid traffic, and a
    timeout that fires while queued reads as a broken endpoint."""
    name: str = field(default="hosted_free", init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.api_key is None:
            self.api_key = (os.environ.get("DESK_LLM_API_KEY")
                            or os.environ.get("OPENROUTER_API_KEY") or None)
        if not self.model.endswith(FREE_SUFFIX):
            # REFUSED, not silently allowed. This provider's whole premise is
            # that the call costs nothing, and the budget meter prices it at
            # zero on the strength of that suffix. A paid model reaching here
            # would be metered as free and spend real money unrecorded.
            raise ValueError(
                f"HostedFreeProvider is for free-tier models only, and "
                f"{self.model!r} does not end in {FREE_SUFFIX!r}. Use "
                f"OpenAIProvider for a paid model so the spend is metered.")

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def complete(self, messages: list[Message], *, max_output_tokens: int,
                 temperature: float) -> Completion:
        if not self.api_key:
            raise LLMUnavailable(
                "no key for the hosted free endpoint, so Stage 3 cannot run. "
                "Set DESK_LLM_API_KEY to an OpenRouter key - creating one is "
                "free and needs no payment method. Reported as a check that "
                "did not run; the deterministic stages are unaffected.")

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content}
                         for m in messages],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(4096).decode("utf-8", "replace")
            except Exception:                            # noqa: BLE001
                pass
            if exc.code in (401, 403):
                raise LLMUnavailable(
                    f"the hosted endpoint rejected the key (HTTP {exc.code}). "
                    f"Stage 3 is reported as not run.") from exc
            if exc.code == 429:
                # A free tier rate-limits HARD and this is its normal state
                # under load, not a fault. LLMUnavailable rather than
                # RateLimited: the desk makes one call per session, so there
                # is nothing to back off and retry within - the honest answer
                # is that the check did not run today.
                raise LLMUnavailable(
                    f"the free tier is rate-limited right now (HTTP 429). "
                    f"That is expected on a free plan rather than a fault - "
                    f"Stage 3 is reported as not run and the plan is "
                    f"unaffected. {detail[:140]}") from exc
            if exc.code == 404:
                raise LLMUnavailable(
                    f"model {self.model!r} is not available at "
                    f"{self.base_url} (HTTP 404). Free models are withdrawn "
                    f"and renamed without notice; try another from "
                    f"FREE_MODEL_CHAIN.") from exc
            raise LLMError(
                f"the hosted endpoint returned HTTP {exc.code}: "
                f"{detail[:400]}") from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailable(
                f"{self.base_url} is unreachable ({exc.reason}). Stage 3 is "
                f"reported as not run.") from exc

        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LLMError("the hosted response exceeded the size cap")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"the hosted endpoint did not return JSON "
                           f"(first 200 bytes: {raw[:200]!r})") from exc
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"the hosted response had no message content: "
                           f"{json.dumps(payload)[:300]}") from exc

        u = payload.get("usage") or {}
        if "prompt_tokens" in u and "completion_tokens" in u:
            usage = Usage(prompt_tokens=int(u["prompt_tokens"]),
                          completion_tokens=int(u["completion_tokens"]))
        else:
            usage = Usage.unknown(sum(len(m.content) for m in messages),
                                  len(text))

        # Report what the ROUTER actually served. OpenRouter can substitute a
        # model, and recording what we asked for rather than what answered
        # would make the ledger claim a model that never ran. The suffix is
        # preserved so the meter still prices it at zero.
        served = str(payload.get("model") or self.model)
        if not served.endswith(FREE_SUFFIX):
            served = f"{served}{FREE_SUFFIX}"
        return Completion(text=text, model=served, usage=usage, raw=payload)
