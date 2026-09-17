"""OpenAI, over its REST API with the standard library.

WHY NOT THE OFFICIAL SDK
------------------------
It would be one more pinned dependency in a project that already pins
pandas to 2.3.3 for Nautilus and joblib to 1.5.3 for Freqtrade, and it would
be pinned for the sake of one POST to one endpoint. Every other network
client here (NSE, BSE, the RSS reader) is stdlib urllib for the same reason.
If streaming, tool-calling or the assistants API is ever needed, that is the
moment to reconsider - not before.

THE KEY
-------
Read from the OPENAI_API_KEY environment variable and never written to disk,
never logged, and never included in an exception message. `LLMUnavailable`
is raised when it is absent, which Stage 3 treats as "this check did not
run" rather than as a failure - the same distinction quality.py draws with
checks_skipped.
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

__all__ = ["DEFAULT_MODEL", "OpenAIProvider"]

_ENDPOINT = "https://api.openai.com/v1/chat/completions"

#: The cheapest capable model. Stage 3 reads a table of numbers that Stages
#: 0-2 already computed and writes a short structured verdict; it is not
#: doing the analysis, it is judging a shortlist. Starting at the small model
#: and moving up if the output is measurably worse is the right order -
#: starting large and never checking is how the per-call cost argument stops
#: being true.
DEFAULT_MODEL = "gpt-4o-mini"

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass(slots=True)
class OpenAIProvider:
    """One chat completion per call. No streaming, no tools, no state."""

    model: str = DEFAULT_MODEL
    timeout: float = 60.0
    api_key: str | None = None
    name: str = field(default="openai", init=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY") or None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: list[Message], *, max_output_tokens: int,
                 temperature: float) -> Completion:
        if not self.api_key:
            raise LLMUnavailable(
                "OPENAI_API_KEY is not set, so Stage 3 cannot run. This is "
                "reported as a check that did not run, not as a failed one - "
                "the deterministic stages are unaffected.")

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content}
                         for m in messages],
            "max_completion_tokens": max_output_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            _ENDPOINT, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            # The body can carry a useful message, but it is bounded and the
            # key is never echoed into it by us.
            detail = ""
            try:
                detail = exc.read(4096).decode("utf-8", "replace")
            except Exception:                        # noqa: BLE001
                pass
            if exc.code in (401, 403):
                raise LLMUnavailable(
                    f"OpenAI rejected the credentials (HTTP {exc.code}). "
                    f"Stage 3 will be reported as not run.") from exc
            raise LLMError(
                f"OpenAI returned HTTP {exc.code}: {detail[:400]}") from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"OpenAI unreachable: {exc.reason}") from exc

        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LLMError("OpenAI response exceeded the size cap")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"OpenAI did not return JSON (first 200 bytes: "
                f"{raw[:200]!r})") from exc

        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"OpenAI response had no message content: "
                f"{json.dumps(payload)[:300]}") from exc

        u = payload.get("usage") or {}
        if "prompt_tokens" in u and "completion_tokens" in u:
            usage = Usage(prompt_tokens=int(u["prompt_tokens"]),
                          completion_tokens=int(u["completion_tokens"]))
        else:
            # Never silently zero. An unmeasured call still cost money, and
            # the estimate is flagged so the ledger can say so.
            usage = Usage.unknown(sum(len(m.content) for m in messages),
                                  len(text))

        return Completion(text=text, model=payload.get("model", self.model),
                          usage=usage, raw=payload)
