"""A locally served model over an OpenAI-compatible REST API.

WHAT THIS REPLACES AND WHY IT IS NOT AN IMPORT
----------------------------------------------
OpenTerminalUI ships `backend/services/lm_studio_client.py`, and the salvage
map marked it "take". Reading it changed the plan: it is a 129-line async
wrapper that needs `httpx` and the application's own pydantic settings, and
the protocol it speaks is the one `providers/openai.py` already speaks. So
the right move was not to import it but to notice what it proves - LM Studio,
Ollama and llama.cpp all expose `/v1/chat/completions` - and reuse the
provider path this project has already tested, metered and handled errors on.

ONE IDEA WAS GENUINELY WORTH TAKING: `parse_json_response`. Local models are
markedly worse than gpt-4o-mini at returning clean JSON - they wrap it in
```json fences, prepend "Here is the analysis:", and sometimes trail
commentary after the closing brace. Upstream's fence-stripping plus
regex-fallback extraction is the accumulated result of someone hitting all
three, and it is reproduced here with attribution rather than re-derived.

WHY IT IS FREE, AND WHY THE LEDGER STILL RECORDS IT
---------------------------------------------------
The call is a POST to a port on this machine, so it cannot cost money. The
model name is stamped with `LOCAL_PREFIX` so `price_for` prices it at zero
by route rather than by name - the server picks its own id and we cannot
enumerate those in advance. Every call is still appended to the spend ledger
at Rs 0, because "how many times did Stage 3 actually run" is a question the
ledger answers independently of money.

NO API KEY IS SENT. That is deliberate belt-and-braces: if `DESK_LLM_BASE_URL`
is ever mis-set to a paid endpoint, the request fails with 401 instead of
quietly billing an account against a ceiling that believes it is free.

WHAT THIS DOES NOT DO
---------------------
It does not make Stage 3's output good. A 4B model judging a numeric table
is not a small gpt-4o-mini, and nothing here measures the difference - that
comparison is a separate piece of work against the same shortlist. This
removes a billing blocker; it does not establish quality.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from desk.llm.base import (
    Completion, LLMError, LLMUnavailable, Message, Usage,
)
from desk.llm.budget import LOCAL_PREFIX

__all__ = ["DEFAULT_BASE_URL", "DEFAULT_LOCAL_MODEL", "LocalProvider",
           "parse_json_response"]

#: LM Studio's default. Ollama serves 11434 and llama.cpp 8080; all three
#: speak the same route, so only the port changes.
DEFAULT_BASE_URL = "http://localhost:1234/v1"

#: Deliberately generic. Most local servers ignore the field entirely and
#: serve whatever single model is loaded, so a wrong name here is usually
#: harmless - but it is reported back in `Completion.model`, so the ledger
#: records what the server said it used rather than what we asked for.
DEFAULT_LOCAL_MODEL = "local-model"

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class LocalProvider:
    """One chat completion per call against a local OpenAI-compatible server.

    Deliberately the same shape as `OpenAIProvider` - `model`, `configured`,
    `complete()` - so `MeteredClient` cannot tell them apart and the metered
    path is identical for both.
    """

    model: str = DEFAULT_LOCAL_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 120.0
    """Longer than OpenAI's 60s on purpose. A local model on CPU can take
    tens of seconds for a first token, and a timeout that fires mid-generation
    reads as "the server is broken" when it is merely slow."""
    name: str = field(default="local", init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        # The prefix is what tells the meter this route is free. Stamping it
        # here - rather than trusting a caller to pass it - means there is no
        # way to construct a LocalProvider that bills as a paid model.
        if not self.model.startswith(LOCAL_PREFIX):
            self.model = f"{LOCAL_PREFIX}{self.model}"

    @property
    def served_model(self) -> str:
        """The name to send on the wire, without our routing prefix."""
        return self.model[len(LOCAL_PREFIX):]

    @property
    def configured(self) -> bool:
        """True whenever a base URL exists.

        Unlike OpenAI there is no key to check, so this cannot tell whether
        the server is actually up. `health()` answers that, and `complete()`
        raises LLMUnavailable if it is not - which Stage 3 already treats as
        "this check did not run".
        """
        return bool(self.base_url)

    def health(self) -> bool:
        """Is a model actually being served? Never raises."""
        req = urllib.request.Request(f"{self.base_url}/models", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(5.0, self.timeout)):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def complete(self, messages: list[Message], *, max_output_tokens: int,
                 temperature: float) -> Completion:
        body = json.dumps({
            "model": self.served_model,
            "messages": [{"role": m.role, "content": m.content}
                         for m in messages],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json"})

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
                # The one case worth naming precisely: a local server does
                # not ask for credentials, so this almost certainly means
                # DESK_LLM_BASE_URL points at a PAID endpoint. Saying so is
                # more useful than "auth failed".
                raise LLMUnavailable(
                    f"the endpoint at {self.base_url} demanded credentials "
                    f"(HTTP {exc.code}), which a local server does not. "
                    f"DESK_LLM_BASE_URL is probably pointing at a paid API - "
                    f"nothing was sent but the prompt, and no key was "
                    f"included.") from exc
            if exc.code == 404:
                raise LLMUnavailable(
                    f"no model is loaded at {self.base_url} (HTTP 404). In "
                    f"LM Studio, load a model and start the server from the "
                    f"Developer tab.") from exc
            raise LLMError(
                f"the local model server returned HTTP {exc.code}: "
                f"{detail[:400]}") from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailable(
                f"nothing is listening at {self.base_url} ({exc.reason}). "
                f"Start LM Studio's local server, or unset "
                f"DESK_LLM_BASE_URL to go back to OpenAI. Stage 3 is "
                f"reported as not run and the deterministic stages are "
                f"unaffected.") from exc

        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LLMError("the local model's response exceeded the size cap")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"the local model server did not return JSON (first 200 "
                f"bytes: {raw[:200]!r})") from exc

        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"the local model's response had no message content: "
                f"{json.dumps(payload)[:300]}") from exc

        u = payload.get("usage") or {}
        if "prompt_tokens" in u and "completion_tokens" in u:
            usage = Usage(prompt_tokens=int(u["prompt_tokens"]),
                          completion_tokens=int(u["completion_tokens"]))
        else:
            # Local servers are far likelier than OpenAI to omit usage. It
            # costs nothing either way, but the token counts are still how we
            # tell a truncated answer from a short one, so estimate and flag
            # rather than zeroing.
            usage = Usage.unknown(sum(len(m.content) for m in messages),
                                  len(text))

        # Whatever the server called itself, reported back with our routing
        # prefix so the ledger records the real model at the free rate.
        served = str(payload.get("model") or self.served_model)
        return Completion(text=text, model=f"{LOCAL_PREFIX}{served}",
                          usage=usage, raw=payload)


def parse_json_response(content: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response, tolerating the usual mess.

    Adapted from OpenTerminalUI's `backend/services/lm_studio_client.py`
    (MIT). Three tolerances, each earned from a real failure mode of small
    local models rather than added defensively:

      1. ```json fences around the object
      2. prose before it ("Here is the analysis:")
      3. commentary after the closing brace

    Raises rather than returning {} on failure: an empty verdict that looks
    like a parsed one is exactly the silent-degradation this project keeps
    finding, and Stage 3 must be able to report "the model did not answer
    usably" as a caveat rather than as an empty pass.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Greedy match, so a trailing comment after the object is discarded
    # while a nested object inside it is kept.
    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise LLMError(
        f"could not find a JSON object in the model's response "
        f"(first 200 chars: {text[:200]!r})")
