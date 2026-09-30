"""A local model must be free, recorded, and never mistaken for a paid one.

The risk being tested is not "does the HTTP call work" - it is the two ways
this could go wrong quietly:

  1. A local call priced as gpt-4o, which would eat the monthly ceiling and
     stop Stage 3 for reasons that look like a budget problem.
  2. The `local:` exemption widening into "any unknown model is free", which
     would silently under-report a real bill.

Nothing here reaches the network. The provider is exercised against a stub
urlopen, because the point is the metering and the error mapping, not urllib.
"""

from __future__ import annotations

import json
import urllib.error
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest

from desk.llm.base import LLMError, LLMUnavailable, Message
from desk.llm.budget import (
    LOCAL_PREFIX, PRICES, CostMeter, price_for,
)
from desk.llm.client import MeteredClient
from desk.llm.providers.local import (
    DEFAULT_BASE_URL, LocalProvider, parse_json_response,
)


class _Resp(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _payload(text: str = "ok", model: str = "gemma-3-4b-it",
             usage: dict | None = None) -> bytes:
    body: dict = {"model": model,
                  "choices": [{"message": {"content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body).encode()


@pytest.fixture
def meter(tmp_path: Path) -> CostMeter:
    return CostMeter(ledger_path=tmp_path / "spend.jsonl",
                     monthly_ceiling_inr=Decimal("500.00"))


# -- pricing: the part that could cost money ---------------------------------

def test_a_local_model_is_free() -> None:
    p = price_for(f"{LOCAL_PREFIX}gemma-3-4b-it")
    assert p.input_per_mtok == 0
    assert p.output_per_mtok == 0


def test_the_exemption_is_the_prefix_only_not_unknown_models() -> None:
    """The guard that keeps this from becoming "unpriced means free".

    If this ever passes for a bare name, a real paid model typo'd into
    DESK_LLM_MODEL would bill silently at zero.
    """
    from desk.llm.base import BudgetExceeded

    with pytest.raises(BudgetExceeded, match="no price on file"):
        price_for("gemma-3-4b-it")
    with pytest.raises(BudgetExceeded, match="no price on file"):
        price_for("some-new-openai-model")


def test_paid_models_are_untouched_by_the_exemption() -> None:
    assert price_for("gpt-4o-mini") is PRICES["gpt-4o-mini"]
    assert price_for("gpt-4o").input_per_mtok > 0


def test_the_prefix_cannot_be_dodged_by_constructing_the_provider() -> None:
    """Stamping happens in __post_init__, so there is no unprefixed path."""
    assert LocalProvider(model="gemma").model == f"{LOCAL_PREFIX}gemma"
    # Already-prefixed input must not double up.
    assert LocalProvider(model=f"{LOCAL_PREFIX}gemma").model == \
        f"{LOCAL_PREFIX}gemma"


def test_the_wire_name_drops_our_prefix() -> None:
    """The server has never heard of `local:` and would 404 on it."""
    assert LocalProvider(model="gemma").served_model == "gemma"


# -- the metered path, end to end, at zero -----------------------------------

def test_a_local_call_costs_nothing_and_is_still_recorded(
    meter: CostMeter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rs 0 in the ledger, but a ROW in the ledger.

    "How many times did Stage 3 run" is a different question from "what did
    it cost", and the ledger has to answer both.
    """
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _Resp(_payload(
            usage={"prompt_tokens": 4000, "completion_tokens": 900})),
    )
    client = MeteredClient(provider=LocalProvider(model="gemma"), meter=meter)
    out = client.complete([Message(role="user", content="x" * 9000)],
                          purpose="stage3")

    assert out.cost_inr == 0
    assert out.text == "ok"
    rows = [json.loads(ln) for ln in
            meter.ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, "a free call must still leave a ledger row"
    # Compared as a Decimal, not a string: the meter quantises to paise, so
    # it writes "0.00" and asserting on the spelling tests the formatter.
    assert Decimal(rows[0]["cost_inr"]) == 0
    assert rows[0]["model"].startswith(LOCAL_PREFIX)
    assert rows[0]["prompt_tokens"] == 4000


def test_the_servers_own_model_name_reaches_the_ledger(
    meter: CostMeter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """We ask for one name; the server may serve another. Record theirs.

    Otherwise the ledger claims a model was used that never was.
    """
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _Resp(_payload(model="qwen2.5-7b-instruct")),
    )
    client = MeteredClient(provider=LocalProvider(model="gemma"), meter=meter)
    out = client.complete([Message(role="user", content="hi")])
    assert out.completion.model == f"{LOCAL_PREFIX}qwen2.5-7b-instruct"


def test_a_huge_local_prompt_is_never_refused_on_budget(
    meter: CostMeter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling must not gate a free route.

    A ceiling that blocks local calls would be the budget system stopping
    Stage 3 for a cost that does not exist.
    """
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _Resp(_payload()))
    tiny = CostMeter(ledger_path=meter.ledger_path,
                     monthly_ceiling_inr=Decimal("0.01"))
    client = MeteredClient(provider=LocalProvider(model="gemma"), meter=tiny)
    assert client.affordable([Message(role="user", content="x" * 500_000)])
    assert client.complete(
        [Message(role="user", content="x" * 500_000)]).cost_inr == 0


def test_missing_usage_is_estimated_and_flagged_not_zeroed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local servers often omit usage. Token counts still matter - they are
    how a truncated answer is told from a short one."""
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _Resp(_payload(text="hello")))
    out = LocalProvider(model="gemma").complete(
        [Message(role="user", content="a question")],
        max_output_tokens=100, temperature=0.0)
    assert out.usage.estimated
    assert out.usage.prompt_tokens > 0


# -- error mapping: each message has to point at the real fix ----------------

def _raise(exc):
    def _f(*a, **k):
        raise exc
    return _f


def test_nothing_listening_reads_as_did_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMUnavailable, not LLMError. Stage 3 treats the first as a caveat and
    the plan still gets produced; the second reads as a pipeline bug."""
    monkeypatch.setattr("urllib.request.urlopen",
                        _raise(urllib.error.URLError("refused")))
    with pytest.raises(LLMUnavailable, match="Start LM Studio"):
        LocalProvider().complete([Message(role="user", content="x")],
                                 max_output_tokens=10, temperature=0.0)


def test_a_401_says_the_base_url_points_at_a_paid_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single most useful diagnostic here.

    A local server never asks for credentials, so a 401 means the URL is
    wrong in a way that could have cost money - and the message must say so
    rather than 'auth failed'.
    """
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _raise(urllib.error.HTTPError(
            DEFAULT_BASE_URL, 401, "Unauthorized", {}, BytesIO(b"nope"))))
    with pytest.raises(LLMUnavailable, match="paid API"):
        LocalProvider().complete([Message(role="user", content="x")],
                                 max_output_tokens=10, temperature=0.0)


def test_a_404_says_to_load_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _raise(urllib.error.HTTPError(
            DEFAULT_BASE_URL, 404, "Not Found", {}, BytesIO(b""))))
    with pytest.raises(LLMUnavailable, match="load a model"):
        LocalProvider().complete([Message(role="user", content="x")],
                                 max_output_tokens=10, temperature=0.0)


def test_health_is_false_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("urllib.request.urlopen",
                        _raise(urllib.error.URLError("down")))
    assert LocalProvider().health() is False


# -- parse_json_response: the one idea worth taking from upstream ------------

@pytest.mark.parametrize("raw,want", [
    ('{"verdict":"pass"}', {"verdict": "pass"}),
    ('```json\n{"verdict":"pass"}\n```', {"verdict": "pass"}),
    ('```\n{"verdict":"pass"}\n```', {"verdict": "pass"}),
    ('Here is the analysis:\n{"verdict":"pass"}', {"verdict": "pass"}),
    ('{"verdict":"pass"}\n\nHope that helps!', {"verdict": "pass"}),
    ('{"a":{"b":1}} trailing', {"a": {"b": 1}}),
])
def test_json_is_recovered_from_the_usual_mess(raw: str, want: dict) -> None:
    assert parse_json_response(raw) == want


def test_unparseable_output_raises_rather_than_returning_empty() -> None:
    """An empty dict that looks parsed is the silent degradation this project
    keeps finding. Stage 3 must be able to report "no usable answer"."""
    for bad in ("", "I cannot help with that.", "[1,2,3]"):
        with pytest.raises(LLMError, match="could not find a JSON object"):
            parse_json_response(bad)


# -- settings: the route must be visible and must not leak -------------------

def test_a_base_url_configures_stage3_without_any_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desk.settings import Settings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DESK_LLM_BASE_URL", DEFAULT_BASE_URL)
    cfg = Settings.from_env(env_file=None)
    assert cfg.llm_local
    assert cfg.llm_configured, "a local server needs no key"


def test_a_local_url_wins_over_a_leftover_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desk.settings import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-leftover")
    monkeypatch.setenv("DESK_LLM_BASE_URL", DEFAULT_BASE_URL)
    cfg = Settings.from_env(env_file=None)
    assert cfg.llm_local
    described = "\n".join(cfg.describe())
    assert "ignored" in described, "the user must be told the key is unused"
    assert "sk-leftover" not in described, "describe() must never echo a key"


def test_no_base_url_leaves_openai_in_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desk.settings import Settings

    monkeypatch.delenv("DESK_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    cfg = Settings.from_env(env_file=None)
    assert not cfg.llm_local
    assert cfg.llm_configured
