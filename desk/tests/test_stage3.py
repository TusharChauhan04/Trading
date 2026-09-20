"""R7: the metered LLM layer and Stage 3.

Two properties dominate this file.

1. STAGE 3 CAN ONLY REMOVE. Every path is checked against the shortlist it
   was given, because the standing constraint is that no LLM may relax a
   risk gate and the only way to guarantee that is structurally.

2. NOTHING REACHES A PAID MODEL WITHOUT BEING AUTHORISED FIRST. The order
   authorise -> call -> record is tested directly, including that a refused
   call never reaches the provider at all.

No test here touches the network. ScriptedProvider implements the real
protocol, so the meter, the ledger and the budget refusal are exercised by
exactly the code path production uses.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pytest

from desk.contracts.enums import Regime, Stance
from desk.llm.base import (
    BudgetExceeded, LLMError, LLMUnavailable, Message, ScriptedProvider, Usage,
)
from desk.llm.budget import IST, CostMeter, ModelPrice, PRICES, price_for
from desk.llm.client import MeteredClient
from desk.scanner.stage2 import DEFAULT_FACTORS, Stage2Result
from desk.scanner.stage3 import (
    Stage3Result, build_prompt, parse_verdicts, run_stage3,
)

DAY = date(2026, 9, 11)


def _stage2(symbols=("AAA.NS", "BBB.NS", "CCC.NS")) -> Stage2Result:
    factors = DEFAULT_FACTORS[:2]
    n = len(symbols)
    data = {"score": [90.0 - 10 * i for i in range(n)],
            "factors_used": [2] * n,
            "penalty": [0.0] * n}
    for f in factors:
        data[f"rank_{f.name}"] = [95.0 - 10 * i for i in range(n)]
    ranked = pd.DataFrame(data, index=list(symbols))
    ranked.index.name = "symbol"
    return Stage2Result(as_of=DAY, regime=Regime.TRENDING_UP, ranked=ranked,
                        factors=factors, universe_in=1598, universe_out=len(symbols))


def _meter(tmp_path, ceiling="1000") -> CostMeter:
    return CostMeter(ledger_path=tmp_path / "ledger.jsonl",
                     monthly_ceiling_inr=Decimal(ceiling),
                     usd_inr=Decimal("90"))


def _client(tmp_path, replies, ceiling="1000") -> MeteredClient:
    return MeteredClient(provider=ScriptedProvider(replies=list(replies)),
                         meter=_meter(tmp_path, ceiling))


def _reply(*pairs) -> str:
    return json.dumps([
        {"symbol": s, "stance": st, "confidence": 0.8,
         "rationale": "because", "concerns": []} for s, st in pairs])


# =========================================================================
# The cost meter
# =========================================================================

def test_an_unpriced_model_is_refused_not_treated_as_free():
    with pytest.raises(BudgetExceeded, match="no price on file"):
        price_for("some-new-model-2030")


def test_cost_is_computed_from_the_published_rate(tmp_path):
    m = _meter(tmp_path)
    PRICES["test-model"] = ModelPrice(Decimal("1.00"), Decimal("2.00"))
    try:
        # 1M input + 1M output = $1 + $2 = $3 = Rs 270 at 90/USD.
        cost = m.cost_inr("test-model", Usage(1_000_000, 1_000_000))
        assert cost == Decimal("270")
    finally:
        del PRICES["test-model"]


def test_money_is_decimal_not_float(tmp_path):
    m = _meter(tmp_path)
    assert isinstance(m.cost_inr("gpt-4o-mini", Usage(1000, 1000)), Decimal)
    assert isinstance(m.month_to_date_inr(), Decimal)


def test_the_estimate_assumes_the_full_output_allowance(tmp_path):
    """A pre-flight check that assumes a short answer is not a ceiling
    check, it is a hope."""
    m = _meter(tmp_path)
    est = m.estimate_inr("gpt-4o", 1000, 700)
    actual_short = m.cost_inr("gpt-4o", Usage(1000, 10))
    assert est > actual_short


def test_authorise_refuses_when_the_ceiling_would_be_breached(tmp_path):
    m = _meter(tmp_path, ceiling="0.01")
    with pytest.raises(BudgetExceeded, match="ceiling"):
        m.authorise("gpt-4o", 500_000, 5_000)


def test_authorise_allows_a_call_within_the_ceiling(tmp_path):
    m = _meter(tmp_path, ceiling="1000")
    assert m.authorise("gpt-4o-mini", 5_000, 700) > 0


def test_spend_accumulates_across_calls(tmp_path):
    m = _meter(tmp_path)
    m.record("gpt-4o-mini", Usage(10_000, 1_000), purpose="a")
    m.record("gpt-4o-mini", Usage(10_000, 1_000), purpose="b")
    assert len(m.entries()) == 2
    assert m.month_to_date_inr() == 2 * m.cost_inr("gpt-4o-mini",
                                                   Usage(10_000, 1_000))


def test_only_this_calendar_month_counts(tmp_path):
    m = _meter(tmp_path)
    m.record("gpt-4o", Usage(100_000, 10_000),
             now=datetime(2026, 8, 31, 23, 0, tzinfo=IST))
    m.record("gpt-4o", Usage(100_000, 10_000),
             now=datetime(2026, 9, 1, 1, 0, tzinfo=IST))
    sept = m.month_to_date_inr(datetime(2026, 9, 15, 12, 0, tzinfo=IST))
    assert sept == m.cost_inr("gpt-4o", Usage(100_000, 10_000))


def test_a_truncated_ledger_line_does_not_destroy_the_history(tmp_path):
    m = _meter(tmp_path)
    m.record("gpt-4o-mini", Usage(1000, 100))
    with m.ledger_path.open("a", encoding="utf-8") as fh:
        fh.write('{"at": "2026-09-11T10:00:00+05:30", "cost_inr":\n')
    m.record("gpt-4o-mini", Usage(1000, 100))
    assert len(m.entries()) == 2


def test_estimated_token_counts_are_flagged_in_the_report(tmp_path):
    m = _meter(tmp_path)
    m.record("gpt-4o-mini", Usage.unknown(4000, 400),
             now=datetime.now(IST))
    assert "ESTIMATED" in m.report()


def test_unknown_usage_rounds_up_and_says_it_is_estimated():
    u = Usage.unknown(10, 10)
    assert u.estimated
    assert u.prompt_tokens == 4        # ceil(10/3), never 3
    assert u.total == 8


# =========================================================================
# The metered client: authorise -> call -> record
# =========================================================================

def test_a_refused_call_never_reaches_the_provider(tmp_path):
    """The budget is checked BEFORE the request leaves. If it were checked
    afterwards it would only report money already spent."""
    provider = ScriptedProvider(replies=["[]"])
    client = MeteredClient(provider=provider,
                           meter=_meter(tmp_path, ceiling="0"))
    PRICES["pricey-1"] = ModelPrice(Decimal("100"), Decimal("100"))
    provider.model = "pricey-1"
    try:
        with pytest.raises(BudgetExceeded):
            client.complete([Message("user", "x" * 10_000)])
    finally:
        del PRICES["pricey-1"]
    assert provider.calls == [], "the provider was called despite the refusal"
    assert client.calls == 0


def test_a_successful_call_is_recorded_with_real_usage(tmp_path):
    client = _client(tmp_path, ["[]"])
    out = client.complete([Message("user", "hello")], purpose="test")
    assert client.calls == 1
    rows = client.meter.entries()
    assert len(rows) == 1
    assert rows[0]["purpose"] == "test"
    assert out.authorised_inr >= out.cost_inr


def test_a_provider_failure_records_nothing(tmp_path):
    """No usage figures came back, and inventing them would corrupt the one
    number that cannot be recomputed."""
    provider = ScriptedProvider(replies=[], fail_with=LLMError("boom"))
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    with pytest.raises(LLMError):
        client.complete([Message("user", "hi")])
    assert client.meter.entries() == []
    assert client.calls == 0


def test_affordable_answers_without_raising(tmp_path):
    assert _client(tmp_path, [], ceiling="1000").affordable(
        [Message("user", "short")])

    # A priced model is needed to be unaffordable at all: ScriptedProvider's
    # own model is free, so no ceiling can refuse it.
    PRICES["pricey-3"] = ModelPrice(Decimal("500"), Decimal("500"))
    broke = _client(tmp_path, [], ceiling="0.0001")
    broke.provider.model = "pricey-3"
    try:
        assert not broke.affordable([Message("user", "x" * 10_000)])
    finally:
        del PRICES["pricey-3"]


def test_temperature_defaults_to_zero(tmp_path):
    """A different answer on a re-run makes the journal's 'why was this
    vetoed on Tuesday' unanswerable."""
    assert _client(tmp_path, []).temperature == 0.0


# =========================================================================
# Stage 3: the subset guarantee
# =========================================================================

def test_without_a_client_the_shortlist_passes_through_unchanged():
    s2 = _stage2()
    out = run_stage3(s2, client=None)
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"]
    assert out.ran is False
    assert out.unavailable and "did not run" in out.unavailable[0]
    assert out.cost_inr == 0


def test_kept_is_always_a_subset_in_ranking_order(tmp_path):
    s2 = _stage2()
    client = _client(tmp_path, [_reply(("AAA.NS", "Buy"), ("BBB.NS", "Sell"),
                                       ("CCC.NS", "Overweight"))])
    out = run_stage3(s2, client=client)
    assert out.kept == ["AAA.NS", "CCC.NS"]
    assert set(out.kept) <= set(s2.ranked.index)


def test_only_buy_and_overweight_survive(tmp_path):
    s2 = _stage2(("A.NS", "B.NS", "C.NS", "D.NS", "E.NS"))
    client = _client(tmp_path, [_reply(
        ("A.NS", "Buy"), ("B.NS", "Overweight"), ("C.NS", "Hold"),
        ("D.NS", "Underweight"), ("E.NS", "Sell"))])
    out = run_stage3(s2, client=client)
    assert out.kept == ["A.NS", "B.NS"]
    assert set(out.vetoed) == {"C.NS", "D.NS", "E.NS"}


def test_a_name_the_model_invents_is_ignored_not_added(tmp_path):
    """Stage 3 may only remove. An extra symbol is not a candidate."""
    s2 = _stage2(("AAA.NS",))
    client = _client(tmp_path, [_reply(("AAA.NS", "Buy"),
                                       ("SURPRISE.NS", "Buy"))])
    out = run_stage3(s2, client=client)
    assert out.kept == ["AAA.NS"]
    assert "SURPRISE.NS" not in out.verdicts


def test_vetoing_everything_is_a_valid_outcome(tmp_path):
    s2 = _stage2()
    client = _client(tmp_path, [_reply(("AAA.NS", "Hold"), ("BBB.NS", "Hold"),
                                       ("CCC.NS", "Sell"))])
    out = run_stage3(s2, client=client)
    assert out.kept == []
    assert out.is_empty and out.ran


def test_empty_shortlist_is_handled_without_calling_the_model(tmp_path):
    empty = pd.DataFrame(columns=["score", "factors_used", "penalty"])
    empty.index.name = "symbol"
    s2 = Stage2Result(as_of=DAY, regime=Regime.RANGE, ranked=empty,
                      factors=DEFAULT_FACTORS[:1])
    provider = ScriptedProvider(replies=["[]"])
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    out = run_stage3(s2, client=client)
    assert out.kept == [] and out.ran
    assert provider.calls == []


def test_only_one_call_is_made_for_the_whole_shortlist(tmp_path):
    """One call per day is the cost argument. One per name is eight."""
    provider = ScriptedProvider(replies=[_reply(
        ("AAA.NS", "Buy"), ("BBB.NS", "Buy"), ("CCC.NS", "Buy"))])
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    run_stage3(_stage2(), client=client)
    assert len(provider.calls) == 1


def test_candidates_bounds_how_many_are_sent(tmp_path):
    s2 = _stage2(("A.NS", "B.NS", "C.NS"))
    client = _client(tmp_path, [_reply(("A.NS", "Buy"), ("B.NS", "Buy"))])
    out = run_stage3(s2, client=client, candidates=2)
    assert out.considered == 2
    assert "C.NS" not in out.verdicts


# =========================================================================
# Stage 3: REVIEW is not HOLD
# =========================================================================

def test_an_unreadable_answer_becomes_review_not_hold(tmp_path):
    s2 = _stage2(("AAA.NS",))
    client = _client(tmp_path, ["I'm afraid I can't do that."])
    out = run_stage3(s2, client=client)
    assert out.kept == []
    assert out.needs_review == ["AAA.NS"]
    assert out.verdicts["AAA.NS"].stance is Stance.REVIEW
    assert out.verdicts["AAA.NS"].parsed is False


def test_a_symbol_omitted_from_the_answer_becomes_review(tmp_path):
    s2 = _stage2(("AAA.NS", "BBB.NS"))
    client = _client(tmp_path, [_reply(("AAA.NS", "Buy"))])
    out = run_stage3(s2, client=client)
    assert out.kept == ["AAA.NS"]
    assert out.needs_review == ["BBB.NS"]
    assert out.verdicts["BBB.NS"].stance is Stance.REVIEW


def test_hold_and_review_are_recorded_differently(tmp_path):
    """HOLD is a neutral the model chose. REVIEW is the absence of an
    answer. Collapsing them lets a parse failure look like a judgement."""
    s2 = _stage2(("AAA.NS", "BBB.NS"))
    client = _client(tmp_path, [_reply(("AAA.NS", "Hold"))])
    out = run_stage3(s2, client=client)
    assert out.verdicts["AAA.NS"].stance is Stance.HOLD
    assert out.verdicts["AAA.NS"].parsed is True
    assert "AAA.NS" not in out.needs_review
    assert out.verdicts["BBB.NS"].stance is Stance.REVIEW
    assert "BBB.NS" in out.needs_review


def test_json_inside_a_code_fence_is_read(tmp_path):
    s2 = _stage2(("AAA.NS",))
    fenced = "```json\n" + _reply(("AAA.NS", "Buy")) + "\n```"
    out = run_stage3(s2, client=_client(tmp_path, [fenced]))
    assert out.kept == ["AAA.NS"]


def test_json_surrounded_by_prose_is_read(tmp_path):
    s2 = _stage2(("AAA.NS",))
    chatty = "Here is my assessment:\n" + _reply(("AAA.NS", "Buy")) + "\nHope that helps!"
    out = run_stage3(s2, client=_client(tmp_path, [chatty]))
    assert out.kept == ["AAA.NS"]


def test_an_unknown_stance_string_is_not_silently_kept():
    v = parse_verdicts(json.dumps([{"symbol": "AAA.NS", "stance": "MAYBE"}]),
                       ["AAA.NS"])
    assert v["AAA.NS"].stance is Stance.REVIEW
    assert not v["AAA.NS"].parsed


def test_confidence_is_clamped_and_never_crashes():
    v = parse_verdicts(json.dumps([
        {"symbol": "A.NS", "stance": "Buy", "confidence": 5},
        {"symbol": "B.NS", "stance": "Buy", "confidence": "nonsense"},
        {"symbol": "C.NS", "stance": "Buy", "confidence": -2},
    ]), ["A.NS", "B.NS", "C.NS"])
    assert v["A.NS"].confidence == 1.0
    assert v["B.NS"].confidence == 0.0
    assert v["C.NS"].confidence == 0.0


def test_a_bare_symbol_matches_a_suffixed_one():
    v = parse_verdicts(json.dumps([{"symbol": "RELIANCE", "stance": "Buy"}]),
                       ["RELIANCE.NS"])
    assert v["RELIANCE.NS"].stance is Stance.BUY


# =========================================================================
# Stage 3: failure modes are skips, not crashes
# =========================================================================

def test_a_budget_refusal_skips_the_stage_and_keeps_the_shortlist(tmp_path):
    provider = ScriptedProvider(replies=["[]"])
    PRICES["pricey-2"] = ModelPrice(Decimal("500"), Decimal("500"))
    provider.model = "pricey-2"
    client = MeteredClient(provider=provider, meter=_meter(tmp_path, "0.0001"))
    try:
        out = run_stage3(_stage2(), client=client)
    finally:
        del PRICES["pricey-2"]
    assert out.ran is False
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"]
    assert any("ceiling" in u for u in out.unavailable)
    assert provider.calls == []


def test_an_unconfigured_provider_skips_the_stage(tmp_path):
    provider = ScriptedProvider(fail_with=LLMUnavailable("no key"))
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    out = run_stage3(_stage2(), client=client)
    assert out.ran is False
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"]


def test_a_model_error_skips_the_stage_rather_than_failing_the_day(tmp_path):
    provider = ScriptedProvider(fail_with=LLMError("500 from upstream"))
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    out = run_stage3(_stage2(), client=client)
    assert out.ran is False
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"]
    assert "NO TRADE" not in out.summary()


# =========================================================================
# The prompt
# =========================================================================

def test_the_prompt_carries_stage2s_own_explanation():
    s2 = _stage2()
    p = build_prompt(s2, ["AAA.NS"])
    assert "AAA.NS" in p
    assert s2.explain("AAA.NS").splitlines()[0] in p


def test_absences_are_stated_rather_than_omitted():
    """An omitted section reads, to the model, like a company with nothing
    notable. 'Not available' is the same discipline as checks_skipped."""
    p = build_prompt(_stage2(), ["AAA.NS"], fundamentals=None, events=None,
                     news=None)
    assert "Fundamentals: not available" in p
    assert "NOT CHECKED" in p
    assert "NEWS CONTEXT: not available" in p


def test_fundamentals_appear_when_supplied():
    table = pd.DataFrame(
        {"nature": ["consolidated"], "period_end": [date(2026, 6, 30)],
         "days_since_filing": [20], "revenue_growth_yoy_pct": [12.5],
         "profit_growth_yoy_pct": [8.0], "net_margin_pct": [11.0],
         "margin_change_pp": [0.5]}, index=["AAA"])
    p = build_prompt(_stage2(), ["AAA.NS"], fundamentals=table)
    assert "12.5%" in p and "consolidated" in p


def test_a_symbol_with_no_filing_says_so():
    table = pd.DataFrame({"nature": []}, index=pd.Index([], name="symbol"))
    p = build_prompt(_stage2(), ["AAA.NS"], fundamentals=table)
    assert "no filing on file" in p


def test_the_prompt_names_the_required_output_order():
    p = build_prompt(_stage2(), ["AAA.NS", "BBB.NS"])
    assert "AAA.NS, BBB.NS" in p


def test_the_summary_reports_a_skipped_stage_honestly():
    out = run_stage3(_stage2(), client=None)
    assert "UNAVAILABLE" in out.summary()


# =========================================================================
# The OpenAI provider, entirely offline
# =========================================================================

def _fake_http(monkeypatch, payload: dict | bytes, status: int = 200):
    """Stand in for urlopen without touching the network."""
    import urllib.error
    import urllib.request

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1):
            return payload if isinstance(payload, bytes) else \
                json.dumps(payload).encode()

    def _open(req, timeout=None):
        if status != 200:
            raise urllib.error.HTTPError(
                "https://api.openai.com/v1/chat/completions", status,
                "err", {}, None)
        _open.seen = req
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _open)
    return _open


def test_openai_without_a_key_is_unavailable_not_an_error(monkeypatch):
    from desk.llm.providers.openai import OpenAIProvider
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAIProvider()
    assert not p.configured
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        p.complete([Message("user", "hi")], max_output_tokens=10,
                   temperature=0.0)


def test_openai_reads_the_reply_and_the_real_token_counts(monkeypatch):
    from desk.llm.providers.openai import OpenAIProvider
    seen = _fake_http(monkeypatch, {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "[]"}}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
    })
    out = OpenAIProvider(api_key="sk-test").complete(
        [Message("system", "s"), Message("user", "u")],
        max_output_tokens=50, temperature=0.0)

    assert out.text == "[]"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (123, 45)
    assert out.usage.estimated is False

    body = json.loads(seen.seen.data)
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.0
    assert body["max_completion_tokens"] == 50
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_openai_never_puts_the_key_in_the_url_or_body(monkeypatch):
    from desk.llm.providers.openai import OpenAIProvider
    seen = _fake_http(monkeypatch, {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    OpenAIProvider(api_key="sk-secret-value").complete(
        [Message("user", "u")], max_output_tokens=5, temperature=0.0)
    assert "sk-secret-value" not in seen.seen.full_url
    assert b"sk-secret-value" not in seen.seen.data
    assert seen.seen.headers["Authorization"] == "Bearer sk-secret-value"


def test_openai_missing_usage_is_estimated_never_zero(monkeypatch):
    """An unmeasured call still cost money. Reporting zero would make a
    month's spend under-report silently."""
    from desk.llm.providers.openai import OpenAIProvider
    _fake_http(monkeypatch, {"choices": [{"message": {"content": "hello"}}]})
    out = OpenAIProvider(api_key="sk-test").complete(
        [Message("user", "some prompt text")], max_output_tokens=5,
        temperature=0.0)
    assert out.usage.estimated is True
    assert out.usage.total > 0


def test_openai_rejected_credentials_are_unavailable_not_a_hard_error(monkeypatch):
    from desk.llm.providers.openai import OpenAIProvider
    _fake_http(monkeypatch, b"", status=401)
    with pytest.raises(LLMUnavailable):
        OpenAIProvider(api_key="sk-bad").complete(
            [Message("user", "u")], max_output_tokens=5, temperature=0.0)


def test_openai_server_error_is_a_model_error(monkeypatch):
    from desk.llm.providers.openai import OpenAIProvider
    _fake_http(monkeypatch, b"", status=500)
    with pytest.raises(LLMError):
        OpenAIProvider(api_key="sk-test").complete(
            [Message("user", "u")], max_output_tokens=5, temperature=0.0)


def test_openai_non_json_response_is_refused(monkeypatch):
    from desk.llm.providers.openai import OpenAIProvider
    _fake_http(monkeypatch, b"<html>maintenance</html>")
    with pytest.raises(LLMError, match="did not return JSON"):
        OpenAIProvider(api_key="sk-test").complete(
            [Message("user", "u")], max_output_tokens=5, temperature=0.0)


def test_openai_response_with_no_content_is_refused(monkeypatch):
    from desk.llm.providers.openai import OpenAIProvider
    _fake_http(monkeypatch, {"choices": []})
    with pytest.raises(LLMError, match="no message content"):
        OpenAIProvider(api_key="sk-test").complete(
            [Message("user", "u")], max_output_tokens=5, temperature=0.0)


def test_the_default_model_is_priced(monkeypatch):
    """A default nobody priced would refuse to run on its first real call."""
    from desk.llm.providers.openai import DEFAULT_MODEL
    assert price_for(DEFAULT_MODEL) is not None


# =========================================================================
# 429 means two different things, and they need opposite responses
# =========================================================================

def _http_error(monkeypatch, status: int, body: bytes):
    import urllib.error
    import urllib.request

    def _open(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions", status, "err", {},
            __import__("io").BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", _open)


def test_no_credits_is_unavailable_not_an_error(monkeypatch):
    """REGRESSION from the first real call: the account had no credits and
    the plan said "the model errored", which sends the reader looking for
    a bug in the pipeline when the fix is to top up the account.

    It is not a failed request - it is the capability being absent, the
    same as a missing key.
    """
    from desk.llm.providers.openai import OpenAIProvider

    _http_error(monkeypatch, 429, b'{"error":{"message":"You have no '
                                  b'credits remaining.","code":'
                                  b'"insufficient_quota"}}')
    with pytest.raises(LLMUnavailable) as exc:
        OpenAIProvider(api_key="sk-valid").complete(
            [Message("user", "u")], max_output_tokens=5, temperature=0.0)

    msg = str(exc.value)
    assert "NO CREDITS" in msg
    assert "key itself is valid" in msg, "must not read as a bad key"
    assert "Nothing was charged" in msg
    assert "billing" in msg, "the message has to name the fix"


def test_a_genuine_rate_limit_is_retryable(monkeypatch):
    """The other meaning of 429. Retrying THIS one helps; retrying a
    no-credits account forever only delays the plan."""
    from desk.marketdata.sources.errors import RateLimited

    from desk.llm.providers.openai import OpenAIProvider

    _http_error(monkeypatch, 429, b'{"error":{"message":"Rate limit '
                                  b'reached","code":"rate_limit_exceeded"}}')
    with pytest.raises(RateLimited):
        OpenAIProvider(api_key="sk-valid").complete(
            [Message("user", "u")], max_output_tokens=5, temperature=0.0)


def test_no_credits_still_lets_the_day_produce_a_plan(tmp_path, monkeypatch):
    """Stage 3 is an enrichment layer. Whatever the provider does, the
    deterministic 98% of the funnel must still answer."""
    from desk.llm.providers.openai import OpenAIProvider

    _http_error(monkeypatch, 429, b'{"error":{"code":"insufficient_quota"}}')
    client = MeteredClient(provider=OpenAIProvider(api_key="sk-valid"),
                           meter=_meter(tmp_path))
    out = run_stage3(_stage2(), client=client)

    assert out.ran is False
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"], \
        "the shortlist must pass through untouched"
    assert any("NO CREDITS" in u for u in out.unavailable)


def test_nothing_is_charged_when_the_call_fails(tmp_path, monkeypatch):
    """record() runs only after a successful completion, so a refused
    request must leave the ledger empty."""
    from desk.llm.providers.openai import OpenAIProvider

    _http_error(monkeypatch, 429, b'{"error":{"code":"insufficient_quota"}}')
    client = MeteredClient(provider=OpenAIProvider(api_key="sk-valid"),
                           meter=_meter(tmp_path))
    run_stage3(_stage2(), client=client)
    assert client.meter.entries() == []
    assert client.meter.month_to_date_inr() == 0


def test_no_provider_error_can_fail_the_day(tmp_path):
    """Stage 3 catches everything. A provider raising its own transport
    type - a SourceError, say - must not take the plan with it."""
    from desk.marketdata.sources.errors import RateLimited

    provider = ScriptedProvider(fail_with=RateLimited("slow down"))
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    out = run_stage3(_stage2(), client=client)
    assert out.ran is False
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"]


# =========================================================================
# The verdict cache - the only thing standing between a refresh loop and
# the monthly ceiling
# =========================================================================

def _cached_client(tmp_path, replies):
    from desk.scanner.stage3 import VerdictCache
    return (_client(tmp_path, replies), VerdictCache(tmp_path))


def test_a_repeat_request_for_the_same_shortlist_spends_nothing(tmp_path):
    """SECURITY/COST. Nothing else in the request path costs money, so
    nothing else needed a throttle - but /plan/today had none, and a
    browser tab set to auto-refresh would buy one real completion per
    reload. Rs 500 at Rs 0.05 a call is ~10,000 reloads; a loop reaches
    that in minutes, after which Stage 3 goes quiet for the month."""
    from desk.scanner.stage3 import VerdictCache

    s2 = _stage2()
    provider = ScriptedProvider(replies=[_reply(
        ("AAA.NS", "Buy"), ("BBB.NS", "Sell"), ("CCC.NS", "Buy"))])
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    cache = VerdictCache(tmp_path)

    first = run_stage3(s2, client=client, cache=cache)
    second = run_stage3(s2, client=client, cache=cache)

    assert len(provider.calls) == 1, "the second request bought a verdict again"
    assert first.kept == second.kept == ["AAA.NS", "CCC.NS"]
    assert second.ran is True
    assert second.cost_inr == 0
    assert any("no second charge" in u for u in second.unavailable)


def test_a_changed_shortlist_re_asks(tmp_path):
    """Keying on the DATE alone would be wrong - the shortlist genuinely
    changes when new bars land, and reusing a verdict for a different set
    of names answers a question nobody asked."""
    from desk.scanner.stage3 import VerdictCache

    provider = ScriptedProvider(replies=[
        _reply(("AAA.NS", "Buy")),
        _reply(("ZZZ.NS", "Buy")),
    ])
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    cache = VerdictCache(tmp_path)

    run_stage3(_stage2(("AAA.NS",)), client=client, cache=cache)
    run_stage3(_stage2(("ZZZ.NS",)), client=client, cache=cache)
    assert len(provider.calls) == 2


def test_reordering_the_shortlist_re_asks(tmp_path):
    """The prompt asks for answers in the order given, so a different
    order is a different prompt."""
    from desk.scanner.stage3 import shortlist_digest
    assert shortlist_digest(["A", "B"]) != shortlist_digest(["B", "A"])


def test_a_cache_hit_goes_through_identical_keep_veto_logic(tmp_path):
    """Two copies of 'which stance keeps a name' would drift invisibly -
    a cached day quietly applying different rules from a fresh one."""
    from desk.scanner.stage3 import VerdictCache

    s2 = _stage2(("A.NS", "B.NS", "C.NS"))
    provider = ScriptedProvider(replies=[_reply(
        ("A.NS", "Buy"), ("B.NS", "Hold"), ("C.NS", "Overweight"))])
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    cache = VerdictCache(tmp_path)

    fresh = run_stage3(s2, client=client, cache=cache)
    hit = run_stage3(s2, client=client, cache=cache)

    assert fresh.kept == hit.kept
    assert set(fresh.vetoed) == set(hit.vetoed)
    assert {k: v.stance for k, v in fresh.verdicts.items()} == \
           {k: v.stance for k, v in hit.verdicts.items()}


def test_a_corrupt_cache_entry_is_a_miss_not_a_partial_verdict(tmp_path):
    """Half a verdict set would silently veto the names it failed to
    parse - the worst possible way to save five paise."""
    from desk.scanner.stage3 import VerdictCache, shortlist_digest

    cache = VerdictCache(tmp_path)
    cache.root.mkdir(parents=True, exist_ok=True)
    digest = shortlist_digest(["AAA.NS", "BBB.NS", "CCC.NS"])
    (cache.root / f"{DAY.isoformat()}_{digest}.json").write_text(
        '{"verdicts": {"AAA.NS": {"stance": "Buy"}, "BBB.NS": {"stance":',
        encoding="utf-8")

    provider = ScriptedProvider(replies=[_reply(
        ("AAA.NS", "Buy"), ("BBB.NS", "Buy"), ("CCC.NS", "Buy"))])
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    out = run_stage3(_stage2(), client=client, cache=cache)

    assert len(provider.calls) == 1, "a corrupt entry must re-ask"
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"]


def test_an_unknown_stance_in_the_cache_is_a_miss(tmp_path):
    from desk.scanner.stage3 import VerdictCache, shortlist_digest
    import json as _j

    cache = VerdictCache(tmp_path)
    cache.root.mkdir(parents=True, exist_ok=True)
    digest = shortlist_digest(["AAA.NS"])
    (cache.root / f"{DAY.isoformat()}_{digest}.json").write_text(
        _j.dumps({"verdicts": {"AAA.NS": {"stance": "MAYBE"}}}),
        encoding="utf-8")

    provider = ScriptedProvider(replies=[_reply(("AAA.NS", "Buy"))])
    client = MeteredClient(provider=provider, meter=_meter(tmp_path))
    run_stage3(_stage2(("AAA.NS",)), client=client, cache=cache)
    assert len(provider.calls) == 1


def test_no_cache_supplied_still_works(tmp_path):
    """The cache is optional - the backtest passes no client at all."""
    out = run_stage3(_stage2(), client=_client(tmp_path, [_reply(
        ("AAA.NS", "Buy"), ("BBB.NS", "Buy"), ("CCC.NS", "Buy"))]))
    assert out.kept == ["AAA.NS", "BBB.NS", "CCC.NS"]


def test_the_funnel_passes_a_cache():
    """Wired, not merely available."""
    import inspect
    from desk.api import main as api
    assert "cache=VerdictCache(" in inspect.getsource(api._run_funnel)
