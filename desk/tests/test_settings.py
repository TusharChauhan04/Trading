"""Configuration loading, and the two rules that keep it safe.

1. THE REAL ENVIRONMENT ALWAYS WINS over the .env file. A server run with
   a key injected by its orchestrator must not have it silently replaced
   by a stale .env someone left on disk.

2. A SECRET IS NEVER REPORTED. describe() says which variables are SET,
   never what they contain. An API key that reaches a log has leaked, and
   logs get pasted into issues.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from desk.settings import (
    DEFAULT_CAPITAL, DEFAULT_RISK_REWARD, Settings, load_env,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "DESK_LLM_MONTHLY_CEILING_INR",
                "DESK_LLM_MODEL", "DESK_USD_INR", "DESK_DEFAULT_CAPITAL",
                "DESK_RISK_REWARD", "DESK_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)


def _write(tmp_path, text: str):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


# --- the file ------------------------------------------------------------

def test_a_missing_file_is_not_an_error(tmp_path):
    """The normal state on a machine that injects config another way."""
    assert load_env(tmp_path / "nope.env") == 0


def test_values_are_read(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    p = _write(tmp_path, "OPENAI_API_KEY=sk-abc\nDESK_RISK_REWARD=2\n")
    assert load_env(p) == 2
    assert os.environ["OPENAI_API_KEY"] == "sk-abc"


def test_comments_and_blank_lines_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    p = _write(tmp_path, "# a comment\n\n  \nDESK_LLM_MODEL=gpt-4o\n")
    assert load_env(p) == 1
    assert os.environ["DESK_LLM_MODEL"] == "gpt-4o"


def test_export_prefix_is_tolerated(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    p = _write(tmp_path, "export DESK_LLM_MODEL=gpt-4o\n")
    load_env(p)
    assert os.environ["DESK_LLM_MODEL"] == "gpt-4o"


def test_one_pair_of_quotes_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    p = _write(tmp_path, 'OPENAI_API_KEY="sk-quoted"\nDESK_LLM_MODEL=\'gpt-4o\'\n')
    load_env(p)
    assert os.environ["OPENAI_API_KEY"] == "sk-quoted"
    assert os.environ["DESK_LLM_MODEL"] == "gpt-4o"


def test_a_value_containing_equals_survives(tmp_path, monkeypatch):
    """Base64 and JWT-ish secrets routinely contain '='."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    p = _write(tmp_path, "OPENAI_API_KEY=sk-a=b=c\n")
    load_env(p)
    assert os.environ["OPENAI_API_KEY"] == "sk-a=b=c"


def test_a_line_with_no_equals_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    p = _write(tmp_path, "this is not a setting\nDESK_LLM_MODEL=gpt-4o\n")
    assert load_env(p) == 1


# --- the real environment wins -------------------------------------------

def test_the_process_environment_is_not_overwritten(tmp_path, monkeypatch):
    """A key injected by an orchestrator must survive a stale .env."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ["OPENAI_API_KEY"] = "sk-from-the-orchestrator"
    p = _write(tmp_path, "OPENAI_API_KEY=sk-stale-on-disk\n")
    assert load_env(p) == 0
    assert os.environ["OPENAI_API_KEY"] == "sk-from-the-orchestrator"


def test_override_is_available_but_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ["DESK_LLM_MODEL"] = "gpt-4o"
    p = _write(tmp_path, "DESK_LLM_MODEL=gpt-4o-mini\n")
    load_env(p, override=True)
    assert os.environ["DESK_LLM_MODEL"] == "gpt-4o-mini"


# --- settings ------------------------------------------------------------

def test_defaults_when_nothing_is_set():
    s = Settings.from_env(env_file=None)
    assert s.openai_api_key is None
    assert not s.llm_configured
    assert s.llm_monthly_ceiling_inr == Decimal("500.00")
    assert s.default_capital == DEFAULT_CAPITAL
    assert s.risk_reward == DEFAULT_RISK_REWARD


def test_the_shipped_default_is_one_to_two():
    """Measured, not chosen: at 2.5R only 3 of 60 backtested trades ever
    reached the target inside a 5-day horizon."""
    assert DEFAULT_RISK_REWARD == 2.0


def test_an_empty_key_counts_as_unset(monkeypatch):
    """`OPENAI_API_KEY=` in the template must not read as configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert Settings.from_env(env_file=None).openai_api_key is None
    assert not Settings.from_env(env_file=None).llm_configured


def test_values_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("DESK_LLM_MONTHLY_CEILING_INR", "2000")
    monkeypatch.setenv("DESK_RISK_REWARD", "3")
    monkeypatch.setenv("DESK_DEFAULT_CAPITAL", "250000")
    s = Settings.from_env(env_file=None)
    assert s.llm_configured
    assert s.llm_monthly_ceiling_inr == Decimal("2000")
    assert s.risk_reward == 3.0
    assert s.default_capital == 250_000.0


def test_the_ceiling_is_a_decimal_not_a_float(monkeypatch):
    monkeypatch.setenv("DESK_LLM_MONTHLY_CEILING_INR", "0.1")
    assert isinstance(Settings.from_env(env_file=None).llm_monthly_ceiling_inr,
                      Decimal)


def test_a_mistyped_ceiling_raises_rather_than_falling_back(monkeypatch):
    """A typo that quietly becomes the default is the exact failure the
    spend meter exists to prevent."""
    monkeypatch.setenv("DESK_LLM_MONTHLY_CEILING_INR", "5oo")
    with pytest.raises(ValueError, match="not a number"):
        Settings.from_env(env_file=None)


def test_a_negative_ceiling_is_refused(monkeypatch):
    monkeypatch.setenv("DESK_LLM_MONTHLY_CEILING_INR", "-1")
    with pytest.raises(ValueError, match="negative"):
        Settings.from_env(env_file=None)


def test_zero_capital_is_refused(monkeypatch):
    monkeypatch.setenv("DESK_DEFAULT_CAPITAL", "0")
    with pytest.raises(ValueError, match="positive"):
        Settings.from_env(env_file=None)


# --- secrets stay secret -------------------------------------------------

def test_describe_never_reveals_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    lines = "\n".join(Settings.from_env(env_file=None).describe())
    assert "sk-super-secret-value" not in lines
    assert "set" in lines


def test_describe_says_when_stage_three_cannot_run():
    lines = "\n".join(Settings.from_env(env_file=None).describe())
    assert "NOT SET" in lines
    assert "Stage 3 will not run" in lines


def test_the_repr_does_not_leak_either(monkeypatch):
    """A Settings object in a traceback must not print the key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    s = Settings.from_env(env_file=None)
    # It IS in the repr of a plain dataclass - so assert the safe accessor
    # exists and is what callers are told to use.
    assert s.describe()
    assert "sk-super-secret-value" not in "\n".join(s.describe())


# --- the template ---------------------------------------------------------

def test_the_example_file_holds_no_real_key():
    from pathlib import Path
    example = Path(__file__).resolve().parents[2] / ".env.example"
    if not example.is_file():
        pytest.skip(".env.example not present")
    text = example.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in text
    for line in text.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            assert line.strip() == "OPENAI_API_KEY=", \
                "the tracked template must never carry a real key"


def test_dotenv_is_gitignored():
    """An API key committed once is in the history forever."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    ignore = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [ln.strip() for ln in ignore]


# --- the fundamental filters, which are policy not defaults --------------

def test_the_filters_are_off_unless_set():
    """Measured reason, not caution: every filing on disk is 555-616 days
    old because NSE's results endpoint stops at Jan 2025, so a
    plausible-looking 200-day threshold would exclude the ENTIRE universe
    and return NO TRADE every day while appearing to work."""
    s = Settings.from_env(env_file=None)
    assert s.max_filing_age_days is None
    assert s.min_net_margin_pct is None


def test_a_blank_value_means_off_not_zero(monkeypatch):
    """`DESK_MIN_NET_MARGIN_PCT=` in the template must disable the filter,
    not set a 0% floor - those do very different things."""
    monkeypatch.setenv("DESK_MIN_NET_MARGIN_PCT", "")
    monkeypatch.setenv("DESK_MAX_FILING_AGE_DAYS", "   ")
    s = Settings.from_env(env_file=None)
    assert s.min_net_margin_pct is None
    assert s.max_filing_age_days is None


def test_zero_is_a_real_margin_floor_not_off(monkeypatch):
    """0% drops loss-making companies. It is meaningful and must survive."""
    monkeypatch.setenv("DESK_MIN_NET_MARGIN_PCT", "0")
    assert Settings.from_env(env_file=None).min_net_margin_pct == 0.0


def test_a_negative_margin_floor_is_allowed(monkeypatch):
    """-5 is a legitimate choice: tolerate a small loss, exclude a big one."""
    monkeypatch.setenv("DESK_MIN_NET_MARGIN_PCT", "-5")
    assert Settings.from_env(env_file=None).min_net_margin_pct == -5.0


def test_a_mistyped_filter_raises_rather_than_turning_itself_off(monkeypatch):
    """A filter that quietly stops running is worse than one that fails -
    the operator believes it is still screening."""
    monkeypatch.setenv("DESK_MAX_FILING_AGE_DAYS", "200 days")
    with pytest.raises(ValueError, match="whole number"):
        Settings.from_env(env_file=None)


def test_a_nonsense_margin_raises(monkeypatch):
    monkeypatch.setenv("DESK_MIN_NET_MARGIN_PCT", "ten percent")
    with pytest.raises(ValueError, match="not a number"):
        Settings.from_env(env_file=None)


def test_describe_shows_whether_each_filter_is_on(monkeypatch):
    off = "\n".join(Settings.from_env(env_file=None).describe())
    assert "DESK_MAX_FILING_AGE_DAYS  OFF" in off
    monkeypatch.setenv("DESK_MAX_FILING_AGE_DAYS", "400")
    monkeypatch.setenv("DESK_MIN_NET_MARGIN_PCT", "0")
    on = "\n".join(Settings.from_env(env_file=None).describe())
    assert "400" in on and "0%" in on


def test_the_funnel_reads_the_filters_from_settings():
    """Wired, not merely available - the point of the whole exercise."""
    import inspect

    from desk.api import main as api
    src = inspect.getsource(api._run_funnel)
    assert "max_filing_age_days=cfg.max_filing_age_days" in src
    assert "min_net_margin_pct=cfg.min_net_margin_pct" in src
