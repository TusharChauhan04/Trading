"""Configuration from the environment, with a .env file as a convenience.

WHY NOT python-dotenv
---------------------
It is a fine library and this is thirty lines. Every other client in this
project is stdlib - urllib for NSE, BSE, the RSS reader and OpenAI - and a
dependency added for a single KEY=VALUE parse is a dependency that has to
be pinned, audited and kept compatible with pandas 2.3.3 and joblib 1.5.3
for the rest of the project's life.

THE REAL ENVIRONMENT ALWAYS WINS
--------------------------------
A variable already set in the process environment is NEVER overwritten by
the file. That ordering is the whole reason this is safe to call at
startup: a server run with a key injected by its orchestrator must not
have it silently replaced by a stale .env someone left on disk. The file
is the fallback for a laptop, not the source of truth for a deployment.

SECRETS ARE NEVER LOGGED
------------------------
`describe()` reports which variables are SET, never what they contain, and
the loader has no debug mode that would print a value. An API key that
reaches a log file has leaked, and log files get pasted into issues.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

__all__ = [
    "DEFAULT_CAPITAL", "DEFAULT_RISK_REWARD", "Settings", "load_env",
    "settings",
]

#: Trading capital, in rupees, when nothing else says otherwise. The web
#: app asks for it explicitly; this is the fallback for CLI and tests.
DEFAULT_CAPITAL = 100_000.0

#: Reward-to-risk. 2.0 means a 1:2 trade - risk one rupee to make two.
#: The risk engine previously targeted 2.5R. See desk/risk/engine.py and
#: the backtest: at 2.5R only 3 of 60 trades ever reached the target.
DEFAULT_RISK_REWARD = 2.0

_TRUTHY = {"1", "true", "yes", "on"}


def load_env(path: str | Path = ".env", *, override: bool = False) -> int:
    """Read KEY=VALUE lines into os.environ. Returns how many were set.

    Missing file is not an error - it is the normal state on a machine
    that injects its configuration some other way.

    Deliberately minimal: no interpolation, no multi-line values, no
    export prefixes. Anything that needs those belongs in a real secret
    store, and a half-implemented shell parser that handles some of them
    is worse than one that handles none.
    """
    p = Path(path)
    if not p.is_file():
        return 0

    count = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip ONE matching pair of quotes. A value that is genuinely
        # quoted in the file is almost always a copy-paste artefact.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # The process environment wins. See the module docstring.
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        count += 1
    return count


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the desk reads from the environment, in one place."""

    openai_api_key: str | None = None
    llm_monthly_ceiling_inr: Decimal = Decimal("500.00")
    llm_model: str = "gpt-4o-mini"
    usd_inr: Decimal = Decimal("90.00")
    default_capital: float = DEFAULT_CAPITAL
    risk_reward: float = DEFAULT_RISK_REWARD
    config_dir: Path | None = None

    @classmethod
    def from_env(cls, *, env_file: str | Path | None = ".env") -> "Settings":
        if env_file is not None:
            load_env(env_file)
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            llm_monthly_ceiling_inr=_decimal("DESK_LLM_MONTHLY_CEILING_INR",
                                             Decimal("500.00")),
            llm_model=os.environ.get("DESK_LLM_MODEL") or "gpt-4o-mini",
            usd_inr=_decimal("DESK_USD_INR", Decimal("90.00")),
            default_capital=_float("DESK_DEFAULT_CAPITAL", DEFAULT_CAPITAL),
            risk_reward=_float("DESK_RISK_REWARD", DEFAULT_RISK_REWARD),
            config_dir=(Path(os.environ["DESK_CONFIG_DIR"])
                        if os.environ.get("DESK_CONFIG_DIR") else None),
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key)

    def describe(self) -> list[str]:
        """What is configured. NEVER what it is set to."""
        return [
            f"OPENAI_API_KEY            {'set' if self.llm_configured else 'NOT SET - Stage 3 will not run'}",
            f"DESK_LLM_MODEL            {self.llm_model}",
            f"DESK_LLM_MONTHLY_CEILING  Rs {self.llm_monthly_ceiling_inr}",
            f"DESK_USD_INR              {self.usd_inr}",
            f"DESK_DEFAULT_CAPITAL      Rs {self.default_capital:,.0f}",
            f"DESK_RISK_REWARD          1:{self.risk_reward:g}",
        ]


def _decimal(name: str, fallback: Decimal) -> Decimal:
    raw = os.environ.get(name)
    if not raw:
        return fallback
    try:
        value = Decimal(raw.strip())
    except (InvalidOperation, ValueError):
        # A malformed ceiling must not silently become the default: that
        # is how a typo turns a Rs 500 cap into whatever the code says.
        raise ValueError(
            f"{name}={raw!r} is not a number. Fix it or unset it - it will "
            f"not be guessed at, because a mistyped spending ceiling that "
            f"quietly falls back to a default is the failure this whole "
            f"meter exists to prevent.") from None
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value


def _float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return fallback
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


#: Loaded once at import. Call Settings.from_env() again to re-read.
settings = Settings.from_env()
