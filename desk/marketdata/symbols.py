"""Canonical symbol registry.

Four dialects are already in this project and they do not agree:

    platform / yfinance / TradingAgents   RELIANCE.NS      BTC-USD
    Vibe-Trading india_broker_loader      RELIANCE  + NSE  (suffix stripped)
    NautilusTrader                        RELIANCE.NSE     BTC/USDT.BINANCE
    Freqtrade                             (crypto only)    BTC/USDT
    Zerodha Kite                          RELIANCE         (exchange separate)

CANONICAL FORM = the yfinance suffix convention: ``RELIANCE.NS`` / ``TCS.BO``.
Chosen because TradingAgents and Vibe-Trading already both use it, so it is
the only dialect that needs no translation on the two most important paths.

Everything crossing a boundary translates here. Nothing guesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from desk.contracts.enums import Exchange

_SUFFIX = {Exchange.NSE: ".NS", Exchange.BSE: ".BO"}
_FROM_SUFFIX = {v: k for k, v in _SUFFIX.items()}

# NSE tickers are uppercase alphanumerics plus & - and a few legacy dots.
_VALID_BASE = re.compile(r"^[A-Z0-9][A-Z0-9&\-]{0,29}$")


class SymbolError(ValueError):
    """Raised rather than guessing. A wrong symbol is a wrong trade."""


@dataclass(frozen=True, slots=True)
class Symbol:
    base: str                 # RELIANCE
    exchange: Exchange        # NSE

    # ---- canonical --------------------------------------------------------
    @property
    def canonical(self) -> str:
        return f"{self.base}{_SUFFIX[self.exchange]}"

    def __str__(self) -> str:
        return self.canonical

    # ---- dialects out -----------------------------------------------------
    @property
    def yfinance(self) -> str:
        return self.canonical

    @property
    def kite(self) -> str:
        """Zerodha carries the exchange separately; the tradingsymbol is bare."""
        return self.base

    @property
    def kite_instrument(self) -> str:
        return f"{self.exchange.value}:{self.base}"

    @property
    def nautilus(self) -> str:
        return f"{self.base}.{self.exchange.value}"

    @property
    def vibe(self) -> str:
        """Vibe-Trading's india_broker_loader strips the suffix and passes the
        exchange code alongside - same shape as Kite."""
        return self.base

    # ---- dialects in ------------------------------------------------------
    @classmethod
    def parse(cls, raw: str, default_exchange: Exchange = Exchange.NSE) -> "Symbol":
        """Accept any dialect this project produces. Reject anything ambiguous."""
        if not raw or not isinstance(raw, str):
            raise SymbolError(f"empty or non-string symbol: {raw!r}")
        s = raw.strip().upper()

        # NSE:RELIANCE  /  BSE:TCS
        if ":" in s:
            ex, _, base = s.partition(":")
            # Resolve the exchange inside the try and validate the base OUTSIDE
            # it. SymbolError subclasses ValueError, so a single try around both
            # reported a bad ticker base as "unknown exchange" - sending whoever
            # read the message hunting the wrong half of the string.
            try:
                exchange = Exchange(ex)
            except ValueError as exc:
                raise SymbolError(
                    f"unknown exchange {ex!r} in {raw!r}; expected NSE or BSE"
                ) from exc
            return cls(_check(base), exchange)

        # RELIANCE.NS / TCS.BO  (canonical)  or  RELIANCE.NSE (nautilus)
        if "." in s:
            base, _, tail = s.rpartition(".")
            suffix = f".{tail}"
            if suffix in _FROM_SUFFIX:
                return cls(_check(base), _FROM_SUFFIX[suffix])
            if tail in {e.value for e in Exchange}:
                return cls(_check(base), Exchange(tail))
            raise SymbolError(
                f"unrecognised suffix {suffix!r} in {raw!r}; "
                f"expected one of {sorted(_FROM_SUFFIX)} or NSE/BSE"
            )

        # Bare RELIANCE - only safe with an explicit default.
        return cls(_check(s), default_exchange)


def _check(base: str) -> str:
    base = base.strip().upper()
    if not _VALID_BASE.match(base):
        raise SymbolError(f"implausible ticker base: {base!r}")
    return base


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------

def to_canonical(raw: str, default_exchange: Exchange = Exchange.NSE) -> str:
    return Symbol.parse(raw, default_exchange).canonical


DIALECTS = ("canonical", "yfinance", "kite", "kite_instrument", "nautilus", "vibe")


def translate(raw: str, dialect: str, default_exchange: Exchange = Exchange.NSE) -> str:
    """translate("RELIANCE", "nautilus") -> "RELIANCE.NSE"

    An explicit whitelist, not getattr. Unbounded getattr returns any attribute
    that happens to exist - `translate(s, "parse")` handed back a bound method
    and `translate(s, "__class__")` the class itself, both typed as `str` and
    both flowing downstream to corrupt whatever formatted them.
    """
    if dialect not in DIALECTS:
        raise SymbolError(
            f"unknown dialect {dialect!r}; expected one of {'|'.join(DIALECTS)}"
        )
    return getattr(Symbol.parse(raw, default_exchange), dialect)


BENCHMARK = {
    Exchange.NSE: "^NSEI",     # Nifty 50
    Exchange.BSE: "^BSESN",    # Sensex
}
