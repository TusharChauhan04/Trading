"""Reading the values an exchange actually sends.

Extracted from desk/marketdata/sources/nse.py because a second source is
coming. R6 adds BSE, whose payloads have the same shape of messiness - a
field that is sometimes a string, sometimes a number, sometimes the literal
"-" - and the alternative was for BSE's parser to import NSE's private
helpers, or copy them. Neither is a boundary; this is.

Every one of these refuses rather than guesses. That is the whole point: a
parser that turns "-" into a URL, or NaN into a shareholding percentage,
produces data that looks right and is not.
"""

from __future__ import annotations

import math

__all__ = ["ABSENT", "as_float", "as_text", "text_or_none"]

#: Every spelling an exchange uses for "there is nothing here". "-" is by far
#: the most common: 456 of RELIANCE's 3,345 announcements carry it in place of
#: an attachment URL, and passed through verbatim it becomes a link that a
#: downstream fetch will dutifully try to open.
ABSENT = {"", "-", "na", "null", "none"}


def as_text(v) -> str:
    """A field as a string, whatever type actually arrived.

    `(v or "").strip()` is the obvious idiom and it is wrong: it substitutes
    "" only for FALSY values, so an int, list or dict passes straight through
    to .strip() and raises AttributeError. These endpoints are undocumented,
    so a field changing type between revisions is a when-not-if - and one
    such field would otherwise cost an entire batch.
    """
    if v is None:
        return ""
    return v.strip() if isinstance(v, str) else str(v).strip()


def text_or_none(v) -> str | None:
    """Text, or None for every spelling of absent."""
    t = as_text(v)
    return None if t.lower() in ABSENT else t


def as_float(v) -> float | None:
    """A number, or None. NaN and Infinity are refused rather than passed on:
    json.loads accepts both as bare literals, and a NaN shareholding compares
    false against every threshold downstream while looking like a real number
    on a dashboard."""
    try:
        f = float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
