"""Failure types shared by every market-data source.

These lived in sources/nse.py, which was fine while NSE was the only source.
BSE needs the same two distinctions, and the alternatives were both worse:
importing them from the NSE module would make every BSE failure nominally an
NSE one, and giving BSE its own parallel pair would force every caller to
catch a tuple and would let the two drift apart.

This module imports nothing from the package, so it cannot participate in an
import cycle.
"""

from __future__ import annotations

__all__ = ["RateLimited", "SourceError", "TransientError"]


class SourceError(Exception):
    """A source returned something unusable, and retrying will not help.

    A renamed endpoint, a missing column, a wrong-day file. The right
    response is to stop and tell a human, not to try again more slowly.
    """


class RateLimited(SourceError):
    """The source is asking us to slow down. Retrying later WILL help.

    Kept distinct from SourceError because "wait and try again" and "the
    endpoint moved, stop" need opposite responses, and before the split they
    were indistinguishable at the call site.
    """


class TransientError(SourceError):
    """The network failed, not the source. Retrying WILL usually help.

    DNS resolution failures, connection resets, read timeouts. These were
    classified as plain SourceError - the "stop and tell a human" class -
    and it cost a whole backfill: a Nifty 200 run made ~2,000 sequential
    requests, exhausted the local resolver, and 117 of 200 symbols died on
    `getaddrinfo failed`. Every one of them would have succeeded a second
    later.

    A SUBCLASS of SourceError on purpose, so existing `except SourceError`
    handlers keep catching it and a caller that does not care about the
    distinction is unaffected. Only `fetch_with_retry` treats it specially.
    """
