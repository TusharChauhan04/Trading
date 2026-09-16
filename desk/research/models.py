"""The shapes exchange-disclosed research comes back in.

ONE RULE GOVERNS THIS ENTIRE MODULE: every record carries `disclosed_at`, the
moment the exchange published it, and it is REQUIRED.

That is not bureaucracy. It is the difference between a research layer that can
be backtested and one that cannot. Master prompt section 26 asks for
fundamentals lagged to the actual filing date rather than the period end - but
stated as a rule it is something you can forget on a Tuesday. Stated as a
required field it is something the type system will not let you forget.

The consequence is the interesting part. NSE genuinely publishes filings it
cannot date: nine of RELIANCE's 130 quarterly results, all from 2005-2007,
carry no broadcast timestamp at all (and their XBRL links are the placeholder
`.../xbrl/-`, so there is no document behind them either). Those records cannot
be represented by the types below, and that is correct - they must not silently
acquire a plausible date.

So every parser here returns `(records, undated)`, exactly as
`parse_corporate_actions` returns `(actions, unparsed)`. The caller gets the
usable records AND the ones that could not be dated, and can do what it likes
about the second list - but it cannot fail to notice them. A research layer that
quietly drops a filing is one that will quietly drop the filing that mattered.

WHAT `disclosed_at` IS NOT: it is not the period the numbers describe. A Q3
result covers Oct-Dec but is disclosed in mid-January, and a backtest simulating
1 January must not see it. `Filing` carries both, named so they cannot be
confused: `period_start`/`period_end` for what it describes, `disclosed_at` for
when it became knowable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

__all__ = [
    "Announcement", "BoardMeeting", "CorporateEvent", "Filing",
    "InsiderDeal", "ShareholdingSnapshot", "Undated", "ResultPeriod",
]


@dataclass(frozen=True, slots=True)
class Undated:
    """A record the exchange published without a usable disclosure time.

    Kept rather than dropped. `raw` is the original row so a human can see
    exactly what arrived, and `reason` says why it could not be used.
    """
    symbol: str
    kind: str
    reason: str
    raw: dict

    def __str__(self) -> str:
        return f"{self.symbol} {self.kind}: {self.reason}"


class ResultPeriod(str, Enum):
    QUARTERLY = "quarterly"
    HALF_YEARLY = "half_yearly"
    ANNUAL = "annual"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Filing:
    """A financial result disclosed to the exchange.

    This is the fundamentals record. `xbrl_url` is where the actual numbers
    live; it is None when NSE published the row without a document, which
    happens for older filings and is why it is optional while `disclosed_at`
    is not.
    """
    symbol: str
    disclosed_at: datetime
    period_start: date | None
    period_end: date | None
    period: ResultPeriod
    audited: bool | None
    """None means NSE did not say. Not the same as unaudited."""
    consolidated: bool | None
    company_name: str = ""
    relating_to: str = ""
    xbrl_url: str | None = None
    isin: str = ""

    @property
    def has_numbers(self) -> bool:
        """Whether there is a document to parse. A Filing without one is a
        record that a result was announced, not the result itself."""
        return self.xbrl_url is not None


@dataclass(frozen=True, slots=True)
class Announcement:
    """A corporate announcement. Tier-1 news: the company's own words to the
    exchange, timestamped to the second."""
    symbol: str
    disclosed_at: datetime
    category: str
    """NSE's own `desc`, e.g. 'Analysts/Institutional Investor Meet'."""
    text: str
    attachment_url: str | None = None

    @property
    def session_phase(self) -> str:
        """before_market / during_market / after_hours, from the real clock.

        Master prompt section 11 asks for exactly this distinction, and it is
        only answerable because `an_dt` carries a time. An announcement at
        17:45 is tradeable tomorrow; one at 11:00 moved the price today.

        NSE's normal session is 09:15-15:30 IST.
        """
        t = self.disclosed_at.time()
        if t < _OPEN:
            return "before_market"
        if t <= _CLOSE:
            return "during_market"
        return "after_hours"


@dataclass(frozen=True, slots=True)
class BoardMeeting:
    """A board meeting the exchange was told about. The `meeting_date` is the
    forward-looking part - it is what the event gate reads to keep a trade
    away from a results announcement."""
    symbol: str
    disclosed_at: datetime
    meeting_date: date | None
    purpose: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class ShareholdingSnapshot:
    """Who owns the company, as at `as_at`, disclosed at `disclosed_at`.

    Promoter holding and pledge are the two fields that actually change a
    trading decision, so they are first-class rather than left in a blob.
    """
    symbol: str
    disclosed_at: datetime
    as_at: date | None
    promoter_pct: float | None = None
    public_pct: float | None = None
    employee_trust_pct: float | None = None


@dataclass(frozen=True, slots=True)
class InsiderDeal:
    """An insider / SAST disclosure: who bought or sold, and how much."""
    symbol: str
    disclosed_at: datetime
    acquirer: str
    mode: str
    """'Off Market', 'Market Sale', etc."""
    shares_after: float | None = None
    pct_after: float | None = None
    from_date: date | None = None
    to_date: date | None = None
    regulation: str = ""


@dataclass(frozen=True, slots=True)
class CorporateEvent:
    """A scheduled, forward-looking event from the market-wide calendar.

    Deliberately NOT carrying `disclosed_at` as a required field: this is the
    one shape that is about the FUTURE rather than a past disclosure, and its
    whole purpose is the `event_date`. It is the input to
    `RegimeState.EventProximity`, the only regime dimension that can force
    NO TRADE on its own.
    """
    symbol: str
    event_date: date | None
    purpose: str
    company: str = ""
    description: str = ""


# NSE's normal equity session, IST. Pre-open (09:00-09:15) and the closing
# session are deliberately not modelled here - `session_phase` answers "could
# this have moved today's price", and for that the normal session is the line
# that matters.
from datetime import time as _time
_OPEN = _time(9, 15)
_CLOSE = _time(15, 30)
