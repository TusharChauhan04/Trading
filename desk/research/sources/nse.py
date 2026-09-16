"""Parsing NSE's research endpoints into the shapes in desk/research/models.py.

Split out of desk/marketdata/sources/nse.py, which had grown to five unrelated
jobs (HTTP session, free-text corporate actions, holiday master, bhavcopy CSV,
and these). The deciding argument was not size: it is that `desk.research`
owns these shapes, so the code building them belongs on the same side of that
boundary. R6 adds a BSE source producing the SAME shapes, and it needs a
symmetric place to live rather than a second reach-across into another
package's module.

Pure functions over already-fetched payloads. Nothing here opens a socket -
`NseSession` in desk/marketdata/sources/nse.py remains the only thing that
does, because a cookie-bearing, throttled HTTP client is genuinely
cross-cutting and both domains need it.

EVERY PARSER RETURNS (records, undated). NSE publishes filings it cannot date;
those must never be dropped and never given a plausible timestamp. See
desk/research/models.py for the full reasoning.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from desk.marketdata.sources.nse import (
    ARCHIVES,
    _num,
    _s,
    _text_or_none,
    parse_nse_date,
)
from desk.research.models import (
    Announcement,
    BoardMeeting,
    CorporateEvent,
    Filing,
    InsiderDeal,
    ResultPeriod,
    ShareholdingSnapshot,
    Undated,
)

__all__ = [
    "parse_announcements", "parse_board_meetings", "parse_event_calendar",
    "parse_insider_deals", "parse_nse_datetime", "parse_results",
    "parse_shareholding",
]

#: NSE writes timestamps four ways across these endpoints, and in mixed case:
#: '16-Jan-2025 20:20:21', '2026-09-16 17:45:33', '26-Apr-2007 18:00',
#: '17-Jul-2026'. Measured across every captured fixture, not guessed.
_DT_FORMATS = (
    "%d-%b-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%b-%Y %H:%M",
    "%d-%b-%Y",
    "%Y-%m-%d",
)

#: NSE emits this when there is no document, rather than omitting the field.
_NO_DOCUMENT = f"{ARCHIVES}/corporate/xbrl/-"


def parse_nse_datetime(s: str | None) -> datetime | None:
    """A timestamp, or None. Never a guess.

    NSE mixes case across endpoints ('16-JUL-2026' and '16-Jul-2026' both
    occur). No special handling is needed: strptime's %b is already
    case-insensitive in CPython. An earlier version of this function carried a
    title-casing retry for that purpose - measured across all 20,556 date-like
    values in the fixtures, it was reached 7,274 times and changed the outcome
    zero times. Removed rather than left as reassuring dead code.
    """
    raw = _s(s)
    if not raw or raw in ("-", "NA", "null"):
        return None
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


def _disclosed(row: dict, *fields: str) -> datetime | None:
    """First usable timestamp among `fields`, in preference order.

    Order matters: broadCastDate is when the exchange DISSEMINATED it, which
    is when the market could act. filingDate is when the company submitted,
    which is earlier and is the honest fallback when NSE published no
    broadcast time.
    """
    for f in fields:
        dt = parse_nse_datetime(row.get(f))
        if dt is not None:
            return dt
    return None


def parse_results(payload: list[dict], symbol: str
                  ) -> tuple[list[Filing], list[Undated]]:
    """Financial results into `Filing` records - the fundamentals shape."""
    out, undated = [], []
    for row in payload:
        if not isinstance(row, dict):
            continue
        dt = _disclosed(row, "broadCastDate", "exchdisstime", "filingDate")
        if dt is None:
            undated.append(Undated(
                symbol=symbol, kind="result",
                reason=f"no broadcast, dissemination or filing time "
                       f"({row.get('relatingTo') or '?'} "
                       f"{row.get('fromDate') or '?'} to "
                       f"{row.get('toDate') or '?'})",
                raw=row))
            continue
        xbrl = _text_or_none(row.get("xbrl"))
        # NSE also emits the archive path with a bare "-" filename, which is
        # not empty but is not a document either.
        if xbrl and (xbrl == _NO_DOCUMENT or xbrl.endswith("/-")):
            xbrl = None
        out.append(Filing(
            symbol=symbol,
            disclosed_at=dt,
            period_start=parse_nse_date(row.get("fromDate")),
            period_end=parse_nse_date(row.get("toDate")),
            period=_result_period(row.get("period")),
            audited=_tri_state(row.get("audited"), "Audited", "Un-Audited"),
            consolidated=_tri_state(row.get("consolidated"),
                                    "Consolidated", "Non-Consolidated"),
            company_name=_s(row.get("companyName")),
            relating_to=_s(row.get("relatingTo")),
            xbrl_url=xbrl,
            isin=_s(row.get("isin")),
        ))
    return out, undated


def _result_period(v) -> ResultPeriod:
    t = _s(v).lower()
    if t.startswith("quarter"):
        return ResultPeriod.QUARTERLY
    if "half" in t:
        return ResultPeriod.HALF_YEARLY
    if t.startswith("annual") or t.startswith("year"):
        return ResultPeriod.ANNUAL
    return ResultPeriod.UNKNOWN


def _tri_state(v, yes: str, no: str) -> bool | None:
    """True / False / None. None means NSE did not say, which is NOT the same
    as saying no - the same third-state discipline as Sizing.checks_skipped."""
    t = _s(v).lower()
    if t == yes.lower():
        return True
    if t == no.lower():
        return False
    return None


def parse_announcements(payload: list[dict], symbol: str
                        ) -> tuple[list[Announcement], list[Undated]]:
    out, undated = [], []
    for row in payload:
        if not isinstance(row, dict):
            continue
        dt = _disclosed(row, "an_dt", "sort_date", "exchdisstime")
        if dt is None:
            undated.append(Undated(symbol=symbol, kind="announcement",
                                   reason="no announcement timestamp", raw=row))
            continue
        att = _text_or_none(row.get("attchmntFile"))
        out.append(Announcement(
            symbol=symbol,
            disclosed_at=dt,
            category=_s(row.get("desc")),
            text=_s(row.get("attchmntText")),
            attachment_url=att,
        ))
    return out, undated


def parse_board_meetings(payload: list[dict], symbol: str
                         ) -> tuple[list[BoardMeeting], list[Undated]]:
    out, undated = [], []
    for row in payload:
        if not isinstance(row, dict):
            continue
        dt = _disclosed(row, "bm_timestamp", "exchdisstime")
        if dt is None:
            undated.append(Undated(symbol=symbol, kind="board_meeting",
                                   reason="no intimation timestamp", raw=row))
            continue
        out.append(BoardMeeting(
            symbol=symbol,
            disclosed_at=dt,
            meeting_date=parse_nse_date(row.get("bm_date")),
            purpose=_s(row.get("bm_purpose")),
            description=_s(row.get("bm_desc")),
        ))
    return out, undated


def parse_shareholding(payload: list[dict], symbol: str
                       ) -> tuple[list[ShareholdingSnapshot], list[Undated]]:
    out, undated = [], []
    for row in payload:
        if not isinstance(row, dict):
            continue
        dt = _disclosed(row, "broadcastDate", "systemDate")
        if dt is None:
            undated.append(Undated(symbol=symbol, kind="shareholding",
                                   reason="no broadcast timestamp", raw=row))
            continue
        out.append(ShareholdingSnapshot(
            symbol=symbol,
            disclosed_at=dt,
            as_at=parse_nse_date(row.get("date")),
            promoter_pct=_num(row.get("pr_and_prgrp")),
            public_pct=_num(row.get("public_val")),
            employee_trust_pct=_num(row.get("employeeTrusts")),
        ))
    return out, undated


def parse_insider_deals(payload: list[dict], symbol: str
                        ) -> tuple[list[InsiderDeal], list[Undated]]:
    out, undated = [], []
    for row in payload:
        if not isinstance(row, dict):
            continue
        # NOT acqtoDt. That is when the TRANSACTION completed, which in every
        # sampled row precedes the actual broadcast by 1-5 days - so using it
        # as `disclosed_at` would tell a backtest the market knew about an
        # insider trade days before it was disclosed. That is precisely the
        # leak this module exists to prevent, so a row we cannot date is
        # reported as undated instead. intimDt (the company's own intimation)
        # IS a disclosure event and is an acceptable last resort.
        dt = _disclosed(row, "date", "exchdisstime", "intimDt")
        if dt is None:
            undated.append(Undated(symbol=symbol, kind="insider_deal",
                                   reason="no disclosure timestamp", raw=row))
            continue
        out.append(InsiderDeal(
            symbol=symbol,
            disclosed_at=dt,
            acquirer=_s(row.get("acqName")),
            mode=_s(row.get("acqMode")),
            shares_after=_num(row.get("afterAcqSharesNo")),
            pct_after=_num(row.get("afterAcqSharesPer")),
            from_date=parse_nse_date(row.get("acqfromDt")),
            to_date=parse_nse_date(row.get("acqtoDt")),
            regulation=_s(row.get("anex")),
        ))
    return out, undated


def parse_event_calendar(payload: list[dict]) -> list[CorporateEvent]:
    """Forward-looking, market-wide. No (records, undated) split: an event
    with no date is simply unusable here and is dropped with the rest of the
    row, because unlike a filing there is no historical record being lost -
    the calendar is refetched every day."""
    out = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        sym = _s(row.get("symbol"))
        if not sym:
            continue
        event_date = parse_nse_date(row.get("date"))
        if event_date is None:
            # The docstring promised this and the code did not do it. An
            # undated event cannot answer "is this within my holding window",
            # and a None here would either crash date arithmetic in the event
            # gate or silently miscount how many events are in range.
            continue
        out.append(CorporateEvent(
            symbol=sym,
            event_date=event_date,
            purpose=_s(row.get("purpose")),
            company=_s(row.get("company")),
            description=_s(row.get("bm_desc")),
        ))
    return out
