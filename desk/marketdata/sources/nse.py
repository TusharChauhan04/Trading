"""NSE as a data source: corporate actions and the trading calendar.

Two halves, kept strictly apart.

`NseSession` is the only part that touches the network. NSE serves its JSON
API to a client that looks like a browser and has been given cookies; a bare
request gets 403. The homepage itself also returns 403 to this handshake, which
does not matter - the cookie jar is populated by the attempt, and the API calls
then succeed. That is odd enough to be worth writing down, because the obvious
"fix" of asserting the homepage returned 200 would break a working client.

Everything else is a **pure function over bytes**, so it is tested offline
against payloads actually captured from NSE (see desk/tests/fixtures/).

THE HARD PART is not fetching, it is `parse_subject`. NSE describes corporate
actions in free text written by humans, and the same event appears in several
spellings across the years:

    "Bonus 1:1"
    "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    "Face Value Split From Rs.10/- To Rs.2/-"
    "Rights 1:15 @ Premium Rs 1247"
    "Annual General Meeting/ Final Dividend Rs 20.50 Per Shar/ Special Dividend..."

A parser that quietly ignores what it cannot read is worse than useless here.
A dropped bonus becomes an unexplained 50% gap, which the backtest reads as a
crash and the strategy reads as a signal. So `parse_subject` NEVER returns
None: it returns either a `CorporateAction` or an `Unparsed` carrying the
original text and the reason, and `parse_corporate_actions` returns both lists.
The caller decides what to do about the ones we could not read - but it cannot
fail to notice them.
"""

from __future__ import annotations

import gzip
import zlib
import http.cookiejar
import io
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from desk.marketdata.corporate_actions import ActionType, CorporateAction
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

BASE = "https://www.nseindia.com"
API = f"{BASE}/api"
ARCHIVES = "https://nsearchives.nseindia.com"     # bhavcopy and other static files
ARCHIVES_HOST = "nsearchives.nseindia.com"

#: Hard ceiling on one response. The largest thing this client legitimately
#: fetches is a full bhavcopy - about 395KB raw, ~140KB gzipped - so 64MB is
#: three orders of magnitude of headroom and still bounds the damage.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024

#: And on what a gzip body may EXPAND to. Measured: 200MB of compressible
#: data gzips to ~204KB, an amplification of about 1029x, so a few-MB
#: response that passes the wire-size cap can still detonate into gigabytes.
#: gzip.decompress() does it in one unbounded allocation; this does not.
MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024

#: Real NSE tickers contain `&` and `-`: M&M, BAJAJ-AUTO, J&KBANK, NAM-INDIA,
#: IL&FSENGG and 17 others in a single day's bhavcopy. An `.isalnum()` check
#: looks conservative and is simply WRONG - it refused a Nifty 50 constituent.
#: This still rejects everything injection-shaped, because no NSE symbol
#: contains a space, `=`, `;`, `/`, `?` or `&index=`-style payloads.
_SYMBOL_RE = re.compile(r"[A-Z0-9&-]{1,30}")

# This module is entirely about NSE, which trades in IST. date.today() alone
# reads server-local time - fine on a laptop in India, silently off by 5:30h
# the moment this runs on a UTC-clock VPS. Same reasoning as desk/api/main.py.
IST = ZoneInfo("Asia/Kolkata")

#: 16 + MAX_WBITS: gzip container rather than bare deflate.
_GZIP_WBITS = 16 + zlib.MAX_WBITS

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer": f"{BASE}/",
    "Connection": "keep-alive",
}

# The equities segment of the holiday master. NSE returns a dict keyed by
# segment; CM is Capital Market, which is what cash equities trade in.
EQUITY_SEGMENT = "CM"


class SourceError(Exception):
    """The source could not be reached or returned something unusable."""


# ===========================================================================
# Fetching - the only part that needs a network
# ===========================================================================

class NseSession:
    """A cookie-bearing session. Construct once and reuse; NSE rate-limits."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self._jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._jar)
        )
        self._warmed = False

    def _warm(self) -> None:
        """Populate the cookie jar.

        The homepage returns 403 to this request and that is FINE - the
        Set-Cookie headers arrive regardless, which is all the API needs. Do
        not "fix" this by requiring a 200.
        """
        if self._warmed:
            return
        req = urllib.request.Request(f"{BASE}/", headers=BROWSER_HEADERS)
        try:
            self._opener.open(req, timeout=self.timeout).read()
        except urllib.error.HTTPError:
            pass                      # cookies still set; see docstring
        except OSError as exc:
            raise SourceError(f"cannot reach {BASE}: {exc}") from exc
        self._warmed = True

    def _fetch_raw(self, url: str) -> bytes:
        """The one place that actually opens a socket, for either JSON or CSV.

        Every caller passes an explicit, fully-formed URL built from a
        constant + validated arguments (never user input verbatim) - see
        get_json's docstring for why that matters.
        """
        self._warm()
        req = urllib.request.Request(url, headers=BROWSER_HEADERS)
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                # A redirect is followed transparently by urllib's default
                # handler, ACROSS HOSTS, and no amount of checking the URL we
                # asked for constrains where we ended up. Verified: a 302 to a
                # different host is followed and its body returned. So the
                # destination is re-checked here, after the fact.
                final = urllib.parse.urlsplit(resp.geturl())
                if final.hostname and not _is_nse_host(final.hostname):
                    raise SourceError(
                        f"request for {url[:60]!r} was redirected off NSE to "
                        f"{final.hostname!r} - refusing the response"
                    )
                # resp.read() with no argument reads to EOF, however large. Cap it
                # and read one byte past the limit so "exactly at the cap" is
                # distinguishable from "truncated".
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise SourceError(
                        f"{url[:60]!r} returned more than "
                        f"{MAX_RESPONSE_BYTES // 1_000_000}MB - refusing to buffer it"
                    )
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = _gunzip_bounded(raw, url)
                return raw
        except urllib.error.HTTPError as exc:
            raise SourceError(f"{url} returned HTTP {exc.code}") from exc
        except OSError as exc:
            raise SourceError(f"{url} unreachable: {exc}") from exc

    def get_json(self, path: str) -> Any:
        """`path` is relative to the API root - this method never accepts a
        caller-supplied absolute URL. The original had a `path.startswith
        ("http")` passthrough branch; a security review flagged it as an
        unreachable-today-but-latent SSRF-shaped primitive, since nothing
        forwarded user input into it. Removed rather than left dormant, now
        that fetch_bhavcopy needs a genuine cross-host request and gets its
        own explicit method instead of overloading this one."""
        url = f"{API}/{path.lstrip('/')}"
        raw = self._fetch_raw(url)
        try:
            # parse_constant rejects NaN/Infinity/-Infinity. json.loads
            # accepts all three as bare literals even though they are not
            # valid JSON, and a NaN promoter holding compares false against
            # every threshold downstream while looking like a real number.
            return json.loads(raw, parse_constant=_reject_constant)
        except RecursionError as exc:
            # Deeply nested JSON raises this, and it is NOT a JSONDecodeError -
            # so it escaped the handler below and killed the caller instead of
            # producing the SourceError every other failure here produces.
            raise SourceError(f"{url} returned pathologically nested JSON") from exc
        except json.JSONDecodeError as exc:
            # Usually an anti-bot HTML interstitial rather than malformed JSON.
            raise SourceError(
                f"{url} did not return JSON (first 120 bytes: {raw[:120]!r})"
            ) from exc

    def fetch_bhavcopy(self, day: date) -> bytes:
        """Raw CSV bytes of NSE's full-market bhavcopy for one trading day.

        ONE request covers the entire listed universe - 2,637 EQ-series
        symbols alone, in a fixture captured 2026-09-11, plus SME/T2T/gilt
        segments alongside them. The alternative (fetch_corporate_actions'
        per-symbol shape, applied to price history instead) would cost
        ~2,000 requests for the same day's information. This is the single
        highest-leverage fetch in this module, and the reason the daily
        universe scan does not need to be a loop over symbols.

        Served from a different host (nsearchives, not the /api layer), so it
        cannot go through get_json - CSV, not JSON, and a different domain
        entirely.
        """
        url = f"{ARCHIVES}/products/content/sec_bhavdata_full_{day.strftime('%d%m%Y')}.csv"
        return self._fetch_raw(url)

    # -- the two JSON endpoints we actually use ----------------------------

    def fetch_corporate_actions(self, symbol: str) -> list[dict]:
        base = symbol.split(".")[0].upper()
        data = self.get_json(
            f"corporates-corporateActions?index=equities&symbol={base}"
        )
        if not isinstance(data, list):
            raise SourceError(f"expected a list of actions, got {type(data).__name__}")
        return data

    # ---- research: filings, announcements, events -------------------------
    #
    # Each is a thin, symbol-validated wrapper. The symbol is upper-cased and
    # stripped of any .NS/.BO suffix before it reaches the URL, for the same
    # reason get_json refuses absolute URLs: the only values that reach a
    # request are built from a constant plus a narrowed argument.

    def fetch_announcements(self, symbol: str) -> list[dict]:
        return self._research("corporate-announcements", symbol)

    def fetch_board_meetings(self, symbol: str) -> list[dict]:
        return self._research("corporate-board-meetings", symbol)

    def fetch_results(self, symbol: str, period: str = "Quarterly") -> list[dict]:
        if period not in ("Quarterly", "Half-Yearly", "Annual"):
            raise SourceError(f"unsupported results period {period!r}")
        return self._research("corporates-financial-results", symbol,
                              extra=f"&period={period}")

    def fetch_shareholding(self, symbol: str) -> list[dict]:
        return self._research("corporate-share-holdings-master", symbol)

    def fetch_insider_deals(self, symbol: str) -> list[dict]:
        return self._research("corporates-pit", symbol)

    def fetch_event_calendar(self) -> list[dict]:
        """Market-wide, not per symbol - this is the forward calendar the
        event gate reads."""
        data = self.get_json("event-calendar")
        rows = data if isinstance(data, list) else data.get("data")
        if not isinstance(rows, list):
            raise SourceError(
                f"expected a list of events, got {type(data).__name__}")
        return rows

    def fetch_xbrl(self, url: str) -> bytes:
        """The document behind a filing. Cross-host (archives), so it goes
        through _fetch_raw directly - but only for a URL NSE itself gave us,
        which is checked rather than assumed."""
        # startswith() is NOT sufficient and was a real bypass:
        # "https://nsearchives.nseindia.com.evil.com/x" starts with the
        # archive prefix and would have sent this cookie-bearing session to
        # an attacker-controlled host. The URL comes from an NSE PAYLOAD, so
        # it is remote data and must be parsed, not pattern-matched.
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.netloc != ARCHIVES_HOST:
            raise SourceError(
                f"refusing to fetch XBRL from {url[:60]!r} - not an NSE "
                f"archive URL. Pass the link NSE supplied on the filing row."
            )
        return self._fetch_raw(url)

    def _research(self, endpoint: str, symbol: str, extra: str = "") -> list[dict]:
        base = symbol.split(".")[0].upper()
        if not _SYMBOL_RE.fullmatch(base):
            raise SourceError(f"refusing to build a URL from symbol {symbol!r}")
        data = self.get_json(f"{endpoint}?index=equities&symbol={base}{extra}")
        rows = data if isinstance(data, list) else data.get("data")
        if not isinstance(rows, list):
            raise SourceError(
                f"{endpoint} for {base}: expected a list, got "
                f"{type(data).__name__}")
        return rows

    def fetch_holiday_master(self) -> dict:
        data = self.get_json("holiday-master?type=trading")
        if not isinstance(data, dict):
            raise SourceError(f"expected a dict of segments, got {type(data).__name__}")
        return data


# ===========================================================================
# Parsing - pure, and tested against captured payloads
# ===========================================================================

@dataclass(frozen=True, slots=True)
class Unparsed:
    """An action we could not turn into a factor, kept rather than dropped."""
    symbol: str
    ex_date: date | None
    subject: str
    reason: str

    def __str__(self) -> str:
        return f"{self.symbol} {self.ex_date} {self.subject!r} - {self.reason}"


def _reject_constant(name: str):
    raise SourceError(f"non-finite JSON constant {name!r} in NSE response")


def _is_nse_host(host: str) -> bool:
    """Exactly nseindia.com or a subdomain of it - never a suffix match.
    `nsearchives.nseindia.com.evil.example` must not pass."""
    h = host.lower().rstrip(".")
    return h == "nseindia.com" or h.endswith(".nseindia.com")


def _gunzip_bounded(raw: bytes, url: str) -> bytes:
    """Decompress incrementally, refusing anything past the output cap."""
    out = bytearray()
    d = zlib.decompressobj(_GZIP_WBITS)
    chunk = d.decompress(raw, MAX_DECOMPRESSED_BYTES + 1 - len(out))
    while chunk:
        out.extend(chunk)
        if len(out) > MAX_DECOMPRESSED_BYTES:
            raise SourceError(
                f"{url[:60]!r} decompressed past "
                f"{MAX_DECOMPRESSED_BYTES // 1_000_000}MB - refusing it"
            )
        if not d.unconsumed_tail:
            break
        chunk = d.decompress(d.unconsumed_tail,
                             MAX_DECOMPRESSED_BYTES + 1 - len(out))
    return bytes(out)


def parse_nse_date(s: str) -> date | None:
    """NSE dates are 'DD-Mon-YYYY', sometimes padded, sometimes '-' for none."""
    s = _s(s)
    if not s or s == "-":
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# "Bonus 1:1", "Bonus issue 1:2". The gap is bounded to 15 chars: a genuine
# bonus subject never puts more than a word or two between the keyword and its
# ratio. Unbounded, this matched straight across an unrelated clause -
# "Bonus Debentures / Rights 1:15 @ Premium Rs 1247" read as a share bonus of
# 1:15, when "Bonus" there describes a debt instrument and the ratio belongs
# to an unrelated rights issue 25+ characters away.
_BONUS = re.compile(r"\bbonus\b[^0-9:]{0,15}(\d+)\s*:\s*(\d+)", re.I)

# Both split spellings seen in the archive. The face values may be written
# "Rs 10/-", "Rs.10/-", "Rs 2" or "Rs. 2.50", hence the loose currency prefix.
#
# Anchored on \bsplit\b (not "splitting", "share-split-adjusted" prose), NOT
# DOTALL (a '.' must never cross a newline into an unrelated sentence - that
# is what let a record-date change get misread as a split), and the from/to
# numbers are bounded close to the keyword so an unrelated later "from X to Y"
# cannot be borrowed - "Splitting Of Share Certificates From Rs 10 To Rs 2"
# under an AGM item is prose, not a corporate action, and \bsplit\b alone
# already excludes it ("Splitting" has no word boundary after "split").
_SPLIT = re.compile(
    r"\bsplit\b[^\n]{0,40}?\bfrom\b\s*(?:rs\.?\s*)?([\d.]+)[^\d\n]{0,20}?"
    r"\bto\b\s*(?:rs\.?\s*)?([\d.]+)",
    re.I,
)

# "Dividend - Rs 6 Per Share", "Interim Dividend - Re 1 Per Share",
# "Dividend-Rs.8.50 Per Share", "Final Dividend Rs 20.50 Per Shar". Commas are
# accepted in the amount ("Rs 1,250") even though no fixture has one yet - a
# four-digit-plus dividend is not implausible and the cost of handling it is
# one character class.
_DIVIDEND_AMOUNT = re.compile(
    r"dividend[^0-9]{0,20}?(?:rs|re)\.?\s*([\d,]*\.?\d+)", re.I)

# Events that move the price but cannot be expressed as one factor.
_NEEDS_REVIEW = re.compile(r"\b(rights?|demerger|de-merger|scheme of arrangement|"
                           r"amalgamation|merger|capital reduction|consolidation)\b", re.I)

# Events that do not adjust the price series at all.
_NOT_PRICE_AFFECTING = re.compile(
    r"\b(annual general meeting|agm|egm|extra[- ]ordinary general meeting|"
    r"buy\s*-?\s*back|buyback|board meeting|postal ballot|record date)\b", re.I)


def _split_leg(text: str) -> tuple[bool, float | None, str | None]:
    """(matched, factor, error). `matched` is True the moment the keyword and
    a from/to pair are found at all, even if the values turn out unusable -
    the caller needs to know a split WAS mentioned, to decide whether a
    parallel bonus mention should be combined with it or reported alone."""
    m = _SPLIT.search(text)
    if not m:
        return False, None, None
    try:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
    except ValueError:
        return True, None, "unreadable face values"
    if old_fv <= 0 or new_fv <= 0:
        return True, None, "non-positive face value"
    if new_fv >= old_fv:
        # A face value that rises is a consolidation (reverse split). It is
        # real, but rare enough that guessing the direction is not worth it.
        return True, None, (f"face value rose {old_fv}->{new_fv}; looks like a "
                            f"consolidation, needs manual review")
    return True, old_fv / new_fv, None          # Rs 10 -> Rs 2 means 5 shares


def _bonus_leg(text: str) -> tuple[bool, float | None, str | None]:
    m = _BONUS.search(text)
    if not m:
        return False, None, None
    free, held = int(m.group(1)), int(m.group(2))
    if free <= 0 or held <= 0:
        return True, None, "non-positive bonus ratio"
    # Indian convention: "Bonus 1:1" is one free share for each held, so a
    # holder ends with two.
    return True, (held + free) / held, None


def _dividend_amount(text: str) -> float | None:
    """Sum every "Rs N Per Share" the subject actually states.

    NSE routinely packs an interim and a special dividend into one line -
    `re.search` (the original approach) took the first and silently dropped
    the rest, understating some real events by 80%+. But a figure already
    stated as inclusive of its own breakdown - "Rs 8.50 (Including Special
    Dividend of Rs 2)" - must NOT be summed on top of itself, so anything
    after "including" is excluded from the scan.
    """
    head = re.split(r"\bincluding\b", text, maxsplit=1, flags=re.I)[0]
    matches = _DIVIDEND_AMOUNT.findall(head)
    if not matches:
        return None
    total = 0.0
    for raw in matches:
        try:
            total += float(raw.replace(",", ""))
        except ValueError:
            return None
    return total


def parse_subject(symbol: str, ex_date: date | None, subject: str
                  ) -> CorporateAction | Unparsed:
    """One NSE subject line -> an action, or an explicit Unparsed.

    Never returns None. A silently dropped bonus becomes an unexplained gap
    that reads as a crash, so "I could not read this" has to be a value the
    caller receives rather than an absence they must notice.

    Structural events are checked before dividends because a combined subject
    line mentioning both must be adjusted for the structural event, which is
    far larger than the payout - and a split AND a bonus on the same line are
    COMBINED (multiplied), not one silently dropped in favour of the other.
    """
    text = (subject or "").strip()
    if not text:
        return Unparsed(symbol, ex_date, subject, "empty subject")
    if ex_date is None:
        return Unparsed(symbol, ex_date, subject, "no usable ex-date")

    # --- structural, price-affecting, expressible as a factor -------------
    split_found, split_factor, split_err = _split_leg(text)
    bonus_found, bonus_factor, bonus_err = _bonus_leg(text)

    if split_found and bonus_found:
        if split_err or bonus_err:
            return Unparsed(symbol, ex_date, subject,
                            f"combined split+bonus but {split_err or bonus_err}")
        return CorporateAction(
            symbol=symbol, ex_date=ex_date, type=ActionType.SPLIT,
            factor=split_factor * bonus_factor, note=text,
        )
    if split_found:
        if split_err:
            return Unparsed(symbol, ex_date, subject, split_err)
        return CorporateAction(symbol=symbol, ex_date=ex_date,
                               type=ActionType.SPLIT, factor=split_factor, note=text)
    if bonus_found:
        if bonus_err:
            return Unparsed(symbol, ex_date, subject, bonus_err)
        return CorporateAction(symbol=symbol, ex_date=ex_date,
                               type=ActionType.BONUS, factor=bonus_factor, note=text)

    # --- things that move price but need a human --------------------------
    if _NEEDS_REVIEW.search(text):
        return Unparsed(symbol, ex_date, subject,
                        "moves the price but cannot be reduced to one factor")

    # --- cash flows --------------------------------------------------------
    amount = _dividend_amount(text)
    if amount is not None:
        return CorporateAction(
            symbol=symbol, ex_date=ex_date, type=ActionType.DIVIDEND,
            amount=amount, note=text,
        )

    # --- explicitly not price-affecting -----------------------------------
    if _NOT_PRICE_AFFECTING.search(text):
        return Unparsed(symbol, ex_date, subject, "not price-affecting")

    return Unparsed(symbol, ex_date, subject, "unrecognised subject")


def parse_corporate_actions(payload: list[dict], symbol: str
                            ) -> tuple[list[CorporateAction], list[Unparsed]]:
    """Whole NSE response -> (actions, unparsed). Both lists always returned."""
    actions: list[CorporateAction] = []
    unparsed: list[Unparsed] = []
    for row in payload:
        ex = parse_nse_date(row.get("exDate", ""))
        # Strip any suffix the row (or the caller-supplied fallback) might
        # already carry before adding .NS - `symbol="ITC.NS"` with a row that
        # omits its own "symbol" field previously produced "ITC.NS.NS", which
        # matches nothing downstream.
        sym = (row.get("symbol") or symbol).strip().upper().split(".")[0]
        canonical = f"{sym}.NS"
        result = parse_subject(canonical, ex, row.get("subject", ""))
        if isinstance(result, CorporateAction):
            actions.append(result)
        else:
            unparsed.append(result)
    actions.sort(key=lambda a: a.ex_date)
    return actions, unparsed


def parse_holiday_master(payload: dict, *, segment: str = EQUITY_SEGMENT
                         ) -> dict:
    """NSE holiday master -> the calendar payload `TradingCalendar.load` wants.

    Muhurat shows up as a holiday row carrying a non-null session window, which
    is exactly the special-session case: the regular session is shut and an
    evening window opens.
    """
    if not isinstance(payload, dict):
        raise SourceError(f"expected a dict of segments, got {type(payload).__name__}")

    if segment not in payload:
        raise SourceError(
            f"segment {segment!r} missing from holiday master "
            f"(present: {', '.join(sorted(payload)) or 'none'})"
        )
    rows = payload[segment]
    if not isinstance(rows, list):
        raise SourceError(
            f"segment {segment!r} is a {type(rows).__name__}, expected a list"
        )

    holidays: list[dict] = []
    specials: list[dict] = []
    needs_timings: list[str] = []
    years: set[int] = set()

    for row in rows:
        d = parse_nse_date(row.get("tradingDate", ""))
        if d is None:
            continue
        years.add(d.year)
        name = (row.get("description") or "").strip()
        window = _session_window(row)
        if window:
            start, end = window
            specials.append({"date": d.isoformat(), "name": name or "Special session",
                             "start": start, "end": end})
            continue

        holidays.append({"date": d.isoformat(), "name": name})

        # NSE marks a Muhurat day by appending '*' to the description and
        # publishing the timings in a separate circular. Treating it as an
        # ordinary holiday is safe (the regular session IS shut), but losing
        # the fact that an evening session exists is not - so it is surfaced
        # rather than dropped.
        if name.endswith("*") or "muhurat" in name.lower():
            needs_timings.append(d.isoformat())

    if not years:
        raise SourceError("holiday master contained no readable dates")

    out = {
        "exchange": "NSE",
        "years": sorted(years),
        "holidays": holidays,
        "special_sessions": specials,
        "_source": "nseindia.com/api/holiday-master?type=trading",
        "_fetched": datetime.now(IST).date().isoformat(),
    }
    if needs_timings:
        out["_needs_session_timings"] = needs_timings
        out["_note"] = (
            "NSE flagged these dates as special-session days but published no "
            "timings in the holiday master. Fetch the Muhurat circular and add "
            "them to special_sessions, or intraday bars on those dates will "
            "have no session window to align to."
        )
    return out


# Captures an optional AM/PM. Dropping the meridiem silently read "6:15 PM"
# as 06:15 - a 12-hour error that put a Muhurat session's is_open_at() window
# exactly opposite the real one: False during the actual evening session,
# True at the same clock hour the next morning.
_TIME = re.compile(r"(\d{1,2})[:.](\d{2})\s*([ap]\.?\s*m\.?)?", re.I)


def _parse_clock(h: str, m: str, meridiem: str | None) -> str | None:
    """A hour of 1-12 with no AM/PM is genuinely ambiguous - '6:15' could be
    morning or evening, and Muhurat is always evening, so guessing morning
    (the literal 24-hour reading) is exactly backwards half the time. This was
    previously accepted as-is despite the module's own docstring claiming
    otherwise - REGRESSION found in the 2026-09-13 correctness re-verification.
    Only hours that are unambiguous WITHOUT a meridiem (0, and 13-23) are
    accepted when none is given; 1-12 with no AM/PM is refused."""
    hour, minute = int(h), int(m)
    if not (0 <= minute < 60):
        return None
    if meridiem:
        is_pm = meridiem.lower().startswith("p")
        if not (1 <= hour <= 12):
            return None
        hour = (hour % 12) + (12 if is_pm else 0)
    elif hour == 0 or hour >= 13:
        pass                          # unambiguous even without AM/PM
    else:
        return None                   # 1-12 with no meridiem: refuse, don't guess
    return f"{hour:02d}:{minute:02d}"


def _session_window(row: dict) -> tuple[str, str] | None:
    """Pull a start/end out of NSE's session fields, which are free text.

    Two failure modes guarded here, both seen in real payloads: the field can
    hold the literal string "Open" or "Closed" instead of a time (segments
    other than CM use this), and a Muhurat session is always evening, so a
    match with no AM/PM marker in a context that clearly needs one is refused
    rather than defaulting to a 24-hour reading nobody wrote.
    """
    for key in ("evening_session", "morning_session"):
        raw = row.get(key)
        if not raw or not str(raw).strip():
            continue
        text = str(raw).strip()
        if text.lower() in ("open", "closed"):
            continue                            # not a time at all
        matches = _TIME.findall(text)
        if len(matches) < 2:
            continue
        (h1, m1, mer1), (h2, m2, mer2) = matches[0], matches[1]
        # NSE writes the meridiem once for a pair ("6:15 PM to 7:15 PM" or
        # "6.15 - 7.15 PM"); if only the second carries it, it applies to both.
        mer1 = mer1 or mer2
        mer2 = mer2 or mer1
        start, end = _parse_clock(h1, m1, mer1), _parse_clock(h2, m2, mer2)
        if start is None or end is None:
            continue
        return start, end
    return None


# ===========================================================================
# Bhavcopy - the whole listed universe, one file, one request per day
# ===========================================================================

# The mainstream cash-equity series. A bhavcopy row can carry any of eleven
# series (measured in a fixture captured 2026-09-11: EQ 2637, SM 373, BE 248,
# ST 91, GS 51, GB 39, BZ 27, IV 11, RR 6, E1 1, SZ 1) and parse_bhavcopy
# drops none of them - this constant is a filter a CALLER opts into, not
# something baked into parsing. BE/BZ are trade-for-trade (no intraday,
# restricted - a different regime, not merely a smaller one); SM/ST are the
# SME platforms (a different liquidity profile entirely); GS/GB are
# government securities, not equity at all. EQ is what "the Indian equity
# universe" means in the sense every other module in this project uses it.
BHAVCOPY_EQUITY_SERIES = frozenset({"EQ"})

_BHAVCOPY_REQUIRED = ("SYMBOL", "SERIES", "DATE1", "OPEN_PRICE", "HIGH_PRICE",
                     "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY")


def parse_bhavcopy(raw: bytes, day: date | None = None) -> pd.DataFrame:
    """NSE's full-market bhavcopy for one day -> a normalized DataFrame.

    One row per (symbol, series) - the same base symbol can legitimately
    appear twice if it trades in more than one series, which is real and is
    not collapsed away here.

    NSE writes "-" for a field it has nothing to report (confirmed in the
    fixture: 297 of 3485 rows, concentrated in BE/BZ/SM/ST/GB where delivery
    figures aren't tracked the same way) - the same missing-value convention
    `parse_nse_date` already handles elsewhere in this module. Those become
    NaN via `pd.to_numeric(errors="coerce")` rather than a parse failure.

    `day`, if given, is a sanity check: every row's own DATE1 must match it,
    or this raises - the wrong day's file silently accepted is exactly the
    kind of error a scanner would not notice until its signals looked odd.
    """
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _BHAVCOPY_REQUIRED if c not in df.columns]
    if missing:
        raise SourceError(f"bhavcopy missing expected columns: {missing}")
    if df.empty:
        raise SourceError("bhavcopy contained no rows")

    parsed_date = pd.to_datetime(df["DATE1"].str.strip(), format="%d-%b-%Y").dt.date

    out = pd.DataFrame({
        "symbol": df["SYMBOL"].str.strip().str.upper() + ".NS",
        "series": df["SERIES"].str.strip(),
        "date": parsed_date,
        "prev_close": df["PREV_CLOSE"],
        "open": df["OPEN_PRICE"],
        "high": df["HIGH_PRICE"],
        "low": df["LOW_PRICE"],
        "close": df["CLOSE_PRICE"],       # NSE's official close, not LAST_PRICE
        "last": df["LAST_PRICE"],
        "volume": df["TTL_TRD_QNTY"],
        "turnover_lacs": df["TURNOVER_LACS"],
        "trades": df["NO_OF_TRADES"],
        "delivery_qty": pd.to_numeric(df["DELIV_QTY"], errors="coerce"),
        "delivery_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
    })

    if day is not None:
        wrong = out[out["date"] != day]
        if len(wrong):
            raise SourceError(
                f"expected bhavcopy for {day}, but {len(wrong)} row(s) carry a "
                f"different DATE1 - fetched the wrong file, or NSE's archive "
                f"for this date is corrupt"
            )

    return out


def bhavcopy_equity_only(df: pd.DataFrame) -> pd.DataFrame:
    """The mainstream cash-equity slice - see BHAVCOPY_EQUITY_SERIES."""
    return df[df["series"].isin(BHAVCOPY_EQUITY_SERIES)].reset_index(drop=True)


# --------------------------------------------------------------------------
# research parsers
#
# Every one returns (records, undated). NSE genuinely publishes filings it
# cannot date - nine of RELIANCE's 130 quarterly results, all 2005-2007 - and
# those must not silently acquire a plausible timestamp. See
# desk/research/models.py for why `disclosed_at` is required.
# --------------------------------------------------------------------------

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


#: NSE spells "there is nothing here" several ways, and "-" is by far the
#: most common - 456 of RELIANCE's 3,345 announcements carry it in place of an
#: attachment URL. Passed through verbatim it becomes a link that a downstream
#: fetch will dutifully try to open.
_ABSENT = {"", "-", "na", "null", "none"}


def _s(v) -> str:
    """A field as a string, whatever NSE actually sent.

    `(v or "").strip()` is the obvious idiom and it is wrong: it only
    substitutes "" for FALSY values, so an int, list or dict passes straight
    through to .strip() and raises AttributeError. These endpoints are
    undocumented, so a field changing type between API revisions is a
    when-not-if - and one such field would otherwise cost the entire batch,
    `undated` list included. _num() already had this right; the string
    helpers did not.
    """
    if v is None:
        return ""
    return v.strip() if isinstance(v, str) else str(v).strip()


def _text_or_none(v) -> str | None:
    t = _s(v)
    return None if t.lower() in _ABSENT else t


def _num(v) -> float | None:
    """A number, or None. NaN and Infinity are refused rather than passed on:
    json.loads accepts both as literals, and a NaN promoter holding compares
    false against every threshold downstream without ever looking wrong."""
    try:
        f = float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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
