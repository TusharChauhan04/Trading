"""Intraday OHLCV from Yahoo's chart endpoint. The desk's only intraday source.

WHY THIS EXISTS AND WHY IT IS NOT AN IMPORT
-------------------------------------------
The salvage map's step 1 is an intraday adapter, because step 2 - "is a 1:2
target reachable on 5-minute bars?" - is the project's stated main goal and
cannot be asked at all without intraday data. The desk has none.

OpenTerminalUI's `core/historical_data_service.py` already fetches exactly this
and the map confirmed it working live (375 five-minute bars over 7 days on
RELIANCE.NS). IT IS ALSO THE MODULE THIS PROJECT QUARANTINED, and for the worst
reason on the list: when the real fetch comes back empty it calls
`_synthetic_ohlcv()`, which returns a SEEDED RANDOM WALK as OHLCV - no warning,
no flag, no exception. A backtest on that symbol would run on fabricated prices
and look entirely normal.

So the URL and parameter shape are borrowed - `/v8/finance/chart/{symbol}` with
`period1`, `period2`, `interval` - and nothing else is. An empty or malformed
response raises here. There is no fallback path that invents a bar.

THREE THINGS LEARNED THE HARD WAY ELSEWHERE, APPLIED HERE
---------------------------------------------------------
1. VALIDATE CONTENT, NOT THE STATUS CODE. BSE answers a file it does not have
   with its own SPA shell under HTTP 200 and `text/html`. Yahoo answers an
   unknown symbol with a 404 but answers a VALID symbol with no data for the
   window with a 200 and `result: [{...}]` carrying empty quote arrays. Both
   are checked, and an empty series raises `SourceError` rather than returning
   an empty frame a caller might store as a day with no trading.

2. THROTTLE, ALWAYS. `NseSession` carries the project's standing note that
   back-to-back requests get the caller rate-limited, and the same applies
   here with less documentation to go on. `min_interval` defaults to 1.5s and
   is enforced inside `fetch`, not left to the caller - a loop over 1,100
   symbols would otherwise issue 1,100 requests as fast as the socket allows.
   HTTP 429 raises `RateLimited`, which the refresh loop treats as a reason to
   stop rather than to retry harder.

3. STDLIB ONLY. Upstream needs `httpx`; the desk's existing sources speak
   `urllib` and nothing is installed for this.

WHAT THE SOURCE ACTUALLY GIVES, MEASURED ON RELIANCE.NS RATHER THAN ASSUMED.
The published retention figures are not the request limits, and the difference
decides what can be tested:

    interval   max span per request   sessions returned
    5m         45 days (60 fails)     31
    15m        45 days (60 fails)     31
    1h         365 days (730 fails)   245

AND THE WINDOW CANNOT BE STITCHED. Walking backwards in 40-day chunks, only the
MOST RECENT chunk returns anything - every earlier one is HTTP 422, including
2026-07-13 to 2026-08-22, which is inside the 60-day retention Yahoo advertises.
So 5-minute history is hard-capped at about 31 sessions and no amount of
chunking extends it.

WHAT THAT MEANS FOR THE MAIN GOAL, stated plainly because it is a constraint on
the question rather than on the code. 31 sessions is enough to MEASURE whether a
2R target is reachable intraday - 31 x 73 bars is ~2,260 entries per symbol -
and nowhere near enough for a walk-forward over disjoint windows at the 30-trade
floor this desk uses. Only the 1-hour interval, at 245 sessions, can carry that.

The way out is time, not cleverness: a daily refresh accumulates history the
source will not hand over in bulk, so in six months the desk owns ~120 sessions
of 5-minute bars that cannot be fetched today. That is the reason to wire the
refresh now despite the sample being thin now.

THE SESSION IS NOT THE SESSION, AND THIS IS THE FINDING THAT MATTERS MOST
-----------------------------------------------------------------------
NSE trades 09:15 to 15:30, which is 75 five-minute bars. Yahoo sends 75-76 and
TWO OF THEM ARE ALWAYS NULL. Measured on RELIANCE, TCS and SBIN across two
sessions - every combination, no exceptions:

    09:15          price present, VOLUME ZERO   (pre-open / opening auction)
    09:20 - 15:15  real bars
    15:20, 15:25   NULL on every symbol, every day
    15:30          sometimes present, volume zero; sometimes absent entirely

So THE LAST RELIABLY TRADED 5-MINUTE BAR IS 15:15, and any intraday rule that
means to exit "at the close" must exit there. Treating 15:30 as the exit would
be filling at a bar that is missing on some days and carries zero volume on the
rest - which is to say, at a price nothing traded at. `session_quality` reports
the count per session so a caller can see this rather than discover it as a
strange result. SBIN on 2026-09-29 lost 15:15 as well, so the hole is at least
ten minutes and occasionally fifteen.

This is a limitation of the free source, not a choice, and it has to travel
with every intraday number this desk produces.

TIMEZONE. Yahoo returns UNIX timestamps in UTC plus the exchange's offset in
`meta.gmtoffset`. Bars are converted to Asia/Kolkata and the session date is
taken from the LOCAL date, because a 09:15 IST bar is 03:45 UTC on the same
calendar day but a 15:30 IST bar is 10:00 UTC - and a naive UTC date would be
correct here by luck rather than by construction. Any exchange east of UTC+9
would break it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from desk.marketdata.sources.errors import (
    RateLimited, SourceError, TransientError,
)

__all__ = ["INTERVAL_RETENTION_DAYS", "IST", "LAST_RELIABLE_BAR",
           "YahooSession", "session_quality", "to_frame"]

IST = timezone(timedelta(hours=5, minutes=30))

#: Yahoo's own retention per interval, in calendar days. Asking for more
#: silently returns only what it has, so a caller that does not know these
#: numbers will think its date range worked.
#: MEASURED per-request maxima, not Yahoo's advertised retention - a request
#: spanning more than this returns HTTP 422 rather than a truncated series, so
#: a caller using the retention figure gets nothing at all. See the module
#: docstring for the measurement and for why chunking does not help.
INTERVAL_RETENTION_DAYS = {
    "1m": 7,
    "2m": 45,
    "5m": 45,
    "15m": 45,
    "30m": 45,
    "60m": 365,
    "1h": 365,
    "1d": 36500,
}

#: The last 5-minute bar that reliably carries a traded price. See the module
#: docstring: 15:20 and 15:25 are null on every symbol and day measured, and
#: 15:30 is absent or zero-volume. An intraday exit "at the close" belongs
#: here, not at 15:30.
LAST_RELIABLE_BAR = "15:15"

_BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"

#: Required columns on every returned frame. A frame missing one is a bug
#: here, not a data condition, so `to_frame` asserts rather than filling.
COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")


@dataclass
class YahooSession:
    """One throttled session against the chart endpoint.

    Deliberately not a module-level singleton: the refresh command owns one and
    the throttle state lives with it, so two concurrent callers cannot each
    think they are respecting the interval.
    """

    timeout: float = 12.0
    min_interval: float = 1.5
    user_agent: str = "Mozilla/5.0 (compatible; india-desk/0.1)"
    _last_request: float = 0.0

    def _wait(self) -> None:
        gap = time.monotonic() - self._last_request
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_request = time.monotonic()

    def fetch(self, symbol: str, *, interval: str = "5m",
              start: date | None = None, end: date | None = None,
              days: int | None = None) -> pd.DataFrame:
        """Intraday bars for one symbol. Raises rather than returning empty.

        `days` is a convenience for "the last N calendar days", clamped to what
        the interval actually retains - asking for 90 days of 5-minute bars
        returns 60 and no warning, so the clamp is reported by the caller
        rather than discovered later as a short series.
        """
        if interval not in INTERVAL_RETENTION_DAYS:
            raise SourceError(
                f"interval {interval!r} is not one Yahoo serves; "
                f"known: {', '.join(sorted(INTERVAL_RETENTION_DAYS))}")
        retention = INTERVAL_RETENTION_DAYS[interval]
        if days is not None:
            days = min(int(days), retention)
            end = end or datetime.now(IST).date()
            start = end - timedelta(days=days)
        if start is None or end is None:
            raise SourceError("pass either `days` or both `start` and `end`")

        # period2 is exclusive of its own second, so push to the end of `end`.
        p1 = int(datetime(start.year, start.month, start.day,
                          tzinfo=IST).timestamp())
        p2 = int((datetime(end.year, end.month, end.day, tzinfo=IST)
                  + timedelta(days=1)).timestamp())
        qs = urllib.parse.urlencode({
            "period1": p1, "period2": p2, "interval": interval,
            "events": "div,splits", "includePrePost": "false",
        })
        url = f"{_BASE}{urllib.parse.quote(symbol)}?{qs}"
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept": "application/json,text/plain,*/*",
        })

        self._wait()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimited(
                    f"Yahoo rate-limited the request for {symbol} (HTTP 429). "
                    f"Raise min_interval; do not retry in a tight loop."
                ) from exc
            if exc.code in (404, 400):
                raise SourceError(
                    f"Yahoo has no chart for {symbol!r} (HTTP {exc.code}). "
                    f"NSE symbols need the .NS suffix.") from exc
            raise TransientError(
                f"Yahoo returned HTTP {exc.code} for {symbol}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TransientError(f"could not reach Yahoo: {exc}") from exc

        # CONTENT, not status. A 200 carrying an HTML error page is the BSE
        # failure mode and it must not be parsed as data.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            head = raw[:80].decode("utf-8", "replace")
            raise SourceError(
                f"Yahoo returned non-JSON for {symbol} (starts {head!r})"
            ) from exc

        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise SourceError(
                f"Yahoo reported an error for {symbol}: {chart['error']}")
        results = chart.get("result") or []
        if not results:
            raise SourceError(f"Yahoo returned no result block for {symbol}")
        return to_frame(symbol, results[0], interval=interval)


def to_frame(symbol: str, node: dict, *, interval: str) -> pd.DataFrame:
    """One chart result block into a tidy frame. Raises on an empty series.

    AN EMPTY SERIES IS AN ERROR, NOT AN EMPTY FRAME. Yahoo answers a valid
    symbol with no data in the window using HTTP 200 and a result block whose
    quote arrays are empty or all-null. Returning an empty frame there would
    let the refresh loop write a file for a session that simply was not
    fetched, and the store cannot tell that from a holiday.
    """
    stamps = node.get("timestamp") or []
    quote = ((node.get("indicators") or {}).get("quote") or [{}])[0]
    if not stamps:
        raise SourceError(
            f"Yahoo returned a result block with no timestamps for {symbol} - "
            f"the symbol is valid but has no {interval} data in that window")

    off = int(((node.get("meta") or {}).get("gmtoffset")) or 0)
    tz = timezone(timedelta(seconds=off)) if off else IST

    frame = pd.DataFrame({
        "timestamp": [datetime.fromtimestamp(int(t), tz=timezone.utc)
                      .astimezone(tz) for t in stamps],
        "open": quote.get("open") or [None] * len(stamps),
        "high": quote.get("high") or [None] * len(stamps),
        "low": quote.get("low") or [None] * len(stamps),
        "close": quote.get("close") or [None] * len(stamps),
        "volume": quote.get("volume") or [None] * len(stamps),
    })
    frame["symbol"] = symbol

    # Yahoo pads the series with null bars for minutes the exchange did not
    # trade. Dropped rather than forward-filled: a filled bar is a price that
    # never printed, and this project quarantines a module for inventing those.
    before = len(frame)
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if frame.empty:
        raise SourceError(
            f"every one of {before} bars Yahoo returned for {symbol} was null")

    # A volume of 0 is legitimate on a thin interval; a volume of null is not
    # known, and the two must not be conflated by a fillna.
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Bars where the high is below the low are corrupt, not tradeable.
    bad = frame["high"] < frame["low"]
    if bool(bad.any()):
        frame = frame[~bad]

    frame["session"] = [ts.date() for ts in frame["timestamp"]]
    return frame[[*COLUMNS, "session"]].sort_values(
        "timestamp").reset_index(drop=True)


def session_quality(frame: pd.DataFrame, *,
                    expected: int = 73) -> pd.DataFrame:
    """Bars per session, so a short day is visible rather than surprising.

    `expected` defaults to 73 rather than the arithmetic 75 because 75 is what
    NSE's hours imply and 73 is what this source actually delivers once the
    two always-null bars are dropped. A session well below 73 is a real gap -
    a half-day, a late start, or a symbol that was halted - and a caller
    sizing a position from a 12-bar session should know that.

    Returns one row per session with `bars`, `first`, `last` and `complete`.
    """
    if frame.empty:
        return pd.DataFrame(columns=["session", "bars", "first", "last",
                                     "complete"])
    g = frame.groupby("session")
    out = pd.DataFrame({
        "bars": g.size(),
        "first": g["timestamp"].min().dt.strftime("%H:%M"),
        "last": g["timestamp"].max().dt.strftime("%H:%M"),
    })
    out["complete"] = out["bars"] >= expected
    return out.reset_index()
