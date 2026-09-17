"""BSE as a SECOND source, for cross-checking NSE.

WHY THIS EXISTS
---------------
desk/marketdata/quality.py has carried a CROSS_SOURCE check since it was
written, and it has never once run: `_check_reference` reports "no second
source supplied" and skips. A single feed cannot tell you it is wrong. An
unapplied split, a decimal shift, a stale file served for the wrong day - all
of them produce a series that is internally consistent and simply false.

BSE is the natural second source: it is free, official, lists most of the
same companies, and is genuinely independent of NSE's matching engine.

WHAT THE DATA ACTUALLY LOOKS LIKE - MEASURED, NOT ASSUMED
---------------------------------------------------------
Every design choice below comes from comparing all six sessions we hold
(2026-09-04 to 2026-09-11, 2,397 matched names per day) rather than from an
expectation about how close two exchanges "should" be.

The disagreement is NOISE, NOT BIAS. Mean signed difference -0.129%, median
-0.006%, NSE higher on 45.6% of names. Neither exchange systematically leads.

But the raw spread is far wider than intuition suggests:

    median   0.14%
    p90      1.11%
    p99      4.60%
    max     61.22%       SANWARIA: NSE 0.19 vs BSE 0.49

None of that is NSE being wrong. It is BSE being THIN. SANWARIA traded
549,342 shares on NSE that day and 6,504 on BSE. A closing print struck on
almost no volume is not an independent measurement of the same thing; it is
one trade. Divergence tracks BSE's traded VALUE almost monotonically:

    BSE turnover        n     median      p99        max
    < 1 lakh          362      0.954     17.258     61.224
    1L - 10L          501      0.338      3.218      6.402
    10L - 1Cr         775      0.114      1.770      4.166
    1Cr - 10Cr        579      0.061      0.753      0.966
    10Cr - 100Cr      171      0.077      0.561      0.752
    > 100Cr             9      0.071      1.321      1.401

So the cross-check is GATED ON BSE LIQUIDITY, and below the gate it reports
CHECK SKIPPED rather than passing. That distinction is the whole point: a
thin BSE print agreeing with NSE is not corroboration, and a thin BSE print
disagreeing with NSE is not evidence NSE is wrong. Neither outcome carries
information, so claiming either would be worse than admitting we cannot tell.

Pooled over the six sessions, above the gate (4,569 observations):

    p99 0.865%      p99.9 2.164%      max 4.393%      none above 5%

Only about a third of matched names clear the gate (759 of 2,397 on
2026-09-11). That is a real limit on the check's reach and callers are told
it, via `coverage` - see desk/marketdata/crosscheck.py.

TWO TRAPS FOUND WHILE PROBING
-----------------------------
1. THE LEGACY URL RETURNS HTTP 200 WITH AN HTML PAGE. BSE's old
   download/BhavCopy/Equity/EQ<ddmmyy>_CSV.ZIP path now serves the site's
   single-page-app shell - 14,287 bytes of "<!DOCTYPE html>", status 200,
   Content-Type text/html. Nothing about the response says failure. Handing
   that to a CSV parser is how a scanner ends up with zero rows and no error,
   so `fetch_bhavcopy` validates the CONTENT, not the status code.

2. BSE requires a Referer header naming its own site. Without it the request
   is refused. That is not a paywall or an anti-automation measure being
   circumvented - the file is published for download - it is just how their
   CDN is configured.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import pandas as pd

from desk.marketdata.sources.errors import RateLimited, SourceError

__all__ = [
    "BSE_EQUITY_SEGMENT", "BseSession", "MAX_RESPONSE_BYTES",
    "bhavcopy_url", "parse_bhavcopy",
]

#: BSE publishes the daily file under a stable, dated name. Verified live for
#: all six sessions we hold.
_BHAVCOPY_URL = ("https://www.bseindia.com/download/BhavCopy/Equity/"
                 "BhavCopy_BSE_CM_0_0_0_{day:%Y%m%d}_F_0000.CSV")

#: The file is ~850KB. A ceiling well above that turns a redirect to
#: something enormous into an error instead of a memory problem.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024

#: `FinInstrmTp` in the cash file. Every row in all six sessions was STK, but
#: it is filtered explicitly rather than assumed - the column exists because
#: the format is shared with the derivatives segment.
BSE_EQUITY_SEGMENT = "STK"

_REQUIRED = ("TradDt", "FinInstrmTp", "ISIN", "TckrSymb", "SctySrs",
             "OpnPric", "HghPric", "LwPric", "ClsPric", "TtlTradgVol",
             "TtlTrfVal")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 "
                   "Safari/537.36"),
    # Required. See trap 2 in the module docstring.
    "Referer": "https://www.bseindia.com/",
    "Accept": "text/csv,application/octet-stream,*/*",
}

_BSE_HOSTS = frozenset({"bseindia.com", "www.bseindia.com"})


def bhavcopy_url(day: date) -> str:
    return _BHAVCOPY_URL.format(day=day)


def _is_bse_host(host: str) -> bool:
    """Exact host match on the parsed netloc.

    Deliberately not `endswith`: "bseindia.com.evil.example" ends with the
    right string. This is the same hardening fetch_xbrl needed in the NSE
    client, where a suffix check passed a hostile host.
    """
    return (host or "").split(":")[0].lower() in _BSE_HOSTS


class BseSession:
    """A self-throttling BSE client. Construct once and reuse.

    Simpler than NseSession - BSE needs no cookie handshake - but the
    throttle discipline is identical and for the same reason: a loop that
    issues requests as fast as urllib manages is how an IP stops being able
    to reach the exchange at all.
    """

    def __init__(self, timeout: float = 20.0, min_interval: float = 1.5,
                 max_requests: int | None = None) -> None:
        self.timeout = timeout
        self.min_interval = max(0.0, min_interval)
        #: Hard ceiling for the life of this session. None means no ceiling.
        #: A bug that loops is otherwise an all-night hammering session whose
        #: first symptom is the block.
        self.max_requests = max_requests
        self.requests_made = 0
        self._last_request = 0.0
        self._opener = urllib.request.build_opener()

    def _wait_turn(self) -> None:
        """Sleep until `min_interval` has elapsed.

        Monotonic clock: time.time() can step backwards over an NTP
        correction, which would silently disable the throttle at exactly the
        wrong moment.
        """
        if self.max_requests is not None and self.requests_made >= self.max_requests:
            raise SourceError(
                f"request budget of {self.max_requests} exhausted for this "
                f"BSE session. Raise max_requests deliberately if that is "
                f"really what you want - it exists to stop a loop becoming a "
                f"ban.")
        if self.min_interval:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()
        self.requests_made += 1

    def fetch_with_retry(self, fn, *args, attempts: int = 4,
                         base_delay: float = 5.0, **kwargs):
        """Retry ONLY a rate limit. A renamed endpoint is not transient, and
        retrying it just reaches the same wrong answer more slowly while
        adding load to something already refusing us."""
        delay = base_delay
        for attempt in range(1, attempts + 1):
            try:
                return fn(*args, **kwargs)
            except RateLimited:
                if attempt == attempts:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def fetch_bhavcopy(self, day: date) -> bytes:
        """The full BSE cash-market file for one day, as raw CSV bytes.

        Validates that the response IS a bhavcopy. BSE answers a missing file
        with its SPA shell under HTTP 200 (trap 1 in the module docstring),
        so the status code proves nothing.
        """
        url = bhavcopy_url(day)
        raw = self._fetch_raw(url)

        head = raw.lstrip()[:200]
        if not head.startswith(b"TradDt,"):
            looks_html = head[:1] == b"<" or b"<!DOCTYPE" in head.upper()
            raise SourceError(
                f"BSE returned {len(raw)} bytes for {day} that are not a "
                f"bhavcopy - "
                + ("an HTML page, which is how BSE answers a file it does "
                   "not have. It does NOT use 404 for this, so the status "
                   "code was 200."
                   if looks_html else
                   f"expected a CSV header starting 'TradDt,', got "
                   f"{head[:60]!r}."))
        return raw

    def _fetch_raw(self, url: str) -> bytes:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or not _is_bse_host(parsed.netloc):
            raise SourceError(f"refusing to fetch a non-BSE URL: {url}")

        self._wait_turn()
        req = urllib.request.Request(url, headers=_HEADERS)
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                # Where it LANDED, not where it was aimed. A redirect can
                # move the request off BSE after the pre-flight check passed.
                final = urllib.parse.urlparse(resp.geturl())
                if final.scheme != "https" or not _is_bse_host(final.netloc):
                    raise SourceError(
                        f"request for {url} was redirected off BSE to "
                        f"{resp.geturl()} - not following it")
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                raise RateLimited(
                    f"BSE returned {exc.code} for {url} - backing off") from exc
            raise SourceError(f"BSE returned HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise SourceError(f"BSE unreachable: {exc.reason}") from exc

        if len(raw) > MAX_RESPONSE_BYTES:
            raise SourceError(
                f"BSE response for {url} exceeded "
                f"{MAX_RESPONSE_BYTES // 1_000_000}MB")
        return raw


def parse_bhavcopy(raw: bytes, day: date | None = None) -> pd.DataFrame:
    """BSE's daily cash file -> a frame shaped like the NSE one.

    Columns mirror desk.marketdata.sources.nse.parse_bhavcopy where they mean
    the same thing, so the two can be compared without a translation layer at
    every call site. BSE names are kept only where there is no NSE
    counterpart.

    `isin` is carried through because it is the ONLY reliable join key
    between the exchanges: NSE's sec_bhavdata_full file has no ISIN column at
    all, so the mapping has to come from NSE's equity master - see
    desk/marketdata/isin.py.

    `day`, if given, is checked against every row's TradDt. A file served for
    the wrong date is the error a scanner would not notice until its signals
    looked strange weeks later.
    """
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise SourceError(f"BSE bhavcopy missing expected columns: {missing}")
    if df.empty:
        raise SourceError("BSE bhavcopy contained no rows")

    df = df[df["FinInstrmTp"].astype(str).str.strip() == BSE_EQUITY_SEGMENT]
    if df.empty:
        raise SourceError(
            f"BSE bhavcopy has no {BSE_EQUITY_SEGMENT} rows - the file is "
            f"for a different segment")

    traded = pd.to_datetime(df["TradDt"].astype(str).str.strip(),
                            format="%Y-%m-%d").dt.date

    out = pd.DataFrame({
        "symbol": df["TckrSymb"].astype(str).str.strip().str.upper() + ".BO",
        "isin": df["ISIN"].astype(str).str.strip().str.upper(),
        "group": df["SctySrs"].astype(str).str.strip(),
        "date": traded,
        "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        # Rupees, not lakhs - NSE's file reports lakhs. Keeping BSE's own
        # unit and naming it plainly beats converting silently.
        "turnover": pd.to_numeric(df["TtlTrfVal"], errors="coerce"),
    }).reset_index(drop=True)

    if "PrvsClsgPric" in df.columns:
        out["prev_close"] = pd.to_numeric(
            df["PrvsClsgPric"], errors="coerce").reset_index(drop=True)
    if "TtlNbOfTxsExctd" in df.columns:
        out["trades"] = pd.to_numeric(
            df["TtlNbOfTxsExctd"], errors="coerce").reset_index(drop=True)

    if day is not None:
        wrong = out[out["date"] != day]
        if len(wrong):
            raise SourceError(
                f"expected a BSE bhavcopy for {day}, but {len(wrong)} row(s) "
                f"carry a different TradDt - fetched the wrong file, or BSE's "
                f"archive for this date is wrong")

    return out
