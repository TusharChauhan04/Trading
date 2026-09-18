"""News from RSS, deduplicated.

WHAT THIS IS AND IS NOT
-----------------------
These feeds are MARKET-WIDE HEADLINES. They are not per-symbol, and this
module does not pretend to map a headline to a ticker - that is entity
resolution, it is a genuinely hard problem, and a wrong mapping is worse than
no mapping because it attaches someone else's news to your trade. So R5's
output is market CONTEXT (what the regime engine and Stage 3 read), not a
per-symbol signal.

Per-symbol news already exists and is better: NSE's corporate announcements
are the company's own words to the exchange, timestamped to the second, and
already parsed in desk/research/sources/nse.py. That is Tier 1. This module
is Tier 3.

THE DUPLICATE PROBLEM, which is the whole reason this is not just a fetcher
-------------------------------------------------------------------------
Indian financial outlets republish the same PTI/wire copy. A result covered
by three of them is ONE piece of evidence appearing three times, and a naive
"three sources agree" scorer reads that as confirmation. That is section 37's
ensemble-bias failure applied to news, and section 11 already demanded the
fix: distinguish NEW from DUPLICATE.

So items are clustered, and a cluster carries ONE vote weighted by its BEST
source tier - never by how many outlets picked it up. The earliest timestamp
in a cluster is the one kept, because that is when the market could first
have known.

SOURCES
-------
Economic Times and Business Standard serve RSS to this client and are used.
MONEYCONTROL RETURNS 403 to an automated request on both its markets and
business feeds, tested. That is the site declining automated access, and this
module does not work around it - the whole reason RSS is the sanctioned path
is that a published feed is an invitation, and a 403 is the opposite. If
Moneycontrol is wanted, it needs a route they actually offer.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

__all__ = [
    "FEEDS", "NewsCluster", "NewsItem", "NewsSource", "RssError",
    "cluster_news", "fetch_feed", "parse_rss",
]

IST = timezone(timedelta(hours=5, minutes=30))

#: Wire-service and generic words carry no signal about WHICH story this is,
#: so they are dropped before comparing titles. Without this, "Sensex rises
#: 200 points" and "Sensex falls 200 points" look similar for the wrong
#: reason - they share every word but the one that matters.
_STOPWORDS = frozenset("""
a an the and or but of in on at to for from by with as is are was were be
been being it its this that these those will would can could may might said
says say after before over under new latest update updates report reports
""".split())

_MAX_BYTES = 8 * 1024 * 1024


class RssError(Exception):
    """A feed could not be fetched or read."""


@dataclass(frozen=True, slots=True)
class NewsSource:
    name: str
    url: str
    tier: int
    """1 = the company's own disclosure to the exchange. 3 = financial media.
    There is no tier 2 here: that was Reuters and Bloomberg, both paid, both
    ruled out. The gap is real and the daily plan should say so rather than
    renumber to hide it."""


#: Verified serving RSS to this client. Moneycontrol is deliberately absent -
#: see the module docstring.
FEEDS: tuple[NewsSource, ...] = (
    NewsSource("economic-times-markets",
               "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", 3),
    NewsSource("business-standard-markets",
               "https://www.business-standard.com/rss/markets-106.rss", 3),
    NewsSource("business-standard-companies",
               "https://www.business-standard.com/rss/companies-101.rss", 3),
)

_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
}


@dataclass(frozen=True, slots=True)
class NewsItem:
    source: str
    tier: int
    title: str
    url: str
    published_at: datetime
    """Always IST. Feeds publish RFC-2822 with an offset; comparing those
    naively across sources would order stories by nothing in particular."""
    summary: str = ""
    wire: str = ""
    """The originating agency when the feed says - Business Standard tags
    'Bloomberg', 'Reuters', 'PTI'. This is the strongest duplicate signal
    there is, because two outlets running the same wire copy say so."""

    @property
    def session_phase(self) -> str:
        t = self.published_at.astimezone(IST).time()
        if t < _OPEN:
            return "before_market"
        if t <= _CLOSE:
            return "during_market"
        return "after_hours"


@dataclass(slots=True)
class NewsCluster:
    """One story, however many outlets ran it."""

    items: list[NewsItem]

    @property
    def first(self) -> NewsItem:
        """The earliest report. When the market could first have known."""
        return min(self.items, key=lambda i: i.published_at)

    @property
    def published_at(self) -> datetime:
        return self.first.published_at

    @property
    def title(self) -> str:
        return self.first.title

    @property
    def tier(self) -> int:
        """The BEST tier in the cluster, not the average and not the count.
        Three tier-3 outlets running one wire story are still tier-3
        evidence."""
        return min(i.tier for i in self.items)

    @property
    def sources(self) -> list[str]:
        return sorted({i.source for i in self.items})

    @property
    def duplicated(self) -> bool:
        return len({i.source for i in self.items}) > 1

    @property
    def weight(self) -> float:
        """One vote, weighted by tier. Deliberately NOT scaled by how many
        outlets carried it - that is the ensemble-bias failure this module
        exists to prevent."""
        return {1: 1.0, 2: 0.6, 3: 0.35}.get(self.tier, 0.2)


def fetch_feed(source: NewsSource, *, timeout: float = 20.0) -> bytes:
    """One feed. Raises RssError rather than returning something unusable.

    A 403 is reported as what it is. Moneycontrol returns one, and the right
    response is to stop asking, not to disguise the client further.
    """
    req = urllib.request.Request(source.url, headers=_BROWSER)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RssError(
                f"{source.name} returned 403 - this feed is declining "
                f"automated access. Do not work around it; find a route the "
                f"publisher offers."
            ) from exc
        raise RssError(f"{source.name} returned HTTP {exc.code}") from exc
    except OSError as exc:
        raise RssError(f"{source.name} unreachable: {exc}") from exc
    if len(raw) > _MAX_BYTES:
        raise RssError(f"{source.name} returned more than {_MAX_BYTES // 1_000_000}MB")
    return raw


def parse_rss(raw: bytes, source: NewsSource) -> tuple[list[NewsItem], list[str]]:
    """(items, caveats).

    `caveats` is NOT a list of dropped items - it mixes two different facts on
    purpose, and both need saying: items this parser refused, and defects in
    the feed itself. A caller that reports "35 items" without them is claiming
    a clean read it did not get.

    An item with no usable timestamp is REFUSED, not dated to now. News whose
    time is unknown cannot be placed before or after a market session, and
    guessing would put a story on the wrong side of the one boundary that
    determines whether it had already moved the price.
    """
    items, caveats = [], []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as first:
        # Real feeds are not always well-formed. Business Standard's
        # companies feed ships bare "&" characters inside headlines
        # ("Co-founder & COO"), which is invalid XML - measured, 3 of them on
        # a single fetch.
        #
        # A bare & not followed by a valid entity has exactly ONE possible
        # meaning: a literal ampersand. Escaping it is a repair, not a guess,
        # and it is narrow enough to be safe. But it is attempted only AFTER
        # strict parsing fails, and it is always reported - a feed quietly
        # repaired is a feed whose breakage nobody ever fixes.
        patched = _BARE_AMP.sub("&amp;", raw.decode("utf-8", "replace"))
        try:
            root = ET.fromstring(patched)
        except ET.ParseError:
            raise RssError(
                f"{source.name}: not parseable as XML even after escaping "
                f"bare ampersands: {first}") from first
        caveats.append(
            f"feed was not well-formed XML ({first}); repaired by escaping "
            f"bare ampersands - the publisher should fix this")

    for node in root.findall(".//item"):
        get = lambda t: _text(node.find(t))                      # noqa: E731
        title = get("title")
        if not title:
            caveats.append("item with no title")
            continue
        when = _as_ist(get("pubDate") or get("lastModification"))
        if when is None:
            caveats.append(f"no usable timestamp: {title[:60]}")
            continue
        items.append(NewsItem(
            source=source.name, tier=source.tier, title=title,
            url=get("link"), published_at=when,
            summary=_strip_html(get("description")),
            wire=get("source"),
        ))
    return items, caveats


def cluster_news(items, *, similarity: float = 0.6) -> list[NewsCluster]:
    """Group reports of the same story, newest cluster first.

    Two passes, cheapest first. An exact normalised-title match is the common
    case and costs a dict lookup; only the leftovers are compared pairwise,
    which keeps this from being quadratic over the whole feed set.

    The second pass is greedy and therefore NOT transitive: if A matches B and
    B matches C but A does not match C, C stays out of A's cluster. That is
    deliberate. Transitive closure chains loosely-related headlines into one
    blob through a series of near-misses, and an over-merged cluster silently
    DELETES a real story - a worse failure than leaving a duplicate in, which
    only over-weights one that is already there.
    """
    by_key: dict[str, list[NewsItem]] = {}
    for it in sorted(items, key=lambda i: i.published_at):
        by_key.setdefault(_normalise(it.title), []).append(it)

    groups = list(by_key.values())
    merged: list[list[NewsItem]] = []
    token_sets = [_tokens(g[0].title) for g in groups]

    used = set()
    for i, g in enumerate(groups):
        if i in used:
            continue
        bucket = list(g)
        for j in range(i + 1, len(groups)):
            if j in used:
                continue
            if (_jaccard(token_sets[i], token_sets[j]) >= similarity
                    and not _opposed(token_sets[i], token_sets[j])):
                bucket.extend(groups[j])
                used.add(j)
        used.add(i)
        merged.append(bucket)

    clusters = [NewsCluster(items=b) for b in merged]
    clusters.sort(key=lambda c: c.published_at, reverse=True)
    return clusters


# --------------------------------------------------------------------------

from datetime import time as _time

_OPEN = _time(9, 15)
_CLOSE = _time(15, 30)
_TAG = re.compile(r"<[^>]+>")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_BARE_AMP = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)")


def _text(node) -> str:
    return (node.text or "").strip() if node is not None else ""


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", s)).strip()


def _as_ist(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        # A feed that omits the offset is stating local time, and these are
        # Indian publishers. Assuming UTC would shift every story by 5.5
        # hours - straight across the market-open boundary.
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def _normalise(title: str) -> str:
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", _NONWORD.sub(" ", t.lower())).strip()


def _tokens(title: str) -> frozenset[str]:
    return frozenset(w for w in _normalise(title).split()
                     if w not in _STOPWORDS and len(w) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


#: Market headlines are short, and after stopwords are removed they are
#: shorter still - which is what makes this necessary. Measured:
#:
#:     "Sensex rises 200 points"  ->  {sensex, rises, 200, points}
#:     "Sensex falls 200 points"  ->  {sensex, falls, 200, points}
#:     jaccard = 0.600, threshold = 0.600  ->  MERGED
#:
#: One token apart, opposite stories, and the merge DELETES one of them
#: because a cluster only ever surfaces its earliest member. No lexical
#: similarity measure can see this: the two headlines really are almost
#: identical as text. The thing that differs is the only thing that matters.
#:
#: Financial headlines have a small, closed direction vocabulary, so an
#: explicit list is both possible and honest - unlike general antonym
#: detection, which is not.
_UP = frozenset("""
rise rises rising rose gain gains gained surge surges surged jump jumps jumped
climb climbs climbed rally rallies rallied soar soars soared advance advances
higher up upbeat gainer gainers top tops surges strengthen strengthens profit
beat beats above outperform outperforms recover recovers rebound rebounds
""".split())

_DOWN = frozenset("""
fall falls falling fell drop drops dropped decline declines declined slip
slips slipped plunge plunges plunged tumble tumbles tumbled sink sinks sank
slide slides slid lower down loser losers weaken weakens loss losses miss
misses below underperform underperforms crash crashes sheds shed
""".split())


def _opposed(a: frozenset[str], b: frozenset[str]) -> bool:
    """Do these two headlines point in opposite directions?

    True only when one says UP and the other says DOWN and neither says both.
    A headline with no direction word does not conflict with anything - most
    have none, and refusing to merge on absence would disable clustering.
    """
    a_up, a_down = bool(a & _UP), bool(a & _DOWN)
    b_up, b_down = bool(b & _UP), bool(b & _DOWN)
    if a_up and a_down or b_up and b_down:
        return False        # mixed direction - cannot tell, so do not veto
    return (a_up and b_down) or (a_down and b_up)


# ===========================================================================
# Persistence
# ===========================================================================
#
# Fetched once per refresh and read by the plan, for the same reason the
# BSE cross-check and the fundamentals table are: three HTTP requests plus
# clustering on every page load, to recompute something that changes a few
# times an hour, is latency spent on a constant.
#
# STALENESS MATTERS MORE HERE THAN ANYWHERE ELSE IN THIS PROJECT. A
# fundamentals table a week old is merely behind; a NEWS snapshot a week
# old is actively misleading, because "market context" that predates the
# session it is describing will be read as current. So the snapshot
# records when it was fetched and the loader refuses anything past a few
# hours - and unlike the calendar, absence here costs nothing but a
# caveat.

import json as _json
from dataclasses import dataclass as _dc
from pathlib import Path as _P

#: Hours before a news snapshot stops being "today's context". Deliberately
#: short: a headline from yesterday's close describes yesterday's market.
DEFAULT_MAX_AGE_HOURS = 18


@_dc(frozen=True, slots=True)
class StoredNews:
    clusters: list
    fetched_at: datetime
    age_hours: float
    stale: bool
    caveats: tuple[str, ...] = ()

    @property
    def caveat(self) -> str | None:
        if not self.stale:
            return None
        return (f"the news snapshot is {self.age_hours:.0f} hours old, so it "
                f"describes an earlier session and was NOT used as market "
                f"context. Refresh it with 'python -m desk.research.refresh "
                f"news'.")


def save_news(clusters, path, *, fetched_at: datetime | None = None,
              caveats=()) -> int:
    """Write the clustered headlines atomically.

    Only the fields the prompt and a human reader need. The full item list
    per cluster is kept because "three outlets ran this" is exactly the
    duplicate signal the module exists to compute, and discarding it would
    make the stored form unable to answer the question it was built for.
    """
    path = _P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in clusters:
        rows.append({
            "title": c.title,
            "published_at": c.published_at.isoformat(),
            "tier": c.tier,
            "weight": c.weight,
            "duplicated": c.duplicated,
            "sources": list(c.sources),
            "session_phase": c.first.session_phase,
            "url": c.first.url,
            "summary": c.first.summary[:400],
        })
    payload = {
        "fetched_at": (fetched_at or datetime.now(IST)).astimezone(IST).isoformat(),
        "caveats": list(caveats),
        "clusters": rows,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    return len(rows)


def load_news(path, *, now: datetime | None = None,
              max_age_hours: float = DEFAULT_MAX_AGE_HOURS) -> "StoredNews | None":
    """Read the snapshot, or None when there is no usable file.

    Returns a lightweight cluster shape carrying exactly what the Stage 3
    prompt reads - `title` and `published_at` - rather than rebuilding
    NewsItem objects whose other fields nothing downstream consumes.
    """
    path = _P(path)
    if not path.is_file():
        return None
    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None

    clusters = []
    for row in payload.get("clusters", []):
        if not isinstance(row, dict):
            continue
        try:
            when = datetime.fromisoformat(row["published_at"])
        except (KeyError, TypeError, ValueError):
            continue
        clusters.append(_StoredCluster(
            title=str(row.get("title", "")), published_at=when,
            tier=int(row.get("tier", 3)),
            weight=float(row.get("weight", 0.0)),
            duplicated=bool(row.get("duplicated", False)),
            sources=tuple(row.get("sources", ())),
            session_phase=str(row.get("session_phase", "")),
            url=str(row.get("url", ""))))

    now = (now or datetime.now(IST)).astimezone(IST)
    age = (now - fetched_at.astimezone(IST)).total_seconds() / 3600.0
    return StoredNews(clusters=clusters, fetched_at=fetched_at,
                      age_hours=age,
                      stale=age > max_age_hours or age < -1.0,
                      caveats=tuple(payload.get("caveats", ())))


@_dc(frozen=True, slots=True)
class _StoredCluster:
    """What a cluster looks like once it has been through a JSON file.

    Deliberately NOT a NewsCluster: that type derives everything from its
    items, and rebuilding fake items to satisfy it would invent
    timestamps. This carries the derived answers directly.
    """

    title: str
    published_at: datetime
    tier: int
    weight: float
    duplicated: bool
    sources: tuple
    session_phase: str
    url: str
