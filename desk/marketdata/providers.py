"""Data providers, scored against the twelve criteria rather than described.

The brief asked that every provider be evaluated on: India coverage, real-time
availability, historical depth, rate limits, cost, reliability, LICENSING,
REDISTRIBUTION restrictions, API quality, Python support, WebSocket support and
corporate-action handling - then classified Essential / Useful / Optional /
Redundant / Not Recommended.

Two of those twelve had never been assessed for any provider, and they are the
two that invalidate an architecture *after* it is built rather than before:
licensing and redistribution. A free source you may not redistribute is fine
for a private desk and fatal the day anything is shared.

`Rating.UNKNOWN` is a first-class value here. An unassessed criterion must not
read as a passing one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Verdict(str, Enum):
    ESSENTIAL = "essential"
    USEFUL = "useful"
    OPTIONAL = "optional"
    REDUNDANT = "redundant"
    NOT_RECOMMENDED = "not_recommended"


class Rating(str, Enum):
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    NONE = "none"
    UNKNOWN = "unknown"          # not assessed - NOT the same as "fine"


class Redistribution(str, Enum):
    """The question nobody asks until it is too late."""
    PERMITTED = "permitted"
    PERSONAL_USE_ONLY = "personal_use_only"
    PROHIBITED = "prohibited"
    UNCLEAR = "unclear"


@dataclass(frozen=True, slots=True)
class Provider:
    key: str
    name: str
    verdict: Verdict
    covers: list[str]                     # what kinds of data
    # --- the twelve criteria ------------------------------------------------
    india_coverage: Rating
    realtime: Rating
    historical_depth: str
    rate_limits: str
    cost: str
    reliability: Rating
    licence: str
    redistribution: Redistribution
    api_quality: Rating
    python_support: Rating
    websocket: Rating
    corporate_actions: Rating
    # ------------------------------------------------------------------------
    why: str
    blockers: list[str] = field(default_factory=list)
    verified: bool = False
    """True only where this project has actually exercised the source."""

    @property
    def unassessed(self) -> list[str]:
        """Criteria still carrying UNKNOWN. An honest gap list."""
        out = []
        for k in ("india_coverage", "realtime", "reliability", "api_quality",
                  "python_support", "websocket", "corporate_actions"):
            if getattr(self, k) is Rating.UNKNOWN:
                out.append(k)
        if self.redistribution is Redistribution.UNCLEAR:
            out.append("redistribution")
        return out

    @property
    def safe_to_redistribute(self) -> bool:
        return self.redistribution is Redistribution.PERMITTED


PROVIDERS: list[Provider] = [
    Provider(
        key="nse_official", name="NSE official site and API",
        verdict=Verdict.ESSENTIAL,
        covers=["corporate actions", "trading calendar", "bhavcopy",
                "index constituents", "F&O ban list", "announcements"],
        india_coverage=Rating.GOOD, realtime=Rating.POOR,
        historical_depth="corporate actions back to ~2000; bhavcopy daily archive",
        rate_limits="undocumented; blocks bare clients, needs a cookie session "
                    "and browser headers. Throttle and cache.",
        cost="free",
        reliability=Rating.ADEQUATE,
        licence="NSE terms of use - no formal data licence for the public API",
        redistribution=Redistribution.PROHIBITED,
        api_quality=Rating.POOR,
        python_support=Rating.NONE,
        websocket=Rating.NONE,
        corporate_actions=Rating.GOOD,
        why="The ONLY authoritative source for Indian corporate actions and the "
            "trading calendar, which are the two things no repository supplies "
            "and the highest-risk gap in the project. Verified working: "
            "desk/marketdata/sources/nse.py fetches and parses both, tested "
            "against captured payloads.",
        blockers=["undocumented API that can change without notice",
                  "corporate actions are FREE TEXT and need parsing per subject",
                  "redistribution prohibited - fine for a private desk, fatal "
                  "the day output is shared"],
        verified=True,
    ),
    Provider(
        key="zerodha_kite", name="Zerodha Kite Connect",
        verdict=Verdict.ESSENTIAL,
        covers=["historical OHLCV", "live quotes", "instrument master",
                "lot sizes", "order execution"],
        india_coverage=Rating.GOOD, realtime=Rating.GOOD,
        historical_depth="minute data ~60 days per request, daily for years",
        rate_limits="3 req/s historical, 10 req/s quote (published)",
        cost="~2,000 INR/month plus a Zerodha account",
        reliability=Rating.GOOD,
        licence="commercial API terms; account holder use",
        redistribution=Redistribution.PERSONAL_USE_ONLY,
        api_quality=Rating.GOOD,
        python_support=Rating.GOOD,
        websocket=Rating.GOOD,
        corporate_actions=Rating.POOR,
        why="The anchor for price data and the only realistic Indian retail "
            "execution path. Vibe-Trading already ships a loader for it.",
        blockers=["NOT SUBSCRIBED - no credentials in this project yet",
                  "does not supply corporate actions; NSE remains authoritative",
                  "access tokens expire daily and need a login flow"],
        verified=False,
    ),
    Provider(
        key="yfinance", name="yfinance (unofficial Yahoo)",
        verdict=Verdict.USEFUL,
        covers=["daily OHLCV", "some fundamentals", "index levels"],
        india_coverage=Rating.ADEQUATE, realtime=Rating.POOR,
        historical_depth="daily back many years for .NS/.BO",
        rate_limits="unofficial; throttles aggressively and without warning",
        cost="free",
        reliability=Rating.POOR,
        licence="no licence - scrapes an undocumented endpoint",
        redistribution=Redistribution.PROHIBITED,
        api_quality=Rating.ADEQUATE,
        python_support=Rating.GOOD,
        websocket=Rating.NONE,
        corporate_actions=Rating.POOR,
        why="Already wired into two vendored repos and fine for research and "
            "prototyping. Its split/dividend adjustment is inconsistent, which "
            "is exactly the failure this project is most exposed to.",
        blockers=["never the sole source for a live decision",
                  "adjustment behaviour differs from NSE's own actions"],
        verified=False,
    ),
    Provider(
        key="dhan_shoonya", name="Dhan / Shoonya (Finvasia)",
        verdict=Verdict.USEFUL,
        covers=["historical OHLCV", "live quotes", "execution"],
        india_coverage=Rating.GOOD, realtime=Rating.GOOD,
        historical_depth="broker-dependent",
        rate_limits="published per broker",
        cost="free or near-free with an account",
        reliability=Rating.UNKNOWN,
        licence="broker API terms",
        redistribution=Redistribution.PERSONAL_USE_ONLY,
        api_quality=Rating.UNKNOWN,
        python_support=Rating.ADEQUATE,
        websocket=Rating.ADEQUATE,
        corporate_actions=Rating.NONE,
        why="Redundancy and a cross-check against Kite. Two feeds disagreeing "
            "on a close is the only way to find out one of them is wrong - "
            "which is what quality.check(reference=...) is for.",
        blockers=["unassessed: reliability and API quality",
                  "no account held"],
        verified=False,
    ),
    Provider(
        key="screener_in", name="Screener.in",
        verdict=Verdict.USEFUL,
        covers=["fundamentals", "filings", "ratios", "promoter holding"],
        india_coverage=Rating.GOOD, realtime=Rating.NONE,
        historical_depth="10 years of annual and quarterly statements",
        rate_limits="no API; scraping is rate-limited by courtesy",
        cost="free tier, paid tier available",
        reliability=Rating.GOOD,
        licence="site terms - CHECK BEFORE AUTOMATING",
        redistribution=Redistribution.PROHIBITED,
        api_quality=Rating.NONE,
        python_support=Rating.NONE,
        websocket=Rating.NONE,
        corporate_actions=Rating.ADEQUATE,
        why="Best-in-class Indian fundamentals, and the fundamentals engine has "
            "no data source at all today.",
        blockers=["NO OFFICIAL API - check the terms of use before automating",
                  "fundamentals must be lagged to FILING date, not period end"],
        verified=False,
    ),
    Provider(
        key="trendlyne_tijori", name="Trendlyne / Tijori",
        verdict=Verdict.OPTIONAL,
        covers=["FII/DII flow", "promoter pledging", "results calendar"],
        india_coverage=Rating.GOOD, realtime=Rating.ADEQUATE,
        historical_depth="varies",
        rate_limits="plan-dependent",
        cost="paid",
        reliability=Rating.UNKNOWN,
        licence="commercial",
        redistribution=Redistribution.PROHIBITED,
        api_quality=Rating.UNKNOWN,
        python_support=Rating.NONE,
        websocket=Rating.NONE,
        corporate_actions=Rating.ADEQUATE,
        why="Genuinely India-specific data nothing else in the fleet supplies - "
            "FII/DII flow and promoter pledging are both named in the brief and "
            "neither has any source today.",
        blockers=["paid, and not yet justified by a measured edge",
                  "buy only after the alpha study shows these factors matter"],
        verified=False,
    ),
    Provider(
        key="alpha_vantage_finnhub", name="Alpha Vantage / Finnhub",
        verdict=Verdict.REDUNDANT,
        covers=["OHLCV", "some fundamentals"],
        india_coverage=Rating.POOR, realtime=Rating.POOR,
        historical_depth="thin for Indian names",
        rate_limits="5 req/min on free tiers",
        cost="free tier, paid tiers",
        reliability=Rating.ADEQUATE,
        licence="commercial terms",
        redistribution=Redistribution.PROHIBITED,
        api_quality=Rating.ADEQUATE,
        python_support=Rating.GOOD,
        websocket=Rating.ADEQUATE,
        corporate_actions=Rating.POOR,
        why="Already configured inside the vendored repos. Leave them wired, "
            "do not depend on them - their India coverage is thin enough that "
            "a gap would read as a data problem rather than a coverage one.",
        blockers=["thin India coverage"],
        verified=False,
    ),
    Provider(
        key="us_news_apis", name="US news / sentiment APIs",
        verdict=Verdict.NOT_RECOMMENDED,
        covers=["news", "sentiment"],
        india_coverage=Rating.POOR, realtime=Rating.GOOD,
        historical_depth="good for US names, poor for Indian ones",
        rate_limits="plan-dependent",
        cost="paid",
        reliability=Rating.ADEQUATE,
        licence="commercial",
        redistribution=Redistribution.PROHIBITED,
        api_quality=Rating.GOOD,
        python_support=Rating.GOOD,
        websocket=Rating.ADEQUATE,
        corporate_actions=Rating.NONE,
        why="Poor coverage of Indian companies. NSE corporate announcements "
            "plus the Indian financial press beat them decisively, and the "
            "announcements are free and authoritative.",
        blockers=["wrong market for company-level news"],
        verified=False,
    ),
]

BY_KEY: dict[str, Provider] = {p.key: p for p in PROVIDERS}


def provider_status() -> list[dict]:
    """JSON-safe view for the API."""
    out = []
    for p in PROVIDERS:
        d = asdict(p)
        for k, v in list(d.items()):
            if isinstance(v, Enum):
                d[k] = v.value
        d["unassessed"] = p.unassessed
        d["safe_to_redistribute"] = p.safe_to_redistribute
        out.append(d)
    return out


def essential() -> list[Provider]:
    return [p for p in PROVIDERS if p.verdict is Verdict.ESSENTIAL]


def unassessed_criteria() -> dict[str, list[str]]:
    """Everything still unknown, so the gap cannot hide behind a verdict."""
    return {p.key: p.unassessed for p in PROVIDERS if p.unassessed}
