"""R5: news fetching, parsing and deduplication.

The property that matters most here is NOT that clustering works - it is that
a cluster carries ONE vote regardless of how many outlets ran the story. That
is the ensemble-bias failure this module exists to prevent, so it is asserted
directly rather than inferred from the clustering tests.

The second-most important property is that clustering NEVER merges two
different stories. An unmerged duplicate only over-weights news that is
already present; an over-merge deletes a story outright, because a cluster
surfaces only its earliest member.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from desk.research.news import (
    FEEDS, IST, NewsCluster, NewsItem, NewsSource, RssError,
    cluster_news, parse_rss,
)

FIXTURES = Path(__file__).parent / "fixtures"

ET_SRC = NewsSource("economic-times-markets", "https://example.invalid/et", 3)


def _feed(*items: str) -> bytes:
    body = "".join(items)
    return ('<?xml version="1.0" encoding="UTF-8"?><rss version="2.0">'
            '<channel><title>t</title>' + body
            + "</channel></rss>").encode()


def _item(title="Story", when="Wed, 17 Sep 2025 19:30:00 +0530",
          link="https://example.invalid/1", desc="") -> str:
    return ("<item><title>" + title + "</title><link>" + link + "</link>"
            "<description>" + desc + "</description>"
            "<pubDate>" + when + "</pubDate></item>")


def _mk(title, minutes, source="a", tier=3) -> NewsItem:
    return NewsItem(
        source=source, tier=tier, title=title, url="https://example.invalid/x",
        published_at=(datetime(2025, 9, 17, 10, 0, tzinfo=IST)
                      + timedelta(minutes=minutes)))


# --- the real feeds ------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("economic-times-markets", 50),
    ("business-standard-markets", 35),
    ("business-standard-companies", 35),
])
def test_live_fixtures_parse(name, expected):
    """Captured from the real endpoints on 2025-09-17."""
    src = next(s for s in FEEDS if s.name == name)
    raw = (FIXTURES / ("rss_" + name + ".xml")).read_bytes()
    items, _caveats = parse_rss(raw, src)
    assert len(items) == expected
    assert all(i.title and i.url for i in items)
    assert all(i.published_at.utcoffset() == timedelta(hours=5, minutes=30)
               for i in items), "every timestamp must be IST"


def test_business_standard_companies_feed_is_malformed_and_is_repaired():
    """This fixture contains literal '&' characters inside headlines.

    It is invalid XML exactly as the publisher serves it. The repair must
    happen AND be reported: a silently repaired feed is one whose defect
    nobody ever fixes.
    """
    raw = (FIXTURES / "rss_business-standard-companies.xml").read_bytes()
    assert b"Co-founder & COO" in raw, "fixture no longer carries the defect"

    src = next(s for s in FEEDS if s.name == "business-standard-companies")
    items, caveats = parse_rss(raw, src)

    assert len(items) == 35
    assert any("not well-formed" in c for c in caveats)
    assert any("publisher should fix" in c for c in caveats)


def test_well_formed_feed_reports_no_repair():
    items, caveats = parse_rss(_feed(_item()), ET_SRC)
    assert len(items) == 1
    assert caveats == []


def test_moneycontrol_is_absent_from_feeds():
    """403, deliberately not worked around. If it ever returns it must be
    because the publisher offered a route, not because a header changed."""
    assert not any("moneycontrol" in s.url.lower() for s in FEEDS)


# --- the ampersand repair ------------------------------------------------

def test_bare_ampersand_is_escaped_and_survives():
    items, caveats = parse_rss(_feed(_item(title="Tata & Sons gain")), ET_SRC)
    assert items[0].title == "Tata & Sons gain"
    assert caveats


def test_real_entities_are_not_double_escaped():
    raw = _feed(_item(title="M&amp;M rises 3&#37; on &lt;order&gt;"))
    items, caveats = parse_rss(raw, ET_SRC)
    assert caveats == [], "well-formed input must not trigger the repair"
    assert items[0].title == "M&M rises 3% on <order>"


def test_unrepairable_xml_still_raises():
    """The repair is narrow. It must not drift into a tolerant HTML parser."""
    with pytest.raises(RssError, match="even after escaping"):
        parse_rss(b"<rss><channel><item><title>x</unclosed></rss>", ET_SRC)


# --- refusing what cannot be placed in time ------------------------------

def test_item_without_timestamp_is_refused_not_dated_to_now():
    items, caveats = parse_rss(_feed(_item(when="")), ET_SRC)
    assert items == []
    assert any("no usable timestamp" in c for c in caveats)


def test_item_with_unparseable_timestamp_is_refused():
    items, caveats = parse_rss(_feed(_item(when="yesterday-ish")), ET_SRC)
    assert items == []
    assert any("no usable timestamp" in c for c in caveats)


def test_item_without_title_is_refused():
    items, caveats = parse_rss(
        _feed("<item><pubDate>Wed, 17 Sep 2025 19:30:00 +0530"
              "</pubDate></item>"), ET_SRC)
    assert items == []
    assert caveats == ["item with no title"]


def test_naive_timestamp_is_read_as_ist_not_utc():
    """Assuming UTC would shift every story 5.5 hours - across market open."""
    items, _ = parse_rss(_feed(_item(when="Wed, 17 Sep 2025 08:00:00")), ET_SRC)
    assert items[0].published_at.hour == 8
    assert items[0].session_phase == "before_market"


def test_offset_timestamp_is_converted_to_ist():
    items, _ = parse_rss(_feed(_item(when="Wed, 17 Sep 2025 04:00:00 +0000")),
                         ET_SRC)
    assert (items[0].published_at.hour, items[0].published_at.minute) == (9, 30)


# --- session phase -------------------------------------------------------

@pytest.mark.parametrize("h,m,phase", [
    (9, 14, "before_market"),
    (9, 15, "during_market"),
    (15, 30, "during_market"),
    (15, 31, "after_hours"),
])
def test_session_phase_boundaries(h, m, phase):
    it = NewsItem(source="a", tier=3, title="t", url="u",
                  published_at=datetime(2025, 9, 17, h, m, tzinfo=IST))
    assert it.session_phase == phase


# --- clustering ----------------------------------------------------------

def test_identical_titles_from_two_outlets_form_one_cluster():
    clusters = cluster_news([_mk("RBI holds repo rate at 6.5%", 0, "et"),
                             _mk("RBI holds repo rate at 6.5%", 5, "bs")])
    assert len(clusters) == 1
    assert clusters[0].duplicated
    assert clusters[0].sources == ["bs", "et"]


def test_near_duplicate_titles_cluster():
    clusters = cluster_news([
        _mk("Infosys wins large deal from European bank", 0, "et"),
        _mk("Infosys bags large European bank deal", 4, "bs"),
    ])
    assert len(clusters) == 1


def test_unrelated_stories_stay_separate():
    clusters = cluster_news([_mk("Reliance completes Jio stake sale", 0),
                             _mk("Nifty ends flat ahead of Fed meeting", 1)])
    assert len(clusters) == 2


def test_cluster_keeps_the_earliest_report():
    """When the market could FIRST have known - not when the slowest outlet
    got around to it."""
    late = _mk("Wipro announces buyback", 45, "bs")
    early = _mk("Wipro announces buyback", 0, "et")
    c = cluster_news([late, early])[0]
    assert c.first is early
    assert c.published_at == early.published_at


def test_clusters_are_newest_first():
    out = cluster_news([_mk("Coal India signs supply pact", 0),
                        _mk("Rupee closes at record versus dollar", 90)])
    assert len(out) == 2
    assert out[0].title == "Rupee closes at record versus dollar"


def test_empty_input():
    assert cluster_news([]) == []


# --- the direction guard -------------------------------------------------
#
# Regression: these two measured jaccard 0.600 against a 0.600 threshold and
# merged, which silently deleted one of them.

@pytest.mark.parametrize("up,down", [
    ("Sensex rises 200 points", "Sensex falls 200 points"),
    ("Nifty ends higher led by banks", "Nifty ends lower led by banks"),
    ("Metal stocks gain on China demand", "Metal stocks decline on China demand"),
    ("TCS shares surge after results", "TCS shares tumble after results"),
])
def test_opposite_direction_headlines_never_merge(up, down):
    assert len(cluster_news([_mk(up, 0, "et"), _mk(down, 3, "bs")])) == 2


def test_direction_guard_does_not_block_same_direction_rewrites():
    clusters = cluster_news([_mk("Sensex rises 200 points on bank buying", 0, "et"),
                             _mk("Sensex gains 200 points on bank buying", 3, "bs")])
    assert len(clusters) == 1


def test_headlines_without_direction_words_still_cluster():
    """Most headlines have no direction word. Absence must not veto."""
    clusters = cluster_news([
        _mk("Reliance board approves rights issue", 0, "et"),
        _mk("Reliance board approves the rights issue", 2, "bs")])
    assert len(clusters) == 1


def test_mixed_direction_headline_does_not_veto():
    """'Sensex falls as IT gains' says both. Ambiguous, so no veto."""
    clusters = cluster_news([
        _mk("Sensex falls as IT stocks gain ground", 0, "et"),
        _mk("Sensex falls while IT stocks gain ground", 2, "bs")])
    assert len(clusters) == 1


# --- the anti-ensemble-bias property -------------------------------------

def test_weight_does_not_grow_with_outlet_count():
    """THE point of this module. Three outlets running one PTI story are one
    piece of evidence, not three."""
    one = NewsCluster(items=[_mk("Results beat estimates today", 0, "et")])
    three = NewsCluster(items=[_mk("Results beat estimates today", 0, "et"),
                               _mk("Results beat estimates today", 2, "bs"),
                               _mk("Results beat estimates today", 5, "mc")])
    assert one.weight == three.weight


def test_cluster_tier_is_the_best_not_the_average():
    c = NewsCluster(items=[_mk("Board approves dividend now", 0, "nse", tier=1),
                           _mk("Board approves dividend now", 9, "et", tier=3)])
    assert c.tier == 1
    assert c.weight == 1.0


def test_tier_one_outweighs_any_number_of_tier_three():
    exchange = NewsCluster(items=[_mk("Company files results", 0, "nse", tier=1)])
    media = NewsCluster(items=[_mk("Company posts results", i, "o%d" % i)
                               for i in range(6)])
    assert exchange.weight > media.weight


def test_single_outlet_repeating_itself_is_not_duplicated():
    c = NewsCluster(items=[_mk("Only one outlet ran this", 0, "et"),
                           _mk("Only one outlet ran this", 1, "et")])
    assert not c.duplicated
