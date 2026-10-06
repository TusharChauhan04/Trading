"""The one page a human opens.

The desk had 30 endpoints and no way to look at any of them without
constructing a URL - which made it a system its operator could not use. These
tests keep the page reachable and keep the cost arithmetic on it, because that
is the part that stops a small account trading into a fee it cannot beat.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from desk.api.main import app

client = TestClient(app)


def test_the_root_serves_a_page_not_json() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<title>The India Desk</title>" in r.text


def test_the_page_asks_for_capital_and_risk() -> None:
    """Capital is an input that changes the whole analysis, so it must be on
    the page rather than buried in a query string."""
    body = client.get("/").text
    assert 'id="cap"' in body
    assert 'id="risk"' in body
    assert "0.5" in body and "1.0" in body


def test_the_page_carries_the_flat_fee_arithmetic() -> None:
    """THE reason this page is not decoration. The DP charge is ~Rs 16 every
    time you sell, identical at any size, and it is why delivery equity cannot
    work below about Rs 2,048 a position. It has to be visible WHILE the
    operator types the amount, not discovered after a losing month."""
    body = client.get("/").text
    assert "15.93" in body, "the flat DP charge is not in the page"
    assert "2048" in body, "the break-even position is not shown"
    assert "0.08" in body, "the measured edge it is compared against is missing"


def test_the_page_calls_the_real_plan_endpoint() -> None:
    body = client.get("/").text
    assert "/plan/today?" in body


def test_the_page_says_what_was_not_checked() -> None:
    """`warnings` carries what the system could not verify. A plan that hides
    those reads as more confident than it is, and this project's whole
    discipline is that a skipped check is a first-class third state."""
    body = client.get("/").text
    assert "could NOT check" in body
    assert "p.warnings" in body


def test_the_page_disclaims_rather_than_implies_validation() -> None:
    """Nothing in this project has cleared its own validation bar. The page
    must not imply otherwise."""
    body = client.get("/").text
    assert "not advice" in body.lower()
    assert "validation bar" in body


def test_the_page_is_self_contained() -> None:
    """No build step, no CDN, no node toolchain - the pandas pin is
    load-bearing and this project does not add a frontend stack to show six
    numbers."""
    body = client.get("/").text
    assert "<script src=" not in body
    assert "cdn" not in body.lower()
    assert 'rel="stylesheet"' not in body
