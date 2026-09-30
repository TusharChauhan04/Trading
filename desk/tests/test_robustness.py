"""The deflated statistics must deflate, and must never invent a number.

These tests exist because of a specific failure this project already made:
an hourly factor scan reported its best cell at t 6.83, which was really
t 1.84 once overlap was removed and DSR 0.08 once the 615 tried cells were
declared. The arithmetic that catches that is only useful if it cannot be
quietly bypassed - by a default num_trials, or by an absent upstream
returning zero instead of nothing.
"""

from __future__ import annotations

import random

import pytest

from desk.robustness import RobustnessReport, available, deflated_sharpe
from desk.robustness.scorecard import MIN_OBSERVATIONS


def _series(n: int = 120, mean: float = 0.004, sd: float = 0.02,
            seed: int = 7) -> list[float]:
    """Seeded noise with a small positive mean.

    Deliberately not a repeating pattern: a short cycle repeated has almost
    no between-cycle variance, which flatters the Sharpe so much that DSR
    barely moves and any assertion about deflation stops testing anything.
    """
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


requires_upstream = pytest.mark.skipif(
    not available(),
    reason="agents/openterminal_ui/upstream absent - gitignored by design",
)


def test_num_trials_has_no_default() -> None:
    """The whole point of DSR is lost if the caller can forget the search.

    A default of 1 silently asserts "I tried exactly one thing", which is
    almost never true of anything worth deflating.
    """
    with pytest.raises(TypeError):
        deflated_sharpe([0.1, -0.2, 0.3])           # type: ignore[call-arg]


def test_num_trials_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="num_trials"):
        deflated_sharpe([0.1, -0.2, 0.3], num_trials=0)


@requires_upstream
def test_dsr_falls_as_the_search_widens() -> None:
    """Same returns, more trials admitted, less credible. Monotonically."""
    s = _series()
    dsrs = [deflated_sharpe(s, num_trials=n).dsr for n in (1, 10, 100, 615)]
    assert all(d is not None for d in dsrs)
    for earlier, later in zip(dsrs, dsrs[1:]):
        assert later <= earlier, f"DSR rose as trials grew: {dsrs}"
    assert dsrs[0] > dsrs[-1], "declaring 615 trials changed nothing"


@requires_upstream
def test_psr_ignores_the_search_but_dsr_does_not() -> None:
    """PSR answers a different question and must not move with num_trials.

    If PSR ever tracks num_trials the two statistics have been confused,
    and the distinction this package exists to preserve is gone.
    """
    s = _series()
    a = deflated_sharpe(s, num_trials=1)
    b = deflated_sharpe(s, num_trials=615)
    assert a.psr == b.psr
    assert a.dsr != b.dsr


@requires_upstream
def test_a_short_series_is_underpowered_not_negative() -> None:
    """Too little data must read as "cannot tell", never as "no".

    The hourly study had 20 non-overlapping cohorts against the 60 these
    estimators need. That is a measurement that cannot answer the question,
    which is a different finding from a strategy that loses.
    """
    r = deflated_sharpe(_series(n=20), num_trials=615)
    assert r.underpowered
    assert r.n < MIN_OBSERVATIONS
    assert not deflated_sharpe(_series(n=200), num_trials=615).underpowered


@requires_upstream
def test_periods_per_year_changes_the_annualisation() -> None:
    """A 20-session hold is ~13 periods a year, not 252.

    Passing the default for it inflates the Sharpe by sqrt(20), which is
    how a monthly holding period gets reported as a daily one.
    """
    s = _series()
    daily = deflated_sharpe(s, num_trials=1, periods_per_year=252)
    monthly = deflated_sharpe(s, num_trials=1, periods_per_year=13)
    assert daily.annual_sharpe is not None
    assert monthly.annual_sharpe is not None
    assert abs(daily.annual_sharpe) > abs(monthly.annual_sharpe)


def test_an_empty_series_reports_unavailable_not_zero() -> None:
    r = deflated_sharpe([], num_trials=10)
    assert r.unavailable is not None
    assert r.psr is None and r.dsr is None
    assert "unavailable" in str(r)


def test_a_missing_upstream_yields_nothing_rather_than_a_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recurring failure mode in this project is a real gap that reads
    as a passing check. agents/*/upstream/ is gitignored and OneDrive has
    already dehydrated one mid-session, so this path is load-bearing."""
    import desk.robustness.scorecard as sc

    monkeypatch.setattr(sc, "_load", lambda: None)
    r = sc.deflated_sharpe(_series(), num_trials=615)
    assert r.unavailable is not None
    assert r.psr is None and r.dsr is None
    assert r.verdict == "unavailable"
    assert "8c46cbc" in r.unavailable, "the fix should name the pinned commit"


def test_report_str_never_hides_an_unavailable_result() -> None:
    r = RobustnessReport(n=0, num_trials=5, unavailable="upstream missing")
    assert "unavailable" in str(r)
    assert "0.0" not in str(r)
