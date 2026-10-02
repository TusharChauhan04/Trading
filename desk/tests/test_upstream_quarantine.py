"""Three upstream modules fabricate data. This stops one being imported by habit.

A comment in a manifest is not a guard. These three return invented numbers that
look real, and the quietest of them - historical_data_service - would poison a
backtest with synthetic prices and no warning at all. So the quarantine is
asserted:

  1. no desk module may import them
  2. the fabricating code is still THERE, so if upstream ever fixes it the test
     fails and the quarantine can be lifted deliberately rather than forgotten

The second half matters as much as the first. A quarantine nobody revisits
becomes folklore.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "agents" / "openterminal_ui" / "upstream"

#: module path -> (the fabricating symbol, why it matters)
QUARANTINED = {
    "backend/core/historical_data_service.py": (
        "_synthetic_ohlcv",
        "returns a seeded random walk as OHLCV when the real fetch returns "
        "nothing - no warning, no flag, no exception. Would poison a backtest "
        "silently.",
    ),
    "backend/tca/service.py": (
        # The fabricating FUNCTION, not `hashlib` - that was the first marker
        # and it flagged our own journal, which hashes entries for integrity.
        # A marker has to be unique to the fabrication, not merely present in it.
        "generate_tca_report",
        "generate_tca_report builds ten 'trades' from an md5 hash of the "
        "window string and returns them as a transaction cost analysis.",
    ),
    "backend/services/sector_rotation.py": (
        "_generate_mock_rrg",
        "returns invented rotation data on a failed download. At least it "
        "prints a warning, which makes it the least bad of the four.",
    ),
    "backend/services/stress_test_service.py": (
        # The marker is the hash-derived SECTOR inference, not "sha256" -
        # hashing is legitimate elsewhere in that file and in our own
        # journal, which digests records for integrity.
        "_infer_sector_from_hash",
        "derives every factor sensitivity from a sha256 digest of the "
        "TICKER STRING - equity_beta, rate_sensitivity, commodity_beta, "
        "fx_exposure and credit_sensitivity are a sector preset plus "
        "jitter(digest[i]), and for an unknown ticker the SECTOR itself "
        "comes from _infer_sector_from_hash. So both halves of the beta "
        "are a function of the name. It is the only one of the four that "
        "labels itself - the tool returns quality='synthetic' with a note "
        "- but the note says 'sector presets plus deterministic per-ticker "
        "jitter' and omits that the sector can be invented too. Honest "
        "labelling is why this is the least dangerous of the four and not "
        "a reason to import it: a stress test is a risk number, and a risk "
        "number derived from a ticker's spelling is worse than none.",
    ),
}

needs_upstream = pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="agents/openterminal_ui/upstream absent - gitignored by design",
)


def _desk_sources() -> list[pathlib.Path]:
    return [p for p in (ROOT / "desk").rglob("*.py")
            if "__pycache__" not in p.parts]


@pytest.mark.parametrize("module", sorted(QUARANTINED))
def test_no_desk_module_imports_a_fabricating_module(module: str) -> None:
    """The quarantine, enforced.

    Matches the module's PATH or dotted form - `backend/tca/service` or
    `backend.tca.service` - plus its fabricating symbol. NOT the bare filename
    stem: the first version of this test matched `service` and flagged six
    innocent lines containing the English word, including "a full-service
    broker" in the cost model. A guard that cries wolf gets deleted, which
    leaves no guard at all.
    """
    rel = module.removesuffix(".py")
    marker = QUARANTINED[module][0]
    patterns = [
        re.compile(re.escape(rel)),                      # backend/tca/service
        re.compile(re.escape(rel.replace("/", "."))),    # backend.tca.service
        re.compile(rf"\b{re.escape(marker)}\b"),         # the fabricating name
    ]
    offenders = []
    for src in _desk_sources():
        if src.name == pathlib.Path(__file__).name:
            continue                 # this file names them all, deliberately
        for i, line in enumerate(src.read_text(encoding="utf-8",
                                               errors="ignore").splitlines(), 1):
            if not any(p.search(line) for p in patterns):
                continue
            stripped = line.strip()
            # A mention in a comment or docstring is the POINT - the quarantine
            # is documented in the code it applies to. Only live code fails.
            if stripped.startswith(("#", '"', "'", "*", "-")):
                continue
            offenders.append(f"{src.relative_to(ROOT)}:{i}: {stripped[:80]}")
    assert not offenders, (
        f"{module} is quarantined ({QUARANTINED[module][1]}) but is "
        f"referenced by:\n  " + "\n  ".join(offenders))


@needs_upstream
@pytest.mark.parametrize("module", sorted(QUARANTINED))
def test_the_fabricating_code_is_still_there(module: str) -> None:
    """If upstream fixes one of these, this fails and the quarantine gets
    reconsidered on purpose instead of outliving the problem.

    A quarantine nobody revisits becomes folklore, and folklore is how a
    usable module stays unused for years.
    """
    path = UPSTREAM / module
    if not path.is_file():
        pytest.skip(f"{module} no longer exists upstream - re-audit")
    marker, why = QUARANTINED[module]
    text = path.read_text(encoding="utf-8", errors="ignore")
    assert marker in text, (
        f"{module} no longer contains {marker!r}. It was quarantined because "
        f"it {why} If that is genuinely fixed, re-read the failure path and "
        f"lift the quarantine deliberately."
    )


@needs_upstream
def test_the_silent_fallback_is_still_silent() -> None:
    """The specific property that makes historical_data_service the worst one.

    sector_rotation at least prints a warning. This one substitutes invented
    prices with nothing in the output to say so - which is why it is the module
    that would have done real damage.
    """
    path = UPSTREAM / "backend/core/historical_data_service.py"
    if not path.is_file():
        pytest.skip("module no longer exists upstream - re-audit")
    text = path.read_text(encoding="utf-8", errors="ignore")
    # The fallback call site, and no warning/raise anywhere near it.
    m = re.search(r"bars = _synthetic_ohlcv\([^)]*\)", text)
    assert m, "the daily synthetic fallback moved - re-read the failure path"
    window = text[max(0, m.start() - 400):m.end() + 400]
    for shout in ("warn", "logger", "log.", "raise", "print("):
        assert shout not in window, (
            f"found {shout!r} near the synthetic fallback - upstream may have "
            f"made it visible, so re-audit before relying on the quarantine"
        )


@needs_upstream
def test_data_quality_monitor_is_us_shaped() -> None:
    """Not fabricated, but not usable: it hardcodes US session boundaries.

    Pointed at NSE it would mis-assess staleness on every bar. Recorded as a
    test so "we could just use their data quality monitor" is answerable with a
    fact rather than a memory.
    """
    path = UPSTREAM / "backend/services/data_quality_monitor.py"
    if not path.is_file():
        pytest.skip("module no longer exists upstream")
    text = path.read_text(encoding="utf-8", errors="ignore")
    assert "_is_us_regular_or_extended_market_hours" in text
