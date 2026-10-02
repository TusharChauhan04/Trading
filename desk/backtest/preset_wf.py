"""Walk-forward for a preset cell on DISJOINT windows. The promotion gate.

WHY A SECOND HARNESS AND NOT walkforward.py
-------------------------------------------
`walkforward.py` walks a CATALOGUED strategy through the desk's full funnel -
Stage 0, Stage 1, regime routing, the risk engine, structural stops. The cell
that survived measurement is not that: it is a fixed-percentage stop on a
declarative preset rule with a breakout buffer, evaluated on a price panel.
Forcing it through the funnel harness would change what is being tested, and the
whole point of a walk-forward is that the thing in the windows is the thing that
was measured.

So this is the same DISCIPLINE on a different object: disjoint windows, no
overlap between them, promotion refused on an aggregate.

WHAT IT ASKS PER WINDOW, which is more than the original harness did:
  net R           after costs, month-weighted inside the window
  permutation p   did entry timing carry information IN THAT WINDOW
  trades          because a window with nine trades says nothing

And across windows: the Deflated Sharpe Ratio on the window means, deflated for
every cell that was tried to get here.

DISJOINT, NOT ROLLING. Rolling windows share trades, so "positive in 3 of 4"
can be one good period counted three times. Each window here is a contiguous
slice of the panel and no trade crosses a boundary - a trade opened inside a
window is resolved inside it or dropped, which is stricter than letting it run
on and costs only the handful of trades near each edge.

THE BAR IS DELIBERATELY HARSH. Positive in EVERY scored window, permutation
significant in every scored window, and a DSR that clears its threshold. The
surviving cell has t +0.95 and DSR 0.280 on the full sample; it is not expected
to pass, and a harness that only reports what it was built to confirm is
worthless.
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["PresetWindow", "PresetWalkForward", "walk_forward_preset"]

#: Minimum trades for a window to be SCORED rather than merely reported. A
#: window below this is not evidence either way, and averaging it in would let
#: a nine-trade window outvote a nine-hundred-trade one.
MIN_TRADES_PER_WINDOW = 30


@dataclass(frozen=True, slots=True)
class PresetWindow:
    index: int
    start: str
    end: str
    trades: int
    net_r: float | None
    median_net_r: float | None
    target_hit_pct: float | None
    perm_p: float | None
    perm_edge: float | None
    """Observed minus the shuffled-timing null, in R. The number that says the
    signal added information rather than keeping a luckier subset - see
    desk/backtest/permutation.py."""

    @property
    def scored(self) -> bool:
        return self.trades >= MIN_TRADES_PER_WINDOW

    @property
    def positive(self) -> bool:
        return bool(self.scored and self.net_r is not None and self.net_r > 0)

    @property
    def timed(self) -> bool:
        """Entry timing carried information in this window."""
        return bool(self.scored and self.perm_p is not None
                    and self.perm_p < 0.05)


@dataclass(slots=True)
class PresetWalkForward:
    label: str
    stop_pct: float
    target_pct: float
    hold_bars: int
    buffer_pct: float
    cost_r: float
    num_trials: int
    windows: list[PresetWindow] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def scored(self) -> list[PresetWindow]:
        return [w for w in self.windows if w.scored]

    @property
    def windows_positive(self) -> int:
        return sum(1 for w in self.scored if w.positive)

    @property
    def windows_timed(self) -> int:
        return sum(1 for w in self.scored if w.timed)

    @property
    def pooled_net_r(self) -> float | None:
        """Mean of the SCORED window means.

        Equal-weighted across windows on purpose: a walk-forward asks whether
        the edge holds in each period, and trade-weighting would let the
        busiest period - which is the trendiest one - answer for the others.
        """
        vals = [w.net_r for w in self.scored if w.net_r is not None]
        return st.fmean(vals) if vals else None

    def robustness(self):
        """DSR on the window means, deflated for every cell tried."""
        from desk.robustness import deflated_sharpe

        vals = [w.net_r for w in self.scored if w.net_r is not None]
        # A window is roughly a year here when four span three years, so ~1-2
        # periods a year. Passing 252 would annualise a yearly figure as daily.
        return deflated_sharpe(vals, num_trials=self.num_trials,
                               periods_per_year=2)

    def verdict(self) -> str:
        """What this evidence licenses on the maturity ladder. Conservative.

        Three independent hurdles, because each catches a different failure:
        positive in every window (not concentrated in one period), timing
        significant in every window (not market drift), and DSR clearing its
        bar (not the best of many tries).
        """
        n = len(self.scored)
        if n == 0:
            return (f"{self.label}: no window reached "
                    f"{MIN_TRADES_PER_WINDOW} trades. A sample-size result, "
                    f"not a verdict on the rule.")
        pooled = self.pooled_net_r
        rep = self.robustness()
        bits = [f"{self.label}: {n} scored windows, pooled "
                f"{pooled:+.4f}R" if pooled is not None else f"{self.label}:"]
        bits.append(f"positive in {self.windows_positive} of {n}")
        bits.append(f"timing significant in {self.windows_timed} of {n}")
        if rep.dsr is not None:
            bits.append(f"DSR {rep.dsr:.4f} at {self.num_trials} trials")
        if rep.unavailable:
            bits.append(f"DSR unavailable ({rep.unavailable})")
        head = ", ".join(bits)

        passes = (pooled is not None and pooled > 0
                  and self.windows_positive == n
                  and self.windows_timed == n
                  and n >= 3
                  and rep.dsr is not None and rep.dsr > 0.95)
        if passes:
            return (head + ". Clears every hurdle - this is what "
                    "WALK_FORWARD is meant to look like, and promoting it is "
                    "a judgement for a human rather than this function.")
        why = []
        if pooled is None or pooled <= 0:
            why.append("pooled net R is not positive")
        if self.windows_positive < n:
            why.append(f"only {self.windows_positive} of {n} windows positive")
        if self.windows_timed < n:
            why.append(f"timing significant in only {self.windows_timed} "
                       f"of {n}")
        if rep.dsr is not None and rep.dsr <= 0.95:
            why.append(f"DSR {rep.dsr:.4f} below 0.95")
        if rep.underpowered:
            why.append(f"only {rep.n} window observations - the deflated "
                       f"estimators want 60")
        return head + ". STAYS AT DRAFT: " + "; ".join(why) + "."

    def report(self) -> str:
        head = (f"Walk-forward: {self.label}\n"
                f"  {self.stop_pct:g}% stop, {self.target_pct:g}% target, "
                f"{self.hold_bars} bars max, {self.buffer_pct:g}% buffer, "
                f"cost {self.cost_r:.4f}R\n"
                f"  {'window':<26}{'trades':>8}{'netR':>9}{'median':>9}"
                f"{'targ%':>7}{'perm p':>9}{'edge':>9}")
        rows = []
        for w in self.windows:
            nr = f"{w.net_r:+.4f}" if w.net_r is not None else "        -"
            md = f"{w.median_net_r:+.4f}" if w.median_net_r is not None else "        -"
            th = f"{w.target_hit_pct:.1f}" if w.target_hit_pct is not None else "    -"
            pp = f"{w.perm_p:.4f}" if w.perm_p is not None else "        -"
            ed = f"{w.perm_edge:+.4f}" if w.perm_edge is not None else "        -"
            flag = "" if w.scored else "  (unscored)"
            rows.append(f"  {w.start + ' - ' + w.end:<26}{w.trades:>8}{nr:>9}"
                        f"{md:>9}{th:>7}{pp:>9}{ed:>9}{flag}")
        tail = [""] + [f"  CAVEAT: {c}" for c in self.caveats]
        tail += ["", "  " + self.verdict()]
        return "\n".join([head] + rows + tail)


def walk_forward_preset(panel: dict[str, pd.DataFrame], signals: np.ndarray, *,
                        label: str, stop_pct: float, hold_bars: int,
                        rr: float = 2.0, buffer_pct: float = 0.0,
                        cost_r: float = 0.0, windows: int = 4,
                        num_trials: int = 1,
                        n_permutations: int = 200,
                        seed: int = 7) -> PresetWalkForward:
    """Walk one preset cell through disjoint windows of the panel.

    `signals` is a bars x symbols boolean array already carrying whatever
    filters the cell declares - the buffer included - because a walk-forward
    must test the rule as specified, not a cleaner version of it.

    `num_trials` is how many cells were tried to arrive at this one. Required
    for the same reason it is required everywhere else in this project.
    """
    close = panel["close"]
    idx = list(close.index)
    if signals.shape != close.shape:
        raise ValueError(f"signals {signals.shape} != panel {close.shape}")
    if windows < 2:
        raise ValueError("a walk-forward needs at least 2 windows")

    from desk.backtest.permutation import outcome_lattice, signal_timing_test

    hi = panel["high"].to_numpy(float)
    lo = panel["low"].to_numpy(float)
    cl = close.to_numpy(float)
    r_lat, ex = outcome_lattice(hi, lo, cl, stop_pct=stop_pct, rr=rr,
                                hold=hold_bars)

    bounds = np.linspace(0, len(idx), windows + 1).astype(int)
    out = PresetWalkForward(label=label, stop_pct=stop_pct,
                            target_pct=stop_pct * rr, hold_bars=hold_bars,
                            buffer_pct=buffer_pct, cost_r=cost_r,
                            num_trials=num_trials)
    out.caveats.append(
        f"windows are DISJOINT slices; a trade must resolve inside its own "
        f"window or it is dropped, which costs the trades within {hold_bars} "
        f"bars of each boundary")

    for w in range(windows):
        lo_i, hi_i = int(bounds[w]), int(bounds[w + 1])
        # A trade must RESOLVE inside the window, so entries stop one hold
        # before the edge. Letting it run past the boundary would share data
        # with the next window, which is the thing disjointness buys.
        last_entry = hi_i - hold_bars - 1
        sub = np.zeros_like(signals)
        if last_entry > lo_i:
            sub[lo_i:last_entry] = signals[lo_i:last_entry]

        trades: list[tuple[int, float]] = []
        for c in range(signals.shape[1]):
            busy = -1
            col = sub[:, c]
            for i in range(lo_i, max(lo_i, last_entry)):
                if col[i] and i > busy:
                    v = r_lat[i, c]
                    if np.isfinite(v) and ex[i, c] < hi_i:
                        trades.append((i, v - cost_r))
                        busy = ex[i, c]

        net = med = th = pp = edge = None
        if trades:
            by_month: dict[str, list[float]] = {}
            for i, v in trades:
                by_month.setdefault(idx[i].strftime("%Y-%m"), []).append(v)

            cohorts = [st.fmean(v) for _k, v in sorted(by_month.items())]
            net = st.fmean(cohorts) if cohorts else None
            med = st.median([v for _i, v in trades])
            hits = sum(1 for _i, v in trades
                       if math.isclose(v + cost_r, rr, abs_tol=1e-9))
            th = 100.0 * hits / len(trades)
            if len(trades) >= MIN_TRADES_PER_WINDOW:
                try:
                    res = signal_timing_test(
                        sub, r_lat, ex, cost_r=cost_r,
                        n_permutations=n_permutations, seed=seed)
                    pp, edge = res.p_value, res.observed - res.null_mean
                except ValueError:
                    pass

        out.windows.append(PresetWindow(
            # The panel index carries date objects, not Timestamps, so no
            # .date() call - str() is already the ISO day.
            index=w, start=str(idx[lo_i]),
            end=str(idx[hi_i - 1]), trades=len(trades),
            net_r=net, median_net_r=med, target_hit_pct=th,
            perm_p=pp, perm_edge=edge))
    return out
