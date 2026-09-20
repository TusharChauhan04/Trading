import { useCallback, useEffect, useState, Fragment } from "react";
import { api, inr, type DailyPlan } from "../api";

/** Remembered between visits. Nobody wants to retype their capital daily. */
const CAPITAL_KEY = "desk.capital";
const DEFAULT_CAPITAL = 100_000;

function storedCapital(): number {
  try {
    const raw = window.localStorage.getItem(CAPITAL_KEY);
    const n = raw ? Number(raw) : NaN;
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_CAPITAL;
  } catch {
    // Private windows and blocked site data both throw here.
    return DEFAULT_CAPITAL;
  }
}

/** An ATR stop is not a worse stop, but it is a DIFFERENT claim - "this is
 *  how far the stock moves" rather than "this is where the setup fails" -
 *  and the plan should not let the two read alike. `noise_floor` is the
 *  third case: structure was found and then overruled for being too tight. */
function stopPill(basis: string): string {
  if (basis === "swing_low" || basis === "low_20") return "pill ok";
  if (basis === "noise_floor") return "pill warn";
  return "pill";
}

export default function DailyPlanView() {
  const [plan, setPlan] = useState<DailyPlan | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);
  // `capital` is what has been APPLIED; `draft` is what is in the box.
  // Kept apart deliberately: re-running the whole funnel on every
  // keystroke would fire a scan for "1", "10", "100"... and each one is a
  // real full-universe run on the server.
  const [capital, setCapital] = useState<number>(storedCapital);
  const [draft, setDraft] = useState<string>(() => String(storedCapital()));

  const load = useCallback((amount: number) => {
    setLoading(true);
    setErr("");
    api
      .plan(amount)
      .then(setPlan)
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(capital); }, [capital, load]);

  const apply = () => {
    const n = Number(draft.replace(/[,\s₹]/g, ""));
    if (!Number.isFinite(n) || n <= 0) {
      setErr("Enter a capital amount greater than zero.");
      return;
    }
    try { window.localStorage.setItem(CAPITAL_KEY, String(n)); } catch {
      /* not fatal - the plan still runs, it just will not be remembered */
    }
    setCapital(n);
  };

  const capitalBox = (
    <div className="card" style={{ marginBottom: 14 }}>
      <h3 style={{ fontSize: 17 }}>Capital to trade</h3>
      <p className="hint" style={{ marginTop: 2 }}>
        Every position is sized against this. Targets are set at 1:2 —
        risk one rupee to make two. Changing it re-runs the whole funnel,
        because a plan for one lakh is a different plan, not the same one
        scaled.
      </p>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10 }}>
        <span className="mono" style={{ fontSize: 18 }}>₹</span>
        <input
          className="mono"
          inputMode="numeric"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") apply(); }}
          style={{ width: 180, padding: "6px 8px", fontSize: 15 }}
          aria-label="Capital to trade, in rupees"
        />
        <button onClick={apply} disabled={loading}>
          {loading ? "Running…" : "Apply"}
        </button>
        <span className="hint">
          currently sizing against {inr(capital)}
        </span>
      </div>
    </div>
  );

  if (err) return <>{capitalBox}<p className="err">{err}</p></>;
  if (!plan) return <>{capitalBox}<p className="hint">Loading today's plan…</p></>;

  const stage = (k: string, v: number) => (
    <div key={k}><div className="k">{k}</div><div className="v">{v}</div></div>
  );

  return (
    <>
      {capitalBox}
      <p className="lede">
        One plan a day. The funnel below narrows the universe; whatever survives
        gets a full multi-agent read. If nothing survives, the plan says so
        instead of manufacturing a trade.
      </p>

      <div className="funnel">
        {stage("Universe", plan.universe_scanned)}
        {stage("Stage 1", plan.survived_stage1)}
        {stage("Stage 2", plan.survived_stage2)}
        {stage("Analysed", plan.analysed)}
        {stage("Trades", plan.trades.length)}
      </div>

      {plan.no_trade_reason ? (
        <div className="banner crit">
          <h3>NO TRADE — {plan.as_of}</h3>
          <p>{plan.no_trade_reason}</p>
        </div>
      ) : (
        <div className="banner good">
          <h3>{plan.trades.length} trade{plan.trades.length === 1 ? "" : "s"} — {plan.as_of}</h3>
          <p>Each one cleared every risk gate. Sizes are already capped.</p>
        </div>
      )}

      <div className="card">
        <h3 style={{ fontSize: 17 }}>Regime</h3>
        <p style={{ margin: "6px 0 0", color: "var(--ink-2)" }}>
          <span className="pill mute">{plan.regime}</span>{" "}
          {plan.regime_note}
        </p>
        {plan.regime_detail?.length > 0 && (
          <ul className="reasons">
            {plan.regime_detail.map((d) => <li key={d} className="mono">{d}</li>)}
          </ul>
        )}
      </div>

      {plan.market_risks?.length > 0 && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 style={{ fontSize: 17 }}>Market risks today</h3>
          <p className="hint" style={{ marginTop: 2 }}>
            Events, news and data affecting the trading day — separate from the
            system caveats below.
          </p>
          <ul className="reasons">
            {plan.market_risks.map((m) => <li key={m}>{m}</li>)}
          </ul>
        </div>
      )}

      {plan.trades.length > 0 && (
        <div className="tw" style={{ marginTop: 14 }}>
          <table>
            <thead>
              <tr>
                <th>Symbol</th><th>Stance</th><th>Conf</th><th>Entry</th>
                <th>Stop</th><th>Target</th><th>Qty</th><th>At risk</th>
              </tr>
            </thead>
            <tbody>
              {plan.trades.map((t) => (
                <Fragment key={t.symbol}>
                  <tr>
                    <td><strong>{t.symbol}</strong></td>
                    <td><span className="pill ok">{t.stance}</span></td>
                    <td className="num">{(t.confidence * 100).toFixed(0)}%</td>
                    <td className="num">{t.entry}</td>
                    <td className="num">
                      {t.stop}
                      {t.stop_basis && (
                        <span className={stopPill(t.stop_basis)}
                              style={{ marginLeft: 6 }}>
                          {t.stop_basis.replace("_", " ")}
                        </span>
                      )}
                    </td>
                    <td className="num">{t.target}</td>
                    <td className="num">{t.qty}</td>
                    <td className="num">{inr(t.capital_at_risk)}</td>
                  </tr>
                  {t.invalidation && (
                    /* The stop's REASON, not a repeat of its price. A number
                       on its own cannot be argued with; a named level can be
                       disagreed with before the order is placed, which is the
                       whole point of a plan a human executes. */
                    <tr className="sub">
                      <td />
                      <td colSpan={7} className="muted"
                          style={{ fontSize: 12, paddingTop: 0 }}>
                        <strong>Invalidation:</strong> {t.invalidation}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {plan.warnings.length > 0 && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 style={{ fontSize: 17 }}>What is missing before this is trustworthy</h3>
          <ul className="reasons">
            {plan.warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
        </div>
      )}
    </>
  );
}
