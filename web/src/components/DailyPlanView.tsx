import { useEffect, useState } from "react";
import { api, inr, type DailyPlan } from "../api";

export default function DailyPlanView() {
  const [plan, setPlan] = useState<DailyPlan | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.plan().then(setPlan).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="err">Backend unreachable — {err}</p>;
  if (!plan) return <p className="hint">Loading today's plan…</p>;

  const stage = (k: string, v: number) => (
    <div key={k}><div className="k">{k}</div><div className="v">{v}</div></div>
  );

  return (
    <>
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
                <tr key={t.symbol}>
                  <td><strong>{t.symbol}</strong></td>
                  <td><span className="pill ok">{t.stance}</span></td>
                  <td className="num">{(t.confidence * 100).toFixed(0)}%</td>
                  <td className="num">{t.entry}</td>
                  <td className="num">{t.stop}</td>
                  <td className="num">{t.target}</td>
                  <td className="num">{t.qty}</td>
                  <td className="num">{inr(t.capital_at_risk)}</td>
                </tr>
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
