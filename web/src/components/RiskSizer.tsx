import { useEffect, useState } from "react";
import { api, inr, type PositionIn, type RiskConfig, type Sizing } from "../api";

/** Five of the engine's gates — max positions, sector, cluster, portfolio heat
 *  and the daily loss cap — read the open book. Without it they evaluate
 *  against an empty portfolio and always pass, which looks like protection and
 *  is not. Two worked-in positions so the caps are live from first render. */
const SEED_BOOK: PositionIn[] = [
  { symbol: "HDFCBANK.NS", qty: 90, entry: 1650, stop: 1590,
    sector: "BANK", corr_group: "RATE_SENSITIVE" },
  { symbol: "INFY.NS", qty: 110, entry: 1520, stop: 1460, sector: "IT" },
];

/** The only screen in the app that is fully alive: every number below comes
 *  from desk.risk.engine, which no model is allowed to touch. */
export default function RiskSizer() {
  const [cfg, setCfg] = useState<RiskConfig | null>(null);
  const [symbol, setSymbol] = useState("RELIANCE.NS");
  const [entry, setEntry] = useState(1400);
  const [stop, setStop] = useState(1370);
  const [target, setTarget] = useState(1490);
  const [sector, setSector] = useState("ENERGY");
  const [corrGroup, setCorrGroup] = useState("");
  const [adv, setAdv] = useState("");
  const [atr, setAtr] = useState("");
  const [slip, setSlip] = useState("");
  const [riskOff, setRiskOff] = useState(false);
  const [book, setBook] = useState<PositionIn[]>(SEED_BOOK);
  const [pnl, setPnl] = useState(0);
  const [out, setOut] = useState<Sizing | null>(null);
  const [err, setErr] = useState("");

  const blank = (s: string) => (s.trim() === "" ? null : Number(s));

  useEffect(() => {
    api.defaultConfig().then(setCfg).catch((e) => setErr(String(e)));
  }, []);

  // Re-price on every keystroke. The engine is pure, so this is safe.
  useEffect(() => {
    if (!cfg) return;
    let dead = false;
    api
      .size({
        symbol, entry, stop, target, sector,
        corr_group: corrGroup.trim() === "" ? null : corrGroup.trim(),
        adv_shares: blank(adv),
        atr_pct: blank(atr),
        expected_slippage_pct: blank(slip),
        market_risk_off: riskOff,
        config: cfg,
        open_positions: book,
        realised_pnl_today: pnl,
      })
      .then((s) => { if (!dead) { setOut(s); setErr(""); } })
      .catch((e) => { if (!dead) setErr(String(e)); });
    return () => { dead = true; };
  }, [cfg, symbol, entry, stop, target, sector, corrGroup, adv, atr, slip,
      riskOff, book, pnl]);

  if (err && !cfg) return <p className="err">Backend unreachable — {err}</p>;
  if (!cfg) return <p className="hint">Loading risk config…</p>;

  const num = (k: keyof RiskConfig, label: string, step = "0.1") => (
    <div className="field" key={k}>
      <label htmlFor={k}>{label}</label>
      {/* null means "disabled", and an empty input is how that reads on screen.
          Coercing it to 0 would silently turn a disabled scaler into an active
          one pinned at zero. */}
      <input id={k} type="number" step={step} value={cfg[k] ?? ""}
        onChange={(e) => setCfg({
          ...cfg,
          [k]: e.target.value === "" ? null : Number(e.target.value),
        })} />
    </div>
  );

  return (
    <>
      <p className="lede">
        Position sizing is arithmetic, not judgement. Change anything and the
        answer moves instantly — including to a refusal. A refusal is a correct
        answer.
      </p>

      <div className="grid2">
        <div className="card">
          <h3 style={{ fontSize: 17, marginBottom: 14 }}>The setup</h3>
          <div className="field">
            <label htmlFor="sym">Symbol</label>
            <input id="sym" value={symbol} onChange={(e) => setSymbol(e.target.value)} />
          </div>
          <div className="row2">
            <div className="field">
              <label htmlFor="e">Entry</label>
              <input id="e" type="number" value={entry}
                onChange={(e) => setEntry(Number(e.target.value))} />
            </div>
            <div className="field">
              <label htmlFor="s">Stop</label>
              <input id="s" type="number" value={stop}
                onChange={(e) => setStop(Number(e.target.value))} />
            </div>
          </div>
          <div className="row2">
            <div className="field">
              <label htmlFor="t">Target</label>
              <input id="t" type="number" value={target}
                onChange={(e) => setTarget(Number(e.target.value))} />
            </div>
            <div className="field">
              <label htmlFor="sec">Sector</label>
              <input id="sec" value={sector} onChange={(e) => setSector(e.target.value)} />
            </div>
          </div>
          <div className="field">
            <label htmlFor="cg">Correlation cluster (blank = sector cap only)</label>
            <input id="cg" value={corrGroup} placeholder="e.g. RATE_SENSITIVE"
              onChange={(e) => setCorrGroup(e.target.value)} />
          </div>
          <div className="row2">
            <div className="field">
              <label htmlFor="adv">ADV in shares</label>
              <input id="adv" type="number" value={adv} placeholder="blank = unchecked"
                onChange={(e) => setAdv(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="atr">ATR % of price</label>
              <input id="atr" type="number" step="0.1" value={atr}
                placeholder="blank = unchecked"
                onChange={(e) => setAtr(e.target.value)} />
            </div>
          </div>
          <div className="row2">
            <div className="field">
              <label htmlFor="slip">Expected slippage %</label>
              <input id="slip" type="number" step="0.05" value={slip}
                placeholder="blank = unchecked"
                onChange={(e) => setSlip(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="pnl">Realised P&amp;L today</label>
              <input id="pnl" type="number" step="1000" value={pnl}
                onChange={(e) => setPnl(Number(e.target.value))} />
            </div>
          </div>
          <div className="field">
            <label htmlFor="ro">Market regime</label>
            <select id="ro" value={riskOff ? "off" : "on"}
              onChange={(e) => setRiskOff(e.target.value === "off")}>
              <option value="on">normal</option>
              <option value="off">risk-off</option>
            </select>
          </div>

          <h3 style={{ fontSize: 17, margin: "22px 0 14px" }}>Your limits</h3>
          {num("capital", "Capital (INR)", "10000")}
          <div className="row2">
            {num("risk_pct", "Risk per trade %")}
            {num("min_risk_reward", "Min R:R")}
          </div>
          <div className="row2">
            {num("max_position_pct", "Max position %")}
            {num("max_sector_pct", "Max sector %")}
          </div>
          <div className="row2">
            {num("max_open_risk_pct", "Max portfolio heat %")}
            {num("max_adv_participation_pct", "Max % of ADV")}
          </div>
          <div className="row2">
            {num("max_stop_distance_pct", "Max stop distance %")}
            {num("lot_size", "Lot size", "1")}
          </div>
          <div className="row2">
            {num("max_slippage_pct", "Max slippage % of price")}
            {num("max_slippage_share_of_stop_pct", "Max slippage % of stop")}
          </div>
          <div className="row2">
            {num("max_instrument_atr_pct", "Max instrument ATR %")}
            {num("risk_off_exposure_pct", "Risk-off exposure %")}
          </div>

          <h3 style={{ fontSize: 17, margin: "22px 0 6px" }}>Open book</h3>
          <p className="hint" style={{ marginTop: 0 }}>
            Five gates read this. With an empty book they all pass, which looks
            like protection and is not.
          </p>
          {book.map((p, i) => (
            <div key={p.symbol} className="mono"
              style={{ fontSize: 12.5, display: "flex", gap: 8, alignItems: "baseline",
                       padding: "5px 0", borderBottom: "1px solid var(--rule)" }}>
              <span style={{ flex: 1 }}>{p.symbol}</span>
              <span>{p.qty} @ {p.entry}</span>
              <span className="pill mute">{p.corr_group ?? p.sector}</span>
              <button type="button" aria-label={`remove ${p.symbol}`}
                onClick={() => setBook(book.filter((_, j) => j !== i))}
                style={{ background: "none", border: "none", cursor: "pointer",
                         color: "var(--crit)" }}>remove</button>
            </div>
          ))}
          {book.length === 0 && (
            <p className="hint" style={{ color: "var(--warn)" }}>
              Book empty — the portfolio gates cannot bind.{" "}
              <button type="button" onClick={() => setBook(SEED_BOOK)}
                style={{ background: "none", border: "none", cursor: "pointer",
                         color: "var(--accent)", textDecoration: "underline",
                         padding: 0 }}>restore sample</button>
            </p>
          )}
        </div>

        <div>
          {out && !out.approved && (
            <div className="banner crit">
              <h3>Refused</h3>
              <p>The engine will not size this trade.</p>
              <ul className="reasons">
                {out.reasons.map((r) => <li key={r}><strong>{r}</strong></li>)}
                {out.notes.map((n) => <li key={n}>{n}</li>)}
              </ul>
            </div>
          )}

          {out && out.approved && (
            <>
              <div className="readout">
                <div><div className="k">Quantity</div>
                  <div className="v">{out.qty.toLocaleString("en-IN")}</div></div>
                <div><div className="k">Position value</div>
                  <div className="v">{inr(out.position_value)}</div></div>
                <div><div className="k">Capital at risk</div>
                  <div className="v bad">{inr(out.capital_at_risk)}</div></div>
                <div><div className="k">Risk : reward</div>
                  <div className="v ok">{out.risk_reward ?? "—"}</div></div>
              </div>
              <div className="card" style={{ marginTop: 14 }}>
                <div className="mono" style={{ fontSize: 13 }}>
                  {out.symbol} · buy {out.qty} @ {out.entry} · stop {out.stop} · target {out.target}
                </div>
                {out.notes.length > 0 ? (
                  <ul className="reasons">
                    {out.notes.map((n) => <li key={n}>{n}</li>)}
                  </ul>
                ) : (
                  <p className="hint">
                    No cap bound. Size is exactly risk budget ÷ stop distance.
                  </p>
                )}
                <p className="hint">
                  Sizing only. Nothing here places an order — the desk has no
                  broker connection by design.
                </p>
              </div>
            </>
          )}

          {out && out.checks_skipped.length > 0 && (
            <div className="card" style={{ marginTop: 14 }}>
              <h3 style={{ fontSize: 16 }}>
                {out.checks_skipped.length} gate
                {out.checks_skipped.length === 1 ? "" : "s"} could not run
              </h3>
              <p className="hint" style={{ marginTop: 2 }}>
                These were skipped for want of an input, not passed. An approval
                with this list is weaker than one without it.
              </p>
              <ul className="reasons">
                {out.checks_skipped.map((c) => <li key={c} className="mono">{c}</li>)}
              </ul>
            </div>
          )}

          {err && <p className="err">{err}</p>}
        </div>
      </div>
    </>
  );
}
