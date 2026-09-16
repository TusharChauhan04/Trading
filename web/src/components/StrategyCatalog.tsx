import { useEffect, useState } from "react";
import { api, type StrategySpec } from "../api";

const RUNGS = ["draft", "audited", "revalidated", "walk_forward", "paper", "live"];

const REGIMES = [
  "trending_up", "trending_down", "range", "high_vol", "crisis", "unknown",
];

function Ladder({ at }: { at: string }) {
  const i = RUNGS.indexOf(at);
  return (
    <div className="ladder">
      {RUNGS.map((r, n) => (
        <span key={r}
          className={"rung" + (n < i ? " done" : n === i ? " here" : "")}
          title={r}>
          {r.replace("_", " ")}
        </span>
      ))}
    </div>
  );
}

export default function StrategyCatalog() {
  const [rows, setRows] = useState<StrategySpec[] | null>(null);
  const [regime, setRegime] = useState("range");
  const [elig, setElig] = useState<string[]>([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.strategies().then(setRows).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    // trusted_only=false, because right now nothing is trusted and an empty
    // list would tell you nothing about the gating rule.
    api.eligible(regime, false).then(setElig).catch(() => setElig([]));
  }, [regime]);

  if (err) return <p className="err">Backend unreachable — {err}</p>;
  if (!rows) return <p className="hint">Loading strategy catalog…</p>;

  const trusted = rows.filter((r) => r.trusted).length;
  const defects = rows.reduce((n, r) => n + r.defects.length, 0);

  return (
    <>
      <p className="lede">
        Your six strategies, held as data the coordinator can reason about
        rather than six scripts. <strong>{trusted} of {rows.length}</strong> have
        cleared walk-forward, and <strong>{defects}</strong> known defects are
        still on the books — so none of them may size a live trade yet.
      </p>

      <div className="banner warn">
        <h3>Nothing here is trusted yet — on purpose</h3>
        <p>
          A strategy earns the right to influence a plan by climbing the ladder:
          draft → audited → revalidated → walk-forward → paper → live. Four of
          the six are still carrying defects that would make their backtest
          numbers meaningless. They are shown, labelled, and locked.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 14 }}>
        <div className="field" style={{ maxWidth: 280, marginBottom: 10 }}>
          <label htmlFor="reg">Who may speak in regime…</label>
          <select id="reg" value={regime} onChange={(e) => setRegime(e.target.value)}>
            {REGIMES.map((r) => <option key={r} value={r}>{r.replace("_", " ")}</option>)}
          </select>
        </div>
        <div>
          {elig.length === 0
            ? <span className="pill bad">none — every strategy calls this regime hostile</span>
            : elig.map((k) => <span key={k} className="pill ok" style={{ marginRight: 6 }}>{k}</span>)}
        </div>
        <p className="hint">
          A strategy that names the current regime as hostile is silenced, not
          down-weighted. Mean reversion in a crisis is not a weak opinion, it is
          a wrong one.
        </p>
      </div>

      {rows.map((s) => (
        <div className="card" key={s.key}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
            <h3 style={{ fontSize: 19 }}>{s.name}</h3>
            <span className="pill mute">{s.family.replace("_", " ")}</span>
            <span className="pill mute">{s.horizon}</span>
            <span className={"pill " + (s.trusted ? "ok" : "bad")}
              style={{ marginLeft: "auto" }}>
              {s.trusted ? "may trade" : "locked"}
            </span>
          </div>

          <p style={{ margin: "8px 0 12px", color: "var(--ink-2)" }}>{s.thesis}</p>

          <Ladder at={s.maturity} />

          <div className="specgrid">
            <div>
              <div className="k">Parameters</div>
              <div className="mono">
                {Object.entries(s.params).map(([k, v]) => (
                  <div key={k}>{k} = {v}</div>
                ))}
              </div>
            </div>
            <div>
              <div className="k">Needs</div>
              {s.data_needed.map((d) => <div key={d}>{d}</div>)}
            </div>
            <div>
              <div className="k">Works in</div>
              {s.favourable_regimes.map((r) => (
                <span key={r} className="pill ok" style={{ marginRight: 5 }}>
                  {r.replace("_", " ")}
                </span>
              ))}
              <div className="k" style={{ marginTop: 10 }}>Silenced in</div>
              {s.hostile_regimes.map((r) => (
                <span key={r} className="pill bad" style={{ marginRight: 5 }}>
                  {r.replace("_", " ")}
                </span>
              ))}
            </div>
          </div>

          {s.defects.length > 0 && (
            <>
              <div className="k" style={{ marginTop: 14 }}>Open against it</div>
              <ul className="reasons">
                {s.defects.map((d) => <li key={d}>{d}</li>)}
              </ul>
            </>
          )}

          {s.notes && <p className="hint">{s.notes}</p>}
        </div>
      ))}
    </>
  );
}
