import { useEffect, useState } from "react";
import { research, type RegimeView as Regime } from "../api";

/**
 * The regime, with its working shown.
 *
 * A regime silences factors and gates strategies, so the number that
 * matters most on this screen is not the label - it is whether the label
 * was MEASURED. A partly-measured regime presented as a finding is worse
 * than no regime at all, so `sources` is rendered in full rather than
 * tucked behind a details toggle.
 */
export default function RegimeView() {
  const [r, setR] = useState<Regime | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    research.regime().then(setR).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="err">{err}</p>;
  if (!r) return <p className="hint">Measuring the regime…</p>;

  const dim = (k: string, v: string) => (
    <div key={k}>
      <div className="k">{k}</div>
      <div className="v" style={{ fontSize: 15 }}>
        {v === "unknown" ? <span className="pill mute">not measured</span> : v}
      </div>
    </div>
  );

  return (
    <>
      <p className="lede">
        Measured from the market itself — {r.universe.toLocaleString("en-IN")}{" "}
        liquid names on {r.as_of}. No index feed, no VIX: breadth is a
        cross-sectional fact and the bhavcopy is a cross-section.
      </p>

      {!r.measured && (
        <div className="banner crit">
          <h3>Only partly measured</h3>
          <p>
            One or more dimensions had no data behind them. The label{" "}
            <strong>{r.label}</strong> rests on the dimensions that could be
            computed — see the sources below for which could not.
          </p>
        </div>
      )}

      <div className="funnel">
        {dim("Trend", r.trend)}
        {dim("Volatility", r.volatility)}
        {dim("Breadth", r.breadth)}
        {dim("Risk appetite", r.risk_appetite)}
        <div>
          <div className="k">Label</div>
          <div className="v" style={{ fontSize: 15 }}>
            <span className={"pill " + (r.risk_off ? "crit" : "ok")}>
              {r.label}
            </span>
          </div>
        </div>
      </div>

      {r.risk_off && (
        <div className="banner crit" style={{ marginTop: 14 }}>
          <h3>Risk off</h3>
          <p>
            The risk engine is told to reduce exposure. This is the one
            regime state that changes sizing on its own.
          </p>
        </div>
      )}

      <div className="card" style={{ marginTop: 14 }}>
        <h3 style={{ fontSize: 17 }}>Where each number came from</h3>
        <p className="hint" style={{ marginTop: 2 }}>
          A regime is an input to every downstream decision, so it has to be
          arguable rather than asserted.
        </p>
        <table style={{ marginTop: 8 }}>
          <tbody>
            {Object.entries(r.sources).map(([k, v]) => (
              <tr key={k}>
                <td style={{ whiteSpace: "nowrap", verticalAlign: "top" }}>
                  <strong>{k}</strong>
                </td>
                <td className="mono" style={{ fontSize: 12.5 }}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {(r.leading_sectors.length > 0 || r.lagging_sectors.length > 0) && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 style={{ fontSize: 17 }}>Sector rotation</h3>
          <p className="hint" style={{ marginTop: 2 }}>
            Median 20-session return per industry. An industry needs at least
            three classified names to appear — two names is not a sector.
          </p>
          <div style={{ display: "flex", gap: 24, marginTop: 8, flexWrap: "wrap" }}>
            <div>
              <div className="k">Leading</div>
              {r.leading_sectors.map((s) => (
                <div key={s}><span className="pill ok">{s}</span></div>
              ))}
            </div>
            <div>
              <div className="k">Lagging</div>
              {r.lagging_sectors.map((s) => (
                <div key={s}><span className="pill mute">{s}</span></div>
              ))}
            </div>
          </div>
        </div>
      )}

      {r.max_concurrent_positions_hint !== null && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 style={{ fontSize: 17 }}>Position-count hint</h3>
          <p style={{ margin: "6px 0 0", color: "var(--ink-2)" }}>
            Breadth is narrow, so hold at most{" "}
            <strong>{r.max_concurrent_positions_hint}</strong> positions —
            when few things are genuinely working, five positions is one idea
            held five times.
          </p>
        </div>
      )}
    </>
  );
}
