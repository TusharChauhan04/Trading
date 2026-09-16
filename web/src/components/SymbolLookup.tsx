import { useEffect, useState } from "react";
import { api, type SymbolInfo } from "../api";

/** Four libraries, four spellings of the same share. This proves the
 *  translation layer works before any of them is wired in. */
export default function SymbolLookup() {
  const [raw, setRaw] = useState("RELIANCE");
  const [info, setInfo] = useState<SymbolInfo | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    const q = raw.trim();
    if (!q) { setInfo(null); setErr("enter a symbol"); return; }
    let dead = false;
    api.symbol(q)
      .then((i) => { if (!dead) { setInfo(i); setErr(""); } })
      .catch(() => { if (!dead) { setInfo(null); setErr(`"${q}" is not a symbol this desk will guess at`); } });
    return () => { dead = true; };
  }, [raw]);

  return (
    <>
      <p className="lede">
        Try <span className="mono">RELIANCE</span>,{" "}
        <span className="mono">NSE:INFY</span>,{" "}
        <span className="mono">TCS.BO</span>, or something invalid. Ambiguous
        input is refused rather than guessed — a wrong guess routes an order to
        the wrong exchange.
      </p>

      <div className="card" style={{ maxWidth: 620 }}>
        <div className="field">
          <label htmlFor="raw">Symbol, any dialect</label>
          <input id="raw" value={raw} onChange={(e) => setRaw(e.target.value)} />
        </div>

        {err && <p className="err">{err}</p>}

        {info && (
          <table style={{ marginTop: 8 }}>
            <tbody>
              <tr><td><strong>canonical</strong></td>
                  <td className="mono">{info.canonical}</td></tr>
              <tr><td><strong>exchange</strong></td>
                  <td className="mono">{info.exchange}</td></tr>
              {Object.entries(info.dialects).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td className="mono">{v}</td></tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
