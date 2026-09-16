import { useEffect, useState } from "react";
import { api, type FleetEntry } from "../api";

const PILL: Record<string, string> = {
  wired: "ok", verified: "warn", installed: "warn", absent: "bad",
};

export default function FleetPanel() {
  const [rows, setRows] = useState<FleetEntry[] | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.fleet().then(setRows).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="err">Backend unreachable — {err}</p>;
  if (!rows) return <p className="hint">Loading fleet…</p>;

  const wired = rows.filter((r) => r.wired).length;
  const ready = rows.filter((r) => r.india_ready).length;

  return (
    <>
      <p className="lede">
        Every component the desk knows about, and — separately — whether it is
        actually connected. A repository can be installed, verified and still
        useless to the pipeline. <strong>{wired} of {rows.length}</strong> are
        wired; <strong>{ready}</strong> are India-ready.
      </p>

      {wired === 0 && (
        <div className="banner warn">
          <h3>Nothing is wired yet</h3>
          <p>
            All five components are on disk and smoke-tested, but none speaks the
            AnalysisResult contract. Until an adapter exists, the coordinator has
            no one to ask, which is why the daily plan returns NO TRADE.
          </p>
        </div>
      )}

      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>Component</th><th>Kind</th><th>Version</th><th>Licence</th>
              <th>India</th><th>State</th><th>Role &amp; blockers</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td><strong>{r.name}</strong></td>
                <td>{r.kind}</td>
                <td className="mono num">{r.version}</td>
                <td className="mono">{r.licence}</td>
                <td>
                  <span className={"pill " + (r.india_ready ? "ok" : "bad")}>
                    {r.india_ready ? "ready" : "no"}
                  </span>
                </td>
                <td><span className={"pill " + PILL[r.status]}>{r.status}</span></td>
                <td>
                  {r.note}
                  {r.blockers.length > 0 && (
                    <ul className="blockers">
                      {r.blockers.map((b) => <li key={b}>{b}</li>)}
                    </ul>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
