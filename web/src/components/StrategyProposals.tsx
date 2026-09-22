import { useEffect, useState } from "react";
import { api, type Proposal, type ProposalsResponse } from "../api";

/**
 * What every eligible strategy proposed for one session.
 *
 * The screen has one job the plan view cannot do: show that the desk has
 * MORE THAN ONE OPINION, and show which opinions were not asked for and
 * why. Three outcomes are rendered as three separate blocks and never
 * merged - a strategy that proposed nothing, one silenced because the
 * measured regime is hostile to it, and one that has no adapter so it
 * could not be asked at all. Collapsing those would make a correctly
 * silent strategy look like a broken one.
 */

function n2(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(2);
}

function meta(p: Proposal, key: string): number | null {
  const v = p.metadata?.[key];
  return typeof v === "number" ? v : null;
}

/** The strategy's own measure of how pronounced the setup is. Deliberately
 *  not normalised across strategies: percent past a channel and sigma
 *  below a band are different units, and a shared scale would imply a
 *  cross-strategy ranking that nothing has earned yet. */
function strength(p: Proposal): string {
  const brk = meta(p, "break_pct");
  if (brk !== null) return `+${brk.toFixed(2)}% past channel`;
  const sig = meta(p, "stretch_sigma");
  if (sig !== null) return `${sig.toFixed(2)}σ stretched`;
  return "—";
}

function ProposalRow({ p }: { p: Proposal }) {
  const rr = meta(p, "rr");
  const risk = p.entry !== null && p.stop !== null ? p.entry - p.stop : null;
  return (
    <tr>
      <td className="mono">{p.symbol}</td>
      <td>
        {p.setup_type}
        <div className="hint" style={{ marginTop: 3 }}>{strength(p)}</div>
      </td>
      <td className="num mono">{n2(p.entry)}</td>
      <td className="num mono">{n2(p.stop)}</td>
      <td className="num mono">{n2(p.target)}</td>
      <td className="num mono">
        {rr !== null ? rr.toFixed(2)
          : risk && p.target !== null && p.entry !== null
            ? ((p.target - p.entry) / risk).toFixed(2) : "—"}
      </td>
      <td>
        <span className="pill mute">
          {String(p.metadata?.maturity ?? "unknown")}
        </span>
      </td>
    </tr>
  );
}

function StrategyBlock({ name, rows }: { name: string; rows: Proposal[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <h3 style={{ fontSize: 17 }} className="mono">{name}</h3>
        <span className="pill ok">{rows.length} proposal{rows.length === 1 ? "" : "s"}</span>
        <span className="pill warn" style={{ marginLeft: "auto" }}>not sizeable</span>
      </div>

      {rows.length === 0 ? (
        <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
          Asked, and had nothing to say — no name in today's universe fired
          this rule. Different from silenced, and different from unbuilt.
        </p>
      ) : (
        <>
          <div className="tw" style={{ marginTop: 12 }}>
            <table>
              <thead>
                <tr>
                  <th>Symbol</th><th>Setup</th><th>Entry</th>
                  <th>Stop</th><th>Target</th><th>R:R</th><th>Maturity</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((p) => <ProposalRow key={p.symbol} p={p} />)}
              </tbody>
            </table>
          </div>
          {/* Unstyled by design: every other button in this app is a bare
              <button>, and index.css has no variant class. Inventing one
              here would be the only styled button in the project. */}
          <button type="button" style={{ marginTop: 10 }}
            onClick={() => setOpen(!open)}>
            {open ? "Hide" : "Show"} the reasoning for {rows[0].symbol}
          </button>
          {open && (
            <div style={{ marginTop: 10 }}>
              <p style={{ fontSize: 14.5 }}>{rows[0].narrative}</p>
              <p className="hint" style={{ marginBottom: 6 }}>
                <b>Trigger.</b> {rows[0].trigger}
              </p>
              {rows[0].invalidations.map((iv, i) => (
                <p key={i} className="hint" style={{ marginBottom: 6 }}>
                  <b>Invalidation.</b> {iv}
                </p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function StrategyProposals() {
  const [data, setData] = useState<ProposalsResponse | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.proposals().then(setData).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="err">Backend unreachable — {err}</p>;
  if (!data) return <p className="hint">Asking every eligible strategy…</p>;

  const total = Object.values(data.counts).reduce((a, b) => a + b, 0);
  const silenced = Object.entries(data.silenced);
  const unbuilt = Object.entries(data.unimplemented);

  return (
    <section>
      <p className="lede">
        Every strategy the measured regime permits to speak, asked against
        the same universe on the same bars. The regime is the router: a
        strategy that names today's regime as hostile is silenced outright,
        not down-weighted, because a mean-reversion rule in a crisis is not
        a weak opinion — it is a wrong one.
      </p>

      <div className="readout">
        <div><div className="k">AS OF</div><div className="v mono">{data.as_of}</div></div>
        <div>
          <div className="k">REGIME</div>
          <div className="v mono">{data.regime}</div>
          <div className="hint">{data.regime_measured ? "measured" : "NOT measured"}</div>
        </div>
        <div><div className="k">UNIVERSE</div><div className="v num">{data.universe_scanned}</div></div>
        <div><div className="k">PROPOSALS</div><div className="v num">{total}</div></div>
      </div>

      {data.tradeable_now.length === 0 && (
        <div className="banner warn" style={{ marginTop: 20 }}>
          <h3>Nothing here may be sized</h3>
          <p>
            Every catalogued strategy sits below the <span className="mono">WALK_FORWARD</span>
            {" "}rung, so none is trusted and none can produce a live position. These
            are proposals to read and to measure. A strategy becomes sizeable by
            surviving out-of-sample windows, not by appearing on this screen.
          </p>
        </div>
      )}

      {Object.entries(data.proposals).map(([name, rows]) => (
        <StrategyBlock key={name} name={name} rows={rows} />
      ))}

      {silenced.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 17 }}>Silenced by the regime</h3>
          <p className="hint">
            These were not asked. That is the gating rule working, not a
            failure — and it is why this block exists separately from
            “proposed nothing”.
          </p>
          <table style={{ marginTop: 8 }}>
            <tbody>
              {silenced.map(([k, why]) => (
                <tr key={k}>
                  <td className="mono" style={{ width: 200 }}>{k}</td>
                  <td className="hint">{why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {unbuilt.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 17 }}>Catalogued but not built</h3>
          <p className="hint">
            No adapter exists, so these could not be asked at all. Shown
            rather than omitted: a strategy missing from the screen is
            indistinguishable from one that declined.
          </p>
          <table style={{ marginTop: 8 }}>
            <tbody>
              {unbuilt.map(([k, why]) => (
                <tr key={k}>
                  <td className="mono" style={{ width: 200 }}>{k}</td>
                  <td className="hint">{why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data.caveats.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 17 }}>What could not be checked</h3>
          {data.caveats.map((c, i) => (
            <p key={i} className="hint" style={{ marginBottom: 6 }}>{c}</p>
          ))}
        </div>
      )}
    </section>
  );
}
