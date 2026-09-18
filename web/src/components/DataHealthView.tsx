import { useEffect, useState } from "react";
import {
  research, type CrossCheck, type FundamentalsView, type NewsView,
} from "../api";

/**
 * Whether the data under today's plan can be trusted.
 *
 * Every panel here answers the same question in a different place: what
 * was NOT checked. Coverage is shown as a fraction of the universe rather
 * than as a pass rate, because "665 of 3,483 agreed" and "everything
 * agreed" are wildly different statements and only the first is true.
 */
export default function DataHealthView() {
  const [xc, setXc] = useState<CrossCheck | null>(null);
  const [xcErr, setXcErr] = useState("");
  const [fu, setFu] = useState<FundamentalsView | null>(null);
  const [fuErr, setFuErr] = useState("");
  const [nw, setNw] = useState<NewsView | null>(null);

  useEffect(() => {
    research.fundamentals()
      .then((f) => {
        setFu(f);
        return research.crosscheck(f.requested_as_of);
      })
      .then(setXc)
      .catch((e) => setXcErr(String(e)));
    research.fundamentals().catch((e) => setFuErr(String(e)));
    research.news().then(setNw).catch(() => undefined);
  }, []);

  const pct = (n: number, d: number) => (d ? (100 * n) / d : 0);

  return (
    <>
      <p className="lede">
        What is behind today's numbers, and what is not. A check that could
        not run is reported as <em>not checked</em> — never as passed.
      </p>

      {/* ---------------------------------------------- second source --- */}
      <div className="card">
        <h3 style={{ fontSize: 17 }}>Second source (BSE)</h3>
        {xcErr && (
          <div className="banner crit" style={{ marginTop: 8 }}>
            <h3>Not cross-checked</h3>
            <p>
              Today's prices rest on a single source. Run{" "}
              <span className="mono">
                python -m desk.marketdata.refresh crosscheck
              </span>
            </p>
          </div>
        )}
        {xc && (
          <>
            <p className="hint" style={{ marginTop: 2 }}>
              {xc.as_of} · tolerance {xc.tolerance_pct}% · checked{" "}
              {new Date(xc.checked_at).toLocaleString("en-IN")}
            </p>
            <div className="funnel" style={{ marginTop: 8 }}>
              <div>
                <div className="k">Checkable</div>
                <div className="v">{Number(xc.coverage.checkable)}</div>
              </div>
              <div>
                <div className="k">Thin on BSE</div>
                <div className="v">{Number(xc.coverage.thin_on_bse)}</div>
              </div>
              <div>
                <div className="k">Not on BSE</div>
                <div className="v">{Number(xc.coverage.not_on_bse)}</div>
              </div>
              <div>
                <div className="k">No ISIN</div>
                <div className="v">{Number(xc.coverage.no_isin)}</div>
              </div>
              <div>
                <div className="k">Disagree</div>
                <div className="v">{xc.disagreements.length}</div>
              </div>
            </div>
            <p className="hint" style={{ marginTop: 8 }}>
              Only{" "}
              {pct(Number(xc.coverage.checkable),
                   Number(xc.coverage.nse_symbols)).toFixed(0)}
              % of the universe is checkable. The rest trade too thinly on
              BSE for its close to be independent evidence, or are not listed
              there — those were <strong>not checked</strong>, not cleared. A
              thin print that happens to match is not corroboration.
            </p>
            {xc.disagreements.length > 0 && (
              <div className="tw" style={{ marginTop: 10 }}>
                <table>
                  <thead>
                    <tr>
                      <th>Symbol</th><th>NSE</th><th>BSE</th><th>Diff</th>
                    </tr>
                  </thead>
                  <tbody>
                    {xc.disagreements.slice(0, 10).map((d) => (
                      <tr key={d.symbol}>
                        <td><strong>{d.symbol}</strong></td>
                        <td className="num">{d.nse_close}</td>
                        <td className="num">{d.bse_close}</td>
                        <td className="num">
                          <span className="pill crit">
                            {d.diff_pct.toFixed(2)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>

      {/* ---------------------------------------------- fundamentals --- */}
      <div className="card" style={{ marginTop: 14 }}>
        <h3 style={{ fontSize: 17 }}>Fundamentals coverage</h3>
        {fuErr && <p className="err">{fuErr}</p>}
        {fu && (
          <>
            <p className="hint" style={{ marginTop: 2 }}>
              Table built for {fu.table_as_of}
              {fu.staleness_days > 0 && <> · {fu.staleness_days} day(s) behind</>}
              {" "}· {fu.total} symbols on file
            </p>
            {fu.caveat && (
              <div className={"banner " + (fu.stale ? "crit" : "")}
                   style={{ marginTop: 8 }}>
                <p>{fu.caveat}</p>
              </div>
            )}
            <table style={{ marginTop: 8 }}>
              <tbody>
                {Object.entries(fu.coverage).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="num"><strong>{v}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="hint" style={{ marginTop: 8 }}>
              Everything from NSE's own XBRL filings — free, official, and
              timestamped to when the market actually learned it. Anything
              not on file is <strong>not checked</strong>.
            </p>
          </>
        )}
      </div>

      {/* ----------------------------------------------------- news --- */}
      <div className="card" style={{ marginTop: 14 }}>
        <h3 style={{ fontSize: 17 }}>News snapshot</h3>
        {!nw ? <p className="hint">Loading…</p> : !nw.available ? (
          <div className="banner crit" style={{ marginTop: 8 }}>
            <p>{nw.caveat ?? "No news context available."}</p>
          </div>
        ) : (
          <>
            <p className="hint" style={{ marginTop: 2 }}>
              {nw.total} deduplicated stories, fetched{" "}
              {nw.age_hours?.toFixed(1)}h ago. Clustered by story, not by
              outlet — three papers running the same wire copy is one piece
              of evidence, not three.
            </p>
            {(nw.feed_caveats ?? []).length > 0 && (
              <ul className="reasons">
                {nw.feed_caveats!.slice(0, 3).map((c) => <li key={c}>{c}</li>)}
              </ul>
            )}
          </>
        )}
      </div>
    </>
  );
}
