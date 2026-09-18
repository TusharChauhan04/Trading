import { useEffect, useState } from "react";
import {
  inr, research, type JournalEntry, type JournalIndex,
} from "../api";

/**
 * What the desk decided, and what came of it.
 *
 * NO TRADE days are shown as prominently as trading days. A journal that
 * only displays trades cannot answer the question a cautious system most
 * needs to answer - how often did it correctly stay out - and the
 * surviving sample becomes every day the desk chose to act.
 *
 * `unrecorded_outcomes` is deliberately a warning banner rather than a
 * quiet footnote: "we have no losing trades" and "nobody wrote down how
 * the trades went" look identical from a results table, and the
 * flattering reading is the one people take.
 */
export default function JournalView() {
  const [index, setIndex] = useState<JournalIndex | null>(null);
  const [open, setOpen] = useState<JournalEntry | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    research.journal().then(setIndex).catch((e) => setErr(String(e)));
  }, []);

  const show = (day: string) =>
    research.journalDay(day).then(setOpen).catch((e) => setErr(String(e)));

  if (err) return <p className="err">{err}</p>;
  if (!index) return <p className="hint">Loading the journal…</p>;

  if (index.total === 0) {
    return (
      <>
        <p className="lede">
          Every plan the desk produces is recorded here — including the days
          it decided not to trade.
        </p>
        <div className="banner">
          <h3>Nothing recorded yet</h3>
          <p>
            The journal fills itself: the first plan of each day is written
            automatically and never overwritten afterwards.
          </p>
        </div>
      </>
    );
  }

  return (
    <>
      <p className="lede">
        {index.total} day{index.total === 1 ? "" : "s"} recorded,{" "}
        {index.no_trade_days} of them NO TRADE. A recorded decision is never
        overwritten — a correction is filed alongside it with a reason.
      </p>

      {index.unrecorded_outcomes.length > 0 && (
        <div className="banner crit">
          <h3>
            {index.unrecorded_outcomes.length} day
            {index.unrecorded_outcomes.length === 1 ? "" : "s"} have trades
            with no outcome recorded
          </h3>
          <p>
            Until these are filled in, nothing can be measured. "No losing
            trades" and "nobody wrote down how they went" look the same from
            a results table.
          </p>
          <p className="mono" style={{ fontSize: 12.5 }}>
            {index.unrecorded_outcomes.join(", ")}
          </p>
        </div>
      )}

      <div className="tw" style={{ marginTop: 14 }}>
        <table>
          <thead>
            <tr>
              <th>Date</th><th>Regime</th><th>Decision</th><th>Capital</th>
              <th>Outcomes</th><th>Versions</th><th></th>
            </tr>
          </thead>
          <tbody>
            {index.days.map((d) => (
              <tr key={d.as_of}>
                <td><strong>{d.as_of}</strong></td>
                <td><span className="pill mute">{d.regime ?? "?"}</span></td>
                <td>
                  {d.trades
                    ? <>{d.trades} trade{d.trades === 1 ? "" : "s"}{" "}
                        <span className="mono" style={{ fontSize: 12 }}>
                          {(d.symbols ?? []).join(" ")}
                        </span></>
                    : <span className="pill crit">NO TRADE</span>}
                </td>
                <td className="num">{d.capital ? inr(d.capital) : "—"}</td>
                <td className="num">
                  {d.trades
                    ? `${d.outcomes_recorded ?? 0} / ${d.trades}`
                    : "—"}
                </td>
                <td className="num">{d.versions ?? 1}</td>
                <td>
                  <button onClick={() => show(d.as_of)}>Open</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {open && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 style={{ fontSize: 17 }}>
            {open.as_of} — decision {open.digest}
          </h3>
          <p className="hint" style={{ marginTop: 2 }}>
            Recorded {new Date(open.recorded_at).toLocaleString("en-IN")} ·
            regime {open.regime} · sized against {inr(open.capital)}
            {open.amends && <> · amends {open.amends}</>}
          </p>

          {open.is_no_trade ? (
            <div className="banner crit" style={{ marginTop: 10 }}>
              <h3>NO TRADE</h3>
              <p>{open.no_trade_reason ?? "no reason recorded"}</p>
            </div>
          ) : (
            <div className="tw" style={{ marginTop: 10 }}>
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th><th>Entry</th><th>Stop</th><th>Target</th>
                    <th>Qty</th><th>R:R</th><th>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {open.trades.map((t) => {
                    const o = open.outcomes.find((x) => x.symbol === t.symbol);
                    return (
                      <tr key={t.symbol}>
                        <td><strong>{t.symbol}</strong></td>
                        <td className="num">{t.entry}</td>
                        <td className="num">{t.stop}</td>
                        <td className="num">{t.target}</td>
                        <td className="num">{t.qty}</td>
                        <td className="num">
                          {t.reward_to_risk
                            ? `1:${t.reward_to_risk.toFixed(1)}` : "—"}
                        </td>
                        <td>
                          {!o ? <span className="pill crit">not recorded</span>
                            : o.status !== "closed"
                              ? <span className="pill mute">open</span>
                              : <span className={"pill " +
                                  ((o.r_multiple ?? 0) > 0 ? "ok" : "crit")}>
                                  {o.exit_reason}{" "}
                                  {o.r_multiple?.toFixed(2)}R
                                </span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div style={{ marginTop: 12 }}>
            <div className="k">Funnel that day</div>
            <div className="mono" style={{ fontSize: 12.5 }}>
              {Object.entries(open.funnel)
                .map(([k, v]) => `${k} ${v}`).join("  ·  ")}
            </div>
          </div>

          {open.caveats.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div className="k">
                What the desk did NOT know that day ({open.caveats.length})
              </div>
              <ul className="reasons">
                {open.caveats.map((c) => <li key={c}>{c}</li>)}
              </ul>
            </div>
          )}

          {open.versions.length > 1 && (
            <div style={{ marginTop: 12 }}>
              <div className="k">Amendment chain</div>
              <ul className="reasons">
                {open.versions.map((v) => (
                  <li key={v.digest} className="mono" style={{ fontSize: 12.5 }}>
                    {v.digest} · {new Date(v.recorded_at).toLocaleString("en-IN")}
                    {v.note && <> — {v.note}</>}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </>
  );
}
