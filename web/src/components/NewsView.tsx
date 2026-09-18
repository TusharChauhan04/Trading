import { useEffect, useState } from "react";
import { research, type NewsView as News } from "../api";

/**
 * Market-wide headlines, one row per STORY rather than per outlet.
 *
 * The `duplicated` badge is the point of the whole module: three papers
 * running the same PTI copy is one piece of evidence appearing three
 * times, and a reader who sees three rows will weigh it three times. So
 * they are collapsed into one row that says how many outlets carried it,
 * and the weight does NOT rise with that count.
 *
 * Nothing here is mapped to a ticker. Mapping a headline to a symbol is
 * entity resolution, and a wrong mapping attaches someone else's news to
 * your trade - worse than no mapping at all.
 */
export default function NewsView() {
  const [n, setN] = useState<News | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    research.news().then(setN).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="err">{err}</p>;
  if (!n) return <p className="hint">Loading headlines…</p>;

  if (!n.available) {
    return (
      <>
        <p className="lede">
          Market context for Stage 3 — deduplicated by story.
        </p>
        <div className="banner crit">
          <h3>No usable news</h3>
          <p>{n.caveat ?? "No snapshot on file."}</p>
          <p className="mono" style={{ fontSize: 12.5 }}>
            python -m desk.research.refresh news
          </p>
        </div>
      </>
    );
  }

  const phase = (p: string) =>
    p === "before_market" ? "pre-open"
      : p === "during_market" ? "in session" : "after hours";

  return (
    <>
      <p className="lede">
        {n.total} stories from {n.clusters.length > 0 ? "the serving feeds" : "—"},
        fetched {n.age_hours?.toFixed(1)}h ago. These are MARKET-WIDE and are
        deliberately not mapped to symbols: a headline attached to the wrong
        ticker is worse than one attached to none.
      </p>

      {(n.feed_caveats ?? []).length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 17 }}>Feed problems</h3>
          <ul className="reasons">
            {n.feed_caveats!.map((c) => <li key={c}>{c}</li>)}
          </ul>
        </div>
      )}

      <div className="tw" style={{ marginTop: 14 }}>
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Story</th><th>Session</th><th>Outlets</th>
              <th>Weight</th>
            </tr>
          </thead>
          <tbody>
            {n.clusters.map((c) => (
              <tr key={c.title + c.published_at}>
                <td className="mono" style={{ whiteSpace: "nowrap", fontSize: 12.5 }}>
                  {new Date(c.published_at).toLocaleTimeString("en-IN", {
                    hour: "2-digit", minute: "2-digit",
                  })}
                </td>
                <td>
                  {c.url
                    ? <a href={c.url} target="_blank" rel="noreferrer">{c.title}</a>
                    : c.title}
                </td>
                <td>
                  <span className="pill mute">{phase(c.session_phase)}</span>
                </td>
                <td>
                  {c.duplicated
                    ? <span className="pill mute" title={c.sources.join(", ")}>
                        {c.sources.length} outlets
                      </span>
                    : <span className="hint">{c.sources[0]}</span>}
                </td>
                <td className="num">{c.weight.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <h3 style={{ fontSize: 17 }}>Why weight does not rise with outlets</h3>
        <p style={{ margin: "6px 0 0", color: "var(--ink-2)" }}>
          Indian outlets republish the same wire copy. A story carried by
          three papers is ONE piece of evidence appearing three times, and a
          scorer that reads that as confirmation is counting the same voice
          repeatedly. Each cluster therefore gets a single vote, weighted by
          its best source tier and never by how many outlets picked it up.
        </p>
      </div>
    </>
  );
}
