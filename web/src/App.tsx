import { useEffect, useState } from "react";
import { api } from "./api";
import DailyPlanView from "./components/DailyPlanView";
import DataHealthView from "./components/DataHealthView";
import FleetPanel from "./components/FleetPanel";
import JournalView from "./components/JournalView";
import NewsView from "./components/NewsView";
import RegimeView from "./components/RegimeView";
import RiskSizer from "./components/RiskSizer";
import StrategyCatalog from "./components/StrategyCatalog";
import SymbolLookup from "./components/SymbolLookup";

const TABS = [
  { id: "plan", label: "Today's plan", el: <DailyPlanView /> },
  { id: "regime", label: "Regime", el: <RegimeView /> },
  { id: "journal", label: "Journal", el: <JournalView /> },
  { id: "news", label: "News", el: <NewsView /> },
  { id: "health", label: "Data health", el: <DataHealthView /> },
  { id: "risk", label: "Risk sizer", el: <RiskSizer /> },
  { id: "strategies", label: "Strategies", el: <StrategyCatalog /> },
  { id: "fleet", label: "Fleet", el: <FleetPanel /> },
  { id: "symbols", label: "Symbols", el: <SymbolLookup /> },
] as const;

export default function App() {
  const [tab, setTab] = useState<string>("plan");
  const [up, setUp] = useState<boolean | null>(null);
  const [version, setVersion] = useState("");

  useEffect(() => {
    api.health()
      .then((h) => { setUp(true); setVersion(h.version); })
      .catch(() => setUp(false));
  }, []);

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <h1>The India Desk</h1>
          <span className="sub">NSE · BSE · research and risk only</span>
          <span className="health">
            <span className={"dot " + (up ? "up" : "down")} />
            {up === null ? "checking…" : up ? `api ${version}` : "api offline"}
          </span>
        </div>

        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {up === false ? (
          <div className="banner crit">
            <h3>Backend is not running</h3>
            <p>
              Start it from the project root:{" "}
              <span className="mono">
                C:/Users/TUSHAR/AppData/Local/Python/pythoncore-3.14-64/python.exe -m uvicorn desk.api.main:app --reload --port 8000
              </span>
            </p>
          </div>
        ) : (
          TABS.find((t) => t.id === tab)?.el
        )}
      </main>
    </div>
  );
}
