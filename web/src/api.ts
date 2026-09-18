const BASE = import.meta.env.VITE_API ?? "http://localhost:8000";

export type Stance =
  | "Buy" | "Overweight" | "Hold" | "Underweight" | "Sell" | "REVIEW";

export interface FleetEntry {
  name: string;
  kind: string;
  version: string;
  capabilities: string[];
  asset_classes: string[];
  horizons: string[];
  licence: string;
  installed: boolean;
  verified: boolean;
  wired: boolean;
  india_ready: boolean;
  note: string;
  blockers: string[];
  status: "wired" | "verified" | "installed" | "absent";
}

export interface RiskConfig {
  capital: number;
  risk_pct: number;
  max_position_pct: number;
  max_sector_pct: number;
  max_open_risk_pct: number;
  max_correlated_pct: number;
  max_daily_loss_pct: number;
  max_open_positions: number;
  min_risk_reward: number;
  max_adv_participation_pct: number;
  max_stop_distance_pct: number;
  lot_size: number;
  max_slippage_pct: number;
  max_slippage_share_of_stop_pct: number;
  max_instrument_atr_pct: number;
  vol_reference_atr_pct: number | null;
  risk_off_exposure_pct: number;
}

export interface PositionIn {
  symbol: string;
  qty: number;
  entry: number;
  stop: number;
  sector?: string;
  corr_group?: string | null;
  beta?: number;
}

export interface Sizing {
  approved: boolean;
  symbol: string;
  qty: number;
  entry: number | null;
  stop: number | null;
  target: number | null;
  capital_at_risk: number;
  position_value: number;
  risk_reward: number | null;
  reasons: string[];
  notes: string[];
  /** Gates that could not run for want of an input. An approval with a long
   *  list here is weaker than one with an empty list, and the screen has to
   *  say so — otherwise "not checked" reads exactly like "checked and passed". */
  checks_skipped: string[];
}

export interface PlanTrade {
  symbol: string;
  stance: Stance;
  confidence: number;
  entry: number | null;
  stop: number | null;
  target: number | null;
  qty: number;
  capital_at_risk: number;
  supporting: string[];
  dissenting: string[];
  rationale: string;
}

export interface DailyPlan {
  as_of: string;
  regime: string;
  regime_note: string;
  regime_detail: string[];
  universe_scanned: number;
  survived_stage1: number;
  survived_stage2: number;
  analysed: number;
  trades: PlanTrade[];
  watchlist: PlanTrade[];
  avoid: PlanTrade[];
  no_trade_reason: string | null;
  market_risks: string[];
  warnings: string[];
}

export type Maturity =
  | "draft" | "audited" | "revalidated" | "walk_forward" | "paper" | "live";

export interface StrategySpec {
  key: string;
  name: string;
  family: string;
  horizon: string;
  asset_classes: string[];
  thesis: string;
  params: Record<string, string>;
  data_needed: string[];
  favourable_regimes: string[];
  hostile_regimes: string[];
  maturity: Maturity;
  defects: string[];
  notes: string;
  trusted: boolean;
}

export interface SymbolInfo {
  canonical: string;
  base: string;
  exchange: string;
  dialects: Record<string, string>;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  health: () => get<{ status: string; version: string }>("/health"),
  fleet: () => get<FleetEntry[]>("/fleet"),
  strategies: () => get<StrategySpec[]>("/strategies"),
  eligible: (regime: string, trustedOnly: boolean) =>
    get<string[]>(`/strategies/eligible/${regime}?trusted_only=${trustedOnly}`),
  symbol: (raw: string) => get<SymbolInfo>(`/symbols/${encodeURIComponent(raw)}`),
  defaultConfig: () => get<RiskConfig>("/risk/config/default"),
  // `capital` is what the risk engine sizes every position against, so it
  // is a real input to the answer rather than a display preference - a
  // plan computed for one lakh is a different plan, not the same one
  // shown differently. Passed through on every call.
  plan: (capital?: number) =>
    get<DailyPlan>(
      capital && capital > 0
        ? `/plan/today?capital=${encodeURIComponent(capital)}`
        : "/plan/today",
    ),
  size: (body: {
    symbol: string; entry: number; stop: number; target?: number | null;
    sector?: string; corr_group?: string | null;
    adv_shares?: number | null; atr_pct?: number | null;
    expected_slippage_pct?: number | null; market_risk_off?: boolean;
    tradeable?: boolean; data_as_of?: string | null;
    config: RiskConfig;
    open_positions?: PositionIn[];
    realised_pnl_today?: number;
  }) => post<Sizing>("/risk/size", body),
};

export const inr = (n: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency", currency: "INR", maximumFractionDigits: 0,
  }).format(n);

// --- regime, journal, news, data health ---------------------------------
//
// These mirror the endpoints added when the corresponding subsystems were
// wired. Every one of them carries its own "what could not be measured"
// field, because a screen that shows only the answer teaches the reader to
// trust it more than the data supports.

export interface RegimeView {
  as_of: string;
  label: string;
  measured: boolean;
  risk_off: boolean;
  trend: string;
  volatility: string;
  breadth: string;
  risk_appetite: string;
  leading_sectors: string[];
  lagging_sectors: string[];
  max_concurrent_positions_hint: number | null;
  explain: string[];
  /** Dimension -> what it was computed from. The reason this is arguable. */
  sources: Record<string, string>;
  universe: number;
}

export interface JournalDay {
  as_of: string;
  regime?: string;
  trades?: number;
  symbols?: string[];
  no_trade_reason?: string | null;
  digest?: string;
  versions?: number;
  capital?: number;
  outcomes_recorded?: number;
  unreadable?: boolean;
}

export interface JournalIndex {
  days: JournalDay[];
  total: number;
  no_trade_days: number;
  /** Days with trades whose outcomes were never written down. */
  unrecorded_outcomes: string[];
  open_positions: { decision_date: string; symbol: string }[];
}

export interface JournalEntry {
  as_of: string;
  recorded_at: string;
  regime: string;
  capital: number;
  digest: string;
  amends: string | null;
  is_no_trade: boolean;
  no_trade_reason: string | null;
  funnel: Record<string, number>;
  trades: {
    symbol: string; stance: string; entry: number | null; stop: number | null;
    target: number | null; qty: number; capital_at_risk: number;
    reward_to_risk: number | null; rationale: string;
  }[];
  caveats: string[];
  coverage_note: string;
  versions: { digest: string; recorded_at: string; amends: string | null; note: string }[];
  outcomes: {
    symbol: string; status: string; exit_price: number | null;
    exit_date: string | null; exit_reason: string | null;
    r_multiple: number | null; pnl: number | null;
  }[];
}

export interface NewsCluster {
  title: string;
  published_at: string;
  tier: number;
  weight: number;
  /** Carried by more than one outlet. Still ONE vote - see the module. */
  duplicated: boolean;
  sources: string[];
  session_phase: string;
  url: string;
}

export interface NewsView {
  available: boolean;
  fetched_at?: string;
  age_hours?: number;
  stale?: boolean;
  caveat: string | null;
  feed_caveats?: string[];
  total?: number;
  clusters: NewsCluster[];
}

export interface CrossCheck {
  as_of: string;
  checked_at: string;
  tolerance_pct: number;
  coverage: Record<string, number | boolean>;
  disagreements: {
    symbol: string; nse_close: number; bse_close: number;
    diff_pct: number; bse_turnover: number;
  }[];
}

export interface FundamentalsView {
  requested_as_of: string;
  table_as_of: string;
  staleness_days: number;
  stale: boolean;
  caveat: string | null;
  coverage: Record<string, number>;
  total: number;
  rows: Record<string, unknown>[];
}

export const research = {
  regime: (day?: string) =>
    get<RegimeView>(day ? `/regime?day=${day}` : "/regime"),
  journal: () => get<JournalIndex>("/journal"),
  journalDay: (day: string) => get<JournalEntry>(`/journal/${day}`),
  news: () => get<NewsView>("/news"),
  crosscheck: (day: string) => get<CrossCheck>(`/crosscheck/${day}`),
  fundamentals: (day?: string) =>
    get<FundamentalsView>(day ? `/fundamentals?day=${day}` : "/fundamentals"),
};
