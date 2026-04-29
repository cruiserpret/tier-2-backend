export type Verdict =
  | "launch_aggressively"
  | "launch"
  | "launch_with_changes"
  | "test_before_launch"
  | "reposition"
  | "do_not_launch_yet";

export type Confidence = "high" | "medium-high" | "medium" | "medium-low" | "low";

export type Stance = "for" | "against" | "neutral";

// New 3b verdict labels — what's shown in the UI per agent.
// JSON value is "WON'T BUY" with apostrophe, kept exact for backend match.
export type AgentVerdict = "BUY" | "CONSIDERING" | "WON'T BUY";

export interface Anchor {
  brand: string;
  trial_rate: number;
  confidence_grade: "A" | "B" | "C" | "D";
  match_reason: string;
}

export interface Counterfactual {
  label: string;
  description: string;
  delta_logit: number;
  new_prediction_pct: number;
  direction: "improves" | "worsens" | "neutral";
}

export interface ForecastResponse {
  simulation_id: string;
  version: string;
  verdict: Verdict;
  headline: string;
  trial_rate: { median: number; low: number; high: number; percentage: number };
  confidence: Confidence;
  confidence_reasons: string[];
  anchored_on: Anchor[];
  downweighted_brands: { brand: string; trial_rate: number; penalty_reasons: string; combined_penalty: number }[];
  why_might_be_wrong: string[];
  counterfactuals: Counterfactual[];
  top_drivers: string[];
  top_objections: string[];
  most_receptive_segment: string;
  diagnostics: { rag_prior: number; adjustment_applied: number; coverage_tier: string };

  // Phase 1 — Evidence Panel + Confidence Ledger (added Apr 30)
  // Optional because cached demos pre-Phase-1 won't have these.
  // Frontend falls back to anchored_on when absent (Call 5=A).
  evidence_buckets?: EvidenceBuckets;
  confidence_ledger?: LedgerEntry[];
}

// Phase 1 — Evidence types (added Apr 30 for P1.6)

export type AnchorStrength = "direct" | "adjacent" | "weak";

export type EvidenceBucketKind =
  | "forecast_anchor"
  | "candidate_comparable"
  | "fallback_neighbor"
  | "exploratory_comparable";

export interface EvidenceItem {
  brand: string;
  similarity: number;
  trial_rate: number;
  bucket: EvidenceBucketKind;
  same_category_match: boolean;
  same_subtype_match: boolean;
  anchor_strength: AnchorStrength | null;
  used_in_forecast: boolean;
  display_warning: string | null;
}

export interface EvidenceBuckets {
  forecast_anchors: EvidenceItem[];
  candidate_comparables: EvidenceItem[];
  fallback_neighbors: EvidenceItem[];
  exploratory_comparables: EvidenceItem[];
}

export type LedgerKind = "positive" | "negative" | "neutral";

export interface LedgerEntry {
  kind: LedgerKind;
  signal: string;
  text: string;
}

export interface ProductPayload {
  name: string;
  product_name: string;
  description: string;
  price: number;
  category: string;
  demographic: string;
  competitors: { name: string }[];
  market_tier_override?: string;
  distribution_hint?: string;
  exclude_brand?: string;
  num_agents?: number;
}

export interface DemoProduct {
  key: string;
  label: string;
  story: string;
  payload: ProductPayload;
}

export interface FormState {
  product_name: string;
  description: string;
  price: string;
  category: string;
  age_ranges: string[];
  gender: string;
  income_ranges: string[];
  interests: string[];
  customer_context: string;
  competitors: { name: string }[];
  market_tier_override: string;
  distribution_hint: string;
  exclude_brand: string;
  num_agents: number;
}

export interface AgentRound {
  round: number;
  title: string;
  summary: string;
  for_count: number;
  neutral_count: number;
  against_count: number;
  avg_score: number;
}

// ─────────────────────────────────────────────────────────────────────
// 3b — Per-agent enrichment
// ─────────────────────────────────────────────────────────────────────

export interface RoundResponse {
  round: 1 | 2 | 3;
  title: "First Impression" | "Competitor Comparison" | "Final Verdict" | string;
  response: string;
}

export interface AgentJourney {
  initial_verdict: AgentVerdict;
  final_verdict: AgentVerdict;
  shifted: boolean;
  shift_reason: string;
  key_moment: string;
  key_quote: string;
}

export interface Agent {
  // Backward-compat fields (legacy frontend continues to read these)
  id: string;
  segment: string;
  profile: string;
  stance: Stance;
  score: number;       // 0-1 legacy scale
  reason: string;
  top_objection: string;

  // 3b additive fields — present on every agent from /discuss
  name: string;
  age: number;
  profession: string;
  verdict: AgentVerdict;
  initial_score_10: number;
  current_score_10: number;
  score_10: number;
  initial_stance: Stance;
  current_stance: Stance;
  is_hardcore: boolean;
  shifted: boolean;
  key_moment: string;
  what_would_change_mind: string;
  round_responses: RoundResponse[];
  journey: AgentJourney;
}

// ─────────────────────────────────────────────────────────────────────
// 3c — Top-level panel additions
// ─────────────────────────────────────────────────────────────────────

export interface IntentDistribution {
  buy: number;          // 0..1
  considering: number;  // 0..1
  resistant: number;    // 0..1
}

export interface BuyerJourneySummary {
  agent_id: string;
  name: string;
  segment: string;
  initial_verdict: AgentVerdict;
  final_verdict: AgentVerdict;
  shifted: boolean;
  shift_reason: string;
  key_quote: string;
}

export interface RepresentativeQuote {
  verdict: AgentVerdict;
  agent_id: string;
  name: string;
  segment: string;
  quote: string;
}

export interface ComparablePriceRange {
  user_price: number | null;
  min: number | null;
  max: number | null;
  anchor_brands: string[];
}

export interface RiskFactorsV3 {
  summary: string;
  detail: string;
  holdout_agents: string[];
}

export interface AgentPanel {
  agent_count: number;
  seed: string;
  mode: "template" | "llm";
  rounds: AgentRound[];
  agents: Agent[];

  // Existing top-level fields
  top_drivers: string[];
  top_objections: string[];
  most_receptive_segment: string;
  winning_message: string;
  risk_factors: string[];     // legacy string-list (kept for backward compat)
  consensus: string;
  coverage_warning: string;

  // 3c additive top-level fields
  intent_distribution?: IntentDistribution;
  buyer_journeys?: BuyerJourneySummary[];
  representative_quotes?: RepresentativeQuote[];
  hardest_to_convert_segment?: string;
  comparable_price_range?: ComparablePriceRange;
  actionable_insight?: string;
  risk_factors_v3?: RiskFactorsV3;
}

export interface CachedDemo {
  product: ProductPayload;
  forecast: ForecastResponse;
  agent_panel: AgentPanel;
  cached_at: string;
  version: string;
}

export type PanelSource = "live" | "cached_fallback" | "unavailable";

export interface SimulationRecord {
  id: string;
  payload: ProductPayload;
  forecast: ForecastResponse;
  agent_panel: AgentPanel | null;
  panel_source: PanelSource;
  panel_error: string | null;
  agent_count: number;
  created_at: number;
  source: "live" | "cached_demo";
  demo_key?: string;
}

// ─────────────────────────────────────────────────────────────────────
// UI helpers — display labels + colors
// ─────────────────────────────────────────────────────────────────────

export const VERDICT_LABEL: Record<AgentVerdict, string> = {
  "BUY": "BUY",
  "CONSIDERING": "CONSIDERING",
  "WON'T BUY": "WON'T BUY",
};

export const VERDICT_COLOR: Record<AgentVerdict, string> = {
  "BUY": "#7CFC9C",         // green
  "CONSIDERING": "#F0C24A", // amber
  "WON'T BUY": "#FF6E80",   // red/pink
};

// Helper: derive AgentVerdict from a stance, when only stance is available.
export function stanceToVerdict(stance: Stance): AgentVerdict {
  if (stance === "for") return "BUY";
  if (stance === "neutral") return "CONSIDERING";
  return "WON'T BUY";
}
