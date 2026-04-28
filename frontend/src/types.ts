export type Verdict =
  | "launch_aggressively"
  | "launch"
  | "launch_with_changes"
  | "test_before_launch"
  | "reposition"
  | "do_not_launch_yet";

export type Confidence = "high" | "medium-high" | "medium" | "medium-low" | "low";

export type Stance = "for" | "against" | "neutral";

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

export interface Agent {
  id: string;
  segment: string;
  profile: string;
  stance: Stance;
  score: number;
  reason: string;
  top_objection: string;
}

export interface AgentPanel {
  agent_count: number;
  seed: string;
  rounds: AgentRound[];
  agents: Agent[];
  top_drivers: string[];
  top_objections: string[];
  most_receptive_segment: string;
  consensus: string;
}

export interface SimulationRecord {
  id: string;
  payload: ProductPayload;
  forecast: ForecastResponse;
  agent_panel?: AgentPanel;
  created_at: number;
  source: "live" | "cached_demo";
  demo_key?: string;
}
