import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { DemoProduct, FormState, ProductPayload, AgentPanel, ForecastResponse, PanelSource } from "../types";
import { DEMOS } from "../data/demoProducts";
import { runForecast, runDiscussion, loadCachedDemo } from "../api/assembly_v3";
import { saveSim, newSimId } from "../lib/simulationStore";
import { Hero } from "../components/Hero";
import { DemoSelector } from "../components/DemoSelector";
import { ProductForm } from "../components/ProductForm";
import { Deliverables } from "../components/Deliverables";

const EMPTY_FORM: FormState = {
  product_name: "",
  description: "",
  price: "",
  category: "",
  age_ranges: [],
  gender: "all",
  income_ranges: [],
  interests: [],
  customer_context: "",
  competitors: [{ name: "" }, { name: "" }, { name: "" }],
  market_tier_override: "",
  distribution_hint: "",
  exclude_brand: "",
  num_agents: 20,
};

function payloadToForm(p: ProductPayload): FormState {
  const competitors = p.competitors.length
    ? [...p.competitors, ...Array(Math.max(0, 3 - p.competitors.length)).fill({ name: "" })]
    : [{ name: "" }, { name: "" }, { name: "" }];
  return {
    product_name: p.product_name,
    description: p.description,
    price: String(p.price),
    category: p.category === "default" ? "" : p.category,
    age_ranges: [],
    gender: "all",
    income_ranges: [],
    interests: [],
    customer_context: p.demographic,
    competitors: competitors.slice(0, 3),
    market_tier_override: p.market_tier_override || "",
    distribution_hint: p.distribution_hint || "",
    exclude_brand: p.exclude_brand || "",
    num_agents: 20,
  };
}

export function HomeView() {
  const navigate = useNavigate();
  const [activeDemoKey, setActiveDemoKey] = useState<string | null>(null);
  const [, setActiveDemoCount] = useState<20 | 50>(20);
  const [formState, setFormState] = useState<FormState>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDemoClick = (d: DemoProduct) => {
    setActiveDemoKey(d.key);
    setFormState(payloadToForm(d.payload));
    setError(null);
  };

  const handleClear = () => {
    setActiveDemoKey(null);
    setFormState(EMPTY_FORM);
    setError(null);
  };

  // Hybrid: try live /discuss with 3s timeout, fall back to cachedPanel if provided
  async function getAgentPanel(
    payload: ProductPayload,
    forecast: ForecastResponse,
    agentCount: 20 | 50,
    cachedPanel: AgentPanel | null,
  ): Promise<{ panel: AgentPanel | null; source: PanelSource; error: string | null }> {
    try {
      const panel = await runDiscussion(payload, forecast, agentCount, "llm");
      return { panel, source: "live", error: null };
    } catch (e: any) {
      if (cachedPanel) {
        return { panel: cachedPanel, source: "cached_fallback", error: e.message };
      }
      return { panel: null, source: "unavailable", error: e.message };
    }
  }

  const handleSubmit = async (payload: ProductPayload) => {
    setLoading(true);
    setError(null);

    try {
      const isDemo = activeDemoKey !== null;
      let forecast: ForecastResponse;
      let cachedPanel: AgentPanel | null = null;
      const agentCount = (formState.num_agents === 50 ? 50 : 20) as 20 | 50;
      const sourceTag: "live" | "cached_demo" = isDemo ? "cached_demo" : "live";

      if (isDemo) {
        try {
          const cached = await loadCachedDemo(activeDemoKey!);
          forecast = cached.forecast;
          cachedPanel = cached.agent_panel;
        } catch {
          forecast = await runForecast(payload);
        }
      } else {
        forecast = await runForecast(payload);
      }

      const { panel, source, error: panelError } = await getAgentPanel(
        payload, forecast, agentCount, cachedPanel,
      );

      const id = newSimId();
      saveSim({
        id,
        payload,
        forecast,
        agent_panel: panel,
        panel_source: source,
        panel_error: panelError,
        agent_count: agentCount,
        created_at: Date.now(),
        source: sourceTag,
        demo_key: activeDemoKey || undefined,
      });
      setActiveDemoCount(agentCount);
      navigate(`/dtc-v3/simulation/${id}`);
    } catch (e: any) {
      setError(e.message || "Forecast failed. Try again.");
      setLoading(false);
    }
  };

  return (
    <div className="dtc-home">
      <Hero />
      <DemoSelector demos={DEMOS} active={activeDemoKey} onSelect={handleDemoClick} onClear={handleClear} />
      <ProductForm initial={formState} onSubmit={handleSubmit} loading={loading} />
      {error && (
        <div style={{ width: "100%", maxWidth: 720, margin: "-40px auto 24px" }}>
          <div className="error-msg">⚠ {error}</div>
        </div>
      )}
      <Deliverables />
    </div>
  );
}
