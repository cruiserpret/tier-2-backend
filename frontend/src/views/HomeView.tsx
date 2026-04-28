import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { DemoProduct, FormState, ProductPayload } from "../types";
import { DEMOS } from "../data/demoProducts";
import { runForecast, loadCachedDemo } from "../api/assembly_v3";
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
  // Best-effort split of demographic string back into rich fields if possible
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

  const handleSubmit = async (payload: ProductPayload) => {
    setLoading(true);
    setError(null);
    try {
      // If active demo and payload matches → load cache for instant nav, then run live in background
      const isDemo = activeDemoKey !== null;
      let forecast;
      let source: "live" | "cached_demo" = "live";

      if (isDemo) {
        try {
          forecast = await loadCachedDemo(activeDemoKey!);
          source = "cached_demo";
        } catch {
          forecast = await runForecast(payload);
          source = "live";
        }
      } else {
        forecast = await runForecast(payload);
      }

      const id = newSimId();
      saveSim({
        id,
        payload,
        forecast,
        created_at: Date.now(),
        source,
        demo_key: activeDemoKey || undefined,
      });
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
