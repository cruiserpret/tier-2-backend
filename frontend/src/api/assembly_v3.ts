import type { ForecastResponse, ProductPayload, CachedDemo, AgentPanel } from "../types";

const API_BASE = (import.meta.env.VITE_TIER2_API_BASE_URL || "").replace(/\/$/, "");

const DISCUSS_TIMEOUT_MS = 90000;  // 90s covers async-batched LLM (~30-60s) + cache hits (~0.4s)

export async function runForecast(payload: ProductPayload): Promise<ForecastResponse> {
  if (!API_BASE) throw new Error("VITE_TIER2_API_BASE_URL is not set");
  const res = await fetch(`${API_BASE}/api/dtc_v3/forecast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Forecast failed: ${res.status} ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function runDiscussion(
  product: ProductPayload,
  forecast: ForecastResponse,
  agent_count: number = 20,
  mode: "template" | "llm" = "llm",
): Promise<AgentPanel> {
  if (!API_BASE) throw new Error("VITE_TIER2_API_BASE_URL is not set");

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DISCUSS_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}/api/dtc_v3/discuss`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product, forecast, agent_count, mode }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Discussion failed: ${res.status} ${text.slice(0, 200)}`);
    }
    const data = await res.json();
    if (!data?.agent_panel) {
      throw new Error("Discussion response missing agent_panel");
    }
    return data.agent_panel;
  } catch (e: any) {
    if (e.name === "AbortError") {
      throw new Error(`Discussion timed out after ${Math.round(DISCUSS_TIMEOUT_MS/1000)}s`);
    }
    throw e;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function loadCachedDemo(key: string): Promise<CachedDemo> {
  const res = await fetch(`/demos/${key}_v3_demo.json`);
  if (!res.ok) throw new Error(`Cache not found for ${key}`);
  const data = await res.json();
  if (!data?.forecast || !data?.agent_panel || !data?.product) {
    throw new Error(`Cache for ${key} is missing required fields`);
  }
  return data as CachedDemo;
}
