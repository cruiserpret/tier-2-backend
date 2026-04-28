import type { ForecastResponse, ProductPayload } from "../types";

const API_BASE = (import.meta.env.VITE_TIER2_API_BASE_URL || "").replace(/\/$/, "");

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

export async function loadCachedDemo(key: string): Promise<ForecastResponse> {
  const res = await fetch(`/demos/${key}_v3_demo.json`);
  if (!res.ok) throw new Error(`Cache not found for ${key}`);
  return res.json();
}
