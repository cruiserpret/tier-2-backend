import type { SimulationRecord } from "../types";

const KEY = "assembly_v3_sims";

function readAll(): Record<string, SimulationRecord> {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function writeAll(records: Record<string, SimulationRecord>) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(records));
  } catch {
    // quota or private mode — silent fail
  }
}

export function saveSim(rec: SimulationRecord) {
  const all = readAll();
  all[rec.id] = rec;
  writeAll(all);
}

export function getSim(id: string): SimulationRecord | null {
  const all = readAll();
  return all[id] || null;
}

export function newSimId(): string {
  return `sim_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
