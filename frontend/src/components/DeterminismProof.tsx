import { useState } from "react";
import { runForecast } from "../api/assembly_v3";
import type { ProductPayload } from "../types";

interface Run { i: number; pct: number; verdict: string; ms: number; }

export function DeterminismProof({ payload }: { payload: ProductPayload }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run5x = async () => {
    setRunning(true); setError(null); setRuns([]);
    try {
      for (let i = 1; i <= 5; i++) {
        const t0 = performance.now();
        const r = await runForecast(payload);
        const ms = Math.round(performance.now() - t0);
        setRuns((prev) => [...prev, { i, pct: r.trial_rate.percentage, verdict: r.verdict, ms }]);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const allSame = runs.length === 5 && new Set(runs.map((r) => r.pct)).size === 1;

  return (
    <div className="report-card determinism-card fade-up">
      <div className="report-card-eyebrow">Determinism proof — Liquid IV × 5</div>
      <p className="muted">
        Same input. Five consecutive live calls to the v3-lite forecast endpoint.
        Same number every time. No variance, no drift.
      </p>
      <button onClick={run5x} disabled={running} className="btn btn-primary">
        {running ? "Running…" : "▶ Run Liquid IV × 5"}
      </button>
      {error && <div className="error-msg" style={{ marginTop: 12 }}>⚠ {error}</div>}
      {runs.length > 0 && (
        <table className="run-table">
          <thead><tr><th>Run</th><th>Trial rate</th><th>Verdict</th><th>Latency</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.i}>
                <td>#{r.i}</td>
                <td>{r.pct.toFixed(1)}%</td>
                <td>{r.verdict}</td>
                <td>{r.ms}ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {runs.length === 5 && (
        <div className={`determinism-result ${allSame ? "ok" : "fail"}`}>
          {allSame
            ? `✓ DETERMINISTIC — all 5 runs returned ${runs[0].pct.toFixed(1)}%`
            : `✕ NON-DETERMINISTIC — got ${runs.map((r) => r.pct).join(", ")}`}
        </div>
      )}
    </div>
  );
}
