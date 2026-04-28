import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { getSim } from "../lib/simulationStore";
import type { SimulationRecord } from "../types";

const PHASES = [
  { id: "intel", label: "Comparable retrieval", duration: 700 },
  { id: "personas", label: "Buyer personas spawning", duration: 700 },
  { id: "round1", label: "Round 1 — First Reaction", duration: 800 },
  { id: "round2", label: "Round 2 — Comparable Comparison", duration: 800 },
  { id: "round3", label: "Round 3 — Consensus", duration: 800 },
  { id: "report", label: "Compiling Market Report", duration: 600 },
];

export function SimulationView() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [sim, setSim] = useState<SimulationRecord | null>(null);
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!id) return;
    const rec = getSim(id);
    if (!rec) { navigate("/dtc-v3"); return; }
    setSim(rec);
  }, [id, navigate]);

  useEffect(() => {
    if (!sim) return;
    if (phaseIdx >= PHASES.length) { setDone(true); return; }
    const t = setTimeout(() => setPhaseIdx((i) => i + 1), PHASES[phaseIdx].duration);
    return () => clearTimeout(t);
  }, [sim, phaseIdx]);

  if (!sim) return null;
  const f = sim.forecast;
  const panel = sim.agent_panel;

  const stepStatus = (i: number): "complete" | "active" | "pending" => {
    if (phaseIdx > i) return "complete";
    if (phaseIdx === i) return "active";
    return "pending";
  };

  return (
    <div className="dtc-simulation">
      <div className="workbench-header">
        <div className="workbench-meta">
          <div className="workbench-product">
            {!done && <span className="live-dot" />}
            <span>{sim.payload.product_name || "Market Simulation"}</span>
            <span className="product-price mono">${sim.payload.price}</span>
          </div>
          <div className={`workbench-status ${done ? "status-complete" : "status-live"}`}>
            {done ? "Forecast + panel complete" : "Running…"}
          </div>
        </div>

        <div className="steps-row">
          {[
            { id: "intel", label: "Comparable Intel" },
            { id: "personas", label: "Buyer Personas" },
            { id: "debate", label: "Market Debate" },
            { id: "report", label: "Market Report" },
          ].map((s, i) => {
            const status = stepStatus(i < 2 ? i : i === 2 ? 2 : 5);
            return (
              <div key={s.id} className={`step ${status}`}>
                <div className="step-dot">
                  {status === "complete" ? "✓" : status === "active" ? <span className="spinner-sm" /> : "·"}
                </div>
                <div className="step-label">{s.label}</div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="workbench-grid">
        <div className="agent-panel">
          {phaseIdx >= 1 && (
            <div className="market-dist fade-up">
              <div className="dist-label">Forecast Signal</div>
              <div className="dist-bar-wrap">
                <div className="dist-bar-segment seg-buy" style={{ width: `${f.trial_rate.percentage}%` }} />
                <div className="dist-bar-segment seg-considering" style={{ width: `${Math.max(0, 100 - f.trial_rate.percentage - 30)}%` }} />
                <div className="dist-bar-segment seg-wont" style={{ width: "30%" }} />
              </div>
              <div className="dist-legend">
                <span className="legend-item buy">▬ {f.trial_rate.percentage.toFixed(1)}% predicted trial</span>
                <span className="legend-item considering">▬ range {(f.trial_rate.low * 100).toFixed(1)}–{(f.trial_rate.high * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}

          {[1, 2, 3].map((roundN) => {
            const phaseForRound = 1 + roundN;
            if (phaseIdx < phaseForRound) return null;
            const rd = panel?.rounds.find((r) => r.round === roundN);
            if (!rd) return (
              <div key={roundN} className="round-block fade-up">
                <div className="round-header">
                  <span className="round-num">ROUND {roundN}</span>
                  <span className="round-name">DISCUSSION UNAVAILABLE</span>
                </div>
                <div className="round-summary">
                  Buyer-panel discussion did not complete for this run.
                </div>
              </div>
            );
            return (
              <div key={roundN} className="round-block fade-up">
                <div className="round-header">
                  <span className="round-num">ROUND {roundN}</span>
                  <span className="round-name">{rd.title.toUpperCase()}</span>
                </div>
                <div className="round-meta">
                  <span className="round-meta-item" style={{ color: "var(--for)" }}>↑ {rd.for_count} for</span>
                  <span className="round-meta-item" style={{ color: "#f59e0b" }}>— {rd.neutral_count} neutral</span>
                  <span className="round-meta-item" style={{ color: "var(--against)" }}>↓ {rd.against_count} against</span>
                  <span className="round-meta-item">avg score: {rd.avg_score.toFixed(2)}</span>
                </div>
                <div className="round-summary">{rd.summary}</div>
              </div>
            );
          })}

          {!done && phaseIdx >= 1 && phaseIdx < PHASES.length && (
            <div className="round-summary mono" style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span className="spinner-sm" />
              {PHASES[phaseIdx]?.label}…
            </div>
          )}
        </div>

        <div className="stats-panel">
          <div className="stat-block">
            <div className="stat-label">Trial Rate</div>
            <div className="stat-value">{f.trial_rate.percentage.toFixed(1)}%</div>
            <div className="stat-sub">range {(f.trial_rate.low * 100).toFixed(1)} – {(f.trial_rate.high * 100).toFixed(1)}%</div>
          </div>

          <div className="divider" />

          <div className="stat-block">
            <div className="stat-label">Confidence</div>
            <div className="stat-value" style={{ fontSize: 24 }}>{f.confidence}</div>
            <div className="stat-sub">coverage: {f.diagnostics.coverage_tier}</div>
          </div>

          <div className="divider" />

          <div className="stat-block">
            <div className="stat-label">Buyer Panel</div>
            <div className="stat-value" style={{ fontSize: 32 }}>{sim.agent_count}</div>
            <div className="stat-sub">
              {sim.panel_source === "live" && "live AI personas"}
              {sim.panel_source === "cached_fallback" && "cached fallback"}
              {sim.panel_source === "unavailable" && "panel unavailable"}
            </div>
          </div>

          <div className="divider" />

          <div className="stat-block">
            <div className="stat-label">Comparables</div>
            <div className="stat-value" style={{ fontSize: 32 }}>{f.anchored_on.length}</div>
            <div className="stat-sub">anchored brands</div>
          </div>

          {done && (
            <>
              <div className="divider" />
              <div className="report-ready">
                <div className="report-ready-label">
                  <span style={{ color: "var(--for)" }}>✓</span>
                  Market analysis complete
                </div>
                <Link to={`/dtc-v3/report/${id}`} className="btn btn-primary report-btn">
                  View Market Report →
                </Link>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
