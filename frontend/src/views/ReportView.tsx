import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getSim } from "../lib/simulationStore";
import type { SimulationRecord } from "../types";
import { ResultCard } from "../components/ResultCard";
import { Counterfactuals } from "../components/Counterfactuals";
import { DeterminismProof } from "../components/DeterminismProof";

export function ReportView() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [sim, setSim] = useState<SimulationRecord | null>(null);

  useEffect(() => {
    if (!id) return;
    const rec = getSim(id);
    if (!rec) { navigate("/dtc-v3"); return; }
    setSim(rec);
  }, [id, navigate]);

  if (!sim) return null;
  const f = sim.forecast;
  const isLiquidIV = sim.demo_key === "liquid_iv";

  return (
    <div className="dtc-report">
      <header className="masthead fade-up">
        <div className="masthead-eyebrow">
          <span className="live-dot" style={{ background: "var(--accent)" }} />
          Market God's Eye View
        </div>
        <h1 className="masthead-title">{sim.payload.product_name}</h1>
        <p className="masthead-sub">
          Comparable-anchored 12-month trial-rate forecast plus {sim.agent_count}-agent
          AI buyer panel discussion. Anchored on {f.anchored_on.length} real DTC brands and
          their measured adoption rates.
        </p>
        <div className="masthead-meta">
          {f.simulation_id} &nbsp;·&nbsp; {f.version} &nbsp;·&nbsp; coverage: {f.diagnostics.coverage_tier} &nbsp;·&nbsp; source: {sim.source === "cached_demo" ? "cached demo" : "live forecast"}
          &nbsp;·&nbsp; panel: {sim.panel_source.replace("_", " ")}
        </div>
      </header>

      <div className="report-body">
        <div className="report-main">
          <ResultCard data={f} />
          <Counterfactuals cfs={f.counterfactuals} />
          {sim.agent_panel?.rounds && sim.agent_panel.rounds.length > 0 && (
            <div className="report-rounds">
              <div className="report-card-eyebrow">Buyer Panel Discussion Rounds</div>
              {sim.agent_panel.rounds.map((rd) => (
                <div key={rd.round} className="round-block">
                  <div className="round-header">
                    <span className="round-num">ROUND {rd.round}</span>
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
              ))}
            </div>
          )}
          {isLiquidIV && <DeterminismProof payload={sim.payload} />}
        </div>

        <aside className="report-sidebar">
          <div className="sidebar-card fade-up">
            <div className="report-card-eyebrow">Outcome</div>
            <div style={{ marginBottom: 16 }}>
              <div className="stat-label">Predicted Trial</div>
              <div className="stat-value">{f.trial_rate.percentage.toFixed(1)}%</div>
              <div className="stat-sub">12-month adoption</div>
            </div>
            <div className="divider" />
            <div style={{ marginBottom: 16 }}>
              <div className="stat-label">Verdict</div>
              <div style={{ fontFamily: "var(--display)", fontSize: 22, color: "var(--text)", lineHeight: 1.2, marginTop: 4, letterSpacing: "0.02em" }}>
                {f.verdict.replace(/_/g, " ").toUpperCase()}
              </div>
            </div>
            <div className="divider" />
            <div style={{ marginBottom: 16 }}>
              <div className="stat-label">Confidence</div>
              <div className="stat-value" style={{ fontSize: 22 }}>{f.confidence}</div>
              <div className="stat-sub">coverage: {f.diagnostics.coverage_tier}</div>
            </div>
            <div className="divider" />
            <div>
              <div className="stat-label">Buyer Panel</div>
              <div className="stat-value" style={{ fontSize: 22 }}>{sim.agent_count}</div>
              <div className="stat-sub">{sim.panel_source.replace("_", " ")}</div>
            </div>
          </div>

          <div className="sidebar-card fade-up fade-up-1">
            <div className="report-card-eyebrow">Actions</div>
            <button
              className="btn btn-ghost"
              style={{ width: "100%", justifyContent: "center", marginBottom: 8 }}
              onClick={() => window.print()}
            >↓ Print / Save PDF</button>
            <button
              className="btn btn-primary"
              style={{ width: "100%", justifyContent: "center" }}
              onClick={() => navigate("/dtc-v3")}
            >+ New Simulation</button>
          </div>
        </aside>
      </div>

      <footer className="app-footer">
        Assembly v3-lite · {f.version} · Forecast deterministic, panel grounded in comparable evidence
      </footer>
    </div>
  );
}
