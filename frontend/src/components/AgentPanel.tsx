import { useState } from "react";
import type { AgentPanel as AgentPanelType, PanelSource } from "../types";

interface Props {
  panel: AgentPanelType | null;
  source: PanelSource;
  error: string | null;
}

export function AgentPanel({ panel, source, error }: Props) {
  const [showAllAgents, setShowAllAgents] = useState(false);

  if (!panel) {
    return (
      <div className="report-card placeholder-card fade-up">
        <div className="placeholder-eyebrow">AI Buyer Panel</div>
        <p className="placeholder-coming-soon">
          <strong>Discussion unavailable for this run.</strong>{" "}
          {error ? `(${error})` : ""}
        </p>
        <p className="placeholder-now">
          The forecast above remains grounded in the comparable-brand evidence shown.
          The buyer-panel discussion layer can be retried by running the forecast again.
        </p>
      </div>
    );
  }

  const initialCount = 6;
  const displayed = showAllAgents ? panel.agents : panel.agents.slice(0, initialCount);

  const totalFor = panel.rounds[panel.rounds.length - 1]?.for_count ?? 0;
  const totalNeutral = panel.rounds[panel.rounds.length - 1]?.neutral_count ?? 0;
  const totalAgainst = panel.rounds[panel.rounds.length - 1]?.against_count ?? 0;
  const total = panel.agent_count;

  return (
    <div className="report-card fade-up">
      <div className="report-card-eyebrow">
        AI Buyer Panel — {panel.agent_count} synthetic personas
        {source === "cached_fallback" && (
          <span className="panel-source-badge cached" title="Live discussion unavailable, showing cached panel">
            cached fallback
          </span>
        )}
        {source === "live" && (
          <span className="panel-source-badge live">live</span>
        )}
      </div>

      {panel.coverage_warning && (
        <div className="confidence-hint" style={{ marginBottom: 18 }}>
          <strong>Note:</strong> {panel.coverage_warning}
        </div>
      )}

      <div className="panel-summary-grid">
        <div className="panel-stance-bars">
          <div className="panel-stance-row">
            <span className="panel-stance-label tag-for">For</span>
            <div className="panel-stance-bar-wrap">
              <div className="panel-stance-bar seg-buy" style={{ width: `${(totalFor / total) * 100}%` }} />
            </div>
            <span className="panel-stance-num">{totalFor}/{total}</span>
          </div>
          <div className="panel-stance-row">
            <span className="panel-stance-label tag-neutral">Neutral</span>
            <div className="panel-stance-bar-wrap">
              <div className="panel-stance-bar seg-considering" style={{ width: `${(totalNeutral / total) * 100}%` }} />
            </div>
            <span className="panel-stance-num">{totalNeutral}/{total}</span>
          </div>
          <div className="panel-stance-row">
            <span className="panel-stance-label tag-against">Against</span>
            <div className="panel-stance-bar-wrap">
              <div className="panel-stance-bar seg-wont" style={{ width: `${(totalAgainst / total) * 100}%` }} />
            </div>
            <span className="panel-stance-num">{totalAgainst}/{total}</span>
          </div>
        </div>
      </div>

      <div className="panel-consensus">
        <div className="panel-block-label">Panel consensus</div>
        <p>{panel.consensus}</p>
      </div>

      <div className="panel-meta-grid">
        <div className="panel-meta-block">
          <div className="panel-block-label">Most receptive segment</div>
          <p className="panel-meta-text">{panel.most_receptive_segment}</p>
        </div>
        <div className="panel-meta-block">
          <div className="panel-block-label">Winning message</div>
          <p className="panel-meta-text">{panel.winning_message}</p>
        </div>
      </div>

      <div className="panel-meta-grid">
        <div className="panel-meta-block">
          <div className="panel-block-label">Top drivers</div>
          <ul className="panel-list">
            {panel.top_drivers.map((d, i) => <li key={`d${i}`}>{d}</li>)}
          </ul>
        </div>
        <div className="panel-meta-block">
          <div className="panel-block-label">Top objections</div>
          <ul className="panel-list against">
            {panel.top_objections.map((o, i) => <li key={`o${i}`}>{o}</li>)}
          </ul>
        </div>
      </div>

      {panel.risk_factors.length > 0 && (
        <div className="panel-meta-block" style={{ marginTop: 12 }}>
          <div className="panel-block-label">Risk factors</div>
          <ul className="panel-list against">
            {panel.risk_factors.map((r, i) => <li key={`r${i}`}>{r}</li>)}
          </ul>
        </div>
      )}

      <div className="panel-rounds">
        <div className="panel-block-label">Three-round panel discussion</div>
        {panel.rounds.map((rd) => (
          <div key={rd.round} className="panel-round">
            <div className="panel-round-head">
              <span className="panel-round-num">R{rd.round}</span>
              <span className="panel-round-title">{rd.title}</span>
              <span className="panel-round-counts mono">
                {rd.for_count}↑ {rd.neutral_count}— {rd.against_count}↓
              </span>
            </div>
            <p className="panel-round-summary">{rd.summary}</p>
          </div>
        ))}
      </div>

      <div className="panel-agents">
        <div className="panel-block-label">
          Individual agent reasoning
          <span className="panel-block-hint mono">
            ({panel.agents.length} AI personas — explanations of the forecast, not real buyers)
          </span>
        </div>
        <div className="panel-agent-grid">
          {displayed.map((a) => (
            <div key={a.id} className={`panel-agent-card stance-${a.stance}`}>
              <div className="panel-agent-head">
                <span className="panel-agent-segment">{a.segment}</span>
                <span className={`dtc-stance-tag stance-${a.stance}`}>{a.stance}</span>
              </div>
              <div className="panel-agent-profile">{a.profile}</div>
              <div className="panel-agent-reason">{a.reason}</div>
              <div className="panel-agent-objection">{a.top_objection}</div>
              <div className="panel-agent-score-bar">
                <div
                  className={`panel-agent-score-fill stance-${a.stance}`}
                  style={{ width: `${a.score * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
        {panel.agents.length > initialCount && (
          <button
            className="btn btn-ghost panel-show-all"
            onClick={() => setShowAllAgents((s) => !s)}
          >
            {showAllAgents
              ? `↑ Hide ${panel.agents.length - initialCount} more agents`
              : `↓ Show all ${panel.agents.length} agents`}
          </button>
        )}
      </div>
    </div>
  );
}
