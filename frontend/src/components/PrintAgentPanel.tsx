// Print-only template for the AI Buyer Panel section.
// Plain HTML tags, no CSS variables, no dark theme classes, no animations.
// Designed to render reliably under @media print regardless of screen state.

import type { Agent } from "../types";

interface Props {
  agents: Agent[];
}

export function PrintAgentPanel({ agents }: Props) {
  if (!agents || agents.length === 0) return null;

  return (
    <section className="print-agent-panel">
      <h2>AI Buyer Panel — Full Discussion</h2>

      {agents.map((agent, index) => (
        <article className="print-agent-card" key={agent.id ?? index}>
          <header className="print-agent-header">
            <h3>{agent.name || `Agent ${index + 1}`}</h3>
            <p>
              {agent.profession ? `${agent.profession}` : null}
              {agent.profession && agent.age ? ` · age ${agent.age}` : null}
              {agent.segment ? ` · ${agent.segment}` : null}
            </p>
            <p>
              Verdict: <strong>{agent.verdict}</strong>
              {typeof agent.score_10 === "number" ? ` · score ${agent.score_10.toFixed(1)}/10` : null}
            </p>
          </header>

          {agent.profile && (
            <p className="print-agent-profile">"{agent.profile}"</p>
          )}

          <div className="print-agent-rounds">
            {(agent.round_responses ?? []).map((round, ri) => (
              <div className="print-agent-round" key={ri}>
                <h4>Round {round.round} — {round.title}</h4>
                <p>{round.response}</p>
              </div>
            ))}
          </div>

          {agent.key_moment && (
            <p className="print-agent-extra">
              <strong>Key moment:</strong> {agent.key_moment}
            </p>
          )}
          {agent.top_objection && (
            <p className="print-agent-extra">
              <strong>Top objection:</strong> {agent.top_objection}
            </p>
          )}
          {agent.what_would_change_mind && (
            <p className="print-agent-extra">
              <strong>What would change mind:</strong> {agent.what_would_change_mind}
            </p>
          )}
        </article>
      ))}
    </section>
  );
}
