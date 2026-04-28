import { useState } from "react";
import type {
  AgentPanel as AgentPanelType,
  PanelSource,
  Agent,
  AgentVerdict,
} from "../types";
import { VERDICT_COLOR } from "../types";

interface Props {
  panel: AgentPanelType | null;
  source: PanelSource;
  error: string | null;
}

type FilterMode = "all" | "BUY" | "CONSIDERING" | "WON'T BUY";

export function AgentPanel({ panel, source, error }: Props) {
  const [showAllAgents, setShowAllAgents] = useState(false);
  const [filter, setFilter] = useState<FilterMode>("all");
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);

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

  // Apply filter
  const filteredAgents = filter === "all"
    ? panel.agents
    : panel.agents.filter((a) => a.verdict === filter);

  const displayed = showAllAgents
    ? filteredAgents
    : filteredAgents.slice(0, initialCount);

  const totalFor = panel.rounds[panel.rounds.length - 1]?.for_count ?? 0;
  const totalNeutral = panel.rounds[panel.rounds.length - 1]?.neutral_count ?? 0;
  const totalAgainst = panel.rounds[panel.rounds.length - 1]?.against_count ?? 0;
  const total = panel.agent_count;

  // Verdict counts for the filter chips
  const verdictCounts = {
    "BUY": panel.agents.filter((a) => a.verdict === "BUY").length,
    "CONSIDERING": panel.agents.filter((a) => a.verdict === "CONSIDERING").length,
    "WON'T BUY": panel.agents.filter((a) => a.verdict === "WON'T BUY").length,
  };

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
            <span className="panel-stance-label tag-for">Buy signal</span>
            <div className="panel-stance-bar-wrap">
              <div className="panel-stance-bar seg-buy" style={{ width: `${(totalFor / total) * 100}%` }} />
            </div>
            <span className="panel-stance-num">{totalFor}/{total}</span>
          </div>
          <div className="panel-stance-row">
            <span className="panel-stance-label tag-neutral">Considering</span>
            <div className="panel-stance-bar-wrap">
              <div className="panel-stance-bar seg-considering" style={{ width: `${(totalNeutral / total) * 100}%` }} />
            </div>
            <span className="panel-stance-num">{totalNeutral}/{total}</span>
          </div>
          <div className="panel-stance-row">
            <span className="panel-stance-label tag-against">Resistant</span>
            <div className="panel-stance-bar-wrap">
              <div className="panel-stance-bar seg-wont" style={{ width: `${(totalAgainst / total) * 100}%` }} />
            </div>
            <span className="panel-stance-num">{totalAgainst}/{total}</span>
          </div>
        </div>
      </div>

      <div className="panel-distribution-caption mono">
        AI buyer-panel signal, not market-size forecast.
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

      {/* ───────────── Slice 2A: enriched individual agent cards ───────────── */}
      <div className="panel-agents v3">
        <div className="panel-block-label">
          Individual buyer opinions
          <span className="panel-block-hint mono">
            ({panel.agents.length} AI personas — explanations of the forecast, not real buyers)
          </span>
        </div>

        <div className="agent-filter-bar">
          <button
            className={`agent-filter-chip ${filter === "all" ? "active" : ""}`}
            onClick={() => { setFilter("all"); setShowAllAgents(false); }}
          >
            All ({panel.agents.length})
          </button>
          <button
            className={`agent-filter-chip ${filter === "BUY" ? "active" : ""}`}
            onClick={() => { setFilter("BUY"); setShowAllAgents(false); }}
            style={{ borderColor: filter === "BUY" ? VERDICT_COLOR["BUY"] : undefined }}
          >
            BUY ({verdictCounts["BUY"]})
          </button>
          <button
            className={`agent-filter-chip ${filter === "CONSIDERING" ? "active" : ""}`}
            onClick={() => { setFilter("CONSIDERING"); setShowAllAgents(false); }}
            style={{ borderColor: filter === "CONSIDERING" ? VERDICT_COLOR["CONSIDERING"] : undefined }}
          >
            CONSIDERING ({verdictCounts["CONSIDERING"]})
          </button>
          <button
            className={`agent-filter-chip ${filter === "WON'T BUY" ? "active" : ""}`}
            onClick={() => { setFilter("WON'T BUY"); setShowAllAgents(false); }}
            style={{ borderColor: filter === "WON'T BUY" ? VERDICT_COLOR["WON'T BUY"] : undefined }}
          >
            WON'T BUY ({verdictCounts["WON'T BUY"]})
          </button>
        </div>

        <div className="agent-card-grid">
          {displayed.map((a) => (
            <AgentCardV3
              key={a.id}
              agent={a}
              expanded={expandedAgent === a.id}
              onToggle={() => setExpandedAgent(expandedAgent === a.id ? null : a.id)}
            />
          ))}
        </div>

        {filteredAgents.length > initialCount && (
          <button
            className="btn btn-ghost panel-show-all"
            onClick={() => setShowAllAgents((s) => !s)}
          >
            {showAllAgents
              ? `↑ Hide ${filteredAgents.length - initialCount} more agents`
              : `↓ Show all ${filteredAgents.length} ${filter === "all" ? "agents" : `${filter} agents`}`}
          </button>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// AgentCardV3 — individual buyer card with full v3 fields
// ─────────────────────────────────────────────────────────────────────

interface AgentCardProps {
  agent: Agent;
  expanded: boolean;
  onToggle: () => void;
}

function AgentCardV3({ agent: a, expanded, onToggle }: AgentCardProps) {
  // Defensive: cached demos from before Commit 3b lack v3 fields.
  // Derive missing fields from legacy shape so the card still renders.
  const name = a.name || a.segment || "Buyer";
  const verdict =
    a.verdict ||
    (a.stance === "for" ? "BUY" : a.stance === "neutral" ? "CONSIDERING" : "WON'T BUY");
  const score10 =
    typeof a.score_10 === "number"
      ? a.score_10
      : typeof a.score === "number"
        ? a.score * 10
        : 5.0;
  const roundResponses = Array.isArray(a.round_responses) ? a.round_responses : [];
  const journey = a.journey || {
    initial_verdict: verdict,
    final_verdict: verdict,
    shifted: false,
    shift_reason: a.reason || "",
    key_moment: "",
    key_quote: a.reason || "",
  };
  const keyMoment = a.key_moment || journey.key_moment || "";
  const whatWouldChangeMind = a.what_would_change_mind || "";

  const verdictColor = VERDICT_COLOR[verdict as keyof typeof VERDICT_COLOR] || "#888";
  const verdictClass =
    verdict === "BUY"        ? "verdict-buy" :
    verdict === "CONSIDERING" ? "verdict-considering" :
                                "verdict-wont";

  const initials = name
    .split(" ")
    .map((s: string) => s[0] || "")
    .filter((c: string) => c)
    .slice(0, 2)
    .join("")
    .toUpperCase() || "??";

  return (
    <div className={`agent-card-v3 ${verdictClass}`}>
      <div className="agent-card-head">
        <div className="agent-avatar" style={{ borderColor: verdictColor }}>
          {initials}
        </div>
        <div className="agent-identity">
          <div className="agent-name">{name}</div>
          <div className="agent-meta mono">
            {a.profession || a.segment || ""}{a.age ? ` · ${a.age}` : ""}
          </div>
          <div className="agent-segment">{a.segment}</div>
        </div>
        <div
          className={`verdict-tag ${verdictClass}`}
          style={{ borderColor: verdictColor, color: verdictColor }}
        >
          {verdict}
        </div>
      </div>

      {/* Score bar with 1-10 number */}
      <div className="agent-score-row">
        <div className="agent-score-bar-wrap">
          <div
            className="agent-score-bar"
            style={{
              width: `${(score10 / 10) * 100}%`,
              backgroundColor: verdictColor,
            }}
          />
        </div>
        <div className="agent-score-num mono" style={{ color: verdictColor }}>
          {score10.toFixed(1)}/10
        </div>
      </div>

      {/* Profile (one-liner) */}
      <div className="agent-profile-line">{a.profile}</div>

      {/* Round responses — primary content, what Hamza wanted */}
      {roundResponses.length > 0 && (
      <div className="agent-rounds">
        {roundResponses.map((rr) => (
          <div key={rr.round} className="agent-round-row">
            <div className="agent-round-label mono">
              R{rr.round} · {rr.title}
            </div>
            <div className="agent-round-text">"{rr.response}"</div>
          </div>
        ))}
      </div>
      )}

      {/* Journey badge */}
      <div className="agent-journey">
        <div className="agent-journey-arc">
          <span className={`journey-verdict ${verdictTagClass(journey.initial_verdict as any)}`}>
            {journey.initial_verdict}
          </span>
          <span className="journey-arrow">→</span>
          <span className={`journey-verdict ${verdictTagClass(journey.final_verdict as any)}`}>
            {journey.final_verdict}
          </span>
          {journey.shifted ? (
            <span className="journey-shifted-tag mono">shifted</span>
          ) : (
            <span className="journey-held-tag mono">held</span>
          )}
          {a.is_hardcore && (
            <span className="journey-hardcore-tag mono" title="Hardcore resistor — no movement across rounds">
              hardcore
            </span>
          )}
        </div>
        {keyMoment && (
        <div className="agent-key-moment mono">
          <span className="agent-key-moment-label">Key moment:</span> {keyMoment}
        </div>
        )}
      </div>

      {/* Expandable: top objection + what would change mind */}
      <button className="agent-card-expand-btn mono" onClick={onToggle}>
        {expanded ? "▴ Hide details" : "▾ Show objection & change-mind"}
      </button>

      {expanded && (
        <div className="agent-card-details">
          <div className="agent-detail-row">
            <div className="agent-detail-label mono">Top objection</div>
            <div className="agent-detail-text">{a.top_objection}</div>
          </div>
          <div className="agent-detail-row">
            <div className="agent-detail-label mono">
              {verdict === "BUY"
                ? "What might make me reconsider"
                : verdict === "CONSIDERING"
                ? "What would push me to buy"
                : "What would soften my no"}
            </div>
            <div className="agent-detail-text">{whatWouldChangeMind || "n/a"}</div>
          </div>
          {journey.shifted && (
            <div className="agent-detail-row">
              <div className="agent-detail-label mono">Why they shifted</div>
              <div className="agent-detail-text">{journey.shift_reason}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function verdictTagClass(v: AgentVerdict): string {
  if (v === "BUY") return "tag-buy";
  if (v === "CONSIDERING") return "tag-considering";
  return "tag-wont";
}
