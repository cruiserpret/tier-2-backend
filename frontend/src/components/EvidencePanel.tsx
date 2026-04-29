import type {
  EvidenceBuckets,
  EvidenceItem,
  Anchor,
  AnchorStrength,
} from "../types";

interface EvidencePanelProps {
  buckets?: EvidenceBuckets;
  // Backward compat (Call 5=A): legacy anchored_on for pre-Phase-1 caches
  legacyAnchors?: Anchor[];
}

const STRENGTH_LABEL: Record<AnchorStrength, string> = {
  direct: "Direct match",
  adjacent: "Adjacent match",
  weak: "Weak match",
};

const STRENGTH_COLOR: Record<AnchorStrength, string> = {
  direct: "var(--text)",
  adjacent: "#c9a86b",   // muted amber
  weak: "#cc8a8a",       // muted rose
};

function ItemRow({ item }: { item: EvidenceItem }) {
  const strengthLabel = item.anchor_strength
    ? STRENGTH_LABEL[item.anchor_strength]
    : item.bucket === "fallback_neighbor"
    ? "Retrieved, not used"
    : item.bucket === "candidate_comparable"
    ? "Below threshold"
    : item.bucket === "exploratory_comparable"
    ? "Exploratory"
    : "";

  const strengthColor = item.anchor_strength
    ? STRENGTH_COLOR[item.anchor_strength]
    : "var(--text-muted, #888)";

  return (
    <div className="evidence-item">
      <div className="evidence-item-head">
        <span className="evidence-item-brand">{item.brand}</span>
        <span className="evidence-item-rate">
          {(item.trial_rate * 100).toFixed(1)}% trial
        </span>
        <span className="evidence-item-sim">sim {item.similarity.toFixed(2)}</span>
      </div>
      {strengthLabel && (
        <div
          className="evidence-item-strength"
          style={{ color: strengthColor }}
        >
          {strengthLabel}
        </div>
      )}
      {item.display_warning && (
        <div className="evidence-item-warning">{item.display_warning}</div>
      )}
    </div>
  );
}

function LegacyAnchorRow({ anchor }: { anchor: Anchor }) {
  return (
    <div className="evidence-item">
      <div className="evidence-item-head">
        <span className="evidence-item-brand">{anchor.brand}</span>
        <span className="evidence-item-rate">
          {(anchor.trial_rate * 100).toFixed(1)}% trial
        </span>
        <span className="evidence-item-sim">grade {anchor.confidence_grade}</span>
      </div>
      <div
        className="evidence-item-strength"
        style={{ color: "var(--text-muted, #888)" }}
      >
        Legacy anchor data
      </div>
      <div className="evidence-item-warning">
        Detailed evidence classification unavailable for this cached result.
      </div>
    </div>
  );
}

export function EvidencePanel({ buckets, legacyAnchors }: EvidencePanelProps) {
  // Backward compat path (Call 5=A)
  if (!buckets && legacyAnchors && legacyAnchors.length > 0) {
    return (
      <section className="evidence-panel sidebar-card fade-up">
        <div className="report-card-eyebrow">Forecast Anchors (legacy)</div>
        <div className="evidence-section-body">
          {legacyAnchors.map((a) => (
            <LegacyAnchorRow key={a.brand} anchor={a} />
          ))}
        </div>
      </section>
    );
  }

  if (!buckets) {
    return null;
  }

  // Defensive defaults per friend Apr 30 hardening — old/partial responses may
  // omit any of these arrays. Treat missing arrays as empty so the component
  // never crashes on bucket destructuring.
  const forecast_anchors = buckets.forecast_anchors ?? [];
  const candidate_comparables = buckets.candidate_comparables ?? [];
  const fallback_neighbors = buckets.fallback_neighbors ?? [];
  const exploratory_comparables = buckets.exploratory_comparables ?? [];

  const hasAnyAnchors =
    forecast_anchors.length > 0 ||
    candidate_comparables.length > 0 ||
    fallback_neighbors.length > 0 ||
    exploratory_comparables.length > 0;

  if (!hasAnyAnchors) {
    return (
      <section className="evidence-panel sidebar-card fade-up">
        <div className="report-card-eyebrow">Forecast Anchors</div>
        <div className="evidence-empty">No eligible forecast anchors found.</div>
      </section>
    );
  }

  return (
    <section className="evidence-panel sidebar-card fade-up">
      {forecast_anchors.length > 0 && (
        <div className="evidence-section">
          <div className="report-card-eyebrow">
            Forecast Anchors ({forecast_anchors.length})
          </div>
          <div className="evidence-section-sub">
            Eligible comparables actually used in forecast math.
          </div>
          <div className="evidence-section-body">
            {forecast_anchors.map((item) => (
              <ItemRow key={`fa-${item.brand}`} item={item} />
            ))}
          </div>
        </div>
      )}

      {candidate_comparables.length > 0 && (
        <div className="evidence-section">
          <div className="report-card-eyebrow">
            Candidate Comparables ({candidate_comparables.length})
          </div>
          <div className="evidence-section-sub">
            Considered, but did not meet the forecast anchor threshold.
          </div>
          <div className="evidence-section-body">
            {candidate_comparables.map((item) => (
              <ItemRow key={`cc-${item.brand}`} item={item} />
            ))}
          </div>
        </div>
      )}

      {fallback_neighbors.length > 0 && (
        <div className="evidence-section">
          <div className="report-card-eyebrow">
            Retrieved but Not Used ({fallback_neighbors.length})
          </div>
          <div className="evidence-section-sub">
            Search returned these, but the forecast fell back to a category
            prior because Assembly did not find enough eligible direct comparables.
          </div>
          <div className="evidence-section-body">
            {fallback_neighbors.map((item) => (
              <ItemRow key={`fn-${item.brand}`} item={item} />
            ))}
          </div>
        </div>
      )}

      {/* Phase 1: render exploratory only when populated. Empty until Phase 3. */}
      {exploratory_comparables.length > 0 && (
        <div className="evidence-section">
          <div className="report-card-eyebrow">
            Exploratory Comparables ({exploratory_comparables.length})
          </div>
          <div className="evidence-section-sub">
            Adjacent or externally-discovered comparables. Directional context only;
            not used in forecast math.
          </div>
          <div className="evidence-section-body">
            {exploratory_comparables.map((item) => (
              <ItemRow key={`ec-${item.brand}`} item={item} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
