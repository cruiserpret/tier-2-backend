// Print-only template for the Phase 1 Evidence Panel + Confidence
// Ledger + Evidence Gap Card sections. Plain HTML tags, no CSS
// variables, no dark theme classes, no animations. Designed to
// render reliably under @media print regardless of screen state.
import type {
  EvidenceBuckets,
  EvidenceItem,
  LedgerEntry,
  Anchor,
} from "../types";

interface PrintEvidenceProps {
  buckets?: EvidenceBuckets;
  legacyAnchors?: Anchor[];
  ledger?: LedgerEntry[];
  legacyReasons?: string[];
}

const SYSTEMIC_GAP_SIGNALS = new Set([
  "no_external_evidence",
  "no_shopify_outcomes",
]);

function strengthLabel(item: EvidenceItem): string {
  if (item.anchor_strength === "direct") return "Direct match";
  if (item.anchor_strength === "adjacent") return "Adjacent match";
  if (item.anchor_strength === "weak") return "Weak match";
  if (item.bucket === "fallback_neighbor") return "Retrieved, not used";
  if (item.bucket === "candidate_comparable") return "Below threshold";
  if (item.bucket === "exploratory_comparable") return "Exploratory";
  return "";
}

function PrintItem({ item }: { item: EvidenceItem }) {
  return (
    <div className="print-evidence-item">
      <p className="print-evidence-item-head">
        <strong>{item.brand}</strong>
        {" \u2014 "}
        {(item.trial_rate * 100).toFixed(1)}% trial
        {" \u00b7 "}
        sim {item.similarity.toFixed(2)}
        {" \u00b7 "}
        {strengthLabel(item)}
      </p>
      {item.display_warning && (
        <p className="print-evidence-item-warning">{item.display_warning}</p>
      )}
    </div>
  );
}

function PrintLegacyItem({ anchor }: { anchor: Anchor }) {
  return (
    <div className="print-evidence-item">
      <p className="print-evidence-item-head">
        <strong>{anchor.brand}</strong>
        {" \u2014 "}
        {(anchor.trial_rate * 100).toFixed(1)}% trial
        {" \u00b7 "}
        grade {anchor.confidence_grade}
        {" \u00b7 "}
        Legacy anchor data
      </p>
      <p className="print-evidence-item-warning">
        Detailed evidence classification unavailable for this cached result.
      </p>
    </div>
  );
}

export function PrintEvidence({
  buckets,
  legacyAnchors,
  ledger,
  legacyReasons,
}: PrintEvidenceProps) {
  // Backward compat: legacy path
  const showLegacyAnchors =
    !buckets && legacyAnchors && legacyAnchors.length > 0;
  const showLegacyReasons =
    !ledger && legacyReasons && legacyReasons.length > 0;

  // Defensive defaults per P1.6 hardening
  const forecast_anchors = buckets?.forecast_anchors ?? [];
  const candidate_comparables = buckets?.candidate_comparables ?? [];
  const fallback_neighbors = buckets?.fallback_neighbors ?? [];
  const exploratory_comparables = buckets?.exploratory_comparables ?? [];

  const ledgerEntries = (ledger ?? []).filter(
    (e) => !SYSTEMIC_GAP_SIGNALS.has(e.signal)
  );
  const gapEntries = (ledger ?? []).filter((e) =>
    SYSTEMIC_GAP_SIGNALS.has(e.signal)
  );

  const hasAnyAnchors =
    forecast_anchors.length > 0 ||
    candidate_comparables.length > 0 ||
    fallback_neighbors.length > 0 ||
    exploratory_comparables.length > 0;

  // Don't render the section header if there's literally nothing.
  if (
    !showLegacyAnchors &&
    !showLegacyReasons &&
    !hasAnyAnchors &&
    ledgerEntries.length === 0 &&
    gapEntries.length === 0
  ) {
    return null;
  }

  return (
    <section className="print-evidence-panel">
      <h2>Evidence Behind The Forecast</h2>

      {showLegacyAnchors && (
        <div className="print-evidence-section">
          <h3>Forecast Anchors (legacy)</h3>
          <p className="print-evidence-section-sub">
            Detailed evidence classification unavailable for this cached result.
          </p>
          {legacyAnchors!.map((a) => (
            <PrintLegacyItem key={a.brand} anchor={a} />
          ))}
        </div>
      )}

      {forecast_anchors.length > 0 && (
        <div className="print-evidence-section">
          <h3>Forecast Anchors ({forecast_anchors.length})</h3>
          <p className="print-evidence-section-sub">
            Eligible comparables actually used in forecast math.
          </p>
          {forecast_anchors.map((item) => (
            <PrintItem key={`fa-${item.brand}`} item={item} />
          ))}
        </div>
      )}

      {candidate_comparables.length > 0 && (
        <div className="print-evidence-section">
          <h3>Candidate Comparables ({candidate_comparables.length})</h3>
          <p className="print-evidence-section-sub">
            Considered, but did not meet the forecast anchor threshold.
          </p>
          {candidate_comparables.map((item) => (
            <PrintItem key={`cc-${item.brand}`} item={item} />
          ))}
        </div>
      )}

      {fallback_neighbors.length > 0 && (
        <div className="print-evidence-section">
          <h3>Retrieved but Not Used ({fallback_neighbors.length})</h3>
          <p className="print-evidence-section-sub">
            Search returned these, but the forecast fell back to a category
            prior because Assembly did not find enough eligible direct
            comparables.
          </p>
          {fallback_neighbors.map((item) => (
            <PrintItem key={`fn-${item.brand}`} item={item} />
          ))}
        </div>
      )}

      {exploratory_comparables.length > 0 && (
        <div className="print-evidence-section">
          <h3>Exploratory Comparables ({exploratory_comparables.length})</h3>
          <p className="print-evidence-section-sub">
            Adjacent or external market evidence. Directional context only;
            not used in forecast math.
          </p>
          {exploratory_comparables.map((item) => (
            <PrintItem key={`ec-${item.brand}`} item={item} />
          ))}
        </div>
      )}

      {(ledgerEntries.length > 0 || showLegacyReasons) && (
        <div className="print-evidence-section">
          <h3>Confidence Ledger</h3>
          <ul className="print-ledger-list">
            {showLegacyReasons
              ? legacyReasons!.map((reason, i) => (
                  <li key={i} className="print-ledger-entry">
                    <span className="print-ledger-marker">{"\u00b7"}</span> {reason}
                  </li>
                ))
              : ledgerEntries.map((entry) => (
                  <li key={entry.signal} className="print-ledger-entry">
                    <span className="print-ledger-marker">
                      {entry.kind === "positive"
                        ? "+"
                        : entry.kind === "negative"
                        ? "\u2212"
                        : "\u00b7"}
                    </span>{" "}
                    {entry.text}
                  </li>
                ))}
          </ul>
        </div>
      )}

      {gapEntries.length > 0 && (
        <div className="print-evidence-section">
          <h3>Evidence Not Yet Connected</h3>
          <p className="print-evidence-section-sub">
            Evidence sources not connected to this forecast yet.
          </p>
          <ul className="print-ledger-list">
            {gapEntries.map((entry) => (
              <li key={entry.signal} className="print-ledger-entry">
                <span className="print-ledger-marker">{"\u00b7"}</span> {entry.text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
