import type { LedgerEntry, LedgerKind } from "../types";

interface ConfidenceLedgerProps {
  entries?: LedgerEntry[];
  // Backward compat: legacy confidence_reasons string list
  legacyReasons?: string[];
}

const KIND_MARKER: Record<LedgerKind, string> = {
  positive: "+",
  negative: "\u2212",
  neutral: "\u00B7",
};

const KIND_COLOR: Record<LedgerKind, string> = {
  positive: "var(--for, #7CFC9C)",
  negative: "var(--against, #FF6E80)",
  neutral: "var(--text-muted, #888)",
};

// Phase 1 separation: ledger excludes systemic gap signals
// (those live in EvidenceGapCard). Filter them out here.
const SYSTEMIC_GAP_SIGNALS = new Set([
  "no_external_evidence",
  "no_shopify_outcomes",
]);

export function ConfidenceLedger({ entries, legacyReasons }: ConfidenceLedgerProps) {
  // Backward compat (Call 5=A)
  if (!entries && legacyReasons && legacyReasons.length > 0) {
    return (
      <section className="confidence-ledger sidebar-card fade-up">
        <div className="report-card-eyebrow">Confidence Ledger</div>
        <ul className="ledger-list">
          {legacyReasons.map((reason, i) => (
            <li key={i} className="ledger-entry ledger-entry-legacy">
              <span className="ledger-marker">{KIND_MARKER.neutral}</span>
              <span className="ledger-text">{reason}</span>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  if (!entries || entries.length === 0) {
    return null;
  }

  const displayEntries = entries.filter(
    (e) => !SYSTEMIC_GAP_SIGNALS.has(e.signal)
  );

  if (displayEntries.length === 0) {
    return null;
  }

  return (
    <section className="confidence-ledger sidebar-card fade-up">
      <div className="report-card-eyebrow">Confidence Ledger</div>
      <ul className="ledger-list">
        {displayEntries.map((entry) => (
          <li
            key={entry.signal}
            className={`ledger-entry ledger-entry-${entry.kind}`}
          >
            <span
              className="ledger-marker"
              style={{ color: KIND_COLOR[entry.kind] }}
            >
              {KIND_MARKER[entry.kind]}
            </span>
            <span className="ledger-text">{entry.text}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
