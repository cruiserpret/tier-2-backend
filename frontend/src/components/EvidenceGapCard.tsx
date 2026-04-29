import type { LedgerEntry } from "../types";

interface EvidenceGapCardProps {
  entries?: LedgerEntry[];
}

// Per friend Apr 30 Call 1=B: this card surfaces SYSTEMIC missing
// evidence types (Phase 3 / Phase 7 not shipped yet). Distinct from
// the Confidence Ledger which surfaces forecast-specific signals.
const SYSTEMIC_GAP_SIGNALS = new Set([
  "no_external_evidence",
  "no_shopify_outcomes",
]);

export function EvidenceGapCard({ entries }: EvidenceGapCardProps) {
  if (!entries || entries.length === 0) {
    return null;
  }

  const gaps = entries.filter((e) => SYSTEMIC_GAP_SIGNALS.has(e.signal));

  if (gaps.length === 0) {
    return null;
  }

  return (
    <section className="evidence-gap-card sidebar-card fade-up">
      <div className="report-card-eyebrow">Evidence Not Yet Connected</div>
      <div className="evidence-gap-sub">
        Evidence sources not connected to this forecast yet.
      </div>
      <ul className="ledger-list">
        {gaps.map((entry) => (
          <li key={entry.signal} className="ledger-entry ledger-entry-neutral">
            <span className="ledger-marker">{"\u00b7"}</span>
            <span className="ledger-text">{entry.text}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
