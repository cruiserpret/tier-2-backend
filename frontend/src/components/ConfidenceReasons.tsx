import { explainConfidenceReason, lowCoverageHint } from "../lib/explainMatch";
import type { ForecastResponse } from "../types";

interface Props {
  forecast: ForecastResponse;
}

export function ConfidenceReasons({ forecast }: Props) {
  const reasons = forecast.confidence_reasons || [];
  const whyWrong = forecast.why_might_be_wrong || [];
  const isLowCoverage =
    forecast.confidence === "low" ||
    forecast.diagnostics.coverage_tier === "weak" ||
    forecast.diagnostics.coverage_tier === "thin";

  if (!reasons.length && !whyWrong.length && !isLowCoverage) return null;

  return (
    <div className="report-card fade-up">
      <div className="report-card-eyebrow">
        Why this confidence — and why it may be wrong
      </div>

      {isLowCoverage && (
        <div className="confidence-hint">
          <strong>Note:</strong> {lowCoverageHint()}
        </div>
      )}

      {reasons.length > 0 && (
        <ul className="reason-list">
          {reasons.map((r, i) => (
            <li key={`r${i}`}>{explainConfidenceReason(r)}</li>
          ))}
        </ul>
      )}

      {whyWrong.length > 0 && (
        <div className="reason-block">
          <div className="reason-block-label">Why this forecast might still be wrong:</div>
          <ul className="reason-list muted">
            {whyWrong.map((r, i) => (
              <li key={`w${i}`}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
