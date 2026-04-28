import type { ForecastResponse } from "../types";

const VERDICT_LABELS: Record<string, string> = {
  launch_aggressively: "🚀 Launch Aggressively",
  launch: "✓ Launch",
  launch_with_changes: "⚠ Launch with Changes",
  test_before_launch: "🔬 Test Before Launch",
  reposition: "↻ Reposition",
  do_not_launch_yet: "✕ Do Not Launch Yet",
};

const VERDICT_PLAIN: Record<string, string> = {
  launch_aggressively: "Strong demand signal. Scale with controlled measurement.",
  launch: "Strong demand. Launch and iterate.",
  launch_with_changes: "Decent demand, but address friction or positioning before scaling.",
  test_before_launch: "Pilot before scaling. Forecast is directional, not validated.",
  reposition: "Buyer interest exists, but positioning, pricing, or distribution likely needs work.",
  do_not_launch_yet: "Weak demand signal. Reconsider product-market fit before launching.",
};

export function ResultCard({ data }: { data: ForecastResponse }) {
  const r = data.trial_rate;
  const isFallback = data.diagnostics.coverage_tier === "weak" || data.diagnostics.coverage_tier === "thin";

  return (
    <div className="report-card fade-up">
      <div className="report-card-eyebrow">Predicted 12-month trial rate</div>
      <div className="report-rate-row">
        <div>
          <div className="report-rate-num">{r.percentage.toFixed(1)}%</div>
          <div className="report-rate-meta">
            range {(r.low * 100).toFixed(1)}% – {(r.high * 100).toFixed(1)}% · coverage {data.diagnostics.coverage_tier}
            {isFallback && <span className="report-rate-flag"> · directional only</span>}
          </div>
        </div>
        <div className="report-rate-pills">
          <div className={`verdict-pill v-${data.verdict}`}>
            {VERDICT_LABELS[data.verdict] || data.verdict}
          </div>
          <div className="confidence-pill">confidence: {data.confidence}</div>
        </div>
      </div>
      <div className="report-headline">{data.headline}</div>
      <div className="report-verdict-plain">
        <strong>What this means: </strong>
        {VERDICT_PLAIN[data.verdict] || ""}
      </div>
    </div>
  );
}
