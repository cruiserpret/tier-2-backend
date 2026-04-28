import type { Counterfactual } from "../types";

export function Counterfactuals({ cfs }: { cfs: Counterfactual[] }) {
  if (!cfs.length) return null;
  return (
    <div className="report-card fade-up">
      <div className="report-card-eyebrow">Counterfactual scenarios</div>
      <div className="cf-list">
        {cfs.map((cf, i) => (
          <div key={i} className={`cf cf-${cf.direction}`}>
            <div className="cf-head">
              <span className="cf-label">{cf.label}</span>
              <span className="cf-pct">could move toward {cf.new_prediction_pct.toFixed(1)}%</span>
            </div>
            <div className="cf-desc">{cf.description}</div>
          </div>
        ))}
      </div>
      <p className="cf-footer">
        Counterfactuals are directional strategy simulations, not validated causal estimates.
        Treat as planning guidance, assuming execution matches comparable brands.
      </p>
    </div>
  );
}
