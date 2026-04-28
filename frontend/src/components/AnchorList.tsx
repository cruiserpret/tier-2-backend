import type { Anchor } from "../types";
import { explainMatchReason } from "../lib/explainMatch";

export function AnchorList({ anchors }: { anchors: Anchor[] }) {
  if (!anchors.length) return null;
  return (
    <div className="report-card fade-up">
      <div className="report-card-eyebrow">
        Anchored on real comparable brands
      </div>
      <p className="anchor-intro">
        These are the real DTC brands Assembly anchored your forecast on,
        sorted by retrieval weight. Trial rates shown are measured 12-month
        adoption from public sources.
      </p>
      <table className="anchor-table">
        <thead>
          <tr>
            <th>Brand</th>
            <th className="num">Trial rate</th>
            <th>Source grade</th>
            <th>Why it matched</th>
          </tr>
        </thead>
        <tbody>
          {anchors.map((a) => (
            <tr key={a.brand}>
              <td>{a.brand}</td>
              <td className="num">{(a.trial_rate * 100).toFixed(1)}%</td>
              <td>
                <span className={`grade grade-${a.confidence_grade}`} title={`Source quality: ${a.confidence_grade}`}>
                  {a.confidence_grade}
                </span>
              </td>
              <td className="match-reason">{explainMatchReason(a.match_reason)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="anchor-footer">
        Source grade key: <strong>A</strong> = source-validated, <strong>B</strong> = strong estimate,
        <strong> C</strong> = directional estimate, <strong>D</strong> = weak estimate.
        Higher-grade anchors carry more weight in the final forecast.
      </p>
    </div>
  );
}
