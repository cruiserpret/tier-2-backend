const DELIVERABLES = [
  { icon: "◎", title: "Predicted Trial Rate", desc: "Comparable-anchored estimate of 12-month trial rate, with low/high range and confidence level.", example: "e.g. 20.5% predicted trial, range 16-26%, confidence medium-high" },
  { icon: "⬤", title: "Verdict & Headline", desc: "Decision-oriented call: launch, launch with changes, test before launch, reposition, or do not launch yet.", example: "e.g. ✓ Launch — strong demand signal, friction manageable" },
  { icon: "▲", title: "Anchored Comparables", desc: "The actual brands the forecast was anchored on, their trial rates, and why they matched.", example: "e.g. Liquid IV (20.5%), AG1 (11%), Olipop (27%) — A/B grade" },
  { icon: "⟳", title: "AI Buyer Panel", desc: "20 AI personas debate the forecast across 3 rounds: first reaction, comparable comparison, consensus.", example: "e.g. 12 buy / 5 considering / 3 won't — segment: health optimizers" },
  { icon: "◈", title: "Top Drivers", desc: "What the buyer panel cited as the strongest reasons to buy, ranked by frequency.", example: "e.g. \"Costco availability\" cited by 9/20 agents" },
  { icon: "◻", title: "Top Objections", desc: "What kept against-stance agents from buying — the real friction the launch will face.", example: "e.g. \"Subscription fatigue\" cited by 7/20 agents" },
  { icon: "⬡", title: "Counterfactual Scenarios", desc: "Directional impact of pricing, distribution, and positioning levers — assuming execution matches comparables.", example: "e.g. Add retail distribution → could move toward 25.8%" },
  { icon: "⚠", title: "Why It Might Be Wrong", desc: "Honest disclosure of forecast weak points: thin coverage, neighbor variance, or fallback triggers.", example: "e.g. Coverage weak — comparable database thin in this subtype" },
];

export function Deliverables() {
  return (
    <section className="deliverable-section fade-up">
      <div className="deliverable-header">
        <div className="deliverable-eyebrow">
          <span style={{ color: "var(--accent)" }}>◈</span>
          What Assembly delivers
        </div>
        <h2 className="deliverable-title display">MARKET GOD'S EYE VIEW</h2>
        <p className="deliverable-desc">
          Comparable-anchored forecast plus AI buyer panel discussion.
          Eight outputs, every one grounded in real measured trial rates from 37+ DTC brands —
          not surveys, not optimal-price math.
        </p>
      </div>

      <div className="deliverable-grid">
        {DELIVERABLES.map((d) => (
          <div key={d.title} className="deliverable-card">
            <div className="d-icon">{d.icon}</div>
            <div className="d-title">{d.title}</div>
            <div className="d-desc">{d.desc}</div>
            <div className="d-example">{d.example}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
