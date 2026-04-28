export function Hero() {
  return (
    <section className="hero">
      <div className="hero-eyebrow fade-up mono">
        <span className="live-dot" />
        DTC Market Intelligence Engine
      </div>

      <div className="hero-logo-wrap fade-up fade-up-1">
        <div className="hero-logo-bg">
          <div className="logo-grid-lines" />
        </div>
        <div className="hero-logo-inner">
          <div className="hero-bracket">[</div>
          <div className="display hero-wordmark">MARKET</div>
          <div className="hero-bracket">]</div>
        </div>
        <h1 className="hero-h1">
          Know exactly where your market lands<br />before you spend a dollar.
        </h1>
        <div className="hero-tagline display">SIMULATE BEFORE YOU LAUNCH.</div>
      </div>

      <p className="hero-sub fade-up fade-up-2">
        Comparable-anchored forecast. Trial-rate predictions grounded in 37+ real DTC brands
        and their measured 12-month adoption rates. AI buyer personas debate the result —
        they explain it, they don't fabricate it.
      </p>

      <div className="stats-row fade-up fade-up-3">
        <div className="hero-stat-card">
          <div className="hsc-icon">◎</div>
          <div className="hsc-title">Comparable-anchored</div>
          <div className="hsc-desc">Forecast number comes from the weighted median of real comparable brand trial rates — not surveys, not LLM guesses.</div>
        </div>
        <div className="hero-stat-card">
          <div className="hsc-icon">⟳</div>
          <div className="hsc-title">Deterministic</div>
          <div className="hsc-desc">Same input, same output. Run Liquid IV five times — get five identical predictions. No drift, no variance.</div>
        </div>
        <div className="hero-stat-card">
          <div className="hsc-icon">◈</div>
          <div className="hsc-title">Honest about uncertainty</div>
          <div className="hsc-desc">When comparable coverage is thin, confidence is low and the system says so. No fake precision.</div>
        </div>
      </div>
    </section>
  );
}
