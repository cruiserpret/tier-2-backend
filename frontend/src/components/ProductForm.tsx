import { useState, useEffect, useRef } from "react";
import type { FormState, ProductPayload } from "../types";

const CATEGORIES = [
  { value: "", label: "Auto-detect from description" },
  { value: "supplements_health", label: "Supplements / Health" },
  { value: "food_beverage", label: "Food & Beverage" },
  { value: "beverages_alcohol_free", label: "Non-Alcoholic Beverages" },
  { value: "electronics_tech", label: "Electronics / Wearables" },
  { value: "wearables_health", label: "Health Wearables" },
  { value: "home_lifestyle", label: "Home / Lifestyle" },
  { value: "fashion_apparel", label: "Fashion / Apparel" },
  { value: "beauty_skincare", label: "Beauty / Skincare" },
  { value: "personal_care", label: "Personal Care" },
  { value: "pet_products", label: "Pet Products" },
  { value: "other", label: "Other" },
];

const AGE_RANGES = [
  { value: "18-24", label: "18–24" },
  { value: "25-34", label: "25–34" },
  { value: "35-44", label: "35–44" },
  { value: "45-54", label: "45–54" },
  { value: "55+", label: "55+" },
];

const GENDERS = [
  { value: "all", label: "All genders" },
  { value: "women", label: "Primarily women" },
  { value: "men", label: "Primarily men" },
  { value: "nb", label: "Non-binary inclusive" },
];

const INCOME_RANGES = [
  { value: "under_50k", label: "Under $50K" },
  { value: "50k_75k", label: "$50K–$75K" },
  { value: "75k_100k", label: "$75K–$100K" },
  { value: "100k_150k", label: "$100K–$150K" },
  { value: "over_150k", label: "Over $150K" },
];

const MARKET_TIERS = [
  { value: "", label: "(auto)" },
  { value: "mass_platform", label: "Mass platform" },
  { value: "mass_market", label: "Mass market" },
  { value: "challenger", label: "Challenger" },
  { value: "premium_niche", label: "Premium niche" },
  { value: "niche", label: "Niche" },
  { value: "luxury", label: "Luxury" },
];

const DISTRIBUTION_HINTS = [
  { value: "", label: "(auto)" },
  { value: "mass_retail", label: "Mass retail" },
  { value: "retail_plus_dtc", label: "Retail + DTC" },
  { value: "dtc_led", label: "DTC-led" },
  { value: "marketplace_led", label: "Marketplace-led" },
  { value: "subscription_led", label: "Subscription-led" },
];

interface Props {
  initial: FormState;
  onSubmit: (payload: ProductPayload) => void;
  loading: boolean;
}

export function ProductForm({ initial, onSubmit, loading }: Props) {
  const [form, setForm] = useState<FormState>(initial);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [interestDraft, setInterestDraft] = useState("");
  const [currentSection, setCurrentSection] = useState(1);
  const [error, setError] = useState("");
  const interestInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setForm(initial);
    setError("");
  }, [initial]);

  const update = <K extends keyof FormState>(k: K, v: FormState[K]) => {
    setForm((f) => ({ ...f, [k]: v }));
  };

  const toggleAge = (v: string) => {
    const arr = form.age_ranges.includes(v)
      ? form.age_ranges.filter((x) => x !== v)
      : [...form.age_ranges, v];
    update("age_ranges", arr);
  };
  const toggleIncome = (v: string) => {
    const arr = form.income_ranges.includes(v)
      ? form.income_ranges.filter((x) => x !== v)
      : [...form.income_ranges, v];
    update("income_ranges", arr);
  };

  const addInterest = () => {
    const clean = interestDraft.trim().replace(/,$/, "");
    if (clean && !form.interests.includes(clean)) {
      update("interests", [...form.interests, clean]);
    }
    setInterestDraft("");
  };
  const removeInterest = (i: number) => {
    update("interests", form.interests.filter((_, idx) => idx !== i));
  };

  const updateCompetitor = (i: number, name: string) => {
    const next = form.competitors.map((c, idx) => (idx === i ? { name } : c));
    update("competitors", next);
  };

  const buildDemographic = (): string => {
    const parts: string[] = [];
    if (form.age_ranges.length) parts.push(`Age: ${form.age_ranges.join(", ")}`);
    if (form.gender && form.gender !== "all") parts.push(`Gender focus: ${form.gender}`);
    if (form.income_ranges.length) parts.push(`Income: ${form.income_ranges.join(", ")}`);
    if (form.interests.length) parts.push(`Interests: ${form.interests.join(", ")}`);
    if (form.customer_context.trim()) parts.push(form.customer_context.trim());
    return parts.filter(Boolean).join("; ");
  };

  const isValid =
    form.product_name.trim().length > 1 &&
    form.description.trim().length > 1 &&
    parseFloat(form.price) > 0;

  const handleSubmit = () => {
    if (!isValid) {
      setError("Fill in product name, description, and price to launch.");
      return;
    }
    setError("");
    const competitors = form.competitors.filter((c) => c.name.trim()).map((c) => ({ name: c.name.trim() }));
    const demographic = buildDemographic() || "General consumers";

    const payload: ProductPayload = {
      name: form.product_name.trim(),
      product_name: form.product_name.trim(),
      description: form.description.trim(),
      price: parseFloat(form.price),
      category: form.category || "default",
      demographic,
      competitors,
    };
    if (form.market_tier_override) payload.market_tier_override = form.market_tier_override;
    if (form.distribution_hint) payload.distribution_hint = form.distribution_hint;
    if (form.exclude_brand.trim()) payload.exclude_brand = form.exclude_brand.trim();
    onSubmit(payload);
  };

  return (
    <section className="launch-section fade-up fade-up-2">
      <div className="launch-card">
        <div className="launch-header">
          <span className="mono" style={{ fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-muted)" }}>
            Market Simulation Setup
          </span>
          <div className="launch-step-pills">
            <span className={`step-pill ${currentSection >= 1 ? "active" : ""}`}>01 Product</span>
            <span className="step-divider">→</span>
            <span className={`step-pill ${currentSection >= 2 ? "active" : ""}`}>02 Customer</span>
            <span className="step-divider">→</span>
            <span className={`step-pill ${currentSection >= 3 ? "active" : ""}`}>03 Competitors</span>
          </div>
        </div>

        <div className="divider" style={{ margin: "16px 0" }} />

        {/* SECTION 1: PRODUCT */}
        <div className="form-section">
          <div className="section-label">
            <span className="section-num mono">01</span>
            <span className="section-title display">YOUR PRODUCT</span>
          </div>

          <div className="form-group">
            <label className="form-label">Product Name</label>
            <input
              className="input"
              value={form.product_name}
              onChange={(e) => update("product_name", e.target.value)}
              onFocus={() => setCurrentSection(1)}
              placeholder="e.g. Nova Ring"
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label className="form-label">
              Product Description
              <span className="form-label-hint">What makes it different?</span>
            </label>
            <textarea
              className="textarea"
              rows={4}
              value={form.description}
              onChange={(e) => update("description", e.target.value)}
              onFocus={() => setCurrentSection(1)}
              placeholder="Describe your product, key features, what problem it solves, and what makes it different."
              disabled={loading}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">
                Price (USD)
                <span className="accent mono">${form.price || "—"}</span>
              </label>
              <div className="price-input-wrap">
                <span className="price-symbol">$</span>
                <input
                  type="number"
                  className="input price-input"
                  value={form.price}
                  onChange={(e) => update("price", e.target.value)}
                  onFocus={() => setCurrentSection(1)}
                  placeholder="49"
                  min="1"
                  disabled={loading}
                />
              </div>
              <p className="field-hint">Counterfactuals will test ±15% price variants</p>
            </div>

            <div className="form-group">
              <label className="form-label">Product Category</label>
              <select
                className="select"
                value={form.category}
                onChange={(e) => update("category", e.target.value)}
                onFocus={() => setCurrentSection(1)}
                disabled={loading}
              >
                {CATEGORIES.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
              </select>
            </div>
          </div>
        </div>

        <div className="divider section-divider" />

        {/* SECTION 2: CUSTOMER */}
        <div className="form-section">
          <div className="section-label">
            <span className="section-num mono">02</span>
            <span className="section-title display">YOUR CUSTOMER</span>
          </div>

          <div className="form-group">
            <label className="form-label">
              Age Range
              <span className="form-label-hint">select all that apply</span>
            </label>
            <div className="chip-group">
              {AGE_RANGES.map((r) => (
                <button
                  key={r.value}
                  type="button"
                  className={`chip ${form.age_ranges.includes(r.value) ? "selected" : ""}`}
                  onClick={() => { toggleAge(r.value); setCurrentSection(2); }}
                  disabled={loading}
                >{r.label}</button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Gender Focus</label>
            <div className="chip-group">
              {GENDERS.map((g) => (
                <button
                  key={g.value}
                  type="button"
                  className={`chip ${form.gender === g.value ? "selected" : ""}`}
                  onClick={() => { update("gender", g.value); setCurrentSection(2); }}
                  disabled={loading}
                >{g.label}</button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Household Income
              <span className="form-label-hint">select all that apply</span>
            </label>
            <div className="chip-group">
              {INCOME_RANGES.map((i) => (
                <button
                  key={i.value}
                  type="button"
                  className={`chip ${form.income_ranges.includes(i.value) ? "selected" : ""}`}
                  onClick={() => { toggleIncome(i.value); setCurrentSection(2); }}
                  disabled={loading}
                >{i.label}</button>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Customer Interests & Lifestyle
              <span className="form-label-hint">press Enter to add</span>
            </label>
            <div className="tag-input-wrap" onClick={() => interestInputRef.current?.focus()}>
              {form.interests.map((tag, i) => (
                <span key={i} className="tag-pill">
                  {tag}
                  <button type="button" className="tag-remove" onClick={(e) => { e.stopPropagation(); removeInterest(i); }} disabled={loading}>×</button>
                </span>
              ))}
              <input
                ref={interestInputRef}
                className="tag-input"
                value={interestDraft}
                onChange={(e) => setInterestDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addInterest(); }
                }}
                onFocus={() => setCurrentSection(2)}
                placeholder="e.g. clean beauty, plant-based, fitness..."
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Additional Context
              <span className="form-label-hint">optional — anything else about your customer</span>
            </label>
            <textarea
              className="textarea context-area"
              rows={3}
              value={form.customer_context}
              onChange={(e) => update("customer_context", e.target.value.slice(0, 500))}
              onFocus={() => setCurrentSection(2)}
              placeholder="e.g. Has tried 3+ skincare brands. Shops Amazon Prime. Trusts peer reviews over brand claims."
              disabled={loading}
            />
            <div className="context-counter">{form.customer_context.length}/500</div>
          </div>
        </div>

        <div className="divider section-divider" />

        {/* SECTION 3: COMPETITORS */}
        <div className="form-section">
          <div className="section-label">
            <span className="section-num mono">03</span>
            <span className="section-title display">YOUR COMPETITORS</span>
            <span className="section-badge">Used as anchor seeds for retrieval</span>
          </div>

          <div className="competitor-grid">
            {form.competitors.map((c, i) => (
              <div key={i} className={`competitor-slot ${c.name ? "filled" : ""}`}>
                <div className="competitor-num">{String(i + 1).padStart(2, "0")}</div>
                <div className="competitor-inputs">
                  <input
                    className="input"
                    value={c.name}
                    onChange={(e) => updateCompetitor(i, e.target.value)}
                    onFocus={() => setCurrentSection(3)}
                    placeholder={`Competitor ${i + 1} name`}
                    disabled={loading}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="field-hint" style={{ marginTop: 12 }}>
            Optional. v3-lite will still anchor on the closest comparable brands in the database
            even without explicit competitors.
          </p>
        </div>

        <div className="divider section-divider" />

        {/* SECTION 4: SIMULATION */}
        <div className="form-section">
          <div className="section-label">
            <span className="section-num mono">04</span>
            <span className="section-title display">SIMULATION</span>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">
                Buyer Agents
                <span className="accent mono">{form.num_agents}</span>
              </label>
              <input
                type="range"
                className="range"
                min={20}
                max={50}
                step={30}
                value={form.num_agents}
                onChange={(e) => update("num_agents", parseInt(e.target.value, 10))}
                disabled={loading}
              />
              <div className="range-labels"><span>20 — faster, recommended</span><span>50 — broader coverage</span></div>
              <p className="field-hint">{form.num_agents}-agent panel will discuss the forecast across 3 rounds.</p>
            </div>

            <div className="form-group">
              <label className="form-label">
                Debate Rounds
                <span className="muted mono" style={{ fontSize: 9 }}>locked at 3</span>
              </label>
              <div className="locked-rounds">
                <div className="round-badge active">Round 1 — First Reaction</div>
                <div className="round-badge active">Round 2 — Comparable Comparison</div>
                <div className="round-badge active">Round 3 — Consensus</div>
              </div>
            </div>
          </div>
        </div>

        <button
          type="button"
          className="advanced-toggle"
          onClick={() => setShowAdvanced((s) => !s)}
        >
          {showAdvanced ? "▼ Hide retrieval overrides" : "▶ Retrieval overrides (advanced)"}
        </button>

        {showAdvanced && (
          <div className="form-row" style={{ marginTop: 16 }}>
            <div className="form-group">
              <label className="form-label">Market tier override</label>
              <select
                className="select"
                value={form.market_tier_override}
                onChange={(e) => update("market_tier_override", e.target.value)}
                disabled={loading}
              >
                {MARKET_TIERS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Distribution hint</label>
              <select
                className="select"
                value={form.distribution_hint}
                onChange={(e) => update("distribution_hint", e.target.value)}
                disabled={loading}
              >
                {DISTRIBUTION_HINTS.map((d) => <option key={d.value} value={d.value}>{d.label}</option>)}
              </select>
            </div>
            <div className="form-group" style={{ gridColumn: "1 / -1" }}>
              <label className="form-label">
                Exclude brand from comparables
                <span className="form-label-hint">used in synthetic-product demos</span>
              </label>
              <input
                className="input"
                value={form.exclude_brand}
                onChange={(e) => update("exclude_brand", e.target.value)}
                placeholder="(blank for normal forecast)"
                disabled={loading}
              />
            </div>
          </div>
        )}

        {error && <p className="error-msg" style={{ marginTop: 16 }}>⚠ {error}</p>}

        <button
          type="button"
          className="btn btn-primary launch-btn"
          disabled={!isValid || loading}
          onClick={handleSubmit}
        >
          {loading ? <span className="spinner" /> : <span>▶</span>}
          {loading ? "Running market forecast..." : "Run Market Forecast"}
        </button>

        <p className="launch-hint">
          v3-lite forecast runs in ~2-4s. Anchored on real comparable brands and their measured trial rates.
        </p>
      </div>
    </section>
  );
}
