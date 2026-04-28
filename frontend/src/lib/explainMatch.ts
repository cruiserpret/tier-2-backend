// Frontend-only translator: turns backend debug strings into plain English
// for customer-facing report cards. Backend math is untouched.

interface MatchSignal {
  kind: "tier" | "scale" | "price" | "dist" | "subtype";
  detail: string;
  weight: number;
}

const KIND_PHRASE: Record<MatchSignal["kind"], string> = {
  tier: "market tier",
  scale: "company scale",
  price: "price band",
  dist: "distribution model",
  subtype: "product subtype",
};

const DETAIL_PHRASE: Record<string, string> = {
  challenger: "challenger brands",
  growth_challenger: "growth-stage DTC",
  venture_challenger: "early-stage DTC",
  mass_market: "mass-market brands",
  mass_platform: "mass platform brands",
  premium_niche: "premium niche brands",
  niche: "niche brands",
  luxury: "luxury brands",
  large_public: "large public brands",
  global_giant: "global category leaders",
  niche_private: "niche private brands",
  early_stage: "early-stage brands",
  mass_retail: "mass-retail distribution",
  retail_plus_dtc: "retail-plus-DTC distribution",
  dtc_led: "DTC-led distribution",
  marketplace_led: "marketplace-led distribution",
  subscription_led: "subscription-led distribution",
  hydration_supplement: "hydration supplements",
  functional_soda: "functional sodas",
  nonalcoholic_beer: "non-alcoholic beer",
  coffee_alternative: "coffee alternatives",
  greens_powder: "greens powders",
  branded_water: "branded water",
  premium_drinkware: "premium drinkware",
  functional_fermented: "functional fermented drinks",
  bedding_premium: "premium bedding",
  mattress: "mattresses",
  wearable_health: "health wearables",
  smart_sleep: "sleep wearables",
  fitness_tracker: "fitness trackers",
};

function parseMatchReason(reason: string): MatchSignal[] {
  if (!reason || reason === "match") return [];
  const tokens = reason.split(",").map((t) => t.trim()).filter(Boolean);
  const signals: MatchSignal[] = [];
  for (const tok of tokens) {
    const m = tok.match(/^(tier|scale|price|dist|subtype)(?:\(([^)]+)\))?×([\d.]+)$/);
    if (!m) continue;
    const kind = m[1] as MatchSignal["kind"];
    const detail = m[2] || "";
    const weight = parseFloat(m[3]);
    signals.push({ kind, detail, weight });
  }
  return signals;
}

export function explainMatchReason(reason: string): string {
  const signals = parseMatchReason(reason);
  if (signals.length === 0) {
    if (reason && reason !== "match") return reason;
    return "Strong subtype match across price, scale, and distribution.";
  }

  // Sort by weight ascending (lowest weight = most penalized = most informative)
  const sorted = [...signals].sort((a, b) => a.weight - b.weight);
  const strongMatches = signals.filter((s) => s.weight >= 0.85);
  const weakerMatches = signals.filter((s) => s.weight < 0.85 && s.weight >= 0.6);
  const heavyPenalties = signals.filter((s) => s.weight < 0.6);

  const parts: string[] = [];

  if (strongMatches.length > 0) {
    const phrases = strongMatches.map((s) => KIND_PHRASE[s.kind]);
    parts.push(`Strong on ${joinList(phrases)}`);
  }

  if (weakerMatches.length > 0) {
    const phrases = weakerMatches.map((s) =>
      s.detail && DETAIL_PHRASE[s.detail]
        ? `${KIND_PHRASE[s.kind]} (${DETAIL_PHRASE[s.detail]})`
        : KIND_PHRASE[s.kind]
    );
    parts.push(`Partial fit on ${joinList(phrases)}`);
  }

  if (heavyPenalties.length > 0) {
    const phrases = heavyPenalties.map((s) =>
      s.detail && DETAIL_PHRASE[s.detail]
        ? `${KIND_PHRASE[s.kind]} differs (${DETAIL_PHRASE[s.detail]})`
        : `${KIND_PHRASE[s.kind]} differs`
    );
    parts.push(joinList(phrases));
  }

  return parts.join("; ") + ".";
}

export function explainPenaltyReason(reason: string): string {
  const signals = parseMatchReason(reason);
  if (signals.length === 0) return "Different adoption pattern.";
  const heavy = signals.filter((s) => s.weight < 0.6);
  if (heavy.length > 0) {
    const phrases = heavy.map((s) =>
      s.detail && DETAIL_PHRASE[s.detail]
        ? `${KIND_PHRASE[s.kind]} (${DETAIL_PHRASE[s.detail]})`
        : KIND_PHRASE[s.kind]
    );
    return `Different ${joinList(phrases)} than your product.`;
  }
  return "Same broad category but different product subtype.";
}

function joinList(items: string[]): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return items.join(" and ");
  return items.slice(0, -1).join(", ") + ", and " + items[items.length - 1];
}

// Translate raw confidence_reasons strings into customer-friendly version
export function explainConfidenceReason(raw: string): string {
  const r = raw.toLowerCase();

  if (r.includes("zero eligible") && r.includes("forced to low")) {
    return "No comparable brands cleared the quality threshold — confidence forced to low.";
  }
  if (r.includes("no strong same-subtype") || r.includes("fallback prior")) {
    return "Assembly didn't find enough direct comparables in your product subtype, so the forecast uses broader market references. Treat as directional.";
  }
  if (r.includes("capped at medium") && r.includes("subtype weight")) {
    const m = raw.match(/(\d+)%/);
    return m
      ? `Less than ${m[1]} of forecast weight came from exact-subtype matches — capped at medium confidence.`
      : "Most forecast weight came from approximate matches, not exact-subtype hits.";
  }
  if (r.includes("capped at medium") && r.includes("variance")) {
    const m = raw.match(/([\d.]+)pp/);
    return m
      ? `Trial rates among comparable brands vary by ${m[1]} percentage points — capped at medium confidence.`
      : "Comparable brands vary widely in trial rates — capped at medium confidence.";
  }
  if (r.includes("variance") || r.includes("vary widely")) {
    return "Comparable brand trial rates vary widely — true outcome depends on distribution and positioning.";
  }
  if (r.includes("weight share")) {
    return "Some retrieved comparables are similar but not identical in adoption pattern.";
  }
  if (r.includes("market tier")) {
    return "Comparable brands span different market tiers — mass-retail distribution could meaningfully shift this forecast.";
  }
  if (r.includes("limited coverage") || r.includes("exploratory")) {
    return "Comparable database has limited coverage for this product subtype — forecast is directional.";
  }

  // Pass through if we don't recognize it
  return raw;
}

export function fallbackExplanation(): string {
  return "Assembly didn't find enough direct comparables for this product. The forecast uses broader market references and should be treated as directional, not validated.";
}

export function lowCoverageHint(): string {
  return "This product appears to be in a thinner benchmark area. Assembly is giving a directional forecast until stronger same-subtype comparables are available.";
}
