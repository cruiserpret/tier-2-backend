"""
backend/dtc_v3/coverage_gate.py — v2 (effective coverage scoring)

Per friend's updated spec:
  Replace hard subtype-count gating with effective comparable coverage.
  Let ONE excellent subtype match qualify for strong coverage.
  Give partial credit for related subtype / adoption_curve / market_tier.
  Use 4 tiers: strong / medium / thin / weak.

Principle:
  "Coverage should measure the quality of available anchors,
   not just the count of exact subtype matches."
"""

from __future__ import annotations
import statistics
from .models import Neighbor, ProductBrief
from .ground_truth_db import GROUND_TRUTH_DB


# ═══════════════════════════════════════════════════════════════════════
# COMPARABLE CREDIT (per friend's exact spec)
# ═══════════════════════════════════════════════════════════════════════

# Related subtypes — same as the rerank spec, kept consistent
RELATED_SUBTYPES = {
    ("functional_soda", "hydration_supplement"),
    ("hydration_supplement", "functional_soda"),
    ("coffee_alternative", "greens_powder"),
    ("greens_powder", "coffee_alternative"),
    ("premium_basics", "athletic_apparel"),
    ("athletic_apparel", "premium_basics"),
    ("bedding_premium", "mattress"),
    ("mattress", "bedding_premium"),
    ("wearable_health", "fitness_tracker"),
    ("fitness_tracker", "wearable_health"),
    ("functional_soda", "branded_water"),
    ("branded_water", "functional_soda"),
    ("functional_soda", "functional_fermented"),
    ("functional_fermented", "functional_soda"),
}


def _is_related_subtype(a: str, b: str) -> bool:
    return (a, b) in RELATED_SUBTYPES


def _comparable_credit(
    query_subtype: str,
    query_category: str,
    query_market_tier: str,
    neighbor_subtype: str,
    neighbor_category: str,
    neighbor_market_tier: str,
) -> float:
    """
    Per friend's exact spec:
      exact same subtype:        1.00 credit
      related subtype:           0.60 credit
      same category + same tier: 0.30 credit
      same category only:        0.15 credit
      different category:        0.00 credit
    """
    if neighbor_subtype == query_subtype and query_subtype != "default":
        return 1.0

    if _is_related_subtype(query_subtype, neighbor_subtype):
        return 0.6

    if neighbor_category == query_category and neighbor_market_tier == query_market_tier:
        return 0.3

    if neighbor_category == query_category:
        return 0.15

    return 0.0


# ═══════════════════════════════════════════════════════════════════════
# COVERAGE TIER RULES (per friend's exact spec)
# ═══════════════════════════════════════════════════════════════════════

# Adjustment clamps by tier (logit space)
MAX_ADJUSTMENT = {
    "strong": 0.20,
    "medium": 0.10,
    "thin":   0.00,
    "weak":   0.00,
}


def _build_lookup_maps():
    """Map brand → (subtype, category, market_tier) for fast neighbor enrichment."""
    return {
        r.brand: (r.category_subtype, r.category, r.market_tier)
        for r in GROUND_TRUTH_DB
    }


def _count_subtype_in_db(subtype: str) -> int:
    return sum(1 for r in GROUND_TRUTH_DB if r.category_subtype == subtype)


def assess_coverage(
    neighbors: list[Neighbor],
    query_subtype: str,
    query_category: str,
    query_market_tier: str,
) -> dict:
    """
    Compute effective coverage score and tier.

    Returns:
        {
            "tier": "strong" | "medium" | "thin" | "weak",
            "score": float,                   # composite, informational
            "max_adjustment": float,          # logit cap for persona
            "reason": str,                    # human-readable
            "blend_with_fallback": float,     # 0-1, weight on fallback prior
            "diagnostics": {...}
        }
    """
    if not neighbors:
        return {
            "tier": "weak",
            "score": 0.0,
            "max_adjustment": MAX_ADJUSTMENT["weak"],
            "reason": "no neighbors retrieved",
            "blend_with_fallback": 1.0,
            "diagnostics": {},
        }

    db_lookup = _build_lookup_maps()

    # Enrich neighbors with structural fields
    credits = []
    weights = []
    same_subtype_sims = []
    best_anchor_score = 0.0

    for n in neighbors:
        n_subtype, n_category, n_market_tier = db_lookup.get(
            n.brand, ("default", "unknown", "challenger")
        )
        credit = _comparable_credit(
            query_subtype, query_category, query_market_tier,
            n_subtype, n_category, n_market_tier,
        )
        credits.append(credit)
        weights.append(n.source_weight)

        if n_subtype == query_subtype and query_subtype != "default":
            same_subtype_sims.append(n.similarity)

        anchor = n.similarity * credit
        if anchor > best_anchor_score:
            best_anchor_score = anchor

    # Effective coverage — weighted by forecast weight
    total_weight = sum(weights)
    if total_weight > 0:
        effective_coverage = sum(c * w for c, w in zip(credits, weights)) / total_weight
    else:
        effective_coverage = 0.0

    # Best exact subtype similarity
    best_exact_subtype_sim = max(same_subtype_sims) if same_subtype_sims else 0.0

    # DB density for this subtype
    available_subtype_count = _count_subtype_in_db(query_subtype)

    # ── Tier rules per friend's spec ──
    # Strong: best exact subtype sim ≥ 0.65 (one excellent anchor is enough)
    # Medium: effective_coverage ≥ 0.45 AND best_anchor_score ≥ 0.35
    # Thin:   effective_coverage ≥ 0.25
    # Weak:   below thin

    if best_exact_subtype_sim >= 0.65:
        tier = "strong"
        reason = f"excellent same-subtype anchor (sim={best_exact_subtype_sim:.2f})"
    elif effective_coverage >= 0.45 and best_anchor_score >= 0.35:
        tier = "medium"
        reason = f"effective coverage {effective_coverage:.2f}, best anchor {best_anchor_score:.2f}"
    elif effective_coverage >= 0.25:
        tier = "thin"
        reason = f"thin coverage {effective_coverage:.2f}, no strong anchor"
    else:
        tier = "weak"
        reason = f"weak coverage {effective_coverage:.2f}"

    # Blend with fallback per friend's spec:
    # Strong: rely fully on RAG (blend=0)
    # Medium: rely fully on RAG (blend=0)
    # Thin:   65% RAG + 35% fallback (blend=0.35)
    # Weak:   100% fallback (blend=1.0)
    blend_table = {"strong": 0.0, "medium": 0.0, "thin": 0.35, "weak": 1.0}

    # ── Confidence quality (per friend's "trust layer" advice) ──
    # Computed independently of tier. Asks: do these neighbors form a tight,
    # structurally appropriate forecast neighborhood?
    if total_weight > 0:
        same_subtype_weight = sum(
            n.source_weight for n in neighbors
            if db_lookup.get(n.brand, ("default", "", ""))[0] == query_subtype
            and query_subtype != "default"
        )
        exact_subtype_weight_share = same_subtype_weight / total_weight
    else:
        exact_subtype_weight_share = 0.0

    # Trial rate variance among neighbors
    if len(neighbors) > 1:
        rates = [n.trial_rate_mid for n in neighbors]
        rate_std = statistics.stdev(rates)
    else:
        rate_std = 0.10  # conservative for single-neighbor case

    # Market tier match share
    tier_matches = sum(
        n.source_weight for n in neighbors
        if db_lookup.get(n.brand, ("", "", ""))[2] == query_market_tier
    )
    market_tier_match_share = tier_matches / total_weight if total_weight > 0 else 0.0

    # Quality-aware confidence label (overrides naive tier-based label)
    confidence_reasons = []

    if (exact_subtype_weight_share >= 0.65
        and best_exact_subtype_sim >= 0.60
        and rate_std <= 0.06
        and market_tier_match_share >= 0.50):
        confidence = "high"
    elif (exact_subtype_weight_share >= 0.40
          and best_exact_subtype_sim >= 0.50
          and rate_std <= 0.10):
        confidence = "medium-high"
        if exact_subtype_weight_share < 0.65:
            confidence_reasons.append(f"exact-subtype weight share {exact_subtype_weight_share:.0%} (high needs ≥65%)")
        if rate_std > 0.06:
            confidence_reasons.append(f"neighbor trial rate variance {rate_std*100:.1f}pp (high needs ≤6pp)")
    elif tier in ("strong", "medium"):
        confidence = "medium"
        confidence_reasons.append(f"comparable mix has variance or partial mismatch")
    elif tier == "thin":
        confidence = "medium-low"
        confidence_reasons.append("thin coverage — relying partly on fallback prior")
    else:
        confidence = "low"
        confidence_reasons.append("no strong same-subtype anchor — using fallback prior")

    if market_tier_match_share < 0.30:
        confidence_reasons.append(f"only {market_tier_match_share:.0%} of neighbors match query's market tier")

    # ── HARD CAPS (per friend's non-negotiable rules) ──
    # Rule 1: Zero eligible comparables → confidence cannot exceed "low"
    n_eligible_at_floor = sum(1 for n in neighbors if n.similarity >= 0.45)
    if n_eligible_at_floor == 0:
        confidence = "low"
        confidence_reasons.append("zero eligible comparables — confidence forced to low")

    # Rule 2: Low exact-subtype weight share → cap at medium
    elif exact_subtype_weight_share < 0.50 and confidence in ("high", "medium-high"):
        confidence = "medium"
        confidence_reasons.append(f"capped at medium: subtype weight share {exact_subtype_weight_share:.0%} below 50%")

    # Rule 3: High neighbor variance → cap at medium
    elif rate_std > 0.08 and confidence in ("high", "medium-high"):
        confidence = "medium"
        confidence_reasons.append(f"capped at medium: neighbor variance {rate_std*100:.1f}pp exceeds 8pp")

    return {
        "tier": tier,
        "score": round(effective_coverage, 3),
        "max_adjustment": MAX_ADJUSTMENT[tier],
        "reason": reason,
        "blend_with_fallback": blend_table[tier],
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "diagnostics": {
            "effective_coverage": round(effective_coverage, 3),
            "best_anchor_score": round(best_anchor_score, 3),
            "best_exact_subtype_similarity": round(best_exact_subtype_sim, 3),
            "available_subtype_count": available_subtype_count,
            "n_neighbors": len(neighbors),
            "n_same_subtype": len(same_subtype_sims),
            "exact_subtype_weight_share": round(exact_subtype_weight_share, 3),
            "neighbor_rate_std": round(rate_std, 3),
            "market_tier_match_share": round(market_tier_match_share, 3),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# FALLBACK PRIOR CHAIN (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════

def compute_fallback_prior(
    query_subtype: str,
    query_market_tier: str,
    query_category: str,
) -> dict:
    """Walk fallback chain: subtype → market_tier → category → global."""
    db = GROUND_TRUTH_DB

    # 1. Same subtype
    candidates = [r for r in db if r.category_subtype == query_subtype]
    if len(candidates) >= 2:
        return {
            "prior": _weighted_median([r.trial_rate_mid for r in candidates],
                                       [r.source_weight for r in candidates]),
            "method": "subtype_median",
            "n_candidates": len(candidates),
            "level": 1,
        }

    # 2. Market tier (≥5)
    candidates = [r for r in db if r.market_tier == query_market_tier]
    if len(candidates) >= 5:
        return {
            "prior": _weighted_median([r.trial_rate_mid for r in candidates],
                                       [r.source_weight for r in candidates]),
            "method": "market_tier_median",
            "n_candidates": len(candidates),
            "level": 2,
        }

    # 3. Category
    candidates = [r for r in db if r.category == query_category]
    if len(candidates) >= 5:
        return {
            "prior": _weighted_median([r.trial_rate_mid for r in candidates],
                                       [r.source_weight for r in candidates]),
            "method": "category_median",
            "n_candidates": len(candidates),
            "level": 3,
        }

    # 4. Global
    return {
        "prior": _weighted_median([r.trial_rate_mid for r in db],
                                   [r.source_weight for r in db]),
        "method": "global_dtc_median",
        "n_candidates": len(db),
        "level": 4,
    }


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.08
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total = sum(weights)
    if total == 0:
        return statistics.median(values)
    target = total / 2
    cumulative = 0
    for v, w in pairs:
        cumulative += w
        if cumulative >= target:
            return v
    return pairs[-1][0]
