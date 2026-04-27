"""
backend/dtc_v3/forecast.py — v3-lite coverage-aware forecast orchestrator.

Per friend's final architecture:
  RAG prior owns the number.
  Personas earn the adjustment, but only when coverage is strong/medium.
  Fallback prior chain when coverage is thin/weak.
  Persona layer always provides explanation, even when not adjusting forecast.

Routing logic:
  strong:  forecast = sigmoid(logit(rag_prior) + clamped_persona_adj)  [adj cap ±0.20]
  medium:  forecast = sigmoid(logit(rag_prior) + clamped_persona_adj)  [adj cap ±0.10]
  thin:    forecast = 0.65*rag_prior + 0.35*fallback_prior, no persona adj
  weak:    forecast = fallback_prior, no persona adj
"""

from __future__ import annotations
import math
import json
from pathlib import Path

from .models import (
    ProductBrief, PersonaSignals, DataQuality, Forecast, Neighbor
)
from .rag_retrieval import (
    retrieve_neighbors, compute_rag_prior,
    _infer_query_subtype, _infer_query_market_structure,
)
from .persona_signals import (
    apply_persona_adjustment, build_persona_signals,
    DESIRABILITY_MEAN, DESIRABILITY_SD,
    AWARENESS_MEAN, AWARENESS_SD,
    FRICTION_MEAN, FRICTION_SD,
)
from .coverage_gate import assess_coverage, compute_fallback_prior


# Load fitted beta coefficients
_BETA_PATH = Path(__file__).parent / "calibration" / "beta_coefficients_v1.json"


def _load_betas() -> dict:
    if _BETA_PATH.exists():
        try:
            data = json.loads(_BETA_PATH.read_text())
            return data["final_betas"]
        except Exception:
            pass
    # Conservative fallback if calibration missing
    return {"desirability": 0.05, "awareness": 0.05, "friction": 0.0}


_BETAS = _load_betas()


def logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ═══════════════════════════════════════════════════════════════════════
# COVERAGE-AWARE FORECAST (the main entry point)
# ═══════════════════════════════════════════════════════════════════════

def forecast(
    product: ProductBrief,
    debate_state: dict | None = None,
    market_intel: dict | None = None,
    exclude_brand: str | None = None,
) -> Forecast:
    """
    Main v3-lite forecast.

    If debate_state + market_intel provided → uses persona signals when coverage allows.
    If not provided → RAG-only mode (Mode 1).
    """
    # ── Stage 1: RAG retrieval (returns ALL candidates, no floor filter) ──
    retrieved_candidates = retrieve_neighbors(product, k=8, exclude_brand=exclude_brand)

    # ── Stage 1b: Filter to eligible neighbors (per friend's spec) ──
    # Eligible = passes SIMILARITY_FLOOR. Only these anchor forecast math.
    from .rag_retrieval import get_eligible_neighbors, SIMILARITY_FLOOR
    eligible_neighbors = get_eligible_neighbors(retrieved_candidates)

    # Compute RAG prior ONLY if we have eligible neighbors
    if eligible_neighbors:
        rag_prior = compute_rag_prior(eligible_neighbors)
    else:
        rag_prior = None  # signal: no eligible RAG anchor

    # ── Stage 2: Coverage assessment (uses retrieved candidates for diagnostics) ──
    query_subtype = _infer_query_subtype(product)
    q_tier, q_scale, q_dist, q_band = _infer_query_market_structure(product)
    coverage = assess_coverage(retrieved_candidates, query_subtype, product.category, q_tier)

    # ── Stage 3: Determine prior with HONEST labeling ──
    # Per friend's spec: never label fallback as RAG. Track what was used.
    MIN_ELIGIBLE_FOR_RAG = 2
    fallback_used = False
    fallback_method = None
    fallback_prior_value = None

    # Compute fallback whenever we might need it
    if (rag_prior is None
        or len(eligible_neighbors) < MIN_ELIGIBLE_FOR_RAG
        or coverage["tier"] in ("weak", "thin")):
        fb = compute_fallback_prior(query_subtype, q_tier, product.category)
        fallback_prior_value = fb["prior"]
        fallback_method = fb["method"]

    if rag_prior is None or len(eligible_neighbors) < MIN_ELIGIBLE_FOR_RAG:
        # No RAG possible — pure fallback
        effective_prior = fallback_prior_value
        prior_source = f"fallback_{fallback_method}"
        fallback_used = True
        # Override coverage tier to weak if not already
        if coverage["tier"] not in ("weak",):
            coverage["tier"] = "weak"
            coverage["max_adjustment"] = 0.0
    elif coverage["tier"] == "weak":
        effective_prior = fallback_prior_value
        prior_source = f"fallback_{fallback_method}"
        fallback_used = True
    elif coverage["tier"] == "thin":
        effective_prior = 0.65 * rag_prior + 0.35 * fallback_prior_value
        prior_source = f"blended_rag_and_{fallback_method}"
        fallback_used = True
    else:
        # strong/medium: pure RAG (rag_prior guaranteed not None here)
        effective_prior = rag_prior
        prior_source = "rag_weighted_median"
        fallback_used = False

    # INVARIANT (per friend's spec): RAG prior cannot come from 0 eligible
    if prior_source == "rag_weighted_median":
        assert len(eligible_neighbors) >= MIN_ELIGIBLE_FOR_RAG, \
            f"RAG prior labeled but only {len(eligible_neighbors)} eligible neighbors"

    # Use retrieved_candidates as the "neighbors" downstream (with full transparency)
    # but the math came from eligible only.
    neighbors = retrieved_candidates

    # ── Stage 4: Persona adjustment (only if strong/medium AND we have data) ──
    signals = PersonaSignals()
    adjustment = 0.0
    persona_explanation = "no debate data provided"

    if debate_state and market_intel:
        signals = build_persona_signals(debate_state, market_intel, product)
        if coverage["tier"] in ("strong", "medium"):
            # Compute adjustment using fitted betas
            raw_adj = (
                _BETAS["desirability"] * signals.desirability_z
                + _BETAS["awareness"] * signals.awareness_z
                + _BETAS["friction"] * signals.friction_z
            )
            # Apply coverage-tier clamp (much tighter than original ±0.75)
            cap = coverage["max_adjustment"]
            adjustment = max(-cap, min(cap, raw_adj))
            persona_explanation = (
                f"raw_adj={raw_adj:+.3f}, clamped to ±{cap}, applied={adjustment:+.3f}"
            )
        else:
            persona_explanation = (
                f"signals computed but NOT applied (coverage={coverage['tier']})"
            )

    # ── Stage 5: Final prediction ──
    final_logit_val = logit(effective_prior) + adjustment
    final = sigmoid(final_logit_val)

    # ── Stage 6: Confidence interval (weighted variance + adjustment + coverage) ──
    # Variance is computed around effective_prior — the prior actually used
    # for the forecast, which may be RAG, blended, or fallback.
    if neighbors and len(neighbors) > 1:
        weights = [n.source_weight for n in neighbors]
        total_w = sum(weights)
        if total_w > 0:
            weighted_var = sum(
                w * (n.trial_rate_mid - effective_prior)**2
                for n, w in zip(neighbors, weights)
            ) / total_w
            base_hw = max(0.015, min(0.05, weighted_var ** 0.5 * 1.2))
        else:
            base_hw = 0.025
    else:
        base_hw = 0.05

    # Widen interval based on coverage tier
    coverage_hw_multiplier = {
        "strong": 1.0, "medium": 1.3, "thin": 1.7, "weak": 2.5,
    }
    half_width = base_hw * coverage_hw_multiplier[coverage["tier"]]
    half_width = min(0.10, half_width)  # cap to avoid silly-wide intervals

    # ── Stage 7: Confidence label (quality-aware, per friend's spec) ──
    # Use the assess_coverage's confidence field — it considers neighbor variance,
    # market_tier match share, and exact-subtype weight share, not just tier.
    confidence_label = coverage.get("confidence", "medium")

    # Build DataQuality with confidence reasons
    dq = DataQuality(
        quality_score=coverage["score"],
        quality_warnings=coverage.get("confidence_reasons", []),
    )

    return Forecast(
        trial_rate_median=final,
        trial_rate_low=max(0.001, final - half_width),
        trial_rate_high=min(0.99, final + half_width),
        confidence=confidence_label,
        neighbors=neighbors,
        persona_signals=signals,
        data_quality=dq,
        rag_prior=rag_prior if rag_prior is not None else 0.0,
        adjustment_applied=adjustment,
        prior_source=prior_source,
        fallback_used=fallback_used,
        eligible_neighbor_count=len(eligible_neighbors),
        retrieved_candidate_count=len(retrieved_candidates),
    )


# ═══════════════════════════════════════════════════════════════════════
# LEGACY MODE WRAPPERS (for eval harness compatibility)
# ═══════════════════════════════════════════════════════════════════════

def forecast_rag_only(product: ProductBrief, exclude_brand: str | None = None) -> Forecast:
    """Mode 1: RAG-only — no persona adjustment regardless of coverage."""
    return forecast(product, debate_state=None, market_intel=None, exclude_brand=exclude_brand)


def forecast_rag_plus_persona(
    product: ProductBrief,
    debate_state: dict,
    market_intel: dict,
    exclude_brand: str | None = None,
) -> Forecast:
    """Mode 2: RAG + persona adjustment (when coverage allows)."""
    return forecast(product, debate_state=debate_state, market_intel=market_intel,
                    exclude_brand=exclude_brand)


def explain_forecast(f: Forecast) -> str:
    """Human-readable forecast explanation."""
    lines = []
    lines.append(f"Predicted 12-month trial rate: {f.trial_rate_median*100:.1f}%")
    lines.append(f"Likely range: {f.trial_rate_low*100:.1f}% – {f.trial_rate_high*100:.1f}%")
    lines.append(f"Confidence: {f.confidence}")
    if f.data_quality.quality_warnings:
        lines.append(f"Why this confidence:")
        for r in f.data_quality.quality_warnings:
            lines.append(f"  • {r}")
    lines.append("")
    # Honest prior labeling per friend's spec
    if f.fallback_used:
        lines.append(f"Forecast method: {f.prior_source}")
        lines.append(f"Eligible comparables: {f.eligible_neighbor_count} (out of {f.retrieved_candidate_count} retrieved)")
        if f.eligible_neighbor_count == 0:
            lines.append(f"⚠ No comparables passed quality threshold — using fallback prior")
    else:
        lines.append(f"RAG prior: {f.rag_prior*100:.1f}% (from {f.eligible_neighbor_count} eligible comparables)")
    if abs(f.adjustment_applied) > 0.001:
        direction = "lifted" if f.adjustment_applied > 0 else "lowered"
        lines.append(f"Persona simulation {direction} prediction by {abs(f.adjustment_applied):.3f} logit points")
    lines.append("")
    lines.append("Top comparables:")
    for n in f.neighbors[:5]:
        lines.append(f"  • {n.brand:<35} GT={n.trial_rate_mid*100:.1f}% conf={n.confidence}")
    return "\n".join(lines)
