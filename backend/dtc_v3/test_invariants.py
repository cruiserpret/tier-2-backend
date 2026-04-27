"""
Invariant tests for v3-lite forecast pipeline.

These tests guard the trust contract:
  1. No fallback forecast is ever labeled as RAG
  2. Zero-eligible forecasts always have low confidence
  3. Liquid IV forecast is deterministic across runs

Run from repo root:
    python -m pytest backend/dtc_v3/test_invariants.py -v
"""
import sys
from pathlib import Path

# Allow running this file from anywhere by ensuring repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.dtc_v3.models import ProductBrief
from backend.dtc_v3.forecast import forecast


# ─── Helper builders ─────────────────────────────────────────────────

def _yeti_brief() -> ProductBrief:
    """Zero-eligible case (no DB analogs for insulated drinkware)."""
    return ProductBrief(
        name="YETI Rambler 20oz Tumbler",
        description="Stainless steel insulated tumbler with magnetic slider lid.",
        price=42,
        category="home_lifestyle",
        demographic="Outdoor enthusiasts",
        competitors=[{"name": "Stanley"}],
    )


def _liquid_iv_brief() -> ProductBrief:
    """Strong-coverage RAG case (mass-market hydration)."""
    return ProductBrief(
        name="Liquid IV Hydration Multiplier",
        description="Electrolyte drink mix powder. Costco, Target, Walmart.",
        price=25,
        category="supplements_health",
        demographic="Active adults 22-50",
        competitors=[{"name": "LMNT"}, {"name": "Pedialyte"}],
        market_tier_override="mass_market",
        distribution_hint="mass_retail",
    )


# ─── Invariant 1: No fallback labeled as RAG ─────────────────────────

def test_no_fallback_labeled_as_rag():
    """
    When fallback fires (zero eligible), the forecast must NOT claim
    its prior came from RAG. This is the core trust invariant.
    """
    f = forecast(_yeti_brief(), exclude_brand="YETI")

    assert f.fallback_used is True, (
        "YETI should hit fallback path; got fallback_used=False"
    )
    assert f.prior_source != "rag_weighted_median", (
        f"Fallback forecast must NOT be labeled RAG; got prior_source={f.prior_source!r}"
    )
    assert f.prior_source.startswith("fallback_"), (
        f"Expected fallback_* prior_source; got {f.prior_source!r}"
    )
    assert f.eligible_neighbor_count == 0, (
        f"YETI should have zero eligible neighbors; got {f.eligible_neighbor_count}"
    )


# ─── Invariant 2: Zero eligible → low confidence ─────────────────────

def test_zero_eligible_confidence_low():
    """
    A forecast with zero eligible comparables cannot show medium or
    higher confidence. The hard cap must fire.
    """
    f = forecast(_yeti_brief(), exclude_brand="YETI")

    assert f.eligible_neighbor_count == 0, "precondition: YETI eligible=0"
    assert f.confidence == "low", (
        f"Zero-eligible forecast must be confidence=low; got {f.confidence!r}"
    )

    # Robust to wherever reasons are stored: top-level confidence_reasons
    # OR data_quality.quality_warnings (per friend's review).
    warnings = []
    if hasattr(f, "confidence_reasons"):
        warnings.extend(getattr(f, "confidence_reasons") or [])
    if hasattr(f, "data_quality") and hasattr(f.data_quality, "quality_warnings"):
        warnings.extend(f.data_quality.quality_warnings or [])

    reason_text = " ".join(warnings).lower()
    assert "zero eligible" in reason_text, (
        f"Expected 'zero eligible' reason in warnings; got {warnings!r}"
    )


# ─── Invariant 3: Deterministic Liquid IV ────────────────────────────

def test_deterministic_liquid_iv_5_runs():
    """
    Same input must produce same output across 5 runs. This is the
    YC demo moment: 22.0% × 5.
    """
    forecasts = [
        forecast(_liquid_iv_brief(), exclude_brand="Liquid IV")
        for _ in range(5)
    ]
    rates = [round(f.trial_rate_median * 100, 1) for f in forecasts]

    assert rates == [rates[0]] * 5, (
        f"Forecast not deterministic across 5 runs; got {rates}"
    )

    # Sanity: this case must use RAG, not fallback
    first = forecasts[0]
    assert first.fallback_used is False, "Liquid IV should use RAG, not fallback"
    assert first.prior_source == "rag_weighted_median", (
        f"Liquid IV expected rag_weighted_median; got {first.prior_source!r}"
    )
