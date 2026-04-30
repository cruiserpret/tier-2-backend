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


# ── Step 7 routing tests (energy_drink subtype) ────────────────────
def _energy_text_brief():
    return ProductBrief(
        name="Triton drinks",
        description="energy drink for athletes and gamers",
        price=3.99,
        category="food_beverage",
        demographic="young adults",
        competitors=[],
    )


def _energy_competitor_brief():
    return ProductBrief(
        name="Triton drinks",
        description="A new beverage",
        price=3.99,
        category="food_beverage",
        demographic="young adults",
        competitors=[{"name": "Red Bull"}, {"name": "Monster"}],
    )


def _liquid_death_brief():
    return ProductBrief(
        name="Liquid Death",
        description="Mountain water canned",
        price=2,
        category="food_beverage",
        demographic="adults",
        competitors=[{"name": "Aquafina"}, {"name": "Dasani"}],
    )


def _sobercraft_brief():
    return ProductBrief(
        name="SoberCraft IPA",
        description="Premium nonalcoholic craft IPA",
        price=14,
        category="food_beverage",
        demographic="adults",
        competitors=[{"name": "Athletic Brewing"}, {"name": "Heineken 0.0"}],
    )


def test_energy_drink_text_routes_to_energy_subtype():
    from backend.dtc_v3.rag_retrieval import _infer_query_subtype
    assert _infer_query_subtype(_energy_text_brief()) == "energy_drink"


def test_energy_drink_competitor_heuristic_routes_to_energy_subtype():
    from backend.dtc_v3.rag_retrieval import _infer_query_subtype
    assert _infer_query_subtype(_energy_competitor_brief()) == "energy_drink"


def test_liquid_iv_still_routes_to_hydration_supplement():
    from backend.dtc_v3.rag_retrieval import _infer_query_subtype
    assert _infer_query_subtype(_liquid_iv_brief()) == "hydration_supplement"


def test_water_brand_still_routes_to_branded_water():
    from backend.dtc_v3.rag_retrieval import _infer_query_subtype
    assert _infer_query_subtype(_liquid_death_brief()) == "branded_water"


def test_na_beer_still_routes_to_nonalcoholic_beer():
    from backend.dtc_v3.rag_retrieval import _infer_query_subtype
    assert _infer_query_subtype(_sobercraft_brief()) == "nonalcoholic_beer"


def test_triton_drinks_anchored_on_energy_drinks():
    """
    Per friend Step 7 spec: do NOT assert exact forecast number or confidence.
    Assert at least 2 energy-drink anchors appear, and Liquid IV invariant
    is unaffected.
    """
    triton = ProductBrief(
        name="Triton drinks",
        description="A new energy drink for the US market",
        price=3.99,
        category="food_beverage",
        demographic="active adults 18-35",
        competitors=[{"name": "Red Bull"}, {"name": "Monster"}, {"name": "Celsius"}],
    )
    f = forecast(triton)

    energy_brands = {
        "Red Bull", "Monster Energy", "Celsius", "C4 Energy",
        "Alani Nu Energy", "Ghost Energy", "G Fuel", "Prime Energy",
    }
    anchor_brands = {n.brand for n in f.neighbors}
    overlap = anchor_brands & energy_brands
    assert len(overlap) >= 2, (
        f"Triton Drinks should anchor on >=2 energy drink brands, got {sorted(overlap)} "
        f"(all neighbors: {sorted(anchor_brands)})"
    )
    assert not f.fallback_used or len(overlap) == 0, (
        "If energy drink anchors found, prior_source should not be fallback"
    )


# ─── Phase 1 invariants — added Apr 30 (P1.5) ────────────────────────

def test_discussion_response_preserves_forecast_core_clean_run():
    """Clean-run preservation: /discuss preserves trial_rate, confidence,
    and verdict in the response forecast.

    NOTE: This test does NOT verify in-place mutation of the input dict
    — that is impossible to test through Flask's test_client because
    the endpoint receives a deserialized copy of the JSON, not the
    same Python object. In-place mutation is verified separately by
    test_discuss_mutation_guard_catches_synthetic_mutation, which
    patches generate_discussion at its source module.
    """
    from backend.main import app
    c = app.test_client()

    fr = c.post('/api/dtc_v3/forecast', json={
        'product_name': 'Liquid IV Hydration Multiplier',
        'description': 'Electrolyte hydration drink mix',
        'price': 25,
        'category': 'supplements_health',
        'demographic': 'Active adults',
        'competitors': [{'name': 'LMNT'}],
        'exclude_brand': 'Liquid IV',
    }).get_json()

    captured_trial_rate = fr['trial_rate']
    captured_confidence = fr['confidence']
    captured_verdict = fr['verdict']

    r = c.post('/api/dtc_v3/discuss?mode=template', json={
        'product': {
            'product_name': 'Liquid IV Hydration Multiplier',
            'price': 25,
            'category': 'supplements_health',
            'demographic': 'Active adults',
            'competitors': [{'name': 'LMNT'}],
        },
        'forecast': fr,
        'agent_count': 20,
    })
    assert r.status_code == 200, f"discuss failed: {r.get_data()[:300]}"
    j = r.get_json()

    response_forecast = j.get('forecast') if isinstance(j, dict) else None
    if isinstance(response_forecast, dict):
        assert response_forecast.get('trial_rate') == captured_trial_rate, \
            "response trial_rate differs from forecast call"
        assert response_forecast.get('confidence') == captured_confidence, \
            "response confidence differs from forecast call"
        assert response_forecast.get('verdict') == captured_verdict, \
            "response verdict differs from forecast call"

    diag = (j.get('diagnostics') or {}) if isinstance(j, dict) else {}
    assert diag.get('forecast_mutation_blocked') is not True, \
        "clean run must NOT set forecast_mutation_blocked"


def test_discuss_mutation_guard_catches_synthetic_mutation():
    """Mutation guard (P1.3 api.py) catches malicious in-place + result
    mutation, restores captured fields, and sets the diagnostic flag.

    Patches backend.dtc_v3.discussion.generate_discussion (source module)
    per friend Apr 30 ruling. api.py uses lazy import inside the route
    (`from .discussion import generate_discussion` at line 259), so the
    function reads the attribute from the source module at call time.
    Patching the api module attribute would NOT intercept this.

    mock_gen.called assertion below proves the malicious function ran
    — without that, a wrong patch target would silently pass the test
    for unrelated reasons (false confidence).
    """
    from unittest.mock import patch
    from backend.main import app

    c = app.test_client()
    fr = c.post('/api/dtc_v3/forecast', json={
        'product_name': 'Liquid IV Hydration Multiplier',
        'description': 'Electrolyte hydration drink mix',
        'price': 25,
        'category': 'supplements_health',
        'demographic': 'Active adults',
        'competitors': [{'name': 'LMNT'}],
        'exclude_brand': 'Liquid IV',
    }).get_json()

    original_trial_rate = fr['trial_rate']
    original_confidence = fr['confidence']
    original_verdict = fr['verdict']

    def malicious_generate_discussion(product, forecast, agent_count, mode, comparison_context=None):
        """Mutates request-local forecast in place AND returns mutated forecast."""
        forecast['trial_rate'] = {
            'percentage': 99.9, 'median': 0.999, 'low': 0.0, 'high': 1.0,
        }
        forecast['confidence'] = 'high'
        forecast['verdict'] = 'launch_aggressively'
        return {
            'agent_panel': {'agents': []},
            'forecast': dict(forecast),
        }

    with patch('backend.dtc_v3.discussion.generate_discussion',
               side_effect=malicious_generate_discussion) as mock_gen:
        r = c.post('/api/dtc_v3/discuss?mode=template', json={
            'product': {
                'product_name': 'Liquid IV Hydration Multiplier',
                'price': 25,
                'category': 'supplements_health',
                'demographic': 'Active adults',
                'competitors': [{'name': 'LMNT'}],
            },
            'forecast': fr,
            'agent_count': 20,
        })

    assert mock_gen.called, "malicious generate_discussion mock was not used"
    assert r.status_code == 200, f"got {r.status_code}: {r.get_data()[:300]}"
    j = r.get_json()

    diag = (j.get('diagnostics') or {}) if isinstance(j, dict) else {}
    assert diag.get('forecast_mutation_blocked') is True, \
        "mutation guard must set forecast_mutation_blocked=True"

    restored = j.get('forecast') or {}
    assert restored.get('trial_rate') == original_trial_rate, \
        f"trial_rate must be restored, got {restored.get('trial_rate')}"
    assert restored.get('confidence') == original_confidence, \
        f"confidence must be restored, got {restored.get('confidence')}"
    assert restored.get('verdict') == original_verdict, \
        f"verdict must be restored, got {restored.get('verdict')}"


def test_banned_phrase_validator_rejects_batch():
    """LLM output validator (P1.4) rejects banned phrases anywhere in
    narrative fields, round responses, or top-level fields.

    This codifies the integration test passed during P1.4 verification.
    """
    from backend.dtc_v3.llm_dialogue_enricher import _validate_batch_response

    base_clean_agent = {
        'id': 'agent_01',
        'reason': 'I would consider buying this if the price drops',
        'top_objection': 'price seems high vs LMNT',
        'what_would_change_mind': 'a discount or trial pack',
        'key_quote': 'might try it',
        'round_responses': [
            {'round': 1, 'response': 'first impression — interesting product'},
            {'round': 2, 'response': 'I will compare this to LMNT carefully'},
            {'round': 3, 'response': 'final verdict — I would consider it'},
        ],
    }
    batch = [{'id': 'agent_01'}]

    bad_reason = {'agents': [{**base_clean_agent, 'reason': 'This is guaranteed to convert'}]}
    assert _validate_batch_response(bad_reason, batch) is None, \
        "banned phrase in reason must reject batch"

    bad_agent = dict(base_clean_agent)
    bad_agent['round_responses'] = [
        {'round': 1, 'response': 'first impression text here'},
        {'round': 2, 'response': 'These are real buyers reacting to LMNT'},
        {'round': 3, 'response': 'final verdict text here'},
    ]
    bad_round = {'agents': [bad_agent]}
    assert _validate_batch_response(bad_round, batch) is None, \
        "banned phrase in round response must reject batch"

    bad_top = {
        'agents': [base_clean_agent],
        'consensus': 'Customers will buy this product at scale',
    }
    assert _validate_batch_response(bad_top, batch) is None, \
        "banned phrase in top-level consensus must reject batch"

    clean = {'agents': [base_clean_agent]}
    result = _validate_batch_response(clean, batch)
    assert result is not None, "clean text must pass validator"



# ════════════════════════════════════════════════════════════════════
# Phase 1.8.1 — comparison_context tests
#
# Per friend Apr 30 ruling: synthetic evidence_buckets fixtures only.
# Do NOT depend on DB retrieval state. These tests verify the pure
# build_comparison_context() classifier in isolation.
#
# The hard invariant under test:
#   If a brand is not in allowed_comparison_brands, agents must not
#   mention it as a competitor or proof point.
# ════════════════════════════════════════════════════════════════════

from types import SimpleNamespace
from backend.dtc_v3.comparison_context import build_comparison_context


def _stub_forecast(fallback_used=False, confidence="medium"):
    """Minimal Forecast-shaped object for comparison_context tests."""
    return SimpleNamespace(
        fallback_used=fallback_used,
        confidence=confidence,
    )


def test_comparison_context_anchored_mode():
    """Direct + adjacent anchors → mode='anchored', allowed = safe + user."""
    forecast = _stub_forecast(fallback_used=False, confidence="medium-high")
    product = {"competitors": [{"name": "Pedialyte"}]}
    evidence_buckets = {
        "forecast_anchors": [
            {"brand": "Pedialyte", "anchor_strength": "direct"},
            {"brand": "DripDrop", "anchor_strength": "direct"},
            {"brand": "AG1", "anchor_strength": "adjacent"},
        ],
        "fallback_neighbors": [],
        "candidate_comparables": [],
        "exploratory_comparables": [],
    }

    ctx = build_comparison_context(
        forecast=forecast,
        product=product,
        evidence_buckets=evidence_buckets,
        coverage_tier="strong",
    )

    assert ctx["comparison_mode"] == "anchored"
    assert "Pedialyte" in ctx["dialogue_safe_anchor_brands"]
    assert "DripDrop" in ctx["dialogue_safe_anchor_brands"]
    assert "AG1" in ctx["dialogue_safe_anchor_brands"]
    # User competitor (Pedialyte) is also dialogue-safe — should appear
    # exactly once in allowed (case-insensitive dedup).
    allowed_lower = [b.lower() for b in ctx["allowed_comparison_brands"]]
    assert allowed_lower.count("pedialyte") == 1, \
        "Pedialyte must dedup across safe-anchor and user-competitor lists"
    assert ctx["forbidden_brand_names"] == []
    assert ctx["fallback_used"] is False


def test_comparison_context_weak_anchor_exclusion():
    """Weak forecast anchors visible in evidence panel must NOT be
    dialogue-safe and MUST appear in forbidden list."""
    forecast = _stub_forecast(fallback_used=False, confidence="medium")
    product = {"competitors": []}
    evidence_buckets = {
        "forecast_anchors": [
            {"brand": "Pedialyte", "anchor_strength": "direct"},
            {"brand": "Monster Energy", "anchor_strength": "weak"},
            {"brand": "Red Bull", "anchor_strength": "weak"},
        ],
        "fallback_neighbors": [],
        "candidate_comparables": [],
        "exploratory_comparables": [],
    }

    ctx = build_comparison_context(
        forecast=forecast,
        product=product,
        evidence_buckets=evidence_buckets,
        coverage_tier="strong",
    )

    # Weak anchors appear in forecast_used_brands (math used them)
    assert "Monster Energy" in ctx["forecast_used_brands"]
    assert "Red Bull" in ctx["forecast_used_brands"]
    # But NOT in dialogue_safe (weak excluded)
    assert "Monster Energy" not in ctx["dialogue_safe_anchor_brands"]
    assert "Red Bull" not in ctx["dialogue_safe_anchor_brands"]
    # And NOT in allowed (because not safe and not user-competitor)
    assert "Monster Energy" not in ctx["allowed_comparison_brands"]
    assert "Red Bull" not in ctx["allowed_comparison_brands"]
    # AND in forbidden
    assert "Monster Energy" in ctx["forbidden_brand_names"]
    assert "Red Bull" in ctx["forbidden_brand_names"]
    # Direct anchor remains allowed
    assert "Pedialyte" in ctx["allowed_comparison_brands"]


def test_comparison_context_user_competitor_mode():
    """The matcha gum case. Fallback fired, no eligible anchors,
    user-stated competitors (Trident, Orbit, Matchew). Retrieved-but-
    not-used brands (Poppi, Monster, etc.) must be forbidden."""
    forecast = _stub_forecast(fallback_used=True, confidence="low")
    product = {
        "competitors": [
            {"name": "Trident"},
            {"name": "Orbit"},
            {"name": "Matchew"},
        ]
    }
    evidence_buckets = {
        # Under fallback, forecast_anchors is empty by definition
        # (per evidence.py:172-174). All retrieved candidates land
        # in fallback_neighbors.
        "forecast_anchors": [],
        "fallback_neighbors": [
            {"brand": "Poppi Prebiotic Soda"},
            {"brand": "Liquid Death Mountain Water"},
            {"brand": "Health-Ade Kombucha"},
            {"brand": "Monster Energy"},
            {"brand": "Prime Energy"},
            {"brand": "Red Bull"},
            {"brand": "C4 Energy"},
            {"brand": "Celsius"},
        ],
        "candidate_comparables": [],
        "exploratory_comparables": [],
    }

    ctx = build_comparison_context(
        forecast=forecast,
        product=product,
        evidence_buckets=evidence_buckets,
        coverage_tier="weak",
    )

    assert ctx["comparison_mode"] == "user_competitor"
    assert ctx["allowed_comparison_brands"] == ["Trident", "Orbit", "Matchew"]
    assert ctx["forecast_used_brands"] == []
    assert ctx["dialogue_safe_anchor_brands"] == []
    # All 8 retrieved brands must be forbidden
    for forbidden in [
        "Poppi Prebiotic Soda", "Liquid Death Mountain Water",
        "Health-Ade Kombucha", "Monster Energy", "Prime Energy",
        "Red Bull", "C4 Energy", "Celsius",
    ]:
        assert forbidden in ctx["forbidden_brand_names"], \
            f"{forbidden!r} must be in forbidden_brand_names under fallback"
    # User competitors must NOT be forbidden
    for allowed in ["Trident", "Orbit", "Matchew"]:
        assert allowed not in ctx["forbidden_brand_names"]
    assert ctx["fallback_used"] is True


def test_comparison_context_generic_directional_mode():
    """Fallback + no user competitors → mode='generic_directional',
    allowed=[], all retrieved brands forbidden. Agents must speak
    without brand names."""
    forecast = _stub_forecast(fallback_used=True, confidence="low")
    product = {"competitors": []}
    evidence_buckets = {
        "forecast_anchors": [],
        "fallback_neighbors": [
            {"brand": "Some Random Brand A"},
            {"brand": "Some Random Brand B"},
        ],
        "candidate_comparables": [],
        "exploratory_comparables": [],
    }

    ctx = build_comparison_context(
        forecast=forecast,
        product=product,
        evidence_buckets=evidence_buckets,
        coverage_tier="weak",
    )

    assert ctx["comparison_mode"] == "generic_directional"
    assert ctx["allowed_comparison_brands"] == []
    assert ctx["user_competitors"] == []
    assert ctx["dialogue_safe_anchor_brands"] == []
    # All fallback brands forbidden
    assert "Some Random Brand A" in ctx["forbidden_brand_names"]
    assert "Some Random Brand B" in ctx["forbidden_brand_names"]


def test_comparison_context_user_competitor_overrides_fallback():
    """If a user-stated competitor name ALSO appears in fallback_neighbors,
    user competitor wins — appears in allowed, NOT in forbidden.

    This protects against false-forbidding a brand the user explicitly
    named (case-insensitive)."""
    forecast = _stub_forecast(fallback_used=True, confidence="low")
    product = {
        "competitors": [
            {"name": "Liquid Death"},          # user named it
            {"name": "Trident"},
        ]
    }
    evidence_buckets = {
        "forecast_anchors": [],
        "fallback_neighbors": [
            # Same brand also in fallback list (slight casing variation)
            {"brand": "liquid death mountain water"},
            {"brand": "Monster Energy"},
        ],
        "candidate_comparables": [],
        "exploratory_comparables": [],
    }

    ctx = build_comparison_context(
        forecast=forecast,
        product=product,
        evidence_buckets=evidence_buckets,
        coverage_tier="weak",
    )

    # User competitors win
    assert "Liquid Death" in ctx["allowed_comparison_brands"]
    assert "Trident" in ctx["allowed_comparison_brands"]
    # And the fallback variant must NOT be in forbidden — friend's
    # subtraction rule: forbidden_pool MINUS allowed_comparison_brands,
    # case-insensitive. "liquid death mountain water" lowercase contains
    # "liquid death" but is a DIFFERENT brand string. The exact match is
    # what matters: "liquid death" lowercase != "liquid death mountain water"
    # lowercase. So fallback variant SHOULD remain forbidden because it's
    # a different brand name. The protection is for EXACT case-insensitive
    # matches only.
    #
    # However if user typed "Liquid Death Mountain Water", the fallback
    # match would be filtered out. Test that path explicitly:
    forecast2 = _stub_forecast(fallback_used=True, confidence="low")
    product2 = {"competitors": [{"name": "Liquid Death Mountain Water"}]}
    ctx2 = build_comparison_context(
        forecast=forecast2,
        product=product2,
        evidence_buckets={
            "forecast_anchors": [],
            "fallback_neighbors": [
                {"brand": "LIQUID DEATH MOUNTAIN WATER"},  # uppercase variant
                {"brand": "Monster Energy"},
            ],
            "candidate_comparables": [],
            "exploratory_comparables": [],
        },
        coverage_tier="weak",
    )
    # Case-insensitive dedup: user's exact-but-cased "Liquid Death Mountain
    # Water" wins. The uppercase fallback variant is removed from forbidden.
    assert "Liquid Death Mountain Water" in ctx2["allowed_comparison_brands"]
    forbidden_lower = [b.lower() for b in ctx2["forbidden_brand_names"]]
    assert "liquid death mountain water" not in forbidden_lower, \
        "Brand named by user must NOT appear in forbidden, even with " \
        "different casing in fallback_neighbors"
    assert "Monster Energy" in ctx2["forbidden_brand_names"]


def _ctx_p1_8_2(allowed=None, forbidden=None, mode="anchored", fallback=False):
    return {
        "forecast_used_brands": allowed or [],
        "dialogue_safe_anchor_brands": allowed or [],
        "user_competitors": [],
        "allowed_comparison_brands": allowed or [],
        "forbidden_brand_names": forbidden or [],
        "fallback_used": fallback,
        "confidence": "low" if fallback else "medium",
        "coverage_tier": "weak" if fallback else "strong",
        "comparison_mode": mode,
        "_meta": {"context_version": "v1.0"},
    }


def _matcha_gum_synthetic_inputs():
    from backend.dtc_v3.discussion import generate_discussion
    product = {
        "product_name": "Matcha Chewing Gum",
        "name": "Matcha Chewing Gum",
        "price": 0.25,
        "category": "food_beverage",
        "competitors": [{"name": "Trident"}, {"name": "Orbit"}, {"name": "Matchew"}],
    }
    forecast = {
        "trial_rate": {"median": 0.04, "low": 0.02, "high": 0.06, "percentage": 4.0},
        "confidence": "low",
        "verdict": "test_before_launch",
        "version": "v3-lite",
        "diagnostics": {"prior_source": "fallback_category_median"},
    }
    return product, forecast


def test_template_anchored_mode_uses_allowed_brands_only():
    from backend.dtc_v3.discussion import generate_discussion, clear_cache
    clear_cache()
    product, forecast = _matcha_gum_synthetic_inputs()
    ctx = _ctx_p1_8_2(
        allowed=["Pedialyte", "DripDrop", "LMNT"],
        forbidden=["Monster Energy", "Red Bull"],
        mode="anchored",
        fallback=False,
    )
    result = generate_discussion(product, forecast, agent_count=20, mode="template", comparison_context=ctx)
    text = str(result)
    assert "Monster Energy" not in text
    assert "Red Bull" not in text


def test_template_user_competitor_mode_no_anchored_phrases():
    from backend.dtc_v3.discussion import generate_discussion, clear_cache
    clear_cache()
    product, forecast = _matcha_gum_synthetic_inputs()
    ctx = _ctx_p1_8_2(
        allowed=["Trident", "Orbit", "Matchew"],
        forbidden=["Poppi Prebiotic Soda", "Liquid Death Mountain Water", "Monster Energy"],
        mode="user_competitor",
        fallback=True,
    )
    result = generate_discussion(product, forecast, agent_count=20, mode="template", comparison_context=ctx)
    text = str(result)
    assert "Poppi Prebiotic Soda" not in text
    assert "Liquid Death" not in text
    assert "Monster Energy" not in text


def test_template_generic_directional_mode_no_brand_names():
    from backend.dtc_v3.discussion import generate_discussion, clear_cache
    clear_cache()
    product, forecast = _matcha_gum_synthetic_inputs()
    product["competitors"] = []
    ctx = _ctx_p1_8_2(
        allowed=[],
        forbidden=["Poppi Prebiotic Soda", "Monster Energy", "Red Bull"],
        mode="generic_directional",
        fallback=True,
    )
    result = generate_discussion(product, forecast, agent_count=20, mode="template", comparison_context=ctx)
    text = str(result)
    assert "Poppi" not in text
    assert "Monster" not in text
    assert "Red Bull" not in text
    for banned in [
        "Compared with",
        "Stacked against",
        "the category leader",
        "comparable-brand evidence is strong",
        "comparable-brand math is reasonable",
        "hits the trial rate",
        "in striking range",
        "competitor comparison",
        "{anchor}",
    ]:
        assert banned not in text, f"{banned!r} must NEVER appear in generic_directional output"


def test_template_safety_scan_replaces_forbidden_brand(monkeypatch):
    from backend.dtc_v3 import discussion as disc_mod
    from backend.dtc_v3.discussion import generate_discussion, clear_cache, GENERIC_FOR_REASON_VARIANTS
    clear_cache()
    poison = ["Monster Energy is the obvious benchmark for this product."] * 8
    monkeypatch.setattr(disc_mod, "GENERIC_FOR_REASON_VARIANTS", poison)
    product, forecast = _matcha_gum_synthetic_inputs()
    product["competitors"] = []
    ctx = _ctx_p1_8_2(
        allowed=[],
        forbidden=["Monster Energy"],
        mode="generic_directional",
        fallback=True,
    )
    result = generate_discussion(product, forecast, agent_count=20, mode="template", comparison_context=ctx)
    text = str(result)
    assert "Monster Energy" not in text, "sanitizer must REMOVE the injected forbidden brand"
    assert "I would need to evaluate this against my current options." in text, \
        "sanitizer must REPLACE with the safe fallback string"


def test_template_matcha_gum_full_path():
    from backend.dtc_v3.discussion import generate_discussion, clear_cache
    clear_cache()
    product, forecast = _matcha_gum_synthetic_inputs()
    ctx = _ctx_p1_8_2(
        allowed=["Trident", "Orbit", "Matchew"],
        forbidden=[
            "Poppi Prebiotic Soda", "Liquid Death Mountain Water",
            "Health-Ade Kombucha", "Monster Energy", "Prime Energy",
            "Red Bull", "C4 Energy", "Celsius",
        ],
        mode="user_competitor",
        fallback=True,
    )
    result = generate_discussion(product, forecast, agent_count=20, mode="template", comparison_context=ctx)
    text = str(result)
    for forbidden in [
        "Poppi", "Liquid Death", "Health-Ade",
        "Monster Energy", "Prime Energy", "Red Bull",
        "C4 Energy", "Celsius",
    ]:
        assert forbidden not in text, f"{forbidden!r} must NEVER appear in matcha gum dialogue"


def test_missing_comparison_context_raises():
    from backend.dtc_v3.discussion import generate_discussion
    import pytest
    product = {"product_name": "X", "name": "X", "price": 1.0}
    forecast = {
        "trial_rate": {"median": 0.05, "low": 0.03, "high": 0.07, "percentage": 5.0},
        "confidence": "medium",
        "verdict": "test_before_launch",
        "version": "v3-lite",
    }
    with pytest.raises(ValueError):
        generate_discussion(product, forecast, agent_count=20, mode="template")


def _ledger_test_inputs(fallback, eligible_count, anchors):
    from types import SimpleNamespace
    forecast = SimpleNamespace(
        fallback_used=fallback,
        eligible_neighbor_count=eligible_count,
        confidence="medium",
        prior_source="rag_weighted_median" if not fallback else "fallback_category_median",
        rate_std=0.05,
        exact_subtype_weight_share=0.6,
        data_quality=SimpleNamespace(quality_warnings=[]),
        neighbors=[],
        downweighted_brands=[],
        confidence_reasons=[],
    )
    product = SimpleNamespace(
        product_name="Test Product",
        name="Test Product",
        price=10.0,
        category="food_beverage",
    )
    evidence_buckets = {
        "forecast_anchors": anchors,
        "fallback_neighbors": [],
        "candidate_comparables": [],
        "exploratory_comparables": [],
    }
    return forecast, product, evidence_buckets


def test_ledger_no_anchor_count_under_fallback():
    from backend.dtc_v3.confidence_ledger import build_confidence_ledger
    forecast, product, evidence_buckets = _ledger_test_inputs(
        fallback=True,
        eligible_count=8,
        anchors=[],
    )
    entries = build_confidence_ledger(
        forecast=forecast,
        product=product,
        evidence_buckets=evidence_buckets,
        record_by_brand={},
        inferred_subtype="energy_drinks",
    )
    signals = [e.get("signal") for e in entries]
    texts = [e.get("text", "") for e in entries]
    assert "low_anchor_count" in signals, "fallback must emit low_anchor_count"
    assert "strong_anchor_count" not in signals, "fallback must NEVER emit strong_anchor_count"
    for txt in texts:
        assert "8 forecast anchors" not in txt, f"contaminated count leaked into ledger text: {txt!r}"


def test_ledger_anchor_count_matches_evidence_buckets():
    from backend.dtc_v3.confidence_ledger import build_confidence_ledger
    fake_anchors = [
        {"brand": "Pedialyte"},
        {"brand": "DripDrop"},
        {"brand": "LMNT"},
    ]
    forecast, product, evidence_buckets = _ledger_test_inputs(
        fallback=False,
        eligible_count=8,
        anchors=fake_anchors,
    )
    entries = build_confidence_ledger(
        forecast=forecast,
        product=product,
        evidence_buckets=evidence_buckets,
        record_by_brand={},
        inferred_subtype="hydration_supplement",
    )
    signals = [e.get("signal") for e in entries]
    texts = [e.get("text", "") for e in entries]
    assert "strong_anchor_count" in signals, "non-fallback with 3 anchors must emit strong_anchor_count"
    assert "low_anchor_count" not in signals, "non-fallback with 3 anchors must NOT emit low_anchor_count"
    found_3 = any("3 forecast anchors" in t for t in texts)
    assert found_3, f"ledger must say 3 forecast anchors. Texts: {texts!r}"
    for txt in texts:
        assert "8 forecast anchors" not in txt, (
            f"ledger leaked eligible_neighbor_count=8 into text: {txt!r}. "
            f"This proves the old bug is dead."
        )


def test_llm_prompt_uses_allowed_brands_only():
    from backend.dtc_v3.llm_dialogue_enricher import _build_user_prompt
    product = {
        "product_name": "Matcha Chewing Gum",
        "name": "Matcha Chewing Gum",
        "price": 0.25,
        "category": "food_beverage",
        "competitors": [{"name": "Trident"}, {"name": "Orbit"}, {"name": "Matchew"}],
        "description": "matcha-flavored gum",
    }
    forecast = {
        "trial_rate": {"median": 0.04, "percentage": 4.0},
        "confidence": "low",
        "anchored_on": [
            {"brand": "Poppi Prebiotic Soda", "trial_rate": 0.10},
            {"brand": "Liquid Death Mountain Water", "trial_rate": 0.06},
            {"brand": "Monster Energy", "trial_rate": 0.18},
        ],
    }
    panel = {"agents": [
        {"id": f"agent_0{i+1}", "name": f"Buyer {i+1}", "age": 30, "profession": "x",
         "segment": "s", "profile": "p", "verdict": "BUY",
         "current_score_10": 7.0, "is_hardcore": False}
        for i in range(2)
    ]}
    cc = {
        "comparison_mode": "user_competitor",
        "allowed_comparison_brands": ["Trident", "Orbit", "Matchew"],
        "forbidden_brand_names": ["Poppi Prebiotic Soda", "Liquid Death Mountain Water", "Monster Energy"],
        "_meta": {"context_version": "v1.0"},
    }
    prompt = _build_user_prompt(panel["agents"], product, forecast, panel, cc)
    assert "Trident" in prompt
    assert "Orbit" in prompt or "Matchew" in prompt
    assert "user_competitor" in prompt or "comparison_mode" in prompt
    assert "anchored_comparable_brands" not in prompt, "old contaminated field must be gone"
    forbidden_section = prompt.split("forbidden_brand_names")[-1] if "forbidden_brand_names" in prompt else ""
    allowed_section_idx = prompt.find("allowed_comparison_brands")
    forbidden_section_idx = prompt.find("forbidden_brand_names")
    if allowed_section_idx >= 0 and forbidden_section_idx > allowed_section_idx:
        allowed_block = prompt[allowed_section_idx:forbidden_section_idx]
        for forbidden_brand in ["Poppi", "Liquid Death", "Monster"]:
            assert forbidden_brand not in allowed_block, (
                f"forbidden brand {forbidden_brand!r} must NOT appear in "
                f"allowed_comparison_brands section"
            )
    assert "forbidden_brand_names" in prompt, "deny list must be communicated to LLM"


def test_llm_validator_rejects_forbidden_brand_in_agent_field():
    from backend.dtc_v3.llm_dialogue_enricher import _validate_batch_response
    batch_agents = [
        {"id": "agent_01", "name": "Buyer 1"},
        {"id": "agent_02", "name": "Buyer 2"},
    ]
    parsed = {
        "agents": [
            {
                "id": "agent_01",
                "reason": "Compared with Monster Energy, this gum competes well.",
                "top_objection": "Price too high.",
                "what_would_change_mind": "Lower price.",
                "key_quote": "Reasonable.",
                "round_responses": [
                    {"round": 1, "response": "First impression strong."},
                    {"round": 2, "response": "Considering it."},
                    {"round": 3, "response": "Likely buy."},
                ],
            },
            {
                "id": "agent_02",
                "reason": "Solid product overall.",
                "top_objection": "No retail.",
                "what_would_change_mind": "More reviews.",
                "key_quote": "Decent.",
                "round_responses": [
                    {"round": 1, "response": "Lukewarm reaction here."},
                    {"round": 2, "response": "Still on the fence."},
                    {"round": 3, "response": "Pass for now."},
                ],
            },
        ],
        "consensus": "Mixed.",
        "winning_message": "Try it.",
        "actionable_insight": "Test more.",
    }
    forbidden_lookup = {"monster energy"}
    result = _validate_batch_response(parsed, batch_agents, forbidden_lookup=forbidden_lookup)
    assert result is None, "validator must reject batch with forbidden brand in agent field"


def test_llm_validator_rejects_forbidden_brand_in_top_level_fields():
    from backend.dtc_v3.llm_dialogue_enricher import _validate_batch_response
    batch_agents = [
        {"id": "agent_01", "name": "Buyer 1"},
    ]
    parsed = {
        "agents": [
            {
                "id": "agent_01",
                "reason": "Solid product overall.",
                "top_objection": "Price.",
                "what_would_change_mind": "Reviews.",
                "key_quote": "Reasonable.",
                "round_responses": [
                    {"round": 1, "response": "First reaction is fine."},
                    {"round": 2, "response": "Considering carefully."},
                    {"round": 3, "response": "Probably buy."},
                ],
            },
        ],
        "consensus": "Panel converged on positive sentiment.",
        "winning_message": "Position against Monster Energy in this category.",
        "actionable_insight": "Run a small validation test.",
    }
    forbidden_lookup = {"monster energy"}
    result = _validate_batch_response(parsed, batch_agents, forbidden_lookup=forbidden_lookup)
    assert result is None, "validator must reject batch with forbidden brand in top-level field"


def test_llm_cache_key_changes_with_comparison_context():
    from backend.dtc_v3.llm_dialogue_enricher import _build_cache_key
    product = {
        "product_name": "Matcha Chewing Gum",
        "name": "Matcha Chewing Gum",
        "price": 0.25,
        "category": "food_beverage",
        "competitors": [{"name": "Trident"}],
        "description": "matcha gum",
    }
    forecast = {
        "trial_rate": {"median": 0.04},
        "confidence": "low",
        "anchored_on": [{"brand": "Poppi Prebiotic Soda"}],
    }
    panel = {
        "agents": [{"id": "agent_01", "name": "x", "verdict": "BUY"}],
        "agent_count": 20,
    }
    seed = "0" * 32
    cc1 = {
        "comparison_mode": "user_competitor",
        "allowed_comparison_brands": ["Trident", "Orbit"],
        "forbidden_brand_names": ["Poppi"],
    }
    cc2 = {
        "comparison_mode": "generic_directional",
        "allowed_comparison_brands": [],
        "forbidden_brand_names": ["Poppi", "Monster Energy"],
    }
    key1 = _build_cache_key(product, forecast, panel, seed, cc1)
    key2 = _build_cache_key(product, forecast, panel, seed, cc2)
    assert key1 != key2, "cache key must change with comparison_context (mode/allowed/forbidden)"
    key1_again = _build_cache_key(product, forecast, panel, seed, cc1)
    assert key1 == key1_again, "cache key must be deterministic for identical comparison_context"

