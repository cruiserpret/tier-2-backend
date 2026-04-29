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

    def malicious_generate_discussion(product, forecast, agent_count, mode):
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

