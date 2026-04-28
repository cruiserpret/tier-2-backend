"""
Invariant tests for /api/dtc_v3/discuss.

Per friend's spec:
  - Discussion must not change forecast number
  - Agent count enforced
  - Low-confidence forecasts include coverage_warning
  - Same seed returns cached response
"""

import copy
import pytest

from backend.dtc_v3.discussion import (
    generate_discussion,
    generate_seed,
    clear_cache,
    ALLOWED_AGENT_COUNTS,
)


@pytest.fixture
def liquid_iv_product():
    return {
        "product_name": "Liquid IV Hydration Multiplier",
        "name": "Liquid IV Hydration Multiplier",
        "description": "Electrolyte drink mix powder. Costco, Target, Walmart.",
        "price": 25,
        "category": "supplements_health",
        "demographic": "Active adults 22-50",
        "competitors": [{"name": "LMNT"}, {"name": "Pedialyte"}],
    }


@pytest.fixture
def liquid_iv_forecast():
    return {
        "simulation_id": "v3_test001",
        "version": "v3-lite",
        "verdict": "launch",
        "headline": "Strong launch candidate.",
        "trial_rate": {"median": 0.205, "low": 0.16, "high": 0.26, "percentage": 20.5},
        "confidence": "medium-high",
        "confidence_reasons": [],
        "anchored_on": [
            {"brand": "AG1 Athletic Greens", "trial_rate": 0.11, "confidence_grade": "B", "match_reason": "match"},
            {"brand": "Olipop", "trial_rate": 0.27, "confidence_grade": "A", "match_reason": "match"},
        ],
        "downweighted_brands": [],
        "why_might_be_wrong": [],
        "counterfactuals": [],
        "top_drivers": ["Mass retail presence", "Habit-forming category"],
        "top_objections": ["Subscription fatigue"],
        "most_receptive_segment": "Active health optimizers",
        "diagnostics": {
            "rag_prior": 0.21,
            "adjustment_applied": -0.005,
            "coverage_tier": "strong",
            "prior_source": "rag_weighted_median",
        },
    }


@pytest.fixture
def low_confidence_forecast(liquid_iv_forecast):
    f = copy.deepcopy(liquid_iv_forecast)
    f["confidence"] = "low"
    f["diagnostics"]["coverage_tier"] = "weak"
    f["diagnostics"]["prior_source"] = "fallback_category_median"
    f["why_might_be_wrong"] = ["Comparable database has limited coverage for this product subtype."]
    return f


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


def test_discussion_returns_agent_panel(liquid_iv_product, liquid_iv_forecast):
    result = generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=20)
    assert "agent_panel" in result
    panel = result["agent_panel"]
    for key in ("agent_count", "seed", "rounds", "agents", "top_drivers",
                "top_objections", "most_receptive_segment", "winning_message",
                "risk_factors", "consensus", "coverage_warning"):
        assert key in panel, f"missing field: {key}"


def test_discussion_agent_count_20(liquid_iv_product, liquid_iv_forecast):
    result = generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=20)
    panel = result["agent_panel"]
    assert panel["agent_count"] == 20
    assert len(panel["agents"]) == 20
    for i, agent in enumerate(panel["agents"]):
        assert agent["stance"] in ("for", "against", "neutral")
        assert 0.0 <= agent["score"] <= 1.0
        assert agent["id"].startswith("agent_")


def test_discussion_agent_count_50(liquid_iv_product, liquid_iv_forecast):
    result = generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=50)
    panel = result["agent_panel"]
    assert panel["agent_count"] == 50
    assert len(panel["agents"]) == 50


def test_discussion_invalid_agent_count_raises(liquid_iv_product, liquid_iv_forecast):
    with pytest.raises(ValueError):
        generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=30)
    with pytest.raises(ValueError):
        generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=0)


def test_discussion_does_not_change_forecast(liquid_iv_product, liquid_iv_forecast):
    """CRITICAL INVARIANT: discussion must not mutate the forecast number."""
    forecast_snapshot = copy.deepcopy(liquid_iv_forecast)
    rate_before = liquid_iv_forecast["trial_rate"]["median"]

    result = generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=20)

    rate_after = liquid_iv_forecast["trial_rate"]["median"]
    assert rate_before == rate_after, "discussion mutated forecast.trial_rate.median"
    assert liquid_iv_forecast == forecast_snapshot, "discussion mutated the forecast dict"

    panel = result["agent_panel"]
    assert "trial_rate" not in panel, "panel must not include its own trial_rate"
    assert "trial_rate_median" not in panel


def test_discussion_low_confidence_includes_coverage_warning(liquid_iv_product, low_confidence_forecast):
    result = generate_discussion(liquid_iv_product, low_confidence_forecast, agent_count=20)
    panel = result["agent_panel"]
    assert panel["coverage_warning"], "low-confidence forecast must include coverage_warning"
    assert "directional" in panel["coverage_warning"].lower()


def test_discussion_high_confidence_no_coverage_warning(liquid_iv_product, liquid_iv_forecast):
    result = generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=20)
    panel = result["agent_panel"]
    assert panel["coverage_warning"] == ""


def test_discussion_same_seed_cached(liquid_iv_product, liquid_iv_forecast):
    """Same input must return the same output (deterministic)."""
    result_a = generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=20)
    result_b = generate_discussion(liquid_iv_product, liquid_iv_forecast, agent_count=20)
    assert result_a == result_b
    assert result_a["agent_panel"]["seed"] == result_b["agent_panel"]["seed"]


def test_seed_changes_with_product(liquid_iv_product, liquid_iv_forecast):
    seed_a = generate_seed(liquid_iv_product, liquid_iv_forecast, 20)
    different_product = dict(liquid_iv_product, price=99.0)
    seed_b = generate_seed(different_product, liquid_iv_forecast, 20)
    assert seed_a != seed_b


def test_seed_stable_for_simulation_id_change(liquid_iv_product, liquid_iv_forecast):
    """Per friend's spec: seed must NOT depend on simulation_id (which is random per run)."""
    seed_a = generate_seed(liquid_iv_product, liquid_iv_forecast, 20)
    forecast_with_diff_id = copy.deepcopy(liquid_iv_forecast)
    forecast_with_diff_id["simulation_id"] = "v3_completely_different"
    seed_b = generate_seed(liquid_iv_product, forecast_with_diff_id, 20)
    assert seed_a == seed_b, "seed must not include simulation_id (random per run)"
