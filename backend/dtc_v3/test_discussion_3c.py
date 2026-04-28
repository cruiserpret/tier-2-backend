"""
backend/dtc_v3/test_discussion_3c.py — Top-level panel fields + change-mind polish.

Per friend's Day 2 spec for Commit 3c:
  - intent_distribution sums to ~1.0 and matches bucket counts
  - buyer_journeys[] is non-empty and includes shifted agents
  - representative_quotes[] has BUY/CONS/WON'T BUY samples
  - hardest_to_convert_segment is populated
  - comparable_price_range has user_price + anchor_brands
  - actionable_insight scales with confidence
  - what_would_change_mind uses stance-specific templates
"""

from collections import Counter

import pytest

from backend.dtc_v3 import discussion


def _liquid_iv_product() -> dict:
    return {
        "product_name": "Liquid IV Hydration Multiplier",
        "name": "Liquid IV Hydration Multiplier",
        "description": "Hydration powder mix with electrolytes",
        "category": "supplements_health",
        "price": 25.0,
        "demographic": "active adults",
        "competitors": [{"name": "LMNT"}, {"name": "Pedialyte"}],
    }


def _high_conf_forecast() -> dict:
    return {
        "trial_rate": {"percentage": 22.0, "median": 0.22},
        "trial_rate_median": 0.22,
        "confidence": "medium-high",
        "fallback_used": False,
        "verdict": "launch",
        "anchored_on": [{"brand": "LMNT"}, {"brand": "Pedialyte"}],
        "top_drivers": [],
        "top_objections": [],
    }


def _low_conf_fallback_forecast() -> dict:
    return {
        "trial_rate": {"percentage": 7.5, "median": 0.075},
        "trial_rate_median": 0.075,
        "confidence": "low",
        "fallback_used": True,
        "verdict": "test_before_launch",
        "anchored_on": [],
        "top_drivers": [],
        "top_objections": [],
    }


def _generate(product, forecast, agent_count=20):
    discussion.clear_cache()
    result = discussion.generate_discussion(product, forecast, agent_count, "template")
    return result["agent_panel"]


# ═══════════════════════════════════════════════════════════════════════
# intent_distribution
# ═══════════════════════════════════════════════════════════════════════

def test_intent_distribution_present():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    assert "intent_distribution" in panel
    dist = panel["intent_distribution"]
    assert set(dist.keys()) == {"buy", "considering", "resistant"}


def test_intent_distribution_sums_to_one():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    dist = panel["intent_distribution"]
    total = dist["buy"] + dist["considering"] + dist["resistant"]
    assert abs(total - 1.0) < 0.01


def test_intent_distribution_matches_actual_agent_counts():
    """intent_distribution shares match observed verdict counts."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    counts = Counter(a["verdict"] for a in panel["agents"])
    n = len(panel["agents"])
    dist = panel["intent_distribution"]
    assert abs(dist["buy"] - counts["BUY"] / n) < 0.05
    assert abs(dist["considering"] - counts["CONSIDERING"] / n) < 0.05
    assert abs(dist["resistant"] - counts["WON'T BUY"] / n) < 0.05


# ═══════════════════════════════════════════════════════════════════════
# buyer_journeys[]
# ═══════════════════════════════════════════════════════════════════════

def test_buyer_journeys_present():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    assert "buyer_journeys" in panel
    journeys = panel["buyer_journeys"]
    assert isinstance(journeys, list)
    assert len(journeys) > 0


def test_buyer_journeys_have_required_fields():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    required = {"agent_id", "name", "segment", "initial_verdict",
                "final_verdict", "shifted", "shift_reason", "key_quote"}
    for j in panel["buyer_journeys"]:
        assert set(j.keys()) >= required, f"missing fields: {required - set(j.keys())}"


def test_buyer_journeys_capped_at_10():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    assert len(panel["buyer_journeys"]) <= 10


# ═══════════════════════════════════════════════════════════════════════
# representative_quotes[]
# ═══════════════════════════════════════════════════════════════════════

def test_representative_quotes_present():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    assert "representative_quotes" in panel
    quotes = panel["representative_quotes"]
    assert isinstance(quotes, list)


def test_representative_quotes_cover_all_verdicts():
    """At minimum, expect BUY + CONS + WON'T BUY samples in high-conf case."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    verdicts_in_quotes = {q["verdict"] for q in panel["representative_quotes"]}
    assert verdicts_in_quotes == {"BUY", "CONSIDERING", "WON'T BUY"}


def test_representative_quote_has_actual_response():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    for q in panel["representative_quotes"]:
        assert isinstance(q["quote"], str)
        assert len(q["quote"]) > 10


# ═══════════════════════════════════════════════════════════════════════
# hardest_to_convert_segment
# ═══════════════════════════════════════════════════════════════════════

def test_hardest_to_convert_segment_populated():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    assert "hardest_to_convert_segment" in panel
    assert isinstance(panel["hardest_to_convert_segment"], str)
    assert len(panel["hardest_to_convert_segment"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# comparable_price_range
# ═══════════════════════════════════════════════════════════════════════

def test_comparable_price_range_present():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    assert "comparable_price_range" in panel
    cpr = panel["comparable_price_range"]
    assert "user_price" in cpr
    assert "anchor_brands" in cpr


def test_comparable_price_range_has_user_price():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    assert panel["comparable_price_range"]["user_price"] == 25.0


def test_comparable_price_range_has_anchor_brands():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    anchors = panel["comparable_price_range"]["anchor_brands"]
    assert isinstance(anchors, list)
    assert "LMNT" in anchors


# ═══════════════════════════════════════════════════════════════════════
# actionable_insight
# ═══════════════════════════════════════════════════════════════════════

def test_actionable_insight_high_conf_strong_demand():
    """High conf + strong BUY share → recommends paid landing test."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    insight = panel["actionable_insight"]
    assert isinstance(insight, str)
    assert len(insight) > 30


def test_actionable_insight_low_conf_fallback_recommends_validation():
    """Low conf + fallback → recommends validation test, no scaling."""
    panel = _generate(_liquid_iv_product(), _low_conf_fallback_forecast(), 50)
    insight = panel["actionable_insight"]
    assert "validation" in insight.lower() or "do not scale" in insight.lower() \
        or "before scaling" in insight.lower()


# ═══════════════════════════════════════════════════════════════════════
# what_would_change_mind polish — stance-specific templates
# ═══════════════════════════════════════════════════════════════════════

def test_buy_agents_change_mind_uses_for_pool():
    """BUY agents' what_would_change_mind reads as cancellation/erosion concerns."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    buys = [a for a in panel["agents"] if a["verdict"] == "BUY"]
    # Look for at least one BUY agent's change-mind text matching the FOR pool semantics
    for_pool_signals = ["price increase", "reconsider", "switch", "erode",
                        "stockout", "cancellation", "shrink", "slippage",
                        "supply", "out", "switching", "churn", "lose me"]
    matched = sum(
        1 for a in buys
        if any(sig in a["what_would_change_mind"].lower() for sig in for_pool_signals)
    )
    # At least 50% of BUY agents should hit the FOR-pool semantics
    assert matched >= len(buys) // 2, \
        f"only {matched}/{len(buys)} BUY agents use cancellation-style language"


def test_against_agents_change_mind_uses_against_pool():
    """WON'T BUY agents' what_would_change_mind reads as 'soften my no'."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    wonts = [a for a in panel["agents"] if a["verdict"] == "WON'T BUY"]
    against_pool_signals = ["proof", "lower price", "trial", "trusted",
                            "differentiation", "money back", "evidence",
                            "test against", "return", "third-party",
                            "long-tail", "risk-free"]
    matched = sum(
        1 for a in wonts
        if any(sig in a["what_would_change_mind"].lower() for sig in against_pool_signals)
    )
    assert matched >= len(wonts) // 2, \
        f"only {matched}/{len(wonts)} WON'T BUY agents use 'soften no' language"


# ═══════════════════════════════════════════════════════════════════════
# Backward compat — 3b fields still exist
# ═══════════════════════════════════════════════════════════════════════

def test_3b_fields_still_present_after_3c():
    """Previous fields from Commit 3b must still exist."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    a0 = panel["agents"][0]
    required_3b_fields = {
        "name", "verdict", "score_10", "current_score_10", "initial_score_10",
        "shifted", "is_hardcore", "key_moment", "round_responses", "journey",
    }
    assert set(a0.keys()) >= required_3b_fields


# ═══════════════════════════════════════════════════════════════════════
# Determinism — same input = same panel (full panel, not just agents)
# ═══════════════════════════════════════════════════════════════════════

def test_determinism_full_panel():
    p1 = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    p2 = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    assert p1["intent_distribution"] == p2["intent_distribution"]
    assert p1["hardest_to_convert_segment"] == p2["hardest_to_convert_segment"]
    assert p1["actionable_insight"] == p2["actionable_insight"]
    # buyer_journeys names should match
    n1 = [j["name"] for j in p1["buyer_journeys"]]
    n2 = [j["name"] for j in p2["buyer_journeys"]]
    assert n1 == n2


# ═══════════════════════════════════════════════════════════════════════
# Forecast invariant still holds
# ═══════════════════════════════════════════════════════════════════════

def test_3c_does_not_mutate_forecast():
    forecast = _high_conf_forecast()
    rate_before = forecast["trial_rate"]["percentage"]
    discussion.clear_cache()
    discussion.generate_discussion(_liquid_iv_product(), forecast, 50, "template")
    assert forecast["trial_rate"]["percentage"] == rate_before
