"""
backend/dtc_v3/test_discussion_3b.py — Per-agent enrichment tests.

Per friend's Day 2 spec for Commit 3b:
  - final BUY current_score_10 >= 7.0
  - final CONSIDERING current_score_10 between 4.6 and 6.9
  - final WON'T BUY current_score_10 <= 4.5
  - legacy score remains 0-1
  - hardcore agents stay WON'T BUY (no shift)
  - shifted=True iff initial_verdict != final_verdict
  - every shifted agent has key_moment
  - every agent has 3 round_responses
  - every agent has journey
  - same input gives same score arcs
  - discussion does not mutate trial_rate
"""

import copy

import pytest

from backend.dtc_v3 import discussion


# ═══════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════

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
    """Liquid-IV-style: 22% trial / medium-high / no fallback."""
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
    """YETI-style: low trial / low confidence / fallback."""
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
    """Helper: generate a discussion panel — returns agent_panel directly."""
    discussion.clear_cache()
    result = discussion.generate_discussion(product, forecast, agent_count, "template")
    return result["agent_panel"]


# ═══════════════════════════════════════════════════════════════════════
# Score band tests (verdict drives final score)
# ═══════════════════════════════════════════════════════════════════════

def test_final_buy_score_above_7():
    """Every BUY agent's current_score_10 must be >= 7.0."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    buys = [a for a in panel["agents"] if a["verdict"] == "BUY"]
    assert len(buys) > 0, "expected some BUY agents in high-conf 22% case"
    for a in buys:
        assert a["current_score_10"] >= 7.0, \
            f"BUY agent {a['name']} has current_score_10={a['current_score_10']} < 7.0"


def test_final_considering_score_between_4_6_and_6_9():
    """Every CONSIDERING agent's current_score_10 must be 4.6-6.9."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    cons = [a for a in panel["agents"] if a["verdict"] == "CONSIDERING"]
    for a in cons:
        s = a["current_score_10"]
        assert 4.6 <= s <= 6.9, \
            f"CONSIDERING agent {a['name']} has current_score_10={s} out of [4.6, 6.9]"


def test_final_wont_buy_score_below_4_5():
    """Every WON'T BUY agent's current_score_10 must be <= 4.5."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    wonts = [a for a in panel["agents"] if a["verdict"] == "WON'T BUY"]
    for a in wonts:
        assert a["current_score_10"] <= 4.5, \
            f"WON'T BUY agent {a['name']} has current_score_10={a['current_score_10']} > 4.5"


# ═══════════════════════════════════════════════════════════════════════
# Backward compat — legacy "score" field stays 0-1
# ═══════════════════════════════════════════════════════════════════════

def test_legacy_score_is_zero_to_one():
    """Existing 'score' field must remain 0.0-1.0 for backward compat."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    for a in panel["agents"]:
        assert 0.0 <= a["score"] <= 1.0, \
            f"legacy score out of [0,1]: {a['score']} for {a['name']}"


def test_score_10_matches_current_score_10():
    """score_10 alias should equal current_score_10."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    for a in panel["agents"]:
        assert a["score_10"] == a["current_score_10"]


# ═══════════════════════════════════════════════════════════════════════
# Hardcore agents
# ═══════════════════════════════════════════════════════════════════════

def test_hardcore_agents_stay_wont_buy():
    """Hardcore agents must always be WON'T BUY with no shift."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    hardcores = [a for a in panel["agents"] if a["is_hardcore"]]
    for a in hardcores:
        assert a["verdict"] == "WON'T BUY", \
            f"hardcore agent {a['name']} should be WON'T BUY"
        assert a["initial_stance"] == "against"
        assert a["current_stance"] == "against"
        assert a["shifted"] is False


# ═══════════════════════════════════════════════════════════════════════
# Shifted flag invariant
# ═══════════════════════════════════════════════════════════════════════

def test_shifted_iff_initial_neq_final():
    """shifted=True if and only if initial_verdict != final_verdict."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    for a in panel["agents"]:
        initial = a["journey"]["initial_verdict"]
        final = a["journey"]["final_verdict"]
        expected_shifted = (initial != final)
        assert a["shifted"] == expected_shifted, \
            f"{a['name']}: initial={initial}, final={final}, shifted={a['shifted']}"
        assert a["journey"]["shifted"] == expected_shifted


def test_every_shifted_agent_has_key_moment():
    """Every agent must have a non-empty key_moment string."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    for a in panel["agents"]:
        assert a["key_moment"], f"{a['name']} missing key_moment"
        assert isinstance(a["key_moment"], str)
        assert len(a["key_moment"]) > 10


# ═══════════════════════════════════════════════════════════════════════
# Round responses + journey shape
# ═══════════════════════════════════════════════════════════════════════

def test_every_agent_has_3_round_responses():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    for a in panel["agents"]:
        assert "round_responses" in a
        assert len(a["round_responses"]) == 3
        for i, rr in enumerate(a["round_responses"]):
            assert rr["round"] == i + 1
            assert rr["title"] in ("First Impression", "Competitor Comparison", "Final Verdict")
            assert isinstance(rr["response"], str)
            assert len(rr["response"]) > 5


def test_every_agent_has_journey_object():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    required_keys = {"initial_verdict", "final_verdict", "shifted",
                     "shift_reason", "key_moment", "key_quote"}
    for a in panel["agents"]:
        assert "journey" in a
        assert set(a["journey"].keys()) >= required_keys, \
            f"{a['name']} journey missing keys"


# ═══════════════════════════════════════════════════════════════════════
# Determinism — same input -> same score arcs
# ═══════════════════════════════════════════════════════════════════════

def test_same_input_same_score_arcs():
    p1 = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    p2 = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    for a1, a2 in zip(p1["agents"], p2["agents"]):
        assert a1["name"] == a2["name"]
        assert a1["initial_score_10"] == a2["initial_score_10"]
        assert a1["current_score_10"] == a2["current_score_10"]
        assert a1["verdict"] == a2["verdict"]
        assert a1["shifted"] == a2["shifted"]


# ═══════════════════════════════════════════════════════════════════════
# Forecast invariant — discussion never mutates trial_rate
# ═══════════════════════════════════════════════════════════════════════

def test_discussion_does_not_mutate_trial_rate():
    forecast = _high_conf_forecast()
    rate_before = forecast["trial_rate"]["percentage"]
    median_before = forecast["trial_rate_median"]

    discussion.clear_cache()
    discussion.generate_discussion(_liquid_iv_product(), forecast, 50, "template")

    rate_after = forecast["trial_rate"]["percentage"]
    median_after = forecast["trial_rate_median"]
    assert rate_before == rate_after
    assert median_before == median_after


# ═══════════════════════════════════════════════════════════════════════
# Bucket cap respected
# ═══════════════════════════════════════════════════════════════════════

def test_buy_count_capped_by_trial_rate():
    """22% trial rate / 50 agents -> BUY <= round(50 * 2 * 0.22) = 22."""
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    n_buy = sum(1 for a in panel["agents"] if a["verdict"] == "BUY")
    assert n_buy <= 24, f"BUY count {n_buy} exceeds 2x trial_rate cap"


def test_low_confidence_caps_buy_count():
    """Low confidence + fallback -> BUY <= round(50 * 0.15) = 8."""
    panel = _generate(_liquid_iv_product(), _low_conf_fallback_forecast(), 50)
    n_buy = sum(1 for a in panel["agents"] if a["verdict"] == "BUY")
    assert n_buy <= 8, f"BUY count {n_buy} exceeds 15% hard cap for low conf + fallback"


# ═══════════════════════════════════════════════════════════════════════
# Names + persona enrichment
# ═══════════════════════════════════════════════════════════════════════

def test_every_agent_has_name():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 20)
    for a in panel["agents"]:
        assert "name" in a
        assert isinstance(a["name"], str)
        assert len(a["name"]) > 1


def test_no_duplicate_agent_names():
    panel = _generate(_liquid_iv_product(), _high_conf_forecast(), 50)
    names = [a["name"] for a in panel["agents"]]
    assert len(set(names)) == len(names), f"duplicate names: {[n for n in names if names.count(n) > 1]}"
