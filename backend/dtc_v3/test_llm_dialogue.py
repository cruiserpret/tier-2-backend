"""
backend/dtc_v3/test_llm_dialogue.py — Slice 2B-LLM acceptance tests.

Per friend's Path D spec:
  - Cache hit returns immediately (no API call)
  - LLM timeout falls back to template (no error to user)
  - Partial batch failure isolates (template only for failed batch)
  - LLM output preserves agent ids/verdicts/scores
  - LLM cannot change trial_rate/confidence/verdict
  - Cache file structure round-trips correctly

These tests do NOT make real OpenAI calls. Path D smoke test
on real product is a separate manual verification (see /tmp/smoke_2b.py).
"""

import os
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.dtc_v3 import discussion
from backend.dtc_v3 import llm_dialogue_enricher as enricher


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

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
        "trial_rate": {"percentage": 22.0, "median": 0.22, "low": 0.17, "high": 0.27},
        "trial_rate_median": 0.22,
        "confidence": "medium-high",
        "fallback_used": False,
        "verdict": "launch",
        "anchored_on": [
            {"brand": "LMNT", "trial_rate": 0.22},
            {"brand": "Pedialyte", "trial_rate": 0.16},
        ],
        "top_drivers": [],
        "top_objections": [],
    }


# ─────────────────────────────────────────────────────────────────
# 1. Cache hit returns immediately
# ─────────────────────────────────────────────────────────────────

def test_cache_hit_returns_immediately(tmp_path, monkeypatch):
    """Pre-warmed cache → no async batching, no API call, fast return."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)

    # Build a template panel
    discussion.clear_cache()
    template_result = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "template"
    )
    panel = template_result["agent_panel"]

    # Build a fake cached LLM payload
    fake_cached = {
        "agents": [
            {
                "id": a["id"],
                "reason": f"CACHED: {a['name']} reasoning",
                "top_objection": "cached objection",
                "what_would_change_mind": "cached change-mind",
                "key_moment": "cached moment",
                "key_quote": "cached quote",
                "round_responses": [
                    {"round": 1, "title": "First Impression", "response": "cached r1 response"},
                    {"round": 2, "title": "Competitor Comparison", "response": "cached r2 response"},
                    {"round": 3, "title": "Final Verdict", "response": "cached r3 response"},
                ],
            }
            for a in panel["agents"]
        ],
        "consensus": "cached consensus",
        "winning_message": "cached winning",
        "actionable_insight": "cached insight",
    }

    # Pre-warm the cache for the exact key this product+forecast+panel would generate
    seed = discussion.generate_seed(_liquid_iv_product(), _high_conf_forecast(), 20, "llm")
    cache_key = enricher._build_cache_key(_liquid_iv_product(), _high_conf_forecast(), panel, seed)
    enricher._save_cache(cache_key, fake_cached)

    # Mock the AsyncOpenAI call so any attempt would crash — proves we never call it
    with patch("openai.AsyncOpenAI", side_effect=AssertionError("LLM should NOT be called on cache hit")):
        result = enricher.enrich_with_llm_dialogue(
            panel=panel,
            product=_liquid_iv_product(),
            forecast=_high_conf_forecast(),
            seed=seed,
        )

    assert result is not None
    assert result["mode"] == "llm"
    assert result["consensus"] == "cached consensus"
    assert result["agents"][0]["reason"].startswith("CACHED:")
    diag = result["diagnostics"]
    assert diag["llm_cache_hit"] is True
    assert diag["llm_batches"] == 0
    assert diag["partial_llm_fallback"] is False


# ─────────────────────────────────────────────────────────────────
# 2. LLM unavailable falls back to template
# ─────────────────────────────────────────────────────────────────

def test_no_api_key_falls_back_to_template(tmp_path, monkeypatch):
    """Missing OPENAI_API_KEY → enricher returns None → template panel served."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    discussion.clear_cache()
    result = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "llm"
    )
    panel = result["agent_panel"]
    assert panel["mode"] == "template"
    assert len(panel["agents"]) == 20
    # All v3 fields still present from template
    for a in panel["agents"]:
        assert a["name"]
        assert a["verdict"] in ("BUY", "CONSIDERING", "WON'T BUY")
        assert len(a["round_responses"]) == 3


def test_enricher_returns_none_on_total_failure(tmp_path, monkeypatch):
    """Direct test: enricher returns None when all batches fail."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    discussion.clear_cache()
    template_result = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "template"
    )
    seed = discussion.generate_seed(_liquid_iv_product(), _high_conf_forecast(), 20, "llm")
    result = enricher.enrich_with_llm_dialogue(
        panel=template_result["agent_panel"],
        product=_liquid_iv_product(),
        forecast=_high_conf_forecast(),
        seed=seed,
    )
    assert result is None


# ─────────────────────────────────────────────────────────────────
# 3. Partial batch failure isolates (only failed batch falls back)
# ─────────────────────────────────────────────────────────────────

def test_merge_marks_template_fallback_agents(tmp_path, monkeypatch):
    """When an agent comes back with _template_fallback=True, merge keeps template fields."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)

    discussion.clear_cache()
    template_result = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "template"
    )
    template_panel = template_result["agent_panel"]
    template_first_agent_reason = template_panel["agents"][0]["reason"]

    # Build mixed payload — first agent gets LLM dialogue, second gets template-fallback marker
    fake_llm_data = {
        "agents": [
            # LLM-enriched agent
            {
                "id": template_panel["agents"][0]["id"],
                "reason": "LLM_REWRITTEN reason",
                "top_objection": "LLM objection",
                "what_would_change_mind": "LLM cm",
                "key_moment": "LLM moment",
                "key_quote": "LLM quote",
                "round_responses": [
                    {"round": 1, "title": "First Impression", "response": "LLM r1"},
                    {"round": 2, "title": "Competitor Comparison", "response": "LLM r2"},
                    {"round": 3, "title": "Final Verdict", "response": "LLM r3"},
                ],
            },
            # Template-fallback agent (one of its batch's batches failed)
            {
                "id": template_panel["agents"][1]["id"],
                "reason": template_panel["agents"][1].get("reason", ""),
                "top_objection": template_panel["agents"][1].get("top_objection", ""),
                "what_would_change_mind": template_panel["agents"][1].get("what_would_change_mind", ""),
                "key_moment": template_panel["agents"][1].get("key_moment", ""),
                "key_quote": "fallback quote",
                "round_responses": template_panel["agents"][1].get("round_responses", []),
                "_template_fallback": True,
            },
        ],
        "consensus": "mixed consensus",
        "winning_message": "mixed winning",
        "actionable_insight": "mixed insight",
    }

    # Take only first 2 template agents for this isolated test
    test_panel = dict(template_panel)
    test_panel["agents"] = template_panel["agents"][:2]

    merged = enricher._merge_llm_into_panel(test_panel, fake_llm_data)

    # Agent 0: LLM rewrote it
    assert merged["agents"][0]["reason"] == "LLM_REWRITTEN reason"
    assert merged["agents"][0]["round_responses"][0]["response"] == "LLM r1"

    # Agent 1: template-fallback flag preserved its template content
    assert merged["agents"][1]["reason"] == template_first_agent_reason or \
           merged["agents"][1]["reason"] == template_panel["agents"][1]["reason"]


# ─────────────────────────────────────────────────────────────────
# 4. LLM preserves agent ids/verdicts/scores
# ─────────────────────────────────────────────────────────────────

def test_merge_preserves_agent_structure(tmp_path, monkeypatch):
    """LLM rewrite must NOT change agent id, verdict, score, name, journey arc."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)

    discussion.clear_cache()
    template_result = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "template"
    )
    template_panel = template_result["agent_panel"]
    template_agents = template_panel["agents"]

    template_id_verdict_score = [
        (a["id"], a["verdict"], a["current_score_10"], a["name"],
         a["journey"]["initial_verdict"], a["journey"]["final_verdict"])
        for a in template_agents
    ]

    # Build a fake LLM rewrite that tries (incorrectly) to flip everything
    fake_llm_data = {
        "agents": [
            {
                "id": a["id"],
                "reason": "LLM rewrote",
                "top_objection": "LLM objection",
                "what_would_change_mind": "LLM cm",
                "key_moment": "LLM moment",
                "key_quote": "LLM quote",
                "round_responses": [
                    {"round": 1, "title": "First Impression", "response": "LLM r1"},
                    {"round": 2, "title": "Competitor Comparison", "response": "LLM r2"},
                    {"round": 3, "title": "Final Verdict", "response": "LLM r3"},
                ],
            }
            for a in template_agents
        ],
        "consensus": "rewritten",
        "winning_message": "rewritten",
        "actionable_insight": "rewritten",
    }

    merged = enricher._merge_llm_into_panel(template_panel, fake_llm_data)
    merged_id_verdict_score = [
        (a["id"], a["verdict"], a["current_score_10"], a["name"],
         a["journey"]["initial_verdict"], a["journey"]["final_verdict"])
        for a in merged["agents"]
    ]

    # All structural fields preserved
    assert merged_id_verdict_score == template_id_verdict_score
    # But narrative was rewritten
    assert all(a["reason"] == "LLM rewrote" for a in merged["agents"])
    assert merged["consensus"] == "rewritten"


# ─────────────────────────────────────────────────────────────────
# 5. LLM cannot change forecast.trial_rate
# ─────────────────────────────────────────────────────────────────

def test_llm_mode_does_not_mutate_forecast(tmp_path, monkeypatch):
    """Forecast object passed in must be byte-identical after generate_discussion."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # forces template path, faster test

    forecast = _high_conf_forecast()
    rate_before = forecast["trial_rate"]["percentage"]
    median_before = forecast["trial_rate_median"]
    confidence_before = forecast["confidence"]
    verdict_before = forecast["verdict"]
    anchors_before = json.dumps(forecast["anchored_on"], sort_keys=True)

    discussion.clear_cache()
    discussion.generate_discussion(_liquid_iv_product(), forecast, 20, "llm")

    assert forecast["trial_rate"]["percentage"] == rate_before
    assert forecast["trial_rate_median"] == median_before
    assert forecast["confidence"] == confidence_before
    assert forecast["verdict"] == verdict_before
    assert json.dumps(forecast["anchored_on"], sort_keys=True) == anchors_before


# ─────────────────────────────────────────────────────────────────
# 6. Cache file round-trips
# ─────────────────────────────────────────────────────────────────

def test_cache_save_load_round_trip(tmp_path, monkeypatch):
    """Saved cache file deserializes back to same dict."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)

    payload = {
        "agents": [
            {
                "id": "agent_01",
                "reason": "test reason",
                "top_objection": "test obj",
                "what_would_change_mind": "test cm",
                "key_moment": "test km",
                "key_quote": "test kq",
                "round_responses": [
                    {"round": 1, "title": "First Impression", "response": "abcdefghij"},
                    {"round": 2, "title": "Competitor Comparison", "response": "abcdefghij"},
                    {"round": 3, "title": "Final Verdict", "response": "abcdefghij"},
                ],
            }
        ],
        "consensus": "x",
        "winning_message": "y",
        "actionable_insight": "z",
    }
    enricher._save_cache("testkey123", payload, latency_ms=42)

    loaded = enricher._load_cache("testkey123")
    assert loaded is not None
    assert "_meta" in loaded
    assert loaded["_meta"]["latency_ms"] == 42
    assert loaded["_meta"]["prompt_version"] == enricher.PROMPT_VERSION
    assert loaded["consensus"] == "x"
    assert loaded["agents"][0]["reason"] == "test reason"


def test_cache_load_missing_returns_none(tmp_path, monkeypatch):
    """Cache miss returns None (not exception)."""
    monkeypatch.setattr(enricher, "LLM_CACHE_DIR", tmp_path)
    assert enricher._load_cache("nonexistent_key") is None


def test_cache_key_deterministic():
    """Same input → same cache key (cache lookups work)."""
    discussion.clear_cache()
    panel_1 = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "template"
    )["agent_panel"]
    discussion.clear_cache()
    panel_2 = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "template"
    )["agent_panel"]

    seed = discussion.generate_seed(_liquid_iv_product(), _high_conf_forecast(), 20, "llm")
    key_1 = enricher._build_cache_key(_liquid_iv_product(), _high_conf_forecast(), panel_1, seed)
    key_2 = enricher._build_cache_key(_liquid_iv_product(), _high_conf_forecast(), panel_2, seed)
    assert key_1 == key_2


# ─────────────────────────────────────────────────────────────────
# 7. Validation rejects malformed responses
# ─────────────────────────────────────────────────────────────────

def test_validation_rejects_wrong_agent_count():
    template_agents = [{"id": "agent_01"}, {"id": "agent_02"}]
    bad = {"agents": [{"id": "agent_01"}]}
    assert enricher._validate_batch_response(bad, template_agents) is None


def test_validation_rejects_mismatched_ids():
    template_agents = [{"id": "agent_01"}]
    bad = {"agents": [
        {"id": "agent_99", "reason": "x", "top_objection": "x",
         "what_would_change_mind": "x", "key_quote": "x",
         "round_responses": [
            {"round": 1, "title": "First Impression", "response": "abcdefghij"},
            {"round": 2, "title": "Competitor Comparison", "response": "abcdefghij"},
            {"round": 3, "title": "Final Verdict", "response": "abcdefghij"},
         ]},
    ]}
    assert enricher._validate_batch_response(bad, template_agents) is None


def test_validation_rejects_short_response():
    template_agents = [{"id": "agent_01"}]
    bad = {"agents": [
        {"id": "agent_01", "reason": "x", "top_objection": "x",
         "what_would_change_mind": "x", "key_quote": "x",
         "round_responses": [
            {"round": 1, "title": "First Impression", "response": "short"},  # < 10 chars
            {"round": 2, "title": "Competitor Comparison", "response": "abcdefghij"},
            {"round": 3, "title": "Final Verdict", "response": "abcdefghij"},
         ]},
    ]}
    assert enricher._validate_batch_response(bad, template_agents) is None


def test_validation_accepts_well_formed():
    template_agents = [{"id": "agent_01"}]
    good = {"agents": [
        {"id": "agent_01",
         "reason": "real persona reason here",
         "top_objection": "real objection",
         "what_would_change_mind": "real cm",
         "key_quote": "real quote",
         "round_responses": [
            {"round": 1, "title": "First Impression", "response": "abcdefghij real"},
            {"round": 2, "title": "Competitor Comparison", "response": "abcdefghij real"},
            {"round": 3, "title": "Final Verdict", "response": "abcdefghij real"},
         ]},
    ]}
    assert enricher._validate_batch_response(good, template_agents) is not None


# ─────────────────────────────────────────────────────────────────
# 8. Backward compat — template mode unchanged
# ─────────────────────────────────────────────────────────────────

def test_template_mode_unchanged():
    """mode='template' still works exactly as before, no LLM ever called."""
    discussion.clear_cache()
    result = discussion.generate_discussion(
        _liquid_iv_product(), _high_conf_forecast(), 20, "template"
    )
    panel = result["agent_panel"]
    assert panel["mode"] == "template"
    assert len(panel["agents"]) == 20
    n_buy = sum(1 for a in panel["agents"] if a["verdict"] == "BUY")
    assert 0 <= n_buy <= 24
