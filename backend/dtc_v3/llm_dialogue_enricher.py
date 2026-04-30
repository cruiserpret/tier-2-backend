"""
backend/dtc_v3/llm_dialogue_enricher.py — Async-batched product-aware dialogue enrichment.

Per friend's Path-D spec (Day 3):
  - Template panel is structural ground truth (built first, all v3 shape).
  - LLM rewrites narrative fields ONLY. Cannot change forecast/verdict/score.
  - Async batched: chunks of 5 agents, max 4 concurrent batches.
  - Per-batch timeout 40s. Overall: 45s for 20-agent, 90s for 50-agent.
  - Per-batch failure -> template fallback for that batch only (not panel-wide).
  - Disk cache at llm_cache/<hash>.json — pre-warmed for demos.

PUBLIC API:
  enrich_with_llm_dialogue(panel, product, forecast, seed) -> dict | None

Returns:
  - enriched panel dict on success (mode='llm', diagnostics with batch metrics)
  - None on total failure (caller falls back to template)
"""

from __future__ import annotations
import os
import json
import hashlib
import time
import asyncio
from pathlib import Path
from typing import Any

LLM_CACHE_DIR = Path(__file__).parent / "llm_cache"
LLM_CACHE_DIR.mkdir(exist_ok=True)

# ── Tuning ──────────────────────────────────────────────────────────
OPENAI_MODEL = os.environ.get("DTC_DISCUSS_LLM_MODEL", "gpt-4o-mini")
BATCH_SIZE = 5
MAX_CONCURRENT_BATCHES = 4
PER_BATCH_TIMEOUT_S = 55
OVERALL_TIMEOUT_S = {20: 45, 50: 90}  # per agent_count
DEFAULT_OVERALL_TIMEOUT_S = 45
PROMPT_VERSION = "v2.2-comparison-context"


# ════════════════════════════════════════════════════════════════════
# PUBLIC ENTRYPOINT (sync wrapper around async batching)
# ════════════════════════════════════════════════════════════════════

def enrich_with_llm_dialogue(
    panel: dict,
    product: dict,
    forecast: dict,
    seed: str,
    comparison_context: dict | None = None,
) -> dict | None:
    """
    Synchronous entrypoint — wraps the async batch flow.
    Returns enriched panel on success, None on total failure.
    Caller (discussion.py) treats None as 'use template panel as-is'.
    """
    if not panel or not isinstance(panel, dict):
        return None
    agents = panel.get("agents", [])
    if not agents:
        return None

    # 1. Whole-panel cache lookup
    cache_key = _build_cache_key(product, forecast, panel, seed, comparison_context)
    cached = _load_cache(cache_key)
    if cached is not None:
        merged = _merge_llm_into_panel(panel, cached)
        diag = merged.setdefault("diagnostics", {})
        diag["llm_cache_hit"] = True
        diag["llm_batches"] = 0
        diag["llm_batches_failed"] = 0
        diag["partial_llm_fallback"] = False
        diag["llm_latency_ms"] = 0
        return merged

    # 2. API key check
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI  # noqa: F401
    except ImportError:
        return None

    # 3. Run async batched flow with overall timeout
    overall_timeout = OVERALL_TIMEOUT_S.get(len(agents), DEFAULT_OVERALL_TIMEOUT_S)
    try:
        result = asyncio.run(
            _run_batched_enrichment(
                panel=panel,
                product=product,
                forecast=forecast,
                seed=seed,
                api_key=api_key,
                overall_timeout=overall_timeout,
                comparison_context=comparison_context,
            )
        )
    except Exception:
        return None

    if result is None:
        return None

    # 4. Save whole-panel cache (even partial success — beats template-only)
    cacheable = {k: v for k, v in result.items() if not k.startswith("_")}
    cacheable["_partial_fallback_count"] = result.get("_failed_batches", 0)
    _save_cache(cache_key, cacheable, latency_ms=result.get("_latency_ms", 0))

    # 5. Merge into template panel
    merged = _merge_llm_into_panel(panel, result)
    merged.setdefault("diagnostics", {})["llm_cache_hit"] = False
    merged.setdefault("diagnostics", {})["partial_llm_fallback"] = bool(result.get("_partial_fallback"))
    merged.setdefault("diagnostics", {})["llm_latency_ms"] = result.get("_latency_ms", 0)
    merged.setdefault("diagnostics", {})["llm_batches"] = result.get("_batch_count", 0)
    merged.setdefault("diagnostics", {})["llm_batches_failed"] = result.get("_failed_batches", 0)
    return merged


# ════════════════════════════════════════════════════════════════════
# ASYNC BATCHED CORE
# ════════════════════════════════════════════════════════════════════

async def _run_batched_enrichment(
    *,
    panel: dict,
    product: dict,
    forecast: dict,
    seed: str,
    api_key: str,
    overall_timeout: float,
    comparison_context: dict | None = None,
) -> dict | None:
    """Returns merged LLM JSON dict, or None on total failure."""
    from openai import AsyncOpenAI

    agents = panel["agents"]
    batches = [agents[i:i + BATCH_SIZE] for i in range(0, len(agents), BATCH_SIZE)]

    client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=PER_BATCH_TIMEOUT_S)
    sem = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
    seed_int = int(seed[:15], 16)

    t_start = time.time()

    async def _do_batch(batch_idx: int, batch_agents: list) -> tuple[int, list | None]:
        """Returns (batch_idx, agent_dialogue_list_or_None)."""
        async with sem:
            try:
                user_prompt = _build_user_prompt(batch_agents, product, forecast, panel, comparison_context)
                system_prompt = _build_system_prompt()
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=OPENAI_MODEL,
                        temperature=0.4,
                        seed=seed_int + batch_idx * 1009,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    ),
                    timeout=PER_BATCH_TIMEOUT_S,
                )
                raw = response.choices[0].message.content
                if not raw:
                    return batch_idx, None
                parsed = json.loads(raw)
                cc_forbidden = set()
                if comparison_context:
                    cc_forbidden = {
                        b.lower() for b in (comparison_context.get("forbidden_brand_names") or [])
                    }
                validated = _validate_batch_response(parsed, batch_agents, forbidden_lookup=cc_forbidden)
                return batch_idx, validated
            except Exception:
                return batch_idx, None

    # Fire all batches concurrently (semaphore caps actual parallelism)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_do_batch(i, b) for i, b in enumerate(batches)]),
            timeout=overall_timeout,
        )
    except asyncio.TimeoutError:
        # Total wall-clock cap hit — give up entirely
        return None

    elapsed_ms = int((time.time() - t_start) * 1000)

    # Stitch batches back together. None entries = batch failure (template fallback for those).
    all_agents_dialogue: list[dict] = []
    failed_count = 0
    for batch_idx, batch_dialogue in results:
        if batch_dialogue is None:
            failed_count += 1
            # Synthesize template-shaped entries for failed batch (frontend-safe)
            failed_batch_agents = batches[batch_idx]
            for ta in failed_batch_agents:
                all_agents_dialogue.append({
                    "id": ta["id"],
                    "reason": ta.get("reason", ""),
                    "top_objection": ta.get("top_objection", ""),
                    "what_would_change_mind": ta.get("what_would_change_mind", ""),
                    "key_moment": ta.get("key_moment", ""),
                    "key_quote": ta.get("journey", {}).get("key_quote", ta.get("reason", "")),
                    "shift_reason": ta.get("journey", {}).get("shift_reason", ""),
                    "round_responses": ta.get("round_responses", []),
                    "_template_fallback": True,
                })
        else:
            all_agents_dialogue.extend(batch_dialogue["agents"])

    if failed_count == len(batches):
        # Every single batch failed — total failure
        return None

    # Top-level narrative — pulled from FIRST successful batch's response
    consensus = winning_message = actionable_insight = None
    for _, batch_dialogue in results:
        if batch_dialogue:
            consensus = batch_dialogue.get("consensus")
            winning_message = batch_dialogue.get("winning_message")
            actionable_insight = batch_dialogue.get("actionable_insight")
            break

    return {
        "agents": all_agents_dialogue,
        "consensus": consensus,
        "winning_message": winning_message,
        "actionable_insight": actionable_insight,
        "_partial_fallback": failed_count > 0,
        "_failed_batches": failed_count,
        "_batch_count": len(batches),
        "_latency_ms": elapsed_ms,
    }


# ════════════════════════════════════════════════════════════════════
# CACHE
# ════════════════════════════════════════════════════════════════════

def _build_cache_key(product: dict, forecast: dict, panel: dict, seed: str, comparison_context: dict | None = None) -> str:
    persona_signature = "|".join(
        f"{a.get('id')}:{a.get('name','?')}:{a.get('verdict','?')}"
        for a in panel.get("agents", [])
    )
    payload = {
        "product_name": product.get("product_name") or product.get("name"),
        "price": product.get("price"),
        "category": product.get("category"),
        "description": (product.get("description") or "")[:500],
        "competitors": [c.get("name") for c in product.get("competitors", [])[:5]],
        "trial_rate_median": forecast.get("trial_rate", {}).get("median"),
        "confidence": forecast.get("confidence"),
        "comparison_mode": (comparison_context or {}).get("comparison_mode"),
        "allowed_comparison_brands": (comparison_context or {}).get("allowed_comparison_brands", []) or [],
        "forbidden_brand_names": (comparison_context or {}).get("forbidden_brand_names", []) or [],
        "agent_count": panel.get("agent_count"),
        "persona_signature": persona_signature,
        "prompt_version": PROMPT_VERSION,
        "model": OPENAI_MODEL,
        "seed": seed[:16],
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _load_cache(key: str) -> dict | None:
    path = LLM_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(key: str, data: dict, latency_ms: int = 0) -> None:
    path = LLM_CACHE_DIR / f"{key}.json"
    record = {
        "_meta": {
            "latency_ms": latency_ms,
            "prompt_version": PROMPT_VERSION,
            "model": OPENAI_MODEL,
            "saved_at": time.time(),
        },
        **data,
    }
    try:
        with path.open("w") as f:
            json.dump(record, f, indent=2)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════
# PROMPTS — tuned for Path D (5-agent batches, no consensus per batch)
# ════════════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    return (
        "You are writing individual synthetic buyer reasoning for an AI buyer panel.\n\n"
        "STRICT RULES:\n"
        "1. You are NOT forecasting demand. The trial rate is fixed input — never reference, "
        "argue, or change it.\n"
        "2. You are NOT real buyers. You are AI personas explaining what the market would say.\n"
        "3. Each agent has a fixed verdict (BUY/CONSIDERING/WON'T BUY) and score. Do NOT change them. "
        "Your job is to write what THIS specific persona would actually say to justify that verdict.\n"
        "4. Every response MUST mention specific product details: the actual product name, price, a "
        "named competitor or anchor brand, or a feature stated in the product description. Generic "
        "phrases like 'differentiation is small' are FORBIDDEN.\n"
        "5. Each persona must sound distinct — match their profession, age, and segment. A 40-year-old "
        "Air Traffic Controller should not sound like a 22-year-old Graduate Student.\n"
        "6. Round responses escalate naturally:\n"
        "   R1 First Impression: gut reaction to seeing the product/price/claim\n"
        "   R2 Competitor Comparison: how they think about this vs the named competitors/anchors\n"
        "   R3 Final Verdict: locked-in reasoning that justifies their fixed verdict\n"
        "7. WON'T BUY agents give specific reasons rooted in their persona. CONSIDERING agents "
        "articulate what's holding them back. BUY agents say what convinced them.\n"
        "8. No causal guarantees. No 'this WILL win' or 'this is BEST'. Speak as buyers, not analysts.\n"
        "9. Each round response: 1-3 sentences. Conversational. Specific.\n"
        "10. BANNED PHRASES — never use these in any response, in any context: "
        "'guaranteed', 'causal proof', 'scientifically proven', 'real buyers', "
        "'optimal price', 'Van Westendorp', 'Juster', 'r=0.895'. "
        "Also avoid deterministic 'will buy' claims like 'customers will buy' or "
        "'the market will buy' — Phase 1 personas express purchase intent, not market certainty.\n"
        "11. PREFERRED LANGUAGE — when expressing preference or intent, use: "
        "'could buy', 'buy signal', 'purchase intent signal', 'AI buyer personas', "
        "'estimated trial rate', 'directional scenario', 'comparable-anchored forecast', "
        "'coverage-aware confidence'. Speak in directional terms, not certainties.\n"
        "12. Return STRICT JSON. No prose outside JSON.\n"
    )


def _build_user_prompt(batch_agents: list, product: dict, forecast: dict, full_panel: dict, comparison_context: dict | None = None) -> str:
    product_section = {
        "name": product.get("product_name") or product.get("name"),
        "description": product.get("description"),
        "price_usd": product.get("price"),
        "category": product.get("category"),
        "demographic": product.get("demographic"),
        "competitors": [c.get("name") for c in product.get("competitors", [])],
    }

    cc = comparison_context or {}
    cc_mode = cc.get("comparison_mode", "anchored")
    cc_allowed = cc.get("allowed_comparison_brands", []) or []
    cc_forbidden = cc.get("forbidden_brand_names", []) or []
    forecast_anchors = forecast.get("anchored_on", [])
    anchor_brand_to_rate = {a.get("brand"): a.get("trial_rate") for a in forecast_anchors if a.get("brand")}
    allowed_with_rate = [
        {"brand": b,
         "trial_rate_pct": round((anchor_brand_to_rate.get(b) or 0) * 100, 1) if anchor_brand_to_rate.get(b) is not None else None}
        for b in cc_allowed[:5]
    ]
    forecast_section = {
        "predicted_trial_rate_pct": forecast.get("trial_rate", {}).get("percentage"),
        "confidence": forecast.get("confidence"),
        "comparison_mode": cc_mode,
        "allowed_comparison_brands": allowed_with_rate,
        "forbidden_brand_names": cc_forbidden,
    }

    agents_section = []
    for a in batch_agents:
        agents_section.append({
            "id": a.get("id"),
            "name": a.get("name"),
            "age": a.get("age"),
            "profession": a.get("profession"),
            "segment": a.get("segment"),
            "profile": a.get("profile"),
            "verdict": a.get("verdict"),
            "current_score_10": a.get("current_score_10"),
            "is_hardcore": a.get("is_hardcore"),
            "shifted": a.get("shifted"),
            "journey_initial_verdict": a.get("journey", {}).get("initial_verdict"),
            "journey_final_verdict": a.get("journey", {}).get("final_verdict"),
        })

    schema_text = (
        "{\n"
        '  "agents": [\n'
        '    {\n'
        '      "id": "agent_01",\n'
        '      "reason": "1-sentence summary of why this persona has this verdict",\n'
        '      "top_objection": "1-sentence specific objection",\n'
        '      "what_would_change_mind": "1-sentence: BUY=what would make them reconsider, '
        'CONSIDERING=what would push to buy, WON\'T BUY=what would soften their no",\n'
        '      "key_moment": "1-sentence: which round moment crystallized their verdict",\n'
        '      "shift_reason": "1-sentence: only if shifted=true, why they changed",\n'
        '      "key_quote": "the most quotable 1-sentence",\n'
        '      "round_responses": [\n'
        '        {"round": 1, "title": "First Impression", "response": "..."},\n'
        '        {"round": 2, "title": "Competitor Comparison", "response": "..."},\n'
        '        {"round": 3, "title": "Final Verdict", "response": "..."}\n'
        '      ]\n'
        '    }\n'
        '  ],\n'
        '  "consensus": "1-2 sentence summary (only this batch needs to provide this)",\n'
        '  "winning_message": "1 sentence positioning",\n'
        '  "actionable_insight": "1-2 sentence concrete next action"\n'
        "}"
    )

    return (
        f"PRODUCT:\n{json.dumps(product_section, indent=2)}\n\n"
        f"FORECAST CONTEXT (do not change):\n{json.dumps(forecast_section, indent=2)}\n\n"
        f"AGENTS TO WRITE FOR (verdicts/scores/journeys are FIXED — write only their dialogue):\n"
        f"{json.dumps(agents_section, indent=2)}\n\n"
        f"For each agent, generate dialogue. Every response must mention specific product "
        f"details (name, price, competitor brand, description feature). Return ALL "
        f"{len(agents_section)} agents in the same order. Match this schema exactly:\n"
        f"{schema_text}"
    )


# ════════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════════

# Hard-banned phrases (Phase 1 trust safety, friend Apr 29-30 spec).
# Any LLM output containing these triggers per-batch template fallback.
HARD_BANNED_PHRASES = (
    "guaranteed",
    "causal proof",
    "scientifically proven",
    "real buyers",
    "optimal price",
    "van westendorp",
    "juster",
    "r=0.895",
    # Deterministic "will buy" claims — exact phrases only.
    # Generic "will" usage is fine (e.g., "I will compare it to LMNT").
    "customers will buy",
    "the market will buy",
)


def _contains_banned_phrase(text: Any) -> bool:
    """Return True if `text` contains any HARD_BANNED_PHRASES (case-insensitive).

    Non-string inputs return False. Used by _validate_batch_response to
    reject LLM outputs that contain trust-violating language and trigger
    per-batch template fallback.
    """
    if not isinstance(text, str) or not text:
        return False
    haystack = text.lower()
    return any(phrase in haystack for phrase in HARD_BANNED_PHRASES)


def _contains_forbidden_brand(text, forbidden_lookup):
    if not isinstance(text, str) or not text or not forbidden_lookup:
        return False
    haystack = text.lower()
    return any(brand in haystack for brand in forbidden_lookup)


def _validate_batch_response(parsed: Any, batch_agents: list, forbidden_lookup=None) -> dict | None:
    """Strict shape check. Returns None if anything wrong.

    P1.8.3: also rejects batch if any agent narrative or top-level
    field contains a forbidden brand name (per comparison_context).
    """
    if not isinstance(parsed, dict):
        return None
    if "agents" not in parsed or not isinstance(parsed["agents"], list):
        return None
    if len(parsed["agents"]) != len(batch_agents):
        return None

    expected_ids = [a["id"] for a in batch_agents]
    for i, llm_a in enumerate(parsed["agents"]):
        if not isinstance(llm_a, dict):
            return None
        if llm_a.get("id") != expected_ids[i]:
            return None
        for k in ("reason", "top_objection", "what_would_change_mind",
                  "key_quote", "round_responses"):
            if k not in llm_a:
                return None
        # Phase 1: hard-banned-phrase scan on per-agent narrative fields.
        # Per friend Apr 30 ruling: any banned phrase rejects the batch
        # to template fallback. Per-batch (not per-agent) is acceptable
        # for Phase 1.
        for narrative_key in (
            "reason", "top_objection", "what_would_change_mind",
            "key_moment", "key_quote", "shift_reason",
        ):
            if _contains_banned_phrase(llm_a.get(narrative_key)):
                import logging
                logging.warning(
                    "[llm_validator] hard-banned phrase in agent %s field %r — "
                    "rejecting batch to template fallback",
                    llm_a.get("id"), narrative_key,
                )
                return None
            if _contains_forbidden_brand(llm_a.get(narrative_key), forbidden_lookup):
                import logging
                logging.warning(
                    "[llm_validator] forbidden brand in agent %s field %r — "
                    "rejecting batch to template fallback",
                    llm_a.get("id"), narrative_key,
                )
                return None
        rrs = llm_a["round_responses"]
        if not isinstance(rrs, list) or len(rrs) != 3:
            return None
        for j, rr in enumerate(rrs):
            if not isinstance(rr, dict):
                return None
            if rr.get("round") != j + 1:
                return None
            if not isinstance(rr.get("response"), str) or len(rr["response"]) < 10:
                return None
            if _contains_banned_phrase(rr.get("response")):
                import logging
                logging.warning(
                    "[llm_validator] hard-banned phrase in agent %s round %d response — "
                    "rejecting batch to template fallback",
                    llm_a.get("id"), j + 1,
                )
                return None
            if _contains_forbidden_brand(rr.get("response"), forbidden_lookup):
                import logging
                logging.warning(
                    "[llm_validator] forbidden brand in agent %s round %d response — "
                    "rejecting batch to template fallback",
                    llm_a.get("id"), j + 1,
                )
                return None

    # Phase 1: top-level narrative fields also scanned.
    # P1.8.3: also scans for forbidden brands per friend Apr 30 ruling.
    for top_key in ("consensus", "winning_message", "actionable_insight"):
        if _contains_banned_phrase(parsed.get(top_key)):
            import logging
            logging.warning(
                "[llm_validator] hard-banned phrase in top-level field %r — "
                "rejecting batch to template fallback",
                top_key,
            )
            return None
        if _contains_forbidden_brand(parsed.get(top_key), forbidden_lookup):
            import logging
            logging.warning(
                "[llm_validator] forbidden brand in top-level field %r — "
                "rejecting batch to template fallback",
                top_key,
            )
            return None

    return parsed


# ════════════════════════════════════════════════════════════════════
# MERGE
# ════════════════════════════════════════════════════════════════════

def _merge_llm_into_panel(template_panel: dict, llm_data: dict) -> dict:
    """Returns NEW panel dict — does not mutate template_panel."""
    enriched = dict(template_panel)
    enriched["agents"] = []

    llm_by_id = {a["id"]: a for a in llm_data.get("agents", [])}

    for tpl_agent in template_panel.get("agents", []):
        merged = dict(tpl_agent)
        llm_a = llm_by_id.get(tpl_agent.get("id"))
        if llm_a and not llm_a.get("_template_fallback"):
            # Narrative overlay — STRUCTURAL fields untouched
            merged["reason"] = llm_a.get("reason", merged.get("reason"))
            merged["top_objection"] = llm_a.get("top_objection", merged.get("top_objection"))
            merged["what_would_change_mind"] = llm_a.get(
                "what_would_change_mind", merged.get("what_would_change_mind"))
            merged["key_moment"] = llm_a.get("key_moment", merged.get("key_moment"))
            merged["round_responses"] = llm_a.get("round_responses", merged.get("round_responses"))

            # Journey — only narrative fields
            tpl_journey = dict(merged.get("journey", {}))
            if llm_a.get("key_quote"):
                tpl_journey["key_quote"] = llm_a["key_quote"]
            if llm_a.get("shift_reason"):
                tpl_journey["shift_reason"] = llm_a["shift_reason"]
            merged["journey"] = tpl_journey

        enriched["agents"].append(merged)

    # Top-level narrative overlays
    if llm_data.get("consensus"):
        enriched["consensus"] = llm_data["consensus"]
    if llm_data.get("winning_message"):
        enriched["winning_message"] = llm_data["winning_message"]
    if llm_data.get("actionable_insight"):
        enriched["actionable_insight"] = llm_data["actionable_insight"]

    enriched["mode"] = "llm"
    return enriched
