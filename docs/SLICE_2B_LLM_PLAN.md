# Slice 2B-LLM — Buyer Dialogue Implementation Plan

Status: Architecture locked, implementation deferred to next session per friend's call.

Decision: D = pre-cache demos + async parallel batches for live custom products.

## Implementation Spec

Batch size: 5 agents per LLM call
Max concurrent batches: 4
20 agents -> 4 batches -> ~30s wall time
50 agents -> 10 batches, 4 at a time -> ~60-90s wall time

Per-batch timeout: 40s
Overall timeout for 20 agents: 45s
Overall timeout for 50 agents: 90s
On timeout: return improved-template panel.

Demo presets: use pre-cached LLM panels (instant)
Custom products: default to template-improved panel
Optional button "Generate richer buyer dialogue" calls LLM mode

Cache key: hash(product_payload + forecast_core + agent_count + persona_ids + model + prompt_version + mode)
Cache location: backend/dtc_v3/llm_cache/<hash>.json

Pre-cache demos: Liquid IV, Triton, YETI, SoberCraft, Nova Ring, Mushroom Morning, LuxeFoam.
20 agents each, mode=llm.

## Forecast Invariants (Non-negotiable)

LLM may rewrite ONLY:
- reason, top_objection, what_would_change_mind, key_moment
- round_responses[].response
- journey.shift_reason, journey.key_quote
- top-level: consensus, winning_message, actionable_insight

LLM may NOT touch:
- trial_rate, confidence, coverage_tier, fallback_used
- verdict, score, score_10, intent_distribution
- agent.name, agent.segment, journey.initial/final_verdict
- bucket counts, hardcore status, anchored_on

## Failure Handling

Per-batch failure -> use template for THAT batch only
Mark diagnostics.partial_llm_fallback = true
Never fail the entire simulation

## Required Tests

- Cache hit returns immediately (no API call)
- LLM timeout falls back to template
- Partial batch failure falls back only failed batch
- LLM output preserves agent ids/verdicts/scores
- LLM cannot change trial_rate/confidence/verdict
- Cache file structure round-trips correctly

## Recovery From This Session

WIP stashed at: stash@{0} "wip slice 2b llm batching plan - do not deploy"
Contains: discussion.py wiring + llm_dialogue_enricher.py (340 lines, synchronous single-call)

To recover next session:
  git stash list
  git stash show -p stash@{0}
  git stash pop stash@{0}

The synchronous enricher is the WRONG shape for D (needs to become async batch).
Treat the stash as REFERENCE (prompts work, validation works, cache helpers work),
not as the implementation to ship. Rewrite as enrich_agents_llm_batched().

## Floor For This Session (Already Shipped)

- Slice 2A at commit 047324a: report-page agent cards render
- 106 tests passing
- Branch tier2-v3-ui-rebuild clean, pushed
- Pransh can QA agent cards starting now
