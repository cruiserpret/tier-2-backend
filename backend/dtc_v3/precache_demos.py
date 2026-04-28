"""
backend/dtc_v3/precache_demos.py — Pre-warm LLM dialogue cache for demo products.

Per friend's Path D spec. After this runs once and the demo_*.json
files are committed, production runtime mode='llm' calls for these
demos will hit cache instantly without any OpenAI traffic.

Usage:
    cd ~/Desktop/Aseembly/assembly-backend
    set -a; source .env; set +a   # OPENAI_API_KEY required
    PYTHONPATH=. python backend/dtc_v3/precache_demos.py

Adds 6 demo cache files. Re-running is safe (idempotent — overwrites
with the same hash if anchors/personas/prompt-version don't change).
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.dtc_v3 import discussion
from backend.dtc_v3 import llm_dialogue_enricher as enricher
from backend.dtc_v3.models import ProductBrief
from backend.dtc_v3.forecast import forecast as v3_forecast
from backend.dtc_v3.customer_report import build_customer_report
from backend.dtc_v3.rag_retrieval import (
    retrieve_neighbors,
    _infer_query_subtype,
    _infer_query_market_structure,
)
from backend.dtc_v3.coverage_gate import assess_coverage


# ─────────────────────────────────────────────────────────────────
# 6 demos from frontend/src/data/demoProducts.ts
# ─────────────────────────────────────────────────────────────────

DEMOS = [
    {
        "key": "liquid_iv",
        "payload": {
            "name": "Liquid IV Hydration Multiplier",
            "product_name": "Liquid IV Hydration Multiplier",
            "description": "Electrolyte drink mix powder. Costco, Target, Walmart.",
            "price": 25,
            "category": "supplements_health",
            "demographic": "Active adults 22-50",
            "competitors": [{"name": "LMNT"}, {"name": "Pedialyte"}],
            "market_tier_override": "mass_market",
            "distribution_hint": "mass_retail",
            "exclude_brand": "Liquid IV",
        },
    },
    {
        "key": "nova_ring",
        "payload": {
            "name": "Nova Ring",
            "product_name": "Nova Ring",
            "description": "Premium smart ring for sleep, recovery, readiness, and health tracking.",
            "price": 299,
            "category": "electronics_tech",
            "demographic": "Health-conscious professionals and fitness optimizers",
            "competitors": [{"name": "Oura"}, {"name": "Whoop"}],
            "market_tier_override": "premium_niche",
            "distribution_hint": "dtc_led",
            "exclude_brand": "Nova Ring",
        },
    },
    {
        "key": "sobercraft_ipa",
        "payload": {
            "name": "SoberCraft IPA",
            "product_name": "SoberCraft IPA",
            "description": "Premium nonalcoholic craft IPA for sober-curious adults and health-conscious beer drinkers.",
            "price": 14,
            "category": "food_beverage",
            "demographic": "Sober-curious adults, athletes, and craft beer drinkers",
            "competitors": [{"name": "Athletic Brewing"}, {"name": "Heineken 0.0"}, {"name": "Partake"}],
            "market_tier_override": "challenger",
            "distribution_hint": "retail_plus_dtc",
            "exclude_brand": "SoberCraft IPA",
        },
    },
    {
        "key": "mushroom_morning",
        "payload": {
            "name": "Mushroom Morning",
            "product_name": "Mushroom Morning",
            "description": "Mushroom coffee alternative with adaptogens for energy, focus, and lower caffeine.",
            "price": 40,
            "category": "food_beverage",
            "demographic": "Wellness-oriented coffee drinkers and productivity-focused adults",
            "competitors": [{"name": "MUDWTR"}, {"name": "Four Sigmatic"}, {"name": "Ryze"}],
            "market_tier_override": "premium_niche",
            "distribution_hint": "dtc_led",
            "exclude_brand": "Mushroom Morning",
        },
    },
    {
        "key": "luxefoam_mattress",
        "payload": {
            "name": "LuxeFoam Mattress",
            "product_name": "LuxeFoam Mattress",
            "description": "Memory foam mattress in a box, premium tier, DTC.",
            "price": 1295,
            "category": "home_lifestyle",
            "demographic": "Adults 25-45 furnishing first apartment or upgrading",
            "competitors": [{"name": "Casper"}, {"name": "Purple"}, {"name": "Tuft & Needle"}],
            "market_tier_override": "challenger",
            "distribution_hint": "dtc_led",
            "exclude_brand": "LuxeFoam Mattress",
        },
    },
    {
        "key": "yeti",
        "payload": {
            "name": "YETI Rambler 20oz Tumbler",
            "product_name": "YETI Rambler 20oz Tumbler",
            "description": "Insulated stainless steel tumbler.",
            "price": 35,
            "category": "home_lifestyle",
            "demographic": "Outdoor enthusiasts 25-55",
            "competitors": [{"name": "Hydro Flask"}, {"name": "Stanley"}],
            "market_tier_override": "mass_market",
            "distribution_hint": "retail_plus_dtc",
            "exclude_brand": "YETI",
        },
    },
]

AGENT_COUNT = 20


def build_forecast_dict(payload: dict) -> dict:
    """Replicate /api/dtc_v3/forecast logic — produce the JSON shape that
    /discuss receives at runtime. No HTTP."""

    brief = ProductBrief(
        name=payload["name"],
        description=payload.get("description", ""),
        price=float(payload["price"]),
        category=payload.get("category", "fashion_apparel"),
        demographic=payload.get("demographic", ""),
        competitors=payload.get("competitors", []),
        market_tier_override=payload.get("market_tier_override"),
        distribution_hint=payload.get("distribution_hint"),
    )
    exclude_brand = payload.get("exclude_brand")

    f = v3_forecast(brief, None, None, exclude_brand)

    neighbors = retrieve_neighbors(brief, k=6, exclude_brand=exclude_brand)
    subtype = _infer_query_subtype(brief)
    q_tier, _, _, _ = _infer_query_market_structure(brief)
    coverage = assess_coverage(neighbors, subtype, brief.category, q_tier)

    report = build_customer_report(
        f, brief,
        coverage_tier=coverage["tier"],
        coverage_subtype=subtype,
        top_drivers=[],
        top_objections=[],
        most_receptive_segment="",
    )

    return {
        "simulation_id": "precache",
        "version": "v3-lite",
        "verdict": report.verdict,
        "headline": report.headline,
        "trial_rate": {
            "median": round(report.forecast_pct, 4),
            "low": round(report.forecast_low, 4),
            "high": round(report.forecast_high, 4),
            "percentage": round(report.forecast_pct * 100, 1),
        },
        "confidence": report.confidence,
        "confidence_reasons": report.confidence_reasons,
        "anchored_on": report.anchored_on,
        "downweighted_brands": report.downweighted_brands,
        "why_might_be_wrong": report.why_might_be_wrong,
        "counterfactuals": [
            {
                "label": cf.label,
                "description": cf.description,
                "delta_logit": round(cf.delta_logit, 3),
                "new_prediction_pct": round(cf.new_prediction * 100, 1),
                "direction": cf.direction,
            }
            for cf in report.counterfactuals
        ],
        "top_drivers": report.top_drivers,
        "top_objections": report.top_objections,
        "most_receptive_segment": report.most_receptive_segment,
        "diagnostics": {
            "rag_prior": round(report.rag_prior, 4),
            "adjustment_applied": round(report.adjustment_applied, 4),
            "coverage_tier": report.coverage_tier,
        },
    }


def precache_demo(demo: dict) -> dict:
    """Run forecast + LLM enrichment for one demo."""
    key = demo["key"]
    payload = demo["payload"]

    print(f"\n{'=' * 70}")
    print(f"  {key}")
    print(f"{'=' * 70}")

    # 1. Forecast
    t_fc_start = time.time()
    forecast_dict = build_forecast_dict(payload)
    fc_elapsed = time.time() - t_fc_start
    print(f"  Forecast: {forecast_dict['trial_rate']['percentage']:.1f}% / "
          f"{forecast_dict['confidence']} / {fc_elapsed:.1f}s")

    # 2. Build template panel
    discussion.clear_cache()
    template_result = discussion.generate_discussion(
        payload, forecast_dict, AGENT_COUNT, "template"
    )
    template_panel = template_result["agent_panel"]
    print(f"  Template panel: {len(template_panel['agents'])} agents")

    # 3. Compute hash key
    seed = discussion.generate_seed(payload, forecast_dict, AGENT_COUNT, "llm")
    cache_key_hash = enricher._build_cache_key(payload, forecast_dict, template_panel, seed)
    runtime_cache_path = enricher.LLM_CACHE_DIR / f"{cache_key_hash}.json"
    demo_cache_path = enricher.LLM_CACHE_DIR / f"demo_{key}.json"

    print(f"  Cache hash: {cache_key_hash}")

    # 4. Run LLM enrichment
    print(f"  Calling LLM (4 parallel batches of 5)...")
    t_llm_start = time.time()
    enriched = enricher.enrich_with_llm_dialogue(
        panel=template_panel,
        product=payload,
        forecast=forecast_dict,
        seed=seed,
    )
    llm_elapsed = time.time() - t_llm_start

    if enriched is None:
        print(f"  FAILED — enricher returned None")
        return {"key": key, "failed": True, "elapsed_s": llm_elapsed}

    diag = enriched.get("diagnostics", {})
    print(f"  LLM latency: {llm_elapsed:.1f}s "
          f"(batches: {diag.get('llm_batches')}, "
          f"failed: {diag.get('llm_batches_failed')}, "
          f"cache_hit: {diag.get('llm_cache_hit')})")

    # 5. Copy runtime cache to durable demo_*.json
    if runtime_cache_path.exists():
        shutil.copy(runtime_cache_path, demo_cache_path)
        print(f"  Cached -> demo_{key}.json")
    elif demo_cache_path.exists():
        print(f"  Cache hit — demo_{key}.json already exists")
    else:
        print(f"  Neither runtime nor demo cache file present")
        return {"key": key, "failed": True}

    # 6. Quick quality check
    for a in enriched["agents"]:
        if a["verdict"] == "BUY":
            print(f"  Sample BUY: {a['name']}, {a['profession']}")
            print(f"    R1: {a['round_responses'][0]['response'][:120]}...")
            break

    return {
        "key": key,
        "ok": True,
        "elapsed_s": fc_elapsed + llm_elapsed,
        "fc_pct": forecast_dict["trial_rate"]["percentage"],
        "fc_conf": forecast_dict["confidence"],
    }


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("FATAL: OPENAI_API_KEY not set. Run 'set -a; source .env; set +a' first.")
        return 1

    print("=" * 70)
    print("  DEMO PRE-CACHE — Slice 2B-Demo")
    print("=" * 70)
    print(f"  6 demos x ~30s each = ~3 min total wall time")
    print(f"  Cache dir: {enricher.LLM_CACHE_DIR}")

    results = []
    t_total_start = time.time()
    for demo in DEMOS:
        try:
            results.append(precache_demo(demo))
        except Exception as e:
            print(f"  EXCEPTION on {demo['key']}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"key": demo["key"], "failed": True, "error": str(e)})

    total_elapsed = time.time() - t_total_start

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    ok_count = sum(1 for r in results if r.get("ok"))
    failed_count = sum(1 for r in results if r.get("failed"))
    print(f"  OK: {ok_count}  Failed: {failed_count}")
    print(f"  Total wall time: {total_elapsed:.1f}s")
    print()
    for r in results:
        marker = "OK" if r.get("ok") else "FAIL"
        details = ""
        if r.get("ok"):
            details = f"{r.get('fc_pct',0):.1f}% / {r.get('fc_conf','?')}"
        print(f"  [{marker}] {r['key']:25s}  {details}")

    print()
    print(f"  Demo cache files:")
    for f in sorted(enricher.LLM_CACHE_DIR.glob("demo_*.json")):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:40s}  {size_kb} KB")

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
