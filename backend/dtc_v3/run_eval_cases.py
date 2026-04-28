"""
backend/dtc_v3/run_eval_cases.py — Baseline evaluation runner.

Per friend Day 2 Step 6 spec:
  - Direct Python forecast() calls (no HTTP, no Railway)
  - Baseline-first: no expectations, just capture actual outputs
  - Output table for review BEFORE setting regression standards

Usage:
    cd ~/Desktop/Aseembly/assembly-backend
    python -m backend.dtc_v3.run_eval_cases
or:
    python backend/dtc_v3/run_eval_cases.py
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

# Make local imports work whether invoked as module or script
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.dtc_v3.models import ProductBrief
from backend.dtc_v3.forecast import forecast as v3_forecast
from backend.dtc_v3.rag_retrieval import (
    retrieve_neighbors,
    _infer_query_subtype,
    _infer_query_market_structure,
)
from backend.dtc_v3.coverage_gate import assess_coverage

CASES_PATH = _HERE / "eval_cases_v1.json"
OUTPUT_PATH = _HERE / "eval_baseline_v2_after_routing_fixes.json"


def build_brief(payload: dict) -> ProductBrief:
    """Build a ProductBrief from the eval case product payload."""
    return ProductBrief(
        name=payload["product_name"],
        description=payload.get("description", ""),
        price=float(payload.get("price", 0)),
        category=payload.get("category", "default"),
        demographic=payload.get("demographic", ""),
        competitors=payload.get("competitors", []),
        market_tier_override=payload.get("market_tier_override"),
        distribution_hint=payload.get("distribution_hint"),
    )


def run_case(case: dict) -> dict:
    """Run one eval case and capture diagnostic output."""
    case_id = case["id"]
    bucket = case.get("_bucket", "?")
    archetype = case.get("_archetype", "")
    product_payload = case["product"]
    exclude_brand = product_payload.get("exclude_brand")

    brief = build_brief(product_payload)
    inferred_subtype = _infer_query_subtype(brief)

    t0 = time.perf_counter()
    try:
        f = v3_forecast(brief, exclude_brand=exclude_brand)
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        f = None
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if not ok:
        return {
            "id": case_id,
            "bucket": bucket,
            "archetype": archetype,
            "ok": False,
            "error": err,
            "elapsed_ms": elapsed_ms,
        }

    neighbors = retrieve_neighbors(brief, k=6, exclude_brand=exclude_brand)
    q_tier, _, _, _ = _infer_query_market_structure(brief)
    coverage = assess_coverage(neighbors, inferred_subtype, brief.category, q_tier)

    top_anchors = [
        {
            "brand": n.brand,
            "trial_rate_pct": round(n.trial_rate_mid * 100, 1),
            "similarity": round(n.similarity, 3),
            "weight": round(n.source_weight, 3),
            "confidence_grade": n.confidence,
        }
        for n in (f.neighbors or [])[:5]
    ]

    quality_warnings = list(f.data_quality.quality_warnings or [])

    return {
        "id": case_id,
        "bucket": bucket,
        "archetype": archetype,
        "ok": True,
        "inferred_subtype": inferred_subtype,
        "forecast_pct": round(f.trial_rate_median * 100, 1),
        "low_pct": round(f.trial_rate_low * 100, 1),
        "high_pct": round(f.trial_rate_high * 100, 1),
        "confidence": f.confidence,
        "coverage_tier": coverage.get("tier"),
        "fallback_used": f.fallback_used,
        "prior_source": f.prior_source,
        "eligible_count": f.eligible_neighbor_count,
        "retrieved_count": f.retrieved_candidate_count,
        "top_anchors": top_anchors,
        "quality_warnings": quality_warnings,
        "elapsed_ms": elapsed_ms,
    }


def print_table(results: list[dict]) -> None:
    """Pretty-print results grouped by bucket."""

    def fmt_anchors(anchors: list[dict], limit: int = 3) -> str:
        return ", ".join(f"{a['brand']}({a['trial_rate_pct']}%)" for a in anchors[:limit])

    bucket_order = ["synthetic_parallel", "synthetic_new", "weird_sparse"]

    print()
    print("=" * 132)
    print(f"{'BUCKET':<22} {'ID':<28} {'SUBTYPE':<22} {'RATE':>6} {'CONF':<13} {'COV':<8} {'FALLBACK':<10} {'TOP ANCHORS'}")
    print("=" * 132)

    for bucket in bucket_order:
        bucket_results = [r for r in results if r["bucket"] == bucket]
        if not bucket_results:
            continue
        print(f"\n--- {bucket.upper()} ({len(bucket_results)} cases) ---")
        for r in bucket_results:
            if not r["ok"]:
                print(f"{r['bucket']:<22} {r['id']:<28} ERROR: {r['error']}")
                continue
            print(
                f"{r['bucket']:<22} {r['id']:<28} {r['inferred_subtype']:<22} "
                f"{r['forecast_pct']:>5.1f}% {r['confidence']:<13} "
                f"{r['coverage_tier']:<8} {str(r['fallback_used']):<10} "
                f"{fmt_anchors(r['top_anchors'])}"
            )

    print()
    print("=" * 132)


def print_summary(results: list[dict]) -> None:
    n = len(results)
    ok_count = sum(1 for r in results if r["ok"])
    err_count = n - ok_count
    fallback_count = sum(1 for r in results if r.get("fallback_used"))

    by_conf: dict[str, int] = {}
    by_cov: dict[str, int] = {}
    by_subtype: dict[str, int] = {}
    by_verdict_proxy: dict[str, int] = {}

    for r in results:
        if not r["ok"]:
            continue
        by_conf[r["confidence"]] = by_conf.get(r["confidence"], 0) + 1
        by_cov[r["coverage_tier"]] = by_cov.get(r["coverage_tier"], 0) + 1
        by_subtype[r["inferred_subtype"]] = by_subtype.get(r["inferred_subtype"], 0) + 1

    print("\n=== SUMMARY ===")
    print(f"  total cases:    {n}")
    print(f"  ran cleanly:    {ok_count}")
    print(f"  errors:         {err_count}")
    print(f"  fallback fired: {fallback_count}/{ok_count}")
    print()
    print("  by confidence:")
    for k in ("high", "medium-high", "medium", "medium-low", "low"):
        if k in by_conf:
            print(f"    {k:<13} {by_conf[k]}")
    print()
    print("  by coverage tier:")
    for k in ("strong", "medium", "thin", "weak"):
        if k in by_cov:
            print(f"    {k:<8} {by_cov[k]}")
    print()
    print("  by inferred subtype:")
    for k, v in sorted(by_subtype.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28} {v}")


def main() -> int:
    cases = json.loads(CASES_PATH.read_text())
    print(f"Loaded {len(cases)} eval cases from {CASES_PATH.name}")
    print(f"Running direct-Python forecast() — no HTTP, no Railway.\n")

    results: list[dict] = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i:2d}/{len(cases)}] {case['id']:<28} ({case['_bucket']})  ...", end="", flush=True)
        r = run_case(case)
        results.append(r)
        if r["ok"]:
            print(f"  {r['forecast_pct']:>5.1f}% {r['confidence']:<13} cov={r['coverage_tier']}  ({r['elapsed_ms']}ms)")
        else:
            print(f"  ERROR: {r['error']}  ({r['elapsed_ms']}ms)")

    print_table(results)
    print_summary(results)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull JSON results written to: {OUTPUT_PATH}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
