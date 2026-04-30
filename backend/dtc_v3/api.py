"""
backend/dtc_v3/api.py — Flask blueprint for v3 forecasts.

Mounts at /api/dtc_v3/* in parallel with v1.
v1 routes remain untouched.
"""

from __future__ import annotations
import uuid
import time
import json
from pathlib import Path
from flask import Blueprint, request, jsonify

from .models import ProductBrief, PersonaSignals
from .forecast import forecast as v3_forecast
from .customer_report import build_customer_report, render_report_text
from .rag_retrieval import (
    retrieve_neighbors, _infer_query_subtype, _infer_query_market_structure
)
from .coverage_gate import assess_coverage
from .ground_truth_db import GROUND_TRUTH_DB
from .evidence import _normalize_brand_for_lookup


# ── Phase 1 helper: capture forecast core for /discuss mutation guard ──
def _extract_forecast_core(forecast_obj: dict) -> dict:
    """Extract the trust-critical forecast fields for mutation comparison.

    Per friend Apr 30 ruling: /discuss must never mutate trial_rate,
    confidence, verdict, fallback_used, or coverage_tier — neither in
    the input forecast dict (in-place) nor in the response forecast.
    """
    if not isinstance(forecast_obj, dict):
        return {}
    diagnostics = forecast_obj.get("diagnostics", {}) or {}
    nested_forecast = forecast_obj.get("forecast", {}) or {}
    return {
        "trial_rate": forecast_obj.get("trial_rate"),
        "confidence": forecast_obj.get("confidence"),
        "verdict": forecast_obj.get("verdict"),
        "fallback_used": (
            forecast_obj.get("fallback_used")
            if forecast_obj.get("fallback_used") is not None
            else (diagnostics.get("fallback_used")
                  if diagnostics.get("fallback_used") is not None
                  else nested_forecast.get("fallback_used"))
        ),
        "coverage_tier": (
            forecast_obj.get("coverage_tier")
            if forecast_obj.get("coverage_tier") is not None
            else (diagnostics.get("coverage_tier")
                  if diagnostics.get("coverage_tier") is not None
                  else nested_forecast.get("coverage_tier"))
        ),
    }


api_v3 = Blueprint("dtc_v3", __name__, url_prefix="/api/dtc_v3")


# In-memory simulation store (matches v1/v2 pattern)
_simulations: dict = {}


@api_v3.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v3-lite",
        "architecture": "rag_prior + coverage_gate + persona_adjustment",
    })


@api_v3.route("/forecast", methods=["POST"])
def create_forecast():
    """
    Synchronous forecast endpoint.
    Input: ProductBrief fields + optional debate_state/market_intel
    Output: Customer report JSON
    """
    data = request.get_json() or {}

    # Build ProductBrief
    try:
        brief = ProductBrief(
            name=data["product_name"],
            description=data.get("description", ""),
            price=float(data.get("price", 0)),
            category=data.get("category", "fashion_apparel"),
            demographic=data.get("demographic", ""),
            competitors=data.get("competitors", []),
            market_tier_override=data.get("market_tier_override"),
            distribution_hint=data.get("distribution_hint"),
        )
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # Optional: debate_state + market_intel for persona adjustment
    debate_state = data.get("debate_state")
    market_intel = data.get("market_intel")
    exclude_brand = data.get("exclude_brand")

    # ── Run forecast ──
    try:
        f = v3_forecast(brief, debate_state, market_intel, exclude_brand)
    except Exception as e:
        return jsonify({"error": f"Forecast failed: {e}"}), 500

    # ── Get coverage tier for verdict (k=6 unchanged — coverage_tier
    #    flows into verdict, so this call is NOT display-only) ──
    neighbors = retrieve_neighbors(brief, k=6, exclude_brand=exclude_brand)
    subtype = _infer_query_subtype(brief)
    q_tier, _, _, _ = _infer_query_market_structure(brief)
    coverage = assess_coverage(neighbors, subtype, brief.category, q_tier)

    # ── Phase 1: separate display-only retrieval for Evidence Panel
    #    candidate pool (per friend Apr 30 ruling — keep coverage path
    #    untouched, do display retrieval as a separate call). ──
    candidate_pool = retrieve_neighbors(brief, k=12, exclude_brand=exclude_brand)

    # Build record_by_brand lookup map (Option B per friend Apr 29).
    record_by_brand: dict = {}
    for _r in GROUND_TRUTH_DB:
        record_by_brand[_normalize_brand_for_lookup(_r.brand)] = _r
        for _alias in (getattr(_r, "aliases", None) or []):
            record_by_brand[_normalize_brand_for_lookup(_alias)] = _r

    # candidate_neighbors: 0.30 <= sim < 0.45, NOT already used in forecast math.
    used_brand_keys = {
        _normalize_brand_for_lookup(n.brand) for n in f.neighbors
    }
    candidate_neighbors = [
        n for n in candidate_pool
        if 0.30 <= n.similarity < 0.45
        and _normalize_brand_for_lookup(n.brand) not in used_brand_keys
    ][:5]

    # ── Build customer report (classifiers run ONCE inside the report
    #    builder per friend's Apr 30 ruling — no double classification). ──
    report = build_customer_report(
        f, brief,
        coverage_tier=coverage["tier"],
        coverage_subtype=subtype,
        top_drivers=data.get("top_drivers", []),
        top_objections=data.get("top_objections", []),
        most_receptive_segment=data.get("most_receptive_segment", ""),
        record_by_brand=record_by_brand,
        candidate_neighbors=candidate_neighbors,
        inferred_subtype=subtype,
    )

    # ── Generate ID + cache ──
    sim_id = f"v3_{uuid.uuid4().hex[:8]}"
    _simulations[sim_id] = {
        "brief": brief.__dict__,
        "report": report,
        "timestamp": time.time(),
    }

    # ── Serialize report for JSON ──
    return jsonify({
        "simulation_id": sim_id,
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
        # Phase 1 — Evidence Panel + Confidence Ledger
        "evidence_buckets": report.evidence_buckets,
        "confidence_ledger": report.confidence_ledger,
        "comparison_context": report.comparison_context,
        "diagnostics": {
            "rag_prior": round(report.rag_prior, 4),
            "adjustment_applied": round(report.adjustment_applied, 4),
            "coverage_tier": report.coverage_tier,
        },
    })


@api_v3.route("/forecast/<sim_id>", methods=["GET"])
def get_forecast(sim_id: str):
    """Retrieve a previously-computed forecast by ID."""
    sim = _simulations.get(sim_id)
    if not sim:
        return jsonify({"error": "Simulation not found"}), 404
    return jsonify({
        "simulation_id": sim_id,
        "report_text": render_report_text(sim["report"]),
        "timestamp": sim["timestamp"],
    })


@api_v3.route("/comparables/<brand_query>", methods=["GET"])
def lookup_comparables(brand_query: str):
    """Debug endpoint: see what neighbors a query would retrieve."""
    fake_brief = ProductBrief(
        name=brand_query, description=brand_query,
        price=50, category="fashion_apparel",
        demographic="general adults", competitors=[],
    )
    neighbors = retrieve_neighbors(fake_brief, k=8)
    return jsonify({
        "query": brand_query,
        "neighbors": [
            {
                "brand": n.brand,
                "trial_rate": n.trial_rate_mid,
                "similarity": round(n.similarity, 3),
                "weight": round(n.source_weight, 4),
                "confidence": n.confidence,
                "reason": n.reason,
            }
            for n in neighbors
        ],
    })


@api_v3.route("/discuss", methods=["POST"])
def create_discussion():
    """
    AI buyer panel for an existing forecast.

    Input JSON:
      {
        "product": {...},        # full product payload (same as /forecast input)
        "forecast": {...},       # full forecast response from /forecast
        "agent_count": 20 | 50   # default 20
      }

    Output JSON:
      { "agent_panel": {...} }

    Invariants:
      - Discussion does NOT change the forecast number.
      - Same input → same output (deterministic via seed + cache).
    """
    from .discussion import (
        generate_discussion,
        ALLOWED_AGENT_COUNTS,
        DEFAULT_AGENT_COUNT,
        ALLOWED_MODES,
        DEFAULT_MODE,
    )

    data = request.get_json() or {}
    product = data.get("product")
    forecast = data.get("forecast")
    agent_count = data.get("agent_count", DEFAULT_AGENT_COUNT)

    mode = (
        request.args.get("mode")
        or data.get("mode")
        or ("llm" if str(request.args.get("llm", "")).lower() == "true" else None)
        or DEFAULT_MODE
    )

    if not isinstance(product, dict) or not product:
        return jsonify({"error": "Missing or invalid 'product'"}), 400
    if not isinstance(forecast, dict) or not forecast:
        return jsonify({"error": "Missing or invalid 'forecast'"}), 400
    if not isinstance(agent_count, int) or agent_count < 20 or agent_count > 50:
        return jsonify({
            "error": "agent_count must be an integer between 20 and 50",
            "got": agent_count,
        }), 400
    if mode not in ALLOWED_MODES:
        return jsonify({
            "error": f"mode must be one of {list(ALLOWED_MODES)}",
            "got": mode,
        }), 400

    # ── Phase 1: capture forecast core BEFORE generate_discussion ──
    # Per friend Apr 30 ruling: guard at the API trust boundary against
    # both in-place mutation of the input forecast dict AND mutation of
    # result["forecast"]. Both must be checked.
    captured_core = _extract_forecast_core(forecast)

    comparison_context = forecast.get("comparison_context") if isinstance(forecast, dict) else None

    try:
        result = generate_discussion(
            product, forecast, agent_count, mode=mode,
            comparison_context=comparison_context,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Discussion failed: {e}"}), 500

    # ── Phase 1: verify forecast core was not mutated ──
    observed_input_core = _extract_forecast_core(forecast)
    observed_result_core = None
    if isinstance(result, dict) and isinstance(result.get("forecast"), dict):
        observed_result_core = _extract_forecast_core(result["forecast"])

    mutation_detected = (observed_input_core != captured_core) or (
        observed_result_core is not None
        and observed_result_core != captured_core
    )

    if mutation_detected:
        import logging
        logging.critical(
            "[/discuss] forecast mutation detected. captured=%s "
            "observed_input=%s observed_result=%s",
            captured_core, observed_input_core, observed_result_core,
        )
        # Restore captured fields in the input forecast dict (defensive).
        for _k, _v in captured_core.items():
            if _v is None:
                continue
            if _k == "fallback_used":
                if "fallback_used" in forecast:
                    forecast["fallback_used"] = _v
                if isinstance(forecast.get("forecast"), dict):
                    forecast["forecast"]["fallback_used"] = _v
            elif _k == "coverage_tier":
                if "coverage_tier" in forecast:
                    forecast["coverage_tier"] = _v
                if isinstance(forecast.get("diagnostics"), dict):
                    forecast["diagnostics"]["coverage_tier"] = _v
            else:
                if _k in forecast:
                    forecast[_k] = _v
        # Restore captured fields in the result forecast (if present).
        if isinstance(result, dict) and isinstance(result.get("forecast"), dict):
            for _k, _v in captured_core.items():
                if _v is None:
                    continue
                if _k == "fallback_used":
                    if "fallback_used" in result["forecast"]:
                        result["forecast"]["fallback_used"] = _v
                    if isinstance(result["forecast"].get("forecast"), dict):
                        result["forecast"]["forecast"]["fallback_used"] = _v
                elif _k == "coverage_tier":
                    if "coverage_tier" in result["forecast"]:
                        result["forecast"]["coverage_tier"] = _v
                    if isinstance(result["forecast"].get("diagnostics"), dict):
                        result["forecast"]["diagnostics"]["coverage_tier"] = _v
                else:
                    if _k in result["forecast"]:
                        result["forecast"][_k] = _v
        # Surface the diagnostic flag.
        if isinstance(result, dict):
            result.setdefault("diagnostics", {})["forecast_mutation_blocked"] = True

    return jsonify(result)
