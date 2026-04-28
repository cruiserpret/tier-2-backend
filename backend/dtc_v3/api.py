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

    # ── Get coverage tier for verdict ──
    neighbors = retrieve_neighbors(brief, k=6, exclude_brand=exclude_brand)
    subtype = _infer_query_subtype(brief)
    q_tier, _, _, _ = _infer_query_market_structure(brief)
    coverage = assess_coverage(neighbors, subtype, brief.category, q_tier)

    # ── Build customer report ──
    report = build_customer_report(
        f, brief,
        coverage_tier=coverage["tier"],
        coverage_subtype=subtype,
        top_drivers=data.get("top_drivers", []),
        top_objections=data.get("top_objections", []),
        most_receptive_segment=data.get("most_receptive_segment", ""),
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
    from .discussion import generate_discussion, ALLOWED_AGENT_COUNTS, DEFAULT_AGENT_COUNT

    data = request.get_json() or {}
    product = data.get("product")
    forecast = data.get("forecast")
    agent_count = data.get("agent_count", DEFAULT_AGENT_COUNT)

    if not isinstance(product, dict) or not product:
        return jsonify({"error": "Missing or invalid 'product'"}), 400
    if not isinstance(forecast, dict) or not forecast:
        return jsonify({"error": "Missing or invalid 'forecast'"}), 400
    if agent_count not in ALLOWED_AGENT_COUNTS:
        return jsonify({
            "error": f"agent_count must be one of {list(ALLOWED_AGENT_COUNTS)}",
            "got": agent_count,
        }), 400

    try:
        result = generate_discussion(product, forecast, agent_count)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Discussion failed: {e}"}), 500

    return jsonify(result)
