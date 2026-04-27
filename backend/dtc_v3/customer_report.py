"""
backend/dtc_v3/customer_report.py — Decision-oriented customer-facing report.

Per friend's "trust layer" advice:
- Show comparable brands visibly (don't hide RAG)
- Decision verdict, not just numbers
- Include "why this forecast may be wrong"
- Make counterfactuals the star

Target audience: DTC founder considering launch.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from .models import Forecast, ProductBrief


VerdictLabel = Literal[
    "launch_aggressively",
    "launch",
    "launch_with_changes",
    "test_before_launch",
    "reposition",
    "do_not_launch_yet",
]


@dataclass
class Counterfactual:
    """A 'what if' scenario projected for a product."""
    label: str
    description: str
    delta_logit: float      # how much it shifts logit prior
    new_prediction: float   # resulting trial rate
    direction: str          # "improves" | "worsens" | "neutral"


@dataclass
class CustomerReport:
    """Full customer-facing report bundle."""
    verdict: VerdictLabel
    headline: str                          # one-sentence summary
    forecast_pct: float
    forecast_low: float
    forecast_high: float
    confidence: str
    confidence_reasons: list[str]

    # Trust signals
    anchored_on: list[dict]                # neighbor brands shown to user
    downweighted_brands: list[dict]        # which were penalized & why
    why_might_be_wrong: list[str]          # honest uncertainty disclaimers

    # Action signals
    top_drivers: list[str]
    top_objections: list[str]
    most_receptive_segment: str
    counterfactuals: list[Counterfactual]

    # Diagnostics (for debugging / power users)
    rag_prior: float
    adjustment_applied: float
    coverage_tier: str


# ═══════════════════════════════════════════════════════════════════════
# VERDICT LOGIC
# ═══════════════════════════════════════════════════════════════════════

def determine_verdict(
    forecast: Forecast,
    persona_signals,
    coverage_tier: str,
) -> tuple[VerdictLabel, str]:
    """
    Returns (verdict_code, headline_sentence).

    Verdict logic considers:
    - Predicted trial rate (high/medium/low)
    - Confidence in that prediction
    - Persona desirability/friction signals
    - Coverage tier (how trustworthy is the prior)
    """
    rate = forecast.trial_rate_median
    confidence = forecast.confidence
    desire = persona_signals.desirability
    friction = persona_signals.friction

    # Coverage too weak → can't make a confident verdict
    if coverage_tier == "weak":
        return (
            "test_before_launch",
            "Comparable data is thin for this product type — pilot before scaling."
        )

    # High predicted trial + good signals
    if rate >= 0.15 and confidence in ("high", "medium-high") and desire >= 0.55:
        if friction <= 0.40:
            return (
                "launch_aggressively",
                f"Strong demand signal ({rate*100:.0f}% predicted trial) with manageable friction. Aggressive launch warranted."
            )
        return (
            "launch",
            f"Solid demand ({rate*100:.0f}% predicted trial), friction is real but addressable. Launch and iterate."
        )

    # Decent rate but high friction
    if rate >= 0.08 and friction >= 0.55:
        return (
            "launch_with_changes",
            f"Predicted {rate*100:.0f}% trial, but friction signals are concerning. Address pricing/positioning before scaling."
        )

    # Medium rate
    if 0.05 <= rate < 0.15:
        if desire < 0.45:
            return (
                "reposition",
                f"Trial prediction modest ({rate*100:.0f}%) and desirability signal is weak. Reposition product or target."
            )
        return (
            "test_before_launch",
            f"Moderate trial signal ({rate*100:.0f}%). Run controlled pilots to validate before scaling spend."
        )

    # Low rate
    if rate < 0.05:
        if desire >= 0.55:
            return (
                "reposition",
                f"Low predicted trial ({rate*100:.1f}%) despite buyer interest — likely a positioning, distribution, or pricing problem."
            )
        return (
            "do_not_launch_yet",
            f"Low trial signal ({rate*100:.1f}%) and weak buyer enthusiasm. Reconsider product-market fit before launching."
        )

    return (
        "test_before_launch",
        f"Mixed signals — predicted {rate*100:.0f}% trial. Pilot to learn before scaling."
    )


# ═══════════════════════════════════════════════════════════════════════
# WHY-MIGHT-BE-WRONG GENERATOR (per friend: "honest uncertainty disclaimers")
# ═══════════════════════════════════════════════════════════════════════

def generate_uncertainty_disclaimers(
    forecast: Forecast,
    coverage_tier: str,
    coverage_subtype: str,
) -> list[str]:
    """Build honest 'why this forecast might be wrong' bullets."""
    disclaimers = []

    if forecast.confidence == "low":
        disclaimers.append(
            "Comparable database has limited coverage for this product subtype — "
            "forecast is exploratory."
        )
    elif forecast.confidence == "medium-low":
        disclaimers.append(
            "Forecast leans partly on broader category data because exact-subtype "
            "comparables are thin."
        )

    # Surface coverage warnings as plain English
    for warning in forecast.data_quality.quality_warnings:
        if "weight share" in warning:
            disclaimers.append(
                "Some retrieved comparables are similar but not identical in adoption pattern."
            )
        elif "variance" in warning:
            disclaimers.append(
                "Trial rates among comparable brands vary widely — true outcome may shift "
                "based on distribution and positioning specifics."
            )
        elif "market tier" in warning:
            disclaimers.append(
                "Comparable brands span different market tiers — mass-retail distribution "
                "could materially raise trial above this forecast."
            )

    # Adjustment-related disclaimer
    if abs(forecast.adjustment_applied) > 0.05:
        if forecast.adjustment_applied > 0:
            disclaimers.append(
                "Persona simulation lifted forecast above raw comparable median based on "
                "buyer enthusiasm signals."
            )
        else:
            disclaimers.append(
                "Persona simulation lowered forecast based on friction/objection signals "
                "from synthetic buyers."
            )

    # Always include one general disclaimer
    if not disclaimers:
        disclaimers.append(
            "Forecast is directional — actual launch outcome depends on execution, "
            "creative, channel mix, and seasonality."
        )

    return disclaimers


# ═══════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL GENERATION (per friend: "counterfactuals are the star")
# ═══════════════════════════════════════════════════════════════════════

import math


def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def generate_counterfactuals(
    forecast: Forecast,
    persona_signals,
    product: ProductBrief,
) -> list[Counterfactual]:
    """
    Generate 'what if' scenarios.
    Currently rule-based; later can be LLM-backed.
    """
    base_logit = _logit(forecast.trial_rate_median)
    base_pct = forecast.trial_rate_median
    counterfactuals = []

    # 1. Lower price 15%
    # Lower price typically reduces friction → +0.10 to +0.20 logit
    price_drop_delta = 0.15
    cf = Counterfactual(
        label="Lower price 15%",
        description=f"Reduce ${product.price:.0f} to ${product.price*0.85:.0f}",
        delta_logit=price_drop_delta,
        new_prediction=_sigmoid(base_logit + price_drop_delta),
        direction="improves",
    )
    counterfactuals.append(cf)

    # 2. Add strong guarantee/risk-reversal (only if friction is high)
    if persona_signals.friction >= 0.40:
        guarantee_delta = 0.20
        counterfactuals.append(Counterfactual(
            label="Add risk-reversal guarantee",
            description="Money-back guarantee or 30-day free trial reduces trust friction",
            delta_logit=guarantee_delta,
            new_prediction=_sigmoid(base_logit + guarantee_delta),
            direction="improves",
        ))

    # 3. Expand to mass retail (if currently DTC)
    expand_delta = 0.30
    counterfactuals.append(Counterfactual(
        label="Add retail distribution",
        description="Expand from DTC-only to Target/Walmart/grocery presence",
        delta_logit=expand_delta,
        new_prediction=_sigmoid(base_logit + expand_delta),
        direction="improves",
    ))

    # 4. Narrower target audience (if desirability is weak)
    if persona_signals.desirability < 0.50:
        narrow_delta = 0.15
        counterfactuals.append(Counterfactual(
            label="Narrow target to high-intent segment",
            description="Focus marketing on most-receptive buyer profile rather than mainstream",
            delta_logit=narrow_delta,
            new_prediction=_sigmoid(base_logit + narrow_delta),
            direction="improves",
        ))

    # 5. Raise price (negative cf — lifestyle premium positioning)
    if persona_signals.desirability >= 0.55:
        raise_delta = -0.10
        counterfactuals.append(Counterfactual(
            label="Raise price 20% (premium positioning)",
            description=f"Position as premium at ${product.price*1.2:.0f} — fewer trial, higher LTV",
            delta_logit=raise_delta,
            new_prediction=_sigmoid(base_logit + raise_delta),
            direction="worsens",
        ))

    return counterfactuals


# ═══════════════════════════════════════════════════════════════════════
# DOWNWEIGHTED BRANDS EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_downweighted_brands(neighbors: list, all_neighbors_pre_rerank: list = None) -> list[dict]:
    """
    Show brands whose forecast weight was reduced SEVERELY (combined penalty < 0.5).
    These had different adoption patterns and got materially downweighted.
    """
    import re
    downweighted = []
    for n in neighbors:
        if not n.reason or n.reason == "match" or "×" not in n.reason:
            continue
        # Parse all penalty multipliers from reason string
        penalties = re.findall(r'×([\d.]+)', n.reason)
        if not penalties:
            continue
        combined = 1.0
        for p in penalties:
            combined *= float(p)
        # Only show as "downweighted" if combined penalty is severe
        if combined < 0.5:
            downweighted.append({
                "brand": n.brand,
                "trial_rate": n.trial_rate_mid,
                "penalty_reasons": n.reason,
                "combined_penalty": round(combined, 2),
            })
    return downweighted[:3]


# ═══════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_customer_report(
    forecast: Forecast,
    product: ProductBrief,
    coverage_tier: str = "medium",
    coverage_subtype: str = "default",
    top_drivers: list[str] | None = None,
    top_objections: list[str] | None = None,
    most_receptive_segment: str = "",
) -> CustomerReport:
    """
    Build full customer-facing report from forecast + product.
    """
    # 1. Verdict
    verdict, headline = determine_verdict(forecast, forecast.persona_signals, coverage_tier)

    # 2. Anchored-on brands (top 3-5 with positive contribution)
    anchored = [
        {
            "brand": n.brand,
            "trial_rate": n.trial_rate_mid,
            "confidence_grade": n.confidence,
            "match_reason": n.reason if n.reason and n.reason != "match" else "strong subtype match",
        }
        for n in forecast.neighbors[:5]
    ]

    # 3. Downweighted brands
    downweighted = extract_downweighted_brands(forecast.neighbors)

    # 4. Why might be wrong
    disclaimers = generate_uncertainty_disclaimers(forecast, coverage_tier, coverage_subtype)

    # 5. Counterfactuals
    cfs = generate_counterfactuals(forecast, forecast.persona_signals, product)

    return CustomerReport(
        verdict=verdict,
        headline=headline,
        forecast_pct=forecast.trial_rate_median,
        forecast_low=forecast.trial_rate_low,
        forecast_high=forecast.trial_rate_high,
        confidence=forecast.confidence,
        confidence_reasons=forecast.data_quality.quality_warnings,
        anchored_on=anchored,
        downweighted_brands=downweighted,
        why_might_be_wrong=disclaimers,
        top_drivers=top_drivers or [],
        top_objections=top_objections or [],
        most_receptive_segment=most_receptive_segment,
        counterfactuals=cfs,
        rag_prior=forecast.rag_prior,
        adjustment_applied=forecast.adjustment_applied,
        coverage_tier=coverage_tier,
    )


def render_report_text(report: CustomerReport) -> str:
    """Render report as human-readable text (for inspection / demo)."""
    lines = []
    lines.append("═" * 70)
    lines.append(f"  ASSEMBLY MARKET FORECAST")
    lines.append("═" * 70)
    lines.append("")

    # Verdict (top of report — friend said decision-oriented)
    verdict_display = {
        "launch_aggressively":  "🚀 LAUNCH AGGRESSIVELY",
        "launch":               "✅ LAUNCH",
        "launch_with_changes":  "⚠️  LAUNCH WITH CHANGES",
        "test_before_launch":   "🔬 TEST BEFORE LAUNCH",
        "reposition":           "🔄 REPOSITION",
        "do_not_launch_yet":    "🛑 DO NOT LAUNCH YET",
    }
    lines.append(f"VERDICT: {verdict_display.get(report.verdict, report.verdict)}")
    lines.append("")
    lines.append(report.headline)
    lines.append("")

    # Forecast
    lines.append(f"PREDICTED 12-MONTH TRIAL RATE: {report.forecast_pct*100:.1f}%")
    lines.append(f"Likely range: {report.forecast_low*100:.1f}% – {report.forecast_high*100:.1f}%")
    lines.append(f"Confidence: {report.confidence}")
    lines.append("")

    # Anchored on
    lines.append("FORECAST ANCHORED ON (real comparable brands):")
    for a in report.anchored_on:
        lines.append(f"  • {a['brand']:<35} {a['trial_rate']*100:>5.1f}% trial  [{a['confidence_grade']}]")
    lines.append("")

    # Downweighted
    if report.downweighted_brands:
        lines.append("DOWNWEIGHTED (different adoption pattern):")
        for d in report.downweighted_brands:
            lines.append(f"  • {d['brand']:<35} reason: {d['penalty_reasons']}")
        lines.append("")

    # Why might be wrong
    lines.append("WHY THIS FORECAST MAY BE WRONG:")
    for r in report.why_might_be_wrong:
        lines.append(f"  • {r}")
    lines.append("")

    # Counterfactuals
    if report.counterfactuals:
        lines.append("COUNTERFACTUAL SCENARIOS:")
        lines.append(f"  Current forecast: {report.forecast_pct*100:.1f}%")
        for cf in report.counterfactuals:
            arrow = "↑" if cf.direction == "improves" else "↓" if cf.direction == "worsens" else "→"
            lines.append(f"  {arrow} {cf.label:<40} {cf.new_prediction*100:.1f}%  ({cf.description})")
        lines.append("")

    return "\n".join(lines)
