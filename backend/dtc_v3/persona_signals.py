"""
backend/dtc_v3/persona_signals.py — Extract 3 composite signals from persona debate.

Per friend's architecture:
  desirability = preference + urgency + emotional_pull
  awareness    = visibility + social/search/review_signals  
  friction     = price_resistance + trust_gap + switching + objections

Quote: "RAG prior owns the number. Persona simulation earns the adjustment."
"""

from __future__ import annotations
import statistics
from typing import Any

from .models import PersonaSignals, ProductBrief


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════

def extract_desirability(debate_state: dict) -> float:
    """
    Composite of:
      - Preference distribution (% of agents in 'for' stance final round)
      - Score intensity (avg score of 'for' agents)
      - Movement (how many shifted to 'for' from neutral/against)
    Returns 0-1.
    """
    final_round = debate_state.get("rounds", [{}])[-1]
    for_count = final_round.get("for_count", 0)
    avg_score = final_round.get("avg_score", 5.0)
    n_agents = max(1, for_count + final_round.get("against_count", 0)
                   + final_round.get("neutral_count", 0))

    # Preference share (0-1)
    pref_share = for_count / n_agents

    # Score intensity (normalize 0-10 score to 0-1)
    score_norm = avg_score / 10.0

    # Positive movement (shifts to FOR)
    shifts = sum(r.get("shifted_count", 0) for r in debate_state.get("rounds", []))
    shift_signal = min(1.0, shifts / (n_agents * 0.3))  # 30% shifts = max signal

    desirability = (pref_share * 0.50 + score_norm * 0.35 + shift_signal * 0.15)
    return max(0.0, min(1.0, desirability))


def extract_awareness(market_intel: dict) -> float:
    """
    Composite of:
      - Total competitor review count (saturation = high awareness)
      - Reddit signal density (mentions per query)
      - Star rating consensus (high rating with many reviews = strong awareness)
    Returns 0-1.
    """
    competitors = market_intel.get("competitors", [])
    reddit_signals = market_intel.get("reddit_signal_count", 0)

    if not competitors:
        return 0.30  # default floor

    # Total reviews across competitors (log-scaled)
    total_reviews = sum(c.get("total_reviews", 0) or 0 for c in competitors)
    if total_reviews >= 50000:
        review_score = 0.90
    elif total_reviews >= 10000:
        review_score = 0.65
    elif total_reviews >= 1000:
        review_score = 0.45
    elif total_reviews >= 100:
        review_score = 0.30
    else:
        review_score = 0.15

    # Reddit density (signals per query — usually 16 max)
    reddit_score = min(1.0, reddit_signals / 16.0)

    # Average rating (consensus signal)
    ratings = [c.get("rating", 0) or 0 for c in competitors if c.get("rating")]
    if ratings:
        avg_rating = statistics.mean(ratings)
        rating_score = max(0, (avg_rating - 3.0) / 2.0)  # 3.0 → 0, 5.0 → 1.0
    else:
        rating_score = 0.5

    awareness = (review_score * 0.55 + reddit_score * 0.25 + rating_score * 0.20)
    return max(0.10, min(1.0, awareness))


def extract_friction(debate_state: dict, market_intel: dict, product: ProductBrief) -> float:
    """
    Composite of:
      - Against-stance share (% rejecting in final round)
      - Top objections count (price, trust, switching mentions)
      - Price ratio penalty
    Returns 0-1 (HIGHER = MORE FRICTION = BAD)
    """
    final_round = debate_state.get("rounds", [{}])[-1]
    against_count = final_round.get("against_count", 0)
    n_agents = max(1, final_round.get("for_count", 0) + against_count
                   + final_round.get("neutral_count", 0))

    # Against share
    against_share = against_count / n_agents

    # Objection density (LLM-extracted top objections)
    objections = market_intel.get("top_objections", []) or []
    objection_signal = min(1.0, len(objections) / 5.0)  # 5+ objections = max friction

    # Price ratio
    price_ratio = market_intel.get("price_ratio", 1.0) or 1.0
    if price_ratio <= 1.2:
        price_friction = 0.10
    elif price_ratio <= 2.0:
        price_friction = 0.40
    elif price_ratio <= 3.0:
        price_friction = 0.65
    else:
        price_friction = 0.85

    friction = (against_share * 0.45 + objection_signal * 0.25 + price_friction * 0.30)
    return max(0.0, min(1.0, friction))


# ═══════════════════════════════════════════════════════════════════════
# MAIN: Build PersonaSignals + z-scores
# ═══════════════════════════════════════════════════════════════════════

# Reference distributions for z-scoring
# Loaded from empirically-computed normalizer_constants_v1.json (median + IQR/1.349)
# DO NOT hardcode — these come from real signal distribution in the calibration set

import json as _json_for_norm
from pathlib import Path as _Path

_NORMALIZER_PATH = _Path(__file__).parent / "calibration" / "normalizer_constants_v1.json"


def _load_normalizers():
    """Load empirical normalizers, fall back to neutral defaults if missing."""
    if _NORMALIZER_PATH.exists():
        try:
            data = _json_for_norm.loads(_NORMALIZER_PATH.read_text())
            n = data["normalizers"]
            return {
                "desirability_center": n["desirability"]["center"],
                "desirability_scale":  n["desirability"]["scale"],
                "awareness_center":    n["awareness"]["center"],
                "awareness_scale":     n["awareness"]["scale"],
                "friction_center":     n["friction"]["center"],
                "friction_scale":      n["friction"]["scale"],
                "_source": "empirical_v1",
            }
        except Exception as e:
            print(f"[persona_signals] Normalizer load failed: {e}, using defaults")
    return {
        "desirability_center": 0.50, "desirability_scale": 0.15,
        "awareness_center":    0.50, "awareness_scale":    0.15,
        "friction_center":     0.50, "friction_scale":     0.15,
        "_source": "fallback_neutral",
    }


_NORM = _load_normalizers()
DESIRABILITY_MEAN = _NORM["desirability_center"]
DESIRABILITY_SD   = _NORM["desirability_scale"]
AWARENESS_MEAN    = _NORM["awareness_center"]
AWARENESS_SD      = _NORM["awareness_scale"]
FRICTION_MEAN     = _NORM["friction_center"]
FRICTION_SD       = _NORM["friction_scale"]


def build_persona_signals(
    debate_state: dict,
    market_intel: dict,
    product: ProductBrief,
) -> PersonaSignals:
    """Build PersonaSignals with raw composites + z-scores."""
    desirability = extract_desirability(debate_state)
    awareness = extract_awareness(market_intel)
    friction = extract_friction(debate_state, market_intel, product)

    desirability_z = (desirability - DESIRABILITY_MEAN) / DESIRABILITY_SD
    awareness_z = (awareness - AWARENESS_MEAN) / AWARENESS_SD
    friction_z = (friction - FRICTION_MEAN) / FRICTION_SD

    return PersonaSignals(
        desirability=desirability,
        awareness=awareness,
        friction=friction,
        desirability_z=desirability_z,
        awareness_z=awareness_z,
        friction_z=friction_z,
    )


# ═══════════════════════════════════════════════════════════════════════
# LOGIT ADJUSTMENT (per friend's spec)
# ═══════════════════════════════════════════════════════════════════════

import math

ADJUSTMENT_CLAMP = 0.75  # Per friend: "roughly -35% to +45% relative movement"

# Initial coefficients (will be ridge-regression-fit on Day 5)
BETA_DESIRABILITY = 0.35
BETA_AWARENESS = 0.20
BETA_FRICTION = -0.35


def logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def apply_persona_adjustment(rag_prior: float, signals: PersonaSignals) -> tuple[float, float]:
    """
    Per friend's logit-stack architecture:
      adjustment = β1·desirability_z + β2·awareness_z + β3·friction_z
      adjustment = clamp(adjustment, -0.75, 0.75)
      final = sigmoid(logit(prior) + adjustment)

    Returns (final_trial_rate, adjustment_applied).
    """
    raw_adjustment = (
        BETA_DESIRABILITY * signals.desirability_z
        + BETA_AWARENESS * signals.awareness_z
        + BETA_FRICTION * signals.friction_z
    )
    clamped = max(-ADJUSTMENT_CLAMP, min(ADJUSTMENT_CLAMP, raw_adjustment))
    final = sigmoid(logit(rag_prior) + clamped)
    return final, clamped
