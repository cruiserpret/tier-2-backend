"""
backend/dtc_v3/models.py — v3-lite data models.

Architecture: RAG comparable prior + 3-feature persona adjustment + calibrated logit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProductBrief:
    name: str
    description: str
    price: float
    category: str
    demographic: str
    competitors: list[dict]

    # v3.1 — Optional explicit overrides for known brands / improved inference
    market_tier_override: str | None = None  # "mass_platform" | "mass_market" | "challenger" | "premium_niche" | "luxury"
    distribution_hint: str | None = None     # "mass_retail" | "dtc_led" | etc — optional retail signals

    def __post_init__(self):
        self.price = float(self.price)
        if not isinstance(self.competitors, list):
            self.competitors = []


@dataclass
class GroundTruthRecord:
    """A single brand in the benchmark DB."""
    brand: str
    category: str
    trial_rate_low: float
    trial_rate_high: float
    trial_rate_mid: float
    confidence: str  # "A" | "B" | "C" | "D"
    price_band: str  # "budget" | "mid" | "premium" | "luxury"
    purchase_frequency: str  # "consumable" | "frequent" | "occasional" | "durable"
    target_demo: list[str]
    frictions: list[str]
    drivers: list[str]
    source_notes: str = ""
    # v3.1 — Market structure fields for reranking (per friend advice)
    market_tier: str = "challenger"
    # mass_platform | mass_market | challenger | premium_niche | niche | luxury
    brand_scale: str = "venture_challenger"
    # global_giant | large_public | growth_challenger | venture_challenger | niche_private | early_stage
    distribution_model: str = "dtc_led"
    # mass_retail | retail_plus_dtc | dtc_led | marketplace_led | subscription_led
    category_role: str = "specialist"
    # generalist | specialist | platform | substitute
    category_subtype: str = "default"
    # functional_soda | nonalcoholic_beer | coffee_alternative | hydration_supplement
    # premium_basics | luxury_fashion | bedding | mattress | wearable_health
    # razor_subscription | brow_makeup | skincare_active
    # (per friend: catches "demand-pattern similarity" embeddings miss)

    @property
    def source_weight(self) -> float:
        return {"A": 1.0, "B": 0.6, "C": 0.25, "D": 0.1}.get(self.confidence, 0.1)


@dataclass
class Neighbor:
    """A retrieved comparable brand."""
    brand: str
    similarity: float
    trial_rate_mid: float
    confidence: str
    source_weight: float
    reason: str = ""


@dataclass
class PersonaSignals:
    """3 composite scores extracted from persona debate."""
    desirability: float = 0.5    # preference + urgency + emotional pull
    awareness: float = 0.5       # visibility + social + reviews
    friction: float = 0.5        # price + trust + switching + objections

    # z-scores (computed during prediction)
    desirability_z: float = 0.0
    awareness_z: float = 0.0
    friction_z: float = 0.0


@dataclass
class DataQuality:
    """Tracks reliability of input data."""
    amazon_status: str = "ok"           # ok | unreliable_missing | partial
    reddit_status: str = "ok"
    competitor_validation_failures: list[str] = field(default_factory=list)
    quality_score: float = 1.0          # 0-1
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class Forecast:
    """Final v3 prediction output."""
    trial_rate_median: float
    trial_rate_low: float
    trial_rate_high: float
    confidence: str  # "high" | "medium" | "low"
    neighbors: list[Neighbor]
    persona_signals: PersonaSignals
    data_quality: DataQuality
    rag_prior: float
    adjustment_applied: float
    # v3.1 — honest fallback labeling per friend's spec
    prior_source: str = "rag_weighted_median"  # "rag_weighted_median" | "fallback_*" | "blended_*"
    fallback_used: bool = False
    eligible_neighbor_count: int = 0
    retrieved_candidate_count: int = 0
    drivers: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)
    segments: dict = field(default_factory=dict)
    counterfactuals: list[dict] = field(default_factory=list)
