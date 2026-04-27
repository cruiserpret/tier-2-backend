"""
backend/dtc_v3/rag_retrieval.py — Hybrid Comparable Retrieval (v3.1)

Two-stage retrieval per friend's architecture spec:
  Stage 1: Semantic candidate retrieval (top 12 by embedding similarity)
  Stage 2: Structured reranker — apply market-structure penalties

Aggregation: weighted MEDIAN, not weighted mean (robust to outliers like
Apple Watch dominating Oura's prior).

Quote: "Semantic similarity gets a product into the room. Market-structure
similarity decides how much its trial rate matters."
"""

from __future__ import annotations
import os
import json
import statistics
from pathlib import Path

import numpy as np
from openai import OpenAI

from .models import GroundTruthRecord, Neighbor, ProductBrief
from .ground_truth_db import GROUND_TRUTH_DB

EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_PATH = Path("backend/dtc_v3/embedding_cache.json")
SIMILARITY_FLOOR = 0.45
CATEGORY_MATCH_BONUS = 0.10
SEMANTIC_TOP_K = 12  # Stage 1 candidate pool size


# ═══════════════════════════════════════════════════════════════════════
# CLIENT + CACHE
# ═══════════════════════════════════════════════════════════════════════

_client = None
_db_embeddings: dict[str, list[float]] | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        import sys as _sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if repo_root not in _sys.path:
            _sys.path.insert(0, repo_root)
        try:
            import config
            api_key = config.OPENAI_API_KEY
        except (ImportError, AttributeError):
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env or environment")
        _client = OpenAI(api_key=api_key)
    return _client


# Subtype-specific retrieval enrichment — boosts semantic match for query patterns
SUBTYPE_RETRIEVAL_TERMS = {
    "hydration_supplement": "electrolyte powder hydration drink mix oral rehydration sports hydration recovery",
    "nonalcoholic_beer": "non-alcoholic beer NA beer zero alcohol craft beer alternative dry january",
    "coffee_alternative": "coffee replacement mushroom coffee adaptogen morning ritual caffeine alternative",
    "functional_soda": "prebiotic soda gut health functional beverage healthy soda",
    "wearable_health": "smart ring wearable tracker sleep recovery device health monitor",
    "premium_basics": "premium essentials sustainable basics everyday wear",
    "athletic_apparel": "activewear workout clothing fitness apparel athletic wear",
    "mattress": "mattress in box memory foam bed in box online mattress",
    "bedding_premium": "premium sheets bedding luxury linens cotton sheets",
}


def _record_to_text(record: GroundTruthRecord) -> str:
    base = (
        f"Brand: {record.brand}. Category: {record.category}. "
        f"Price band: {record.price_band}. "
        f"Purchase frequency: {record.purchase_frequency}. "
        f"Target demo: {', '.join(record.target_demo)}. "
        f"Drivers: {', '.join(record.drivers)}. "
        f"Frictions: {', '.join(record.frictions)}."
    )
    enrichment = SUBTYPE_RETRIEVAL_TERMS.get(record.category_subtype, "")
    if enrichment:
        base += f" Synonyms: {enrichment}."
    return base


def _product_to_text(product: ProductBrief) -> str:
    comp_names = ", ".join(c.get("name", "") for c in product.competitors[:3])
    return (
        f"Brand: {product.name}. Category: {product.category}. "
        f"Price: ${product.price}. Description: {product.description}. "
        f"Target demo: {product.demographic}. Competitors: {comp_names}."
    )


def _embed(text: str) -> list[float]:
    response = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _load_or_build_db_embeddings() -> dict[str, list[float]]:
    global _db_embeddings
    if _db_embeddings is not None:
        return _db_embeddings
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
            db_brands = {r.brand for r in GROUND_TRUTH_DB}
            if db_brands == set(cache.keys()):
                _db_embeddings = cache
                return _db_embeddings
        except Exception:
            pass
    print(f"[RAG] Building embedding cache for {len(GROUND_TRUTH_DB)} products...")
    embeddings = {}
    for record in GROUND_TRUTH_DB:
        embeddings[record.brand] = _embed(_record_to_text(record))
        print(f"  ✓ {record.brand}")
    CACHE_PATH.write_text(json.dumps(embeddings))
    _db_embeddings = embeddings
    return embeddings


def _cosine(a, b) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / norm) if norm > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: STRUCTURED RERANKING (per friend's spec)
# ═══════════════════════════════════════════════════════════════════════

def _tier_weight(query_tier: str, neighbor_tier: str) -> float:
    if query_tier == neighbor_tier:
        return 1.0
    soft_matches = {
        ("premium_niche", "challenger"), ("challenger", "premium_niche"),
        ("mass_market", "mass_platform"), ("mass_platform", "mass_market"),
        ("premium_niche", "luxury"), ("luxury", "premium_niche"),
        ("challenger", "mass_market"), ("mass_market", "challenger"),
    }
    if (query_tier, neighbor_tier) in soft_matches:
        return 0.65
    return 0.35


def _scale_weight(query_tier: str, neighbor_scale: str) -> float:
    severe_mismatch = {
        ("premium_niche", "global_giant"),
        ("niche", "global_giant"),
        ("luxury", "global_giant"),
    }
    if (query_tier, neighbor_scale) in severe_mismatch:
        return 0.25
    return 0.85  # default mild penalty for any scale mismatch


def _price_band_weight(query_band: str, neighbor_band: str) -> float:
    if query_band == neighbor_band:
        return 1.0
    return 0.70


def _distribution_weight(query_dist: str, neighbor_dist: str) -> float:
    if query_dist == neighbor_dist:
        return 1.0
    if "dtc" in query_dist and "dtc" in neighbor_dist:
        return 0.85
    return 0.65


def _subtype_weight(query_subtype: str, neighbor_subtype: str, same_category: bool) -> float:
    """
    Per friend's spec: catches "demand-pattern similarity" that embeddings miss.
    Same broad category but different subtype = serious penalty.
    """
    if query_subtype == neighbor_subtype:
        return 1.0
    
    # Related demand patterns (loose substitutes)
    related = {
        ("functional_soda", "hydration_supplement"),
        ("hydration_supplement", "functional_soda"),
        ("coffee_alternative", "greens_powder"),
        ("greens_powder", "coffee_alternative"),
        ("premium_basics", "athletic_apparel"),
        ("athletic_apparel", "premium_basics"),
        ("bedding_premium", "mattress"),
        ("mattress", "bedding_premium"),
        ("wearable_health", "fitness_tracker"),
        ("fitness_tracker", "wearable_health"),
        ("functional_soda", "branded_water"),
        ("branded_water", "functional_soda"),
    }
    if (query_subtype, neighbor_subtype) in related:
        return 0.65
    
    # Same broad category, unrelated subtype = serious penalty
    if same_category:
        return 0.40
    
    # Different category entirely = harshest penalty
    return 0.25


def _infer_query_subtype(product: ProductBrief) -> str:
    """Heuristic inference of category_subtype from product description + name."""
    text = (product.name + " " + product.description).lower()
    cat = product.category
    
    # F&B subtypes
    if cat == "food_beverage":
        if any(x in text for x in ["non-alcoholic", "non alcoholic", "na beer", "zero alcohol"]):
            return "nonalcoholic_beer"
        if any(x in text for x in ["mushroom", "coffee alternative", "adaptogen"]):
            return "coffee_alternative"
        if any(x in text for x in ["soda", "prebiotic", "functional drink"]):
            return "functional_soda"
        if any(x in text for x in ["water", "sparkling water", "mineral"]):
            return "branded_water"
        if any(x in text for x in ["kombucha", "fermented"]):
            return "functional_fermented"
        return "default"
    
    # Supplements
    if cat == "supplements_health":
        if any(x in text for x in ["greens", "superfood powder"]):
            return "greens_powder"
        if any(x in text for x in ["electrolyte", "hydration"]):
            return "hydration_supplement"
        if any(x in text for x in ["vitamin", "multivitamin"]):
            return "multivitamin_subscription"
        return "default"
    
    # Electronics
    if cat == "electronics_tech":
        if any(x in text for x in ["ring", "watch", "wearable", "tracker"]):
            return "wearable_health"
        if any(x in text for x in ["sleep", "mattress smart"]):
            return "smart_sleep"
        return "default"
    
    # Home
    if cat == "home_lifestyle":
        if any(x in text for x in ["mattress", "bed in a box"]):
            return "mattress"
        if any(x in text for x in ["sheets", "bedding", "duvet"]):
            return "bedding_premium"
        if any(x in text for x in ["mug", "tumbler", "drinkware", "cup"]):
            return "premium_drinkware"
        return "default"
    
    # Fashion
    if cat == "fashion_apparel":
        if any(x in text for x in ["sock", "t-shirt", "tee", "basics"]):
            return "premium_basics"
        if any(x in text for x in ["sneaker", "running shoe"]):
            return "premium_sneaker"
        if any(x in text for x in ["sports bra", "leggings", "activewear", "athletic"]):
            return "athletic_apparel"
        if any(x in text for x in ["glasses", "eyewear", "frames"]):
            return "eyewear"
        return "default"
    
    # Beauty
    if cat == "beauty_skincare":
        if any(x in text for x in ["razor", "shave"]):
            return "razor_subscription"
        if any(x in text for x in ["brow", "eyebrow"]):
            return "brow_makeup"
        if any(x in text for x in ["niacinamide", "retinol", "active"]):
            return "skincare_active"
        return "default"
    
    return "default"


def _infer_query_market_structure(product: ProductBrief) -> tuple[str, str, str, str]:
    """
    Infer market structure with override support.
    Per friend's spec: explicit override > mass-market signals > price heuristic.
    """
    desc_lower = (product.description + " " + product.name).lower()
    competitor_text = " ".join(c.get("name", "") for c in product.competitors).lower()
    full_text = desc_lower + " " + competitor_text

    # Price band
    if product.price < 25:
        price_band = "budget"
    elif product.price < 75:
        price_band = "mid"
    elif product.price < 200:
        price_band = "premium"
    else:
        price_band = "luxury"

    # ── Market tier ──
    # Priority 1: explicit override
    if product.market_tier_override:
        market_tier = product.market_tier_override
    else:
        # Priority 2: mass-market retail signals
        mass_terms = [
            "costco", "target", "walmart", "cvs", "walgreens",
            "grocery", "national retail", "mass retail", "mass-retail",
            "amazon bestseller", "whole foods", "kroger", "publix",
            "sams club", "sam's club", "dollar general"
        ]
        if any(term in full_text for term in mass_terms):
            market_tier = "mass_market"
        elif any(x in desc_lower for x in ["mass", "everyone", "wide audience", "mainstream"]):
            market_tier = "mass_market"
        elif price_band == "luxury":
            market_tier = "luxury"
        elif price_band == "budget":
            market_tier = "mass_market"
        else:
            market_tier = "premium_niche"

    # ── Distribution ──
    if product.distribution_hint:
        distribution = product.distribution_hint
    elif any(x in full_text for x in ["costco", "walmart", "target", "grocery", "mass retail"]):
        distribution = "mass_retail"
    elif any(x in desc_lower for x in ["subscription", "monthly", "/month", "recurring"]):
        distribution = "subscription_led"
    else:
        distribution = "dtc_led"

    brand_scale = "venture_challenger"
    return market_tier, brand_scale, distribution, price_band


# ═══════════════════════════════════════════════════════════════════════
# MAIN: TWO-STAGE RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════

def retrieve_neighbors(
    product: ProductBrief,
    k: int = 6,
    exclude_brand: str | None = None,
) -> list[Neighbor]:
    """
    Stage 1: Semantic retrieval (top SEMANTIC_TOP_K candidates)
    Stage 2: Structured reranking with market-structure penalties

    Returns top-k neighbors sorted by structure-adjusted weight.
    """
    db_embeddings = _load_or_build_db_embeddings()
    query_emb = _embed(_product_to_text(product))

    # Infer query's market structure for reranking
    q_tier, q_scale, q_dist, q_band = _infer_query_market_structure(product)

    # ── STAGE 1: Semantic candidates ──
    candidates = []
    for record in GROUND_TRUTH_DB:
        record_emb = db_embeddings.get(record.brand)
        if not record_emb:
            continue
        if exclude_brand and exclude_brand.lower() in record.brand.lower():
            continue
        if exclude_brand and record.brand.lower() in exclude_brand.lower():
            continue

        base_sim = _cosine(query_emb, record_emb)
        adj_sim = base_sim
        if product.category and product.category == record.category:
            adj_sim = min(1.0, adj_sim + CATEGORY_MATCH_BONUS)

        # Do not apply SIMILARITY_FLOOR here.
        # retrieve_neighbors returns raw top candidates for transparency;
        # get_eligible_neighbors() applies the floor before forecast math.
        candidates.append((record, adj_sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:SEMANTIC_TOP_K]

    # ── STAGE 2: Structured reranking ──
    neighbors = []
    for record, sim in candidates:
        semantic_weight = sim ** 2

        tier_w = _tier_weight(q_tier, record.market_tier)
        scale_w = _scale_weight(q_tier, record.brand_scale)
        price_w = _price_band_weight(q_band, record.price_band)
        dist_w = _distribution_weight(q_dist, record.distribution_model)

        same_cat = (product.category == record.category)
        q_subtype = _infer_query_subtype(product)
        subtype_w = _subtype_weight(q_subtype, record.category_subtype, same_cat)

        forecast_weight = (
            semantic_weight
            * record.source_weight
            * tier_w * scale_w * price_w * dist_w * subtype_w
        )

        # Build human-readable reason for debugging
        penalties = []
        if tier_w < 1.0:
            penalties.append(f"tier({record.market_tier})×{tier_w}")
        if scale_w < 1.0:
            penalties.append(f"scale({record.brand_scale})×{scale_w}")
        if price_w < 1.0:
            penalties.append(f"price×{price_w}")
        if dist_w < 1.0:
            penalties.append(f"dist×{dist_w}")
        if subtype_w < 1.0:
            penalties.append(f"subtype({record.category_subtype})×{subtype_w}")
        reason_str = "match" if not penalties else ", ".join(penalties)

        neighbors.append(Neighbor(
            brand=record.brand,
            similarity=sim,
            trial_rate_mid=record.trial_rate_mid,
            confidence=record.confidence,
            source_weight=forecast_weight,  # store final forecast weight here
            reason=reason_str,
        ))

    # Sort by forecast weight (not raw similarity)
    neighbors.sort(key=lambda n: n.source_weight, reverse=True)
    return neighbors[:k]


# ═══════════════════════════════════════════════════════════════════════
# WEIGHTED MEDIAN (per friend's spec — robust to outliers)
# ═══════════════════════════════════════════════════════════════════════

def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Compute weighted median. Robust alternative to weighted mean."""
    if not values:
        return 0.08
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total = sum(weights)
    if total == 0:
        return statistics.median(values)
    cumulative = 0
    target = total / 2
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return value
    return pairs[-1][0]


def get_eligible_neighbors(neighbors: list[Neighbor], min_similarity: float = None) -> list[Neighbor]:
    """
    Filter retrieved neighbors to those eligible for forecast math.
    Per friend's spec: similarity must clear SIMILARITY_FLOOR to be eligible.
    """
    threshold = min_similarity if min_similarity is not None else SIMILARITY_FLOOR
    return [n for n in neighbors if n.similarity >= threshold]


def compute_rag_prior(neighbors: list[Neighbor]) -> float:
    """
    Compute prior trial rate using WEIGHTED MEDIAN (not mean).
    Per friend's advice: weighted mean lets outliers like Apple Watch
    dominate. Weighted median is robust.
    """
    if not neighbors:
        return 0.08
    values = [n.trial_rate_mid for n in neighbors]
    weights = [n.source_weight for n in neighbors]  # already includes structure penalties
    return _weighted_median(values, weights)


# ═══════════════════════════════════════════════════════════════════════
# DEBUG: show semantic vs forecast neighbors
# ═══════════════════════════════════════════════════════════════════════

def debug_retrieval(product: ProductBrief, exclude_brand: str | None = None) -> dict:
    """Returns both semantic and forecast neighbor lists for debugging."""
    db_embeddings = _load_or_build_db_embeddings()
    query_emb = _embed(_product_to_text(product))
    q_tier, q_scale, q_dist, q_band = _infer_query_market_structure(product)

    semantic = []
    for record in GROUND_TRUTH_DB:
        if exclude_brand and exclude_brand.lower() in record.brand.lower():
            continue
        record_emb = db_embeddings.get(record.brand)
        if not record_emb:
            continue
        sim = _cosine(query_emb, record_emb)
        if product.category == record.category:
            sim = min(1.0, sim + CATEGORY_MATCH_BONUS)
        semantic.append((record.brand, sim))
    semantic.sort(key=lambda x: x[1], reverse=True)

    forecast = retrieve_neighbors(product, k=6, exclude_brand=exclude_brand)

    return {
        "query_inferred_structure": {
            "market_tier": q_tier, "brand_scale": q_scale,
            "distribution_model": q_dist, "price_band": q_band,
        },
        "semantic_neighbors": [{"brand": b, "sim": round(s, 3)} for b, s in semantic[:6]],
        "forecast_neighbors": [
            {"brand": n.brand, "weight": round(n.source_weight, 4),
             "trial_rate": n.trial_rate_mid, "reason": n.reason}
            for n in forecast
        ],
    }
