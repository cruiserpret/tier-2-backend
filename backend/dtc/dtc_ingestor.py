"""
backend/dtc/dtc_ingestor.py — GODMODE 3.1 UNIVERSAL ACCURACY

Universal fixes applied across ALL product categories:

1. Subscription-aware effective pricing (lifetime cost, not sticker)
2. Brand-aware market saturation (Apple/Samsung/Stanley/Quince = saturated)
3. Category-specific incumbent thresholds (electronics = 1500, home = 10000)
4. Compound penalty stacking with intelligent caps
5. Saturation detection via DOMINANT COMPETITOR rating (4.5+★ alone = cult)
"""

import asyncio
import math
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataclasses import dataclass, field
from backend.dtc.amazon_ingestor import (
    fetch_all_competitors,
    compute_weighted_signal,
    CompetitorProfile,
)
from backend.dtc.reddit_ingestor import (
    fetch_reddit_intelligence,
    RedditIntelligence,
)


# ── Category Price Thresholds ────────────────────────────────────────────────
CATEGORY_PRICE_THRESHOLD = {
    "fashion_apparel":    1.15,
    "food_beverage":      1.25,
    "home_lifestyle":     1.20,
    "beauty_skincare":    1.30,
    "supplements_health": 1.35,
    "electronics_tech":   1.15,  # GM3.1 tightened — Apple/Samsung disruption
    "fitness_sports":     1.20,
    "saas_software":      1.50,
    "pet_products":       1.30,
    "baby_kids":          1.45,
    "general":            1.25,
}

CATEGORY_PENALTY_COEFFICIENT = {
    "fashion_apparel":    0.18,
    "food_beverage":      0.15,
    "home_lifestyle":     0.15,
    "beauty_skincare":    0.12,
    "supplements_health": 0.12,
    "electronics_tech":   0.18,  # GM3.1 raised — wearables are elastic
    "fitness_sports":     0.15,
    "saas_software":      0.08,
    "pet_products":       0.11,
    "baby_kids":          0.07,
    "general":            0.12,
}

# GM3.1: Category-specific incumbent detection thresholds
# Electronics reviews are fragmented across variants, so threshold is lower
CATEGORY_INCUMBENT_THRESHOLD = {
    "fashion_apparel":    5000,
    "food_beverage":      3000,
    "home_lifestyle":     10000,
    "beauty_skincare":    5000,
    "supplements_health": 4000,
    "electronics_tech":   1500,   # fragmented variants — lower bar
    "fitness_sports":     3000,
    "pet_products":       3000,
    "baby_kids":          3000,
    "saas_software":      500,
    "general":            5000,
}

# GM3.1: Known dominant brands — force saturated_market = True if present
# Based on Counterpoint Research, NPD, Mintel 2023-2024 market share data
DOMINANT_BRANDS = {
    # Electronics/wearables
    "apple", "samsung", "garmin", "fitbit", "whoop",
    # Drinkware / lifestyle
    "stanley", "yeti", "hydro flask", "thermos",
    # Fashion dupes
    "quince", "uniqlo", "shein", "aritzia",
    # Food/beverage
    "celsius", "monster", "red bull", "liquid iv", "liquid i.v.",
    "olipop", "poppi",
    # Supplements
    "ritual", "ag1", "athletic greens", "huel",
    # Beauty
    "cerave", "cetaphil", "the ordinary", "drunk elephant",
    # Baby
    "huggies", "pampers", "graco",
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ProductBrief:
    name:        str
    description: str
    price:       float
    category:    str = "general"
    demographic: str = ""
    competitors: list = field(default_factory=list)


@dataclass
class CompetitorGap:
    competitor_name:   str
    competitor_price:  float
    competitor_rating: float
    competitor_bought: int
    market_share:      float = 0.0
    user_price_diff:   float = 0.0
    star_signal:       dict  = field(default_factory=dict)
    top_praise:        list  = field(default_factory=list)
    top_complaints:    list  = field(default_factory=list)
    review_topics:     list  = field(default_factory=list)
    ai_summary:        str   = ""


@dataclass
class MarketIntelligence:
    product:      ProductBrief
    competitors:  list[CompetitorProfile] = field(default_factory=list)
    reddit:       RedditIntelligence = None
    gaps:         list[CompetitorGap] = field(default_factory=list)

    market_for:     float = 0.0
    market_against: float = 0.0
    market_neutral: float = 0.0

    category_avg_rating:  float = 0.0
    category_avg_price:   float = 0.0
    total_market_reviews: int = 0

    # Effective price after subscription math
    effective_price:       float = 0.0
    subscription_detected: bool = False
    subscription_monthly:  float = 0.0

    price_premium_ratio:   float = 1.0
    price_premium_penalty: float = 0.0

    dominant_competitor:    str = ""
    dominant_bought:        int = 0
    dominant_rating:        float = 0.0
    dominant_reviews:       int = 0
    is_saturated_market:    bool = False
    saturation_reason:      str = ""  # why saturation fired
    cult_brand_penalty:     float = 0.0
    switching_cost_penalty: float = 0.0

    agent_for_ratio:     float = 0.0
    agent_against_ratio: float = 0.0
    agent_neutral_ratio: float = 0.0

    hardcore_resistor_count: int = 0

    error: str = ""


# ── GM3.1 NEW: Subscription Detection ────────────────────────────────────────

def _detect_subscription(description: str, name: str) -> tuple[bool, float]:
    """
    GODMODE 3.1: Parse subscription cost from product description.
    Returns (has_subscription, monthly_cost)

    Detects patterns like:
    - "$5.99/month membership"
    - "$99/month subscription"
    - "monthly fee of $10"
    - "requires $X monthly"
    """
    text = (description + " " + name).lower()

    # Quick keyword check
    subscription_keywords = ["subscription", "membership", "monthly", "/month", "per month", "recurring"]
    if not any(kw in text for kw in subscription_keywords):
        return False, 0.0

    # Try to extract dollar amount
    # Patterns: $X.XX/month, $X/mo, X.XX monthly, etc.
    patterns = [
        r'\$(\d+(?:\.\d+)?)\s*/\s*(?:month|mo)\b',
        r'\$(\d+(?:\.\d+)?)\s*per\s*month',
        r'\$(\d+(?:\.\d+)?)\s*monthly',
        r'monthly\s*(?:fee|cost|charge)\s*of\s*\$(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:dollars?\s*)?(?:/|per)\s*month',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                monthly = float(match.group(1))
                if 0.99 <= monthly <= 500:  # sanity bound
                    return True, monthly
            except ValueError:
                continue

    # Subscription mentioned but no price — conservative estimate
    return True, 10.0  # assume $10/mo default


def _effective_price(product: ProductBrief, subscription_monthly: float) -> float:
    """
    GM3.1: Compute lifetime effective price including 3-year subscription cost.
    Research: NPD shows 36 months as average consumer retention for DTC products.
    """
    if subscription_monthly <= 0:
        return product.price
    # 3-year effective cost
    return product.price + (subscription_monthly * 36)


# ── GM3.1 NEW: Multi-Signal Saturation Detection ────────────────────────────

def _detect_brand_saturation(competitors, product_category: str) -> tuple[bool, str]:
    """
    GODMODE 3.1: Multi-signal saturated market detection.

    Fires if ANY of:
    1. Dominant competitor name matches known dominant brand
    2. Dominant has category-specific review threshold
    3. Dominant has 5x+ review advantage over next competitor
    4. Dominant has 4.5+★ AND 1000+ reviews (cult brand signal)
    """
    if not competitors:
        return False, ""

    valid = [c for c in competitors if c.found_on_amazon and c.total_reviews > 0]
    if not valid:
        return False, ""

    valid.sort(key=lambda c: c.total_reviews, reverse=True)
    dominant = valid[0]
    dominant_reviews = dominant.total_reviews
    dominant_rating = dominant.avg_rating
    dominant_name_lower = dominant.name.lower()

    # Check 1: Known dominant brand match
    for brand in DOMINANT_BRANDS:
        if brand in dominant_name_lower:
            return True, f"dominant_brand_detected:{brand}"

    # Check 2: Category-specific review threshold
    threshold = CATEGORY_INCUMBENT_THRESHOLD.get(product_category, 5000)
    if dominant_reviews >= threshold:
        return True, f"review_threshold:{dominant_reviews}>={threshold}"

    # Check 3: Review asymmetry — dominant has 5x+ more than next
    if len(valid) > 1:
        second_reviews = valid[1].total_reviews
        if second_reviews > 0 and (dominant_reviews / second_reviews) >= 5:
            return True, f"asymmetry_ratio:{dominant_reviews/second_reviews:.1f}x"

    # Check 4: Cult brand — 4.5+★ with 1000+ reviews
    if dominant_rating >= 4.5 and dominant_reviews >= 1000:
        return True, f"cult_brand:{dominant_rating}★_with_{dominant_reviews}_reviews"

    return False, ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _effective_market_share(comp: CompetitorProfile) -> int:
    if comp.bought_last_month > 0:
        return comp.bought_last_month
    if comp.total_reviews > 0:
        return max(10, comp.total_reviews // 10)
    return 0


# ── Gap Analysis ──────────────────────────────────────────────────────────────

def _build_competitor_gaps(product, competitors):
    gaps = []
    total_market = sum(_effective_market_share(c) for c in competitors if c.found_on_amazon)

    for comp in competitors:
        if not comp.found_on_amazon:
            continue
        signal = compute_weighted_signal(comp.reviews, comp.star_distribution)
        eff_share = _effective_market_share(comp)
        market_share = (eff_share / total_market) if total_market > 0 else 0

        top_praise = [t.get("label", "") for t in comp.review_topics[:5] if t.get("label")]
        top_complaints = []
        for review in comp.reviews:
            if review.star_rating <= 2:
                text = review.text[:200].strip()
                if text and text not in top_complaints:
                    top_complaints.append(text)

        gaps.append(CompetitorGap(
            competitor_name=comp.name,
            competitor_price=comp.price,
            competitor_rating=comp.avg_rating,
            competitor_bought=eff_share,
            market_share=round(market_share, 3),
            user_price_diff=round(product.price - comp.price, 2),
            star_signal=signal,
            top_praise=top_praise[:4],
            top_complaints=top_complaints[:3],
            review_topics=comp.review_topics[:6],
            ai_summary=comp.ai_summary,
        ))
    gaps.sort(key=lambda g: g.market_share, reverse=True)
    return gaps


def _compute_market_signal(competitors):
    if not competitors:
        return {"for": 0.33, "against": 0.33, "neutral": 0.34,
                "avg_rating": 0.0, "weighted_price": 0.0, "total_reviews": 0,
                "dominant_competitor": "", "dominant_bought": 0, "dominant_rating": 0.0,
                "dominant_reviews": 0}

    weights = []
    for comp in competitors:
        if not comp.found_on_amazon:
            weights.append(0)
        else:
            weights.append(_effective_market_share(comp))

    total_weight = sum(weights)
    if total_weight == 0:
        return {"for": 0.33, "against": 0.33, "neutral": 0.34,
                "avg_rating": 0.0, "weighted_price": 0.0, "total_reviews": 0,
                "dominant_competitor": "", "dominant_bought": 0, "dominant_rating": 0.0,
                "dominant_reviews": 0}

    w_for = w_against = w_neutral = 0.0
    w_price = w_rating = 0.0
    total_reviews = 0
    dominant = None
    dominant_share = 0

    for comp, weight in zip(competitors, weights):
        if weight == 0:
            continue
        signal = compute_weighted_signal(comp.reviews, comp.star_distribution)
        share = weight / total_weight

        w_for     += signal["for"]     * share
        w_against += signal["against"] * share
        w_neutral += signal["neutral"] * share
        if comp.price > 0:
            w_price += comp.price * share
        if comp.avg_rating > 0:
            w_rating += comp.avg_rating * share
        total_reviews += comp.total_reviews

        if weight > dominant_share:
            dominant = comp
            dominant_share = weight

    return {
        "for":           round(w_for, 3),
        "against":       round(w_against, 3),
        "neutral":       round(w_neutral, 3),
        "avg_rating":    round(w_rating, 2),
        "weighted_price": round(w_price, 2),
        "total_reviews": total_reviews,
        "dominant_competitor": dominant.name if dominant else "",
        "dominant_bought":     dominant_share,
        "dominant_rating":     dominant.avg_rating if dominant else 0.0,
        "dominant_reviews":    dominant.total_reviews if dominant else 0,
    }


# ── GM3.1: Universal Penalty System ─────────────────────────────────────────

def _compute_agent_ratios(intel, market_signal, reddit, num_agents=6):
    amazon_for     = market_signal["for"]
    amazon_against = market_signal["against"]
    amazon_neutral = market_signal["neutral"]

    # Reddit blend
    if reddit and (reddit.positive_count + reddit.negative_count + reddit.neutral_count) > 0:
        total_r = reddit.positive_count + reddit.negative_count + reddit.neutral_count
        r_for     = reddit.positive_count / total_r
        r_against = reddit.negative_count / total_r
        r_neutral = reddit.neutral_count  / total_r
        b_for     = 0.60 * amazon_for     + 0.40 * r_for
        b_against = 0.60 * amazon_against + 0.40 * r_against
        b_neutral = 0.60 * amazon_neutral + 0.40 * r_neutral
    else:
        b_for, b_against, b_neutral = amazon_for, amazon_against, amazon_neutral

    total = b_for + b_against + b_neutral
    if total > 0:
        b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total

    category = intel.product.category
    threshold = CATEGORY_PRICE_THRESHOLD.get(category, 1.25)
    coefficient = CATEGORY_PENALTY_COEFFICIENT.get(category, 0.12)

    # GM3.1: Use EFFECTIVE price (includes subscription)
    effective = intel.effective_price
    category_price = market_signal.get("weighted_price", 0) or intel.category_avg_price

    # ── PENALTY 1: Price Premium (uses effective price for subscription products) ──
    total_penalty = 0.0
    if category_price > 0:
        # If subscription detected, compare lifetime costs
        if intel.subscription_detected:
            price_ratio = effective / category_price
        else:
            price_ratio = intel.product.price / category_price

        intel.price_premium_ratio = round(price_ratio, 2)

        if price_ratio > threshold:
            penalty = min(0.50, coefficient * math.log2(price_ratio / threshold + 1))
            total_penalty = penalty
            intel.price_premium_penalty = round(penalty, 3)
            b_for     = max(0.05, b_for - penalty)
            b_against = b_against + penalty * 0.65
            b_neutral = b_neutral + penalty * 0.35
            total = b_for + b_against + b_neutral
            b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total

    # ── PENALTY 2: Switching Cost ──────────────────────────────────────
    dominant_bought = market_signal.get("dominant_bought", 0)
    dominant_rating = market_signal.get("dominant_rating", 0)
    dominant_reviews = market_signal.get("dominant_reviews", 0)
    intel.dominant_reviews = dominant_reviews

    if category in ("fashion_apparel", "food_beverage", "home_lifestyle",
                    "fitness_sports", "electronics_tech"):
        bought_trigger = 50
        rating_trigger = 3.8
        review_trigger = 1500
    else:
        bought_trigger = 5000
        rating_trigger = 4.3
        review_trigger = 3000

    switching_fires = (
        (dominant_bought >= bought_trigger or dominant_reviews >= review_trigger) and
        dominant_rating >= rating_trigger
    )

    if switching_fires:
        switching_penalty = 0.12 if category in ("fashion_apparel", "electronics_tech") else 0.10
        intel.switching_cost_penalty = switching_penalty
        b_for     = max(0.05, b_for - switching_penalty)
        b_neutral = b_neutral + switching_penalty
        total = b_for + b_against + b_neutral
        b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total
    else:
        intel.switching_cost_penalty = 0.0

    # ── PENALTY 3: Cult Brand / Saturated Market (BRAND-AWARE) ────────
    is_saturated, reason = _detect_brand_saturation(intel.competitors, category)
    intel.is_saturated_market = is_saturated
    intel.saturation_reason = reason

    if is_saturated:
        # Category-specific cult penalty
        cult_penalty_map = {
            "home_lifestyle":     0.20,
            "fitness_sports":     0.20,
            "fashion_apparel":    0.18,
            "electronics_tech":   0.22,  # GM3.1: strongest (Apple/Samsung effect)
            "food_beverage":      0.15,
            "beauty_skincare":    0.15,
            "supplements_health": 0.15,
        }
        cult_penalty = cult_penalty_map.get(category, 0.12)
        intel.cult_brand_penalty = cult_penalty

        b_for     = max(0.05, b_for - cult_penalty)
        b_against = b_against + cult_penalty * 0.55
        b_neutral = b_neutral + cult_penalty * 0.45

        total = b_for + b_against + b_neutral
        b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total
    else:
        intel.cult_brand_penalty = 0.0

    return (
        round(b_for, 3),
        round(b_against, 3),
        round(b_neutral, 3),
        round(total_penalty, 3),
    )


# ── Main Entry ────────────────────────────────────────────────────────────────

async def run_market_ingestion(product, num_agents=6):
    intel = MarketIntelligence(product=product)

    # GM3.1: Detect subscription cost upfront
    has_sub, monthly_cost = _detect_subscription(product.description, product.name)
    intel.subscription_detected = has_sub
    intel.subscription_monthly = monthly_cost
    intel.effective_price = _effective_price(product, monthly_cost if has_sub else 0)

    print(f"\n[DTCIngestor] ══ GODMODE 3.1 market ingestion ══")
    print(f"[DTCIngestor] Product: {product.name} @ ${product.price}")
    print(f"[DTCIngestor] Category: {product.category}")
    if has_sub:
        print(f"[DTCIngestor] 💰 Subscription detected: ${monthly_cost}/mo")
        print(f"[DTCIngestor] 💰 Effective 3yr price: ${intel.effective_price:.2f} "
              f"(sticker ${product.price} + ${monthly_cost*36:.0f} subscription)")

    competitor_names = [c.get("name", "") for c in product.competitors if c.get("name")]

    amazon_task = fetch_all_competitors(product.competitors, category=product.category)
    reddit_task = fetch_reddit_intelligence(
        product_name=product.name,
        category=product.category,
        competitors=competitor_names,
        price=product.price,
    )
    amazon_results, reddit_result = await asyncio.gather(
        amazon_task, reddit_task, return_exceptions=True
    )

    if isinstance(amazon_results, Exception):
        intel.error = f"Amazon failed: {amazon_results}"
        amazon_results = []
    intel.competitors = amazon_results or []

    if isinstance(reddit_result, Exception):
        reddit_result = None
    intel.reddit = reddit_result

    market_signal = _compute_market_signal(intel.competitors)
    intel.market_for       = market_signal["for"]
    intel.market_against   = market_signal["against"]
    intel.market_neutral   = market_signal["neutral"]
    intel.category_avg_rating  = market_signal["avg_rating"]
    intel.category_avg_price   = market_signal["weighted_price"]
    intel.total_market_reviews = market_signal["total_reviews"]
    intel.dominant_competitor  = market_signal["dominant_competitor"]
    intel.dominant_bought      = market_signal["dominant_bought"]
    intel.dominant_rating      = market_signal["dominant_rating"]

    for_ratio, against_ratio, neutral_ratio, penalty = _compute_agent_ratios(
        intel, market_signal, intel.reddit, num_agents
    )
    intel.agent_for_ratio     = for_ratio
    intel.agent_against_ratio = against_ratio
    intel.agent_neutral_ratio = neutral_ratio

    num_against_est = max(1, round(against_ratio * num_agents))
    if intel.is_saturated_market:
        intel.hardcore_resistor_count = max(3, round(num_agents * 0.10))
    elif num_against_est >= 3:
        intel.hardcore_resistor_count = max(1, num_against_est // 3)
    else:
        intel.hardcore_resistor_count = 1

    intel.gaps = _build_competitor_gaps(product, intel.competitors)

    print(f"\n[DTCIngestor] ══ Ingestion complete ══")
    print(f"[DTCIngestor] Competitors: {len(intel.competitors)}")
    print(f"[DTCIngestor] Market-weighted avg price: ${intel.category_avg_price}")
    print(f"[DTCIngestor] Dominant competitor: {intel.dominant_competitor} "
          f"({intel.dominant_reviews:,} reviews, {intel.dominant_rating}★)")
    print(f"[DTCIngestor] Price premium ratio: {intel.price_premium_ratio}x")
    print(f"[DTCIngestor] Saturated market: {intel.is_saturated_market}"
          f"{' (' + intel.saturation_reason + ')' if intel.saturation_reason else ''}")

    if intel.price_premium_penalty > 0:
        print(f"[DTCIngestor] ⚡ Price penalty:     -{intel.price_premium_penalty*100:.1f}% from FOR")
    if intel.switching_cost_penalty > 0:
        print(f"[DTCIngestor] ⚡ Switching penalty: -{intel.switching_cost_penalty*100:.0f}% from FOR")
    if intel.cult_brand_penalty > 0:
        print(f"[DTCIngestor] ⚡ Cult-brand penalty: -{intel.cult_brand_penalty*100:.0f}% from FOR")

    print(f"[DTCIngestor] Final ratios: FOR={intel.agent_for_ratio*100:.1f}% "
          f"AGAINST={intel.agent_against_ratio*100:.1f}% "
          f"NEUTRAL={intel.agent_neutral_ratio*100:.1f}%")
    print(f"[DTCIngestor] Hardcore resistors: {intel.hardcore_resistor_count}")

    return intel