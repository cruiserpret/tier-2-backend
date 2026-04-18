"""
backend/dtc/dtc_ingestor.py — GODMODE FINAL

All fixes applied:
 - Fashion threshold lowered to 1.2x (was 1.5x — fashion is price-sensitive)
 - bought_last_month fallback to total_reviews/10 when missing
 - Switching penalty now fires on review-based market share too
 - Price premium penalty scaled more aggressively for apparel
 - Supports 4-50 agents
"""

import asyncio
import math
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


# ── GODMODE: Category-Specific Price Sensitivity Thresholds ──────────────────
# Lower threshold = penalty fires earlier = more aggressive for price-sensitive
# categories. Fashion & food are highly price-elastic. Supplements/beauty less so.
CATEGORY_PRICE_THRESHOLD = {
    "fashion_apparel":    1.15,  # Quince-style disruption is real — fire early
    "food_beverage":      1.25,
    "home_lifestyle":     1.30,
    "beauty_skincare":    1.35,
    "supplements_health": 1.40,
    "electronics_tech":   1.40,
    "fitness_sports":     1.30,
    "saas_software":      1.50,  # B2B less price-sensitive
    "pet_products":       1.35,
    "baby_kids":          1.50,  # necessity, less elastic
    "general":            1.35,
}

# Category-specific penalty coefficients (steeper for elastic categories)
CATEGORY_PENALTY_COEFFICIENT = {
    "fashion_apparel":    0.18,   # 80% more aggressive
    "food_beverage":      0.15,
    "home_lifestyle":     0.12,
    "beauty_skincare":    0.10,
    "supplements_health": 0.10,
    "electronics_tech":   0.10,
    "fitness_sports":     0.12,
    "saas_software":      0.08,
    "pet_products":       0.11,
    "baby_kids":          0.07,
    "general":            0.10,
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

    price_premium_ratio:  float = 1.0
    price_premium_penalty: float = 0.0

    dominant_competitor:   str = ""
    dominant_bought:       int = 0
    dominant_rating:       float = 0.0
    switching_cost_penalty: float = 0.0

    agent_for_ratio:     float = 0.0
    agent_against_ratio: float = 0.0
    agent_neutral_ratio: float = 0.0

    hardcore_resistor_count: int = 0

    error: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _effective_market_share(comp: CompetitorProfile) -> int:
    """
    GODMODE: bought_last_month fallback.
    Some Amazon listings don't return bought_activity. Fall back to review volume
    as a proxy for market penetration (10 reviews ~ 1 bought/month heuristic).
    """
    if comp.bought_last_month > 0:
        return comp.bought_last_month
    if comp.total_reviews > 0:
        return max(10, comp.total_reviews // 10)  # min 10 so non-zero
    return 0


# ── Competitor Gap Analysis ───────────────────────────────────────────────────

def _build_competitor_gaps(
    product:     ProductBrief,
    competitors: list[CompetitorProfile]
) -> list[CompetitorGap]:
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

        price_diff = round(product.price - comp.price, 2)

        gaps.append(CompetitorGap(
            competitor_name=comp.name,
            competitor_price=comp.price,
            competitor_rating=comp.avg_rating,
            competitor_bought=eff_share,  # GODMODE: use effective share
            market_share=round(market_share, 3),
            user_price_diff=price_diff,
            star_signal=signal,
            top_praise=top_praise[:4],
            top_complaints=top_complaints[:3],
            review_topics=comp.review_topics[:6],
            ai_summary=comp.ai_summary,
        ))

    gaps.sort(key=lambda g: g.market_share, reverse=True)
    return gaps


# ── Market Signal with Fallback ───────────────────────────────────────────────

def _compute_market_signal(competitors: list[CompetitorProfile]) -> dict:
    if not competitors:
        return {
            "for": 0.33, "against": 0.33, "neutral": 0.34,
            "avg_rating": 0.0, "weighted_price": 0.0, "total_reviews": 0,
            "dominant_competitor": "", "dominant_bought": 0, "dominant_rating": 0.0,
        }

    # GODMODE: weights use effective market share (bought_last_month OR reviews/10)
    weights = []
    for comp in competitors:
        if not comp.found_on_amazon:
            weights.append(0)
        else:
            weights.append(_effective_market_share(comp))

    total_weight = sum(weights)
    if total_weight == 0:
        return {
            "for": 0.33, "against": 0.33, "neutral": 0.34,
            "avg_rating": 0.0, "weighted_price": 0.0, "total_reviews": 0,
            "dominant_competitor": "", "dominant_bought": 0, "dominant_rating": 0.0,
        }

    w_for = w_against = w_neutral = 0.0
    w_price = 0.0
    w_rating = 0.0
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
        "dominant_bought":     dominant_share,  # effective, non-zero
        "dominant_rating":     dominant.avg_rating if dominant else 0.0,
    }


# ── Agent Ratios with Category-Specific Penalties ─────────────────────────────

def _compute_agent_ratios(
    intel,
    market_signal: dict,
    reddit:        RedditIntelligence,
    num_agents:    int = 6
) -> tuple[float, float, float, float]:
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

    # ── GODMODE: Category-specific price penalty ────────────────────────
    product_price = intel.product.price
    category_price = market_signal.get("weighted_price", 0) or intel.category_avg_price
    category = intel.product.category

    threshold = CATEGORY_PRICE_THRESHOLD.get(category, 1.35)
    coefficient = CATEGORY_PENALTY_COEFFICIENT.get(category, 0.10)

    total_penalty = 0.0
    if category_price > 0:
        price_ratio = product_price / category_price
        intel.price_premium_ratio = round(price_ratio, 2)

        if price_ratio > threshold:
            # Log-scale penalty scaled by category elasticity
            penalty = min(0.50, coefficient * math.log2(price_ratio / threshold + 1))
            total_penalty = penalty
            intel.price_premium_penalty = round(penalty, 3)

            b_for     = max(0.05, b_for - penalty)
            b_against = b_against + penalty * 0.65
            b_neutral = b_neutral + penalty * 0.35

            total = b_for + b_against + b_neutral
            b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total

    # ── GODMODE: Switching cost penalty (with review fallback) ───────────
    dominant_bought = market_signal.get("dominant_bought", 0)
    dominant_rating = market_signal.get("dominant_rating", 0)

    # Fashion/apparel trigger: very dominant competitor OR high review volume
    # Lower threshold because fashion dupes create strong incumbent anchors
    if category in ("fashion_apparel", "food_beverage"):
        switching_threshold_bought = 50    # effective share (reviews/10 at minimum)
        switching_threshold_rating = 3.8
    else:
        switching_threshold_bought = 5000
        switching_threshold_rating = 4.3

    if dominant_bought >= switching_threshold_bought and dominant_rating >= switching_threshold_rating:
        # Fashion gets stronger penalty because price-comparison is trivial (same material, different brand)
        switching_penalty = 0.12 if category == "fashion_apparel" else 0.08
        intel.switching_cost_penalty = switching_penalty

        b_for     = max(0.05, b_for - switching_penalty)
        b_neutral = b_neutral + switching_penalty

        total = b_for + b_against + b_neutral
        b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total
    else:
        intel.switching_cost_penalty = 0.0

    return (
        round(b_for, 3),
        round(b_against, 3),
        round(b_neutral, 3),
        round(total_penalty, 3),
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def run_market_ingestion(
    product:    ProductBrief,
    num_agents: int = 6
) -> MarketIntelligence:
    intel = MarketIntelligence(product=product)

    print(f"\n[DTCIngestor] ══ Starting GODMODE market ingestion ══")
    print(f"[DTCIngestor] Product: {product.name} @ ${product.price}")
    print(f"[DTCIngestor] Category: {product.category}")

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
        print(f"[DTCIngestor] Amazon failed: {amazon_results}")
        intel.error = f"Amazon ingestion failed: {amazon_results}"
        amazon_results = []

    intel.competitors = amazon_results or []

    if isinstance(reddit_result, Exception):
        print(f"[DTCIngestor] Reddit failed: {reddit_result}")
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

    # Hardcore resistors scale with total AGAINST count
    num_against_est = max(1, round(against_ratio * num_agents))
    intel.hardcore_resistor_count = max(1, num_against_est // 3) if num_against_est >= 3 else 1

    intel.gaps = _build_competitor_gaps(product, intel.competitors)

    print(f"\n[DTCIngestor] ══ Ingestion complete ══")
    print(f"[DTCIngestor] Competitors: {len(intel.competitors)}")
    print(f"[DTCIngestor] Market-weighted avg price: ${intel.category_avg_price}")
    print(f"[DTCIngestor] Dominant competitor: {intel.dominant_competitor} "
          f"(share={intel.dominant_bought}, {intel.dominant_rating}★)")
    print(f"[DTCIngestor] Price premium ratio: {intel.price_premium_ratio}x "
          f"(threshold for {product.category}: {CATEGORY_PRICE_THRESHOLD.get(product.category, 1.35)}x)")
    if intel.price_premium_penalty > 0:
        print(f"[DTCIngestor] GODMODE price penalty: -{intel.price_premium_penalty*100:.1f}% from FOR")
    if intel.switching_cost_penalty > 0:
        print(f"[DTCIngestor] GODMODE switching penalty: -{intel.switching_cost_penalty*100:.0f}% from FOR")
    print(f"[DTCIngestor] Agent ratios: FOR={intel.agent_for_ratio*100:.1f}% "
          f"AGAINST={intel.agent_against_ratio*100:.1f}% "
          f"NEUTRAL={intel.agent_neutral_ratio*100:.1f}%")
    print(f"[DTCIngestor] Hardcore resistors: {intel.hardcore_resistor_count}")

    return intel


if __name__ == "__main__":
    async def test():
        product = ProductBrief(
            name="Everlane Cashmere Crew",
            description="100% Mongolian cashmere",
            price=130.0,
            category="fashion_apparel",
            demographic="professionals 28-45",
            competitors=[
                {"name": "Quince Mongolian Cashmere", "asin": ""},
                {"name": "Uniqlo Cashmere Crew", "asin": ""},
            ],
        )
        intel = await run_market_ingestion(product, num_agents=50)
    asyncio.run(test())