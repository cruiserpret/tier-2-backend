"""
backend/dtc/dtc_ingestor.py — GODMODE EDITION

DTC Market Intelligence Orchestrator for Assembly Tier 2.
Upgrades: price premium penalty, competitor market-share weighting,
hardcore resistor flag, recency-weighted reviews.
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
    market_share:      float = 0.0   # GODMODE: weighted by bought_last_month
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

    # GODMODE: price premium context
    price_premium_ratio:  float = 1.0    # user_price / category_weighted_price
    price_premium_penalty: float = 0.0   # penalty applied to agent_for_ratio

    # GODMODE: dominant competitor signal
    dominant_competitor:   str = ""
    dominant_bought:       int = 0
    dominant_rating:       float = 0.0
    switching_cost_penalty: float = 0.0

    agent_for_ratio:     float = 0.0
    agent_against_ratio: float = 0.0
    agent_neutral_ratio: float = 0.0

    # GODMODE: hardcore resistor count for debate engine
    hardcore_resistor_count: int = 0

    error: str = ""


# ── Competitor Gap Analysis (GODMODE) ─────────────────────────────────────────

def _build_competitor_gaps(
    product:     ProductBrief,
    competitors: list[CompetitorProfile]
) -> list[CompetitorGap]:
    """
    GODMODE: Gaps weighted by market penetration (bought_last_month).
    """
    gaps = []

    # Compute total market for share calculation
    total_bought = sum(c.bought_last_month for c in competitors
                       if c.found_on_amazon and c.bought_last_month > 0)

    for comp in competitors:
        if not comp.found_on_amazon:
            continue

        signal = compute_weighted_signal(comp.reviews, comp.star_distribution)
        market_share = (comp.bought_last_month / total_bought) if total_bought > 0 else 0

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
            competitor_bought=comp.bought_last_month,
            market_share=round(market_share, 3),
            user_price_diff=price_diff,
            star_signal=signal,
            top_praise=top_praise[:4],
            top_complaints=top_complaints[:3],
            review_topics=comp.review_topics[:6],
            ai_summary=comp.ai_summary,
        ))

    # Sort by market share — dominant competitor first
    gaps.sort(key=lambda g: g.market_share, reverse=True)
    return gaps


# ── Market Signal with Market-Share Weighting (GODMODE) ──────────────────────

def _compute_market_signal(competitors: list[CompetitorProfile]) -> dict:
    """
    GODMODE: Weighted by bought_last_month (market share) not just total_reviews.

    Chevalier & Mayzlin (2006): market share > review count as revealed preference.
    A product with 100k/month bought reflects current market behavior better than
    a product with 10k reviews that was popular 3 years ago.
    """
    if not competitors:
        return {
            "for": 0.33, "against": 0.33, "neutral": 0.34,
            "avg_rating": 0.0, "weighted_price": 0.0, "total_reviews": 0,
            "dominant_competitor": "", "dominant_bought": 0, "dominant_rating": 0.0,
        }

    # Market share weights from bought_last_month (primary signal)
    # Fall back to total_reviews if bought data is missing
    weights = []
    for comp in competitors:
        if not comp.found_on_amazon:
            weights.append(0)
            continue
        # Prefer bought_last_month; fall back to reviews
        if comp.bought_last_month > 0:
            weights.append(comp.bought_last_month)
        elif comp.total_reviews > 0:
            weights.append(comp.total_reviews / 10)  # normalize to bought-like scale
        else:
            weights.append(0)

    total_weight = sum(weights)
    if total_weight == 0:
        return {
            "for": 0.33, "against": 0.33, "neutral": 0.34,
            "avg_rating": 0.0, "weighted_price": 0.0, "total_reviews": 0,
            "dominant_competitor": "", "dominant_bought": 0, "dominant_rating": 0.0,
        }

    # Weighted signal aggregation
    w_for = w_against = w_neutral = 0.0
    w_price = 0.0
    w_rating = 0.0
    total_reviews = 0

    dominant = None
    dominant_bought = 0

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

        # Track dominant competitor
        if comp.bought_last_month > dominant_bought:
            dominant = comp
            dominant_bought = comp.bought_last_month

    return {
        "for":           round(w_for, 3),
        "against":       round(w_against, 3),
        "neutral":       round(w_neutral, 3),
        "avg_rating":    round(w_rating, 2),
        "weighted_price": round(w_price, 2),
        "total_reviews": total_reviews,
        "dominant_competitor": dominant.name if dominant else "",
        "dominant_bought":     dominant_bought,
        "dominant_rating":     dominant.avg_rating if dominant else 0.0,
    }


# ── Agent Ratios with Price Premium Penalty (GODMODE) ─────────────────────────

def _compute_agent_ratios(
    intel:         "MarketIntelligence",
    market_signal: dict,
    reddit:        RedditIntelligence,
    num_agents:    int = 6
) -> tuple[float, float, float, float]:
    """
    GODMODE: Apply price premium penalty (Monroe 2003) and switching cost (Burnham 2003).

    Returns: (for_ratio, against_ratio, neutral_ratio, total_penalty_applied)
    """
    amazon_for     = market_signal["for"]
    amazon_against = market_signal["against"]
    amazon_neutral = market_signal["neutral"]

    # Reddit blend (60/40 Amazon/Reddit)
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

    # Normalize
    total = b_for + b_against + b_neutral
    if total > 0:
        b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total

    # ── GODMODE: Price Premium Penalty (Monroe 2003) ────────────────────────
    # Log-scale: each doubling of price ratio shifts 10% of FOR to AGAINST/NEUTRAL
    product_price = intel.product.price
    category_price = market_signal.get("weighted_price", 0) or intel.category_avg_price

    total_penalty = 0.0
    if category_price > 0:
        price_ratio = product_price / category_price
        intel.price_premium_ratio = round(price_ratio, 2)

        if price_ratio > 1.5:
            # Log-scale penalty — Magic Spoon at 6.4x → 0.10 * log2(6.4) = 0.27
            penalty = min(0.40, 0.10 * math.log2(price_ratio))
            total_penalty = penalty
            intel.price_premium_penalty = round(penalty, 3)

            # Transfer penalty from FOR: 60% to AGAINST, 40% to NEUTRAL
            b_for     = max(0.05, b_for - penalty)
            b_against = b_against + penalty * 0.6
            b_neutral = b_neutral + penalty * 0.4

            # Re-normalize
            total = b_for + b_against + b_neutral
            b_for, b_against, b_neutral = b_for/total, b_against/total, b_neutral/total

    # ── GODMODE: Switching Cost Penalty (Burnham 2003) ───────────────────────
    # If a dominant competitor has strong market position, reduce FOR further
    dominant_bought = market_signal.get("dominant_bought", 0)
    dominant_rating = market_signal.get("dominant_rating", 0)

    if dominant_bought >= 10000 and dominant_rating >= 4.3:
        switching_penalty = 0.08  # 8% shift from FOR to NEUTRAL
        intel.switching_cost_penalty = switching_penalty

        b_for     = max(0.05, b_for - switching_penalty)
        b_neutral = b_neutral + switching_penalty

        # Re-normalize
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
    """
    Main orchestrator — runs Amazon + Reddit in parallel.
    GODMODE: applies price premium + switching cost penalties to agent ratios.
    """
    intel = MarketIntelligence(product=product)

    print(f"\n[DTCIngestor] ══ Starting GODMODE market ingestion ══")
    print(f"[DTCIngestor] Product: {product.name} @ ${product.price}")
    print(f"[DTCIngestor] Category: {product.category}")

    competitor_names = [c.get("name", "") for c in product.competitors if c.get("name")]

    amazon_task = fetch_all_competitors(
        competitors=product.competitors,
        category=product.category
    )
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

    # Market signal with market-share weighting
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

    # Agent ratios with GODMODE penalties
    for_ratio, against_ratio, neutral_ratio, penalty = _compute_agent_ratios(
        intel, market_signal, intel.reddit, num_agents
    )
    intel.agent_for_ratio     = for_ratio
    intel.agent_against_ratio = against_ratio
    intel.agent_neutral_ratio = neutral_ratio

    # Hardcore resistor count: 1 in every 4 AGAINST agents
    num_against = max(1, round(against_ratio * num_agents))
    intel.hardcore_resistor_count = max(1, num_against // 3) if num_against >= 2 else 1

    # Build gaps (sorted by market share)
    intel.gaps = _build_competitor_gaps(product, intel.competitors)

    print(f"\n[DTCIngestor] ══ Ingestion complete ══")
    print(f"[DTCIngestor] Competitors: {len(intel.competitors)}")
    print(f"[DTCIngestor] Market-weighted avg price: ${intel.category_avg_price}")
    print(f"[DTCIngestor] Dominant competitor: {intel.dominant_competitor} "
          f"({intel.dominant_bought:,}/month, {intel.dominant_rating}★)")
    print(f"[DTCIngestor] Price premium ratio: {intel.price_premium_ratio}x")
    if intel.price_premium_penalty > 0:
        print(f"[DTCIngestor] GODMODE price penalty: -{intel.price_premium_penalty*100:.0f}% from FOR")
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
            name="Magic Spoon Cereal",
            description="High protein low carb cereal. 13g protein. Zero sugar. Keto.",
            price=39.99,
            category="food_beverage",
            demographic="Health-conscious adults 25-45",
            competitors=[
                {"name": "Kashi Go Cereal", "asin": ""},
                {"name": "Three Wishes Cereal", "asin": ""},
            ],
        )
        intel = await run_market_ingestion(product, num_agents=12)
        print("\n── GODMODE Intel ──")
        print(f"Price premium ratio: {intel.price_premium_ratio}x")
        print(f"Price penalty:       -{intel.price_premium_penalty*100:.1f}%")
        print(f"Switching penalty:   -{intel.switching_cost_penalty*100:.1f}%")
        print(f"Agent FOR:           {intel.agent_for_ratio*100:.1f}%")
        print(f"Agent AGAINST:       {intel.agent_against_ratio*100:.1f}%")
        print(f"Hardcore resistors:  {intel.hardcore_resistor_count}")

    asyncio.run(test())