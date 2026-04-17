"""
backend/dtc/dtc_ingestor.py

DTC Market Intelligence Orchestrator for Assembly Tier 2.

Runs Amazon and Reddit ingestion in parallel, combines into
a single MarketIntelligence object used by all downstream modules:
  - buyer_persona_generator.py
  - competitor_intel.py
  - price_sensitivity.py
  - market_debate_engine.py

PIPELINE:
  Input:  Product brief (name, description, price, category, competitors)
  Step 1: Parallel fetch — Amazon DETAIL for each competitor + Reddit signal
  Step 2: Compute market signal (FOR/AGAINST/NEUTRAL ratios)
  Step 3: Build competitor gap analysis
  Step 4: Return MarketIntelligence

TIMING TARGET:
  Total ingestion: < 8 seconds (runs during frontend "Competitor Intel" phase)
  Amazon DETAIL:   ~4s per competitor (parallel, so 4s total regardless of count)
  Reddit:          ~6s (4 parallel Tavily searches)
  Combined:        ~6-8s with asyncio.gather
"""

import asyncio
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
    """
    Input from the DTC simulation form.
    Mirrors DTCHomeView.vue form fields.
    """
    name:        str
    description: str
    price:       float
    category:    str = "general"
    demographic: str = ""
    competitors: list = field(default_factory=list)  # [{"name": str, "asin": str}]


@dataclass
class CompetitorGap:
    """
    Feature gap between user product and one competitor.
    Used in Round 2 (Competitor Comparison) debate prompts.
    """
    competitor_name:   str
    competitor_price:  float
    competitor_rating: float
    competitor_bought: int
    user_price_diff:   float    # user price - competitor price (positive = user more expensive)
    star_signal:       dict     # {"for": 0.92, "against": 0.03, "neutral": 0.05}
    top_praise:        list[str]    # what buyers love about competitor
    top_complaints:    list[str]    # what buyers hate — user's opportunity
    review_topics:     list[dict]   # [{"topic": str, "label": str, "count": int}]
    ai_summary:        str


@dataclass
class MarketIntelligence:
    """
    Complete market intelligence package.
    This is the single object passed to all downstream modules.
    """
    product:      ProductBrief
    competitors:  list[CompetitorProfile] = field(default_factory=list)
    reddit:       RedditIntelligence = None
    gaps:         list[CompetitorGap] = field(default_factory=list)

    # Aggregated market signal across all competitors
    market_for:     float = 0.0
    market_against: float = 0.0
    market_neutral: float = 0.0

    # Category-level context
    category_avg_rating:  float = 0.0
    category_avg_price:   float = 0.0
    total_market_reviews: int = 0

    # Agent initialization ratios
    # Derived from competitor star distributions — ground truth for Deffuant model
    agent_for_ratio:     float = 0.0
    agent_against_ratio: float = 0.0
    agent_neutral_ratio: float = 0.0

    error: str = ""


# ── Gap Analysis ──────────────────────────────────────────────────────────────

def _build_competitor_gaps(
    product:     ProductBrief,
    competitors: list[CompetitorProfile]
) -> list[CompetitorGap]:
    """
    Build feature gap analysis for each competitor.

    The gap analysis answers: what do their buyers complain about
    that our product potentially solves? This is the killer feature
    of Tier 2 — auto-detecting competitor weaknesses from real reviews.

    APPROACH:
    - Top praised features → what we're competing against
    - Top complained features → our opportunity to differentiate
    - Price gap → positioning signal for Van Westendorp PSM
    - Star signal → how satisfied is their market?
    """
    gaps = []

    for comp in competitors:
        if not comp.found_on_amazon:
            continue

        signal = compute_weighted_signal(comp.reviews, comp.star_distribution)

        # Extract praise signals from review topics (high mention count = strong signal)
        top_praise = []
        top_complaints = []

        for topic in comp.review_topics[:5]:
            label = topic.get("label", "")
            if label:
                top_praise.append(label)

        # Extract complaint signals from negative reviews
        for review in comp.reviews:
            if review.star_rating <= 2:
                # Short extract of negative review for complaint signal
                text = review.text[:200].strip()
                if text and text not in top_complaints:
                    top_complaints.append(text)

        # Price positioning
        price_diff = round(product.price - comp.price, 2)

        gap = CompetitorGap(
            competitor_name=comp.name,
            competitor_price=comp.price,
            competitor_rating=comp.avg_rating,
            competitor_bought=comp.bought_last_month,
            user_price_diff=price_diff,
            star_signal=signal,
            top_praise=top_praise[:4],
            top_complaints=top_complaints[:3],
            review_topics=comp.review_topics[:6],
            ai_summary=comp.ai_summary,
        )

        gaps.append(gap)

    return gaps


def _compute_market_signal(competitors: list[CompetitorProfile]) -> dict:
    """
    Aggregate market signal across all competitors.

    Method: weighted average by total_reviews.
    Competitors with more reviews get more weight — they represent
    a larger sample of actual market behavior.

    Returns: {"for": float, "against": float, "neutral": float,
              "avg_rating": float, "avg_price": float, "total_reviews": int}
    """
    if not competitors:
        return {
            "for": 0.33, "against": 0.33, "neutral": 0.34,
            "avg_rating": 0.0, "avg_price": 0.0, "total_reviews": 0
        }

    total_weight  = 0
    weighted_for  = 0.0
    weighted_against = 0.0
    weighted_neutral = 0.0
    total_reviews = 0
    prices        = []
    ratings       = []

    for comp in competitors:
        if not comp.found_on_amazon or comp.total_reviews == 0:
            continue

        signal = compute_weighted_signal(comp.reviews, comp.star_distribution)
        weight = comp.total_reviews

        weighted_for     += signal["for"]     * weight
        weighted_against += signal["against"] * weight
        weighted_neutral += signal["neutral"] * weight
        total_weight     += weight
        total_reviews    += comp.total_reviews

        if comp.price > 0:
            prices.append(comp.price)
        if comp.avg_rating > 0:
            ratings.append(comp.avg_rating)

    if total_weight == 0:
        return {
            "for": 0.33, "against": 0.33, "neutral": 0.34,
            "avg_rating": 0.0, "avg_price": 0.0, "total_reviews": 0
        }

    return {
        "for":           round(weighted_for     / total_weight, 3),
        "against":       round(weighted_against / total_weight, 3),
        "neutral":       round(weighted_neutral / total_weight, 3),
        "avg_rating":    round(sum(ratings) / len(ratings), 2) if ratings else 0.0,
        "avg_price":     round(sum(prices)  / len(prices),  2) if prices  else 0.0,
        "total_reviews": total_reviews,
    }


def _compute_agent_ratios(
    market_signal: dict,
    reddit:        RedditIntelligence,
    num_agents:    int = 6
) -> tuple[float, float, float]:
    """
    Compute agent initialization ratios for the Deffuant debate model.

    BLENDED SIGNAL:
      60% Amazon (verified purchase behavior)
      40% Reddit (organic community opinion)

    Reddit signal is computed from positive/negative/neutral counts.
    If Reddit has no signal, falls back to Amazon only.

    Returns: (for_ratio, against_ratio, neutral_ratio)
    All three sum to 1.0.
    """
    amazon_for     = market_signal["for"]
    amazon_against = market_signal["against"]
    amazon_neutral = market_signal["neutral"]

    # Reddit sentiment ratio
    if reddit and (reddit.positive_count + reddit.negative_count + reddit.neutral_count) > 0:
        total_reddit = reddit.positive_count + reddit.negative_count + reddit.neutral_count
        reddit_for     = reddit.positive_count / total_reddit
        reddit_against = reddit.negative_count / total_reddit
        reddit_neutral = reddit.neutral_count  / total_reddit

        blended_for     = 0.60 * amazon_for     + 0.40 * reddit_for
        blended_against = 0.60 * amazon_against + 0.40 * reddit_against
        blended_neutral = 0.60 * amazon_neutral + 0.40 * reddit_neutral
    else:
        blended_for     = amazon_for
        blended_against = amazon_against
        blended_neutral = amazon_neutral

    # Normalize to sum to 1.0
    total = blended_for + blended_against + blended_neutral
    if total > 0:
        blended_for     /= total
        blended_against /= total
        blended_neutral /= total

    return (
        round(blended_for, 3),
        round(blended_against, 3),
        round(blended_neutral, 3),
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def run_market_ingestion(
    product:    ProductBrief,
    num_agents: int = 6
) -> MarketIntelligence:
    """
    Main orchestration function.
    Runs Amazon + Reddit ingestion in parallel.

    Args:
        product:    ProductBrief from form input
        num_agents: Number of buyer agents to initialize

    Returns:
        MarketIntelligence — complete market picture
    """
    intel = MarketIntelligence(product=product)

    print(f"\n[DTCIngestor] ══ Starting market ingestion ══")
    print(f"[DTCIngestor] Product: {product.name} @ ${product.price}")
    print(f"[DTCIngestor] Category: {product.category}")
    print(f"[DTCIngestor] Competitors: {[c.get('name') for c in product.competitors]}")

    # Run Amazon + Reddit in parallel
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
        amazon_task,
        reddit_task,
        return_exceptions=True
    )

    # Handle Amazon results
    if isinstance(amazon_results, Exception):
        print(f"[DTCIngestor] Amazon ingestion failed: {amazon_results}")
        intel.error = f"Amazon ingestion failed: {amazon_results}"
        amazon_results = []

    intel.competitors = amazon_results or []

    # Handle Reddit results
    if isinstance(reddit_result, Exception):
        print(f"[DTCIngestor] Reddit ingestion failed: {reddit_result}")
        reddit_result = None

    intel.reddit = reddit_result

    # Compute market signal
    market_signal = _compute_market_signal(intel.competitors)

    intel.market_for       = market_signal["for"]
    intel.market_against   = market_signal["against"]
    intel.market_neutral   = market_signal["neutral"]
    intel.category_avg_rating  = market_signal["avg_rating"]
    intel.category_avg_price   = market_signal["avg_price"]
    intel.total_market_reviews = market_signal["total_reviews"]

    # Compute agent ratios
    intel.agent_for_ratio, intel.agent_against_ratio, intel.agent_neutral_ratio = \
        _compute_agent_ratios(market_signal, intel.reddit, num_agents)

    # Build competitor gap analysis
    intel.gaps = _build_competitor_gaps(product, intel.competitors)

    print(f"\n[DTCIngestor] ══ Ingestion complete ══")
    print(f"[DTCIngestor] Competitors analyzed: {len(intel.competitors)}")
    print(f"[DTCIngestor] Reddit signals: {len(intel.reddit.signals) if intel.reddit else 0}")
    print(f"[DTCIngestor] Total market reviews: {intel.total_market_reviews:,}")
    print(f"[DTCIngestor] Category avg price: ${intel.category_avg_price}")
    print(f"[DTCIngestor] Category avg rating: {intel.category_avg_rating}★")
    print(f"[DTCIngestor] Market signal: FOR={intel.market_for*100:.1f}% "
          f"AGAINST={intel.market_against*100:.1f}% "
          f"NEUTRAL={intel.market_neutral*100:.1f}%")
    print(f"[DTCIngestor] Agent ratios: FOR={intel.agent_for_ratio*100:.1f}% "
          f"AGAINST={intel.agent_against_ratio*100:.1f}% "
          f"NEUTRAL={intel.agent_neutral_ratio*100:.1f}%")

    return intel


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("Assembly Tier 2 — DTC Ingestor Test")
        print("=" * 60)

        product = ProductBrief(
            name="CollagenRise Daily Serum",
            description="A vegan collagen-boosting serum with bakuchiol. "
                        "No synthetic fragrance. Clinically tested.",
            price=49.0,
            category="beauty_skincare",
            demographic="Women 28-45, clean beauty enthusiasts",
            competitors=[
                {"name": "The Ordinary Niacinamide", "asin": "B01MDTVZTZ"},
                {"name": "Drunk Elephant", "asin": ""},
            ],
        )

        intel = await run_market_ingestion(product, num_agents=6)

        print("\n── Market Intelligence Summary ──────────────────────")
        print(f"Competitors found:     {len([c for c in intel.competitors if c.found_on_amazon])}")
        print(f"Total market reviews:  {intel.total_market_reviews:,}")
        print(f"Category avg price:    ${intel.category_avg_price}")
        print(f"Category avg rating:   {intel.category_avg_rating}★")
        print(f"Market FOR:            {intel.market_for*100:.1f}%")
        print(f"Market AGAINST:        {intel.market_against*100:.1f}%")
        print(f"Market NEUTRAL:        {intel.market_neutral*100:.1f}%")
        print(f"Agent FOR ratio:       {intel.agent_for_ratio*100:.1f}%")
        print(f"Agent AGAINST ratio:   {intel.agent_against_ratio*100:.1f}%")
        print(f"Agent NEUTRAL ratio:   {intel.agent_neutral_ratio*100:.1f}%")

        print("\n── Competitor Gaps ──────────────────────────────────")
        for gap in intel.gaps:
            print(f"\n  {gap.competitor_name}")
            print(f"    Price:      ${gap.competitor_price} "
                  f"(user is ${abs(gap.user_price_diff):.2f} "
                  f"{'more' if gap.user_price_diff > 0 else 'less'} expensive)")
            print(f"    Rating:     {gap.competitor_rating}★ | "
                  f"{gap.competitor_bought:,} bought/month")
            print(f"    FOR signal: {gap.star_signal['for']*100:.1f}%")
            print(f"    Top praise: {gap.top_praise[:2]}")
            print(f"    Complaints: {len(gap.top_complaints)} found")

        if intel.reddit:
            print(f"\n── Reddit Intelligence ──────────────────────────────")
            print(f"  Signals: {len(intel.reddit.signals)}")
            print(f"  Sentiment: +{intel.reddit.positive_count} "
                  f"-{intel.reddit.negative_count} ~{intel.reddit.neutral_count}")
            print(f"  Positive themes: {intel.reddit.positive_themes}")
            print(f"  Negative themes: {intel.reddit.negative_themes}")

    asyncio.run(test())