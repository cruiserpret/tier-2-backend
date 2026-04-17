"""
backend/dtc/reddit_ingestor.py

Reddit signal ingestor for Assembly Tier 2 DTC Market Simulator.
Uses Tavily web search to pull Reddit discussions — no Reddit API needed.

WHY TAVILY OVER REDDIT API:
  Reddit API requires commercial approval (weeks), costs $0.24/1K calls,
  and has a 1000-post ceiling per subreddit. Tavily indexes Reddit organically
  and returns full post + comment text via search, with no approval needed.

RESEARCH BASIS:
  Noelle-Neumann (1974) — Spiral of Silence:
    People express authentic opinions in communities where their view
    is socially acceptable. r/SkincareAddiction = authentic skincare signal.
    r/Frugal = authentic price resistance signal.
    Subreddit selection is the most important variable in Reddit signal quality.

  Anderson & Magruder (2012) — Yelp reviews predict restaurant survival:
    Online community discussions predict real-world market behavior.
    Reddit discussions in category-specific subreddits are the closest
    analog to focus groups — unprompted, authentic, high-signal.

SUBREDDIT STRATEGY:
  We don't search Reddit generally — we target subreddits by product category.
  Each category has 3-4 subreddits ordered by signal quality:
    1. Category-specific (highest signal, most relevant audience)
    2. General consumer (broader but still relevant)
    3. Price/value focused (captures economic objection voice)

DATA PIPELINE:
  1. Classify product category → select subreddits
  2. Run 3-4 Tavily searches (product + competitors + category discussions)
  3. Parse results → extract post titles, snippets, URLs
  4. Score each result for signal quality
  5. Return RedditSignal objects for buyer_persona_generator.py
"""

import asyncio
import aiohttp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from dataclasses import dataclass, field


# ── Subreddit Map by Category ─────────────────────────────────────────────────
# Ordered by signal quality for each category.
# Noelle-Neumann: match subreddit to product's natural community.

CATEGORY_SUBREDDITS = {
    "beauty_skincare": [
        "SkincareAddiction", "30PlusSkincare", "tretinoin",
        "AsianBeauty", "BeautyGuides",
    ],
    "supplements_health": [
        "Supplements", "nootropics", "nutrition",
        "Fitness", "loseit",
    ],
    "food_beverage": [
        "food", "HealthyFood", "EatCheapAndHealthy",
        "Cooking", "vegan",
    ],
    "saas_software": [
        "SaaS", "entrepreneur", "startups",
        "productivity", "smallbusiness",
    ],
    "fitness_sports": [
        "Fitness", "xxfitness", "homegym",
        "bodyweightfitness", "running",
    ],
    "home_lifestyle": [
        "homeimprovement", "BuyItForLife", "minimalism",
        "malelivingspace", "femalelivingspace",
    ],
    "fashion_apparel": [
        "malefashionadvice", "femalefashionadvice",
        "frugalmalefashion", "frugalfemalefahion",
    ],
    "electronics_tech": [
        "gadgets", "BuyItForLife", "hardware",
        "tech", "techsupport",
    ],
    "pet_products": [
        "dogs", "cats", "pets",
        "DogAdvice", "CatAdvice",
    ],
    "baby_kids": [
        "beyondthebump", "Parenting", "NewParents",
        "BabyBumps", "Mommit",
    ],
    "general": [
        "BuyItForLife", "frugal", "personalfinance",
        "AskReddit", "ProductReviews",
    ],
}

# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RedditSignal:
    """
    Single Reddit signal — a post, comment, or thread snippet.

    signal_type: 'product_mention' | 'category_discussion' | 'competitor_mention'
    sentiment:   'positive' | 'negative' | 'neutral' | 'unknown'
    subreddit:   Source subreddit (community context matters)
    """
    text:        str
    title:       str
    url:         str
    subreddit:   str
    signal_type: str = "product_mention"
    sentiment:   str = "unknown"
    score:       float = 0.5   # 0-1 signal quality score


@dataclass
class RedditIntelligence:
    """
    Complete Reddit intelligence for a product/category.
    Fed into buyer_persona_generator.py for agent enrichment.
    """
    product_name:     str
    category:         str
    subreddits_used:  list[str] = field(default_factory=list)
    signals:          list[RedditSignal] = field(default_factory=list)

    # Aggregated sentiment
    positive_count:   int = 0
    negative_count:   int = 0
    neutral_count:    int = 0

    # Key themes extracted from discussions
    positive_themes:  list[str] = field(default_factory=list)
    negative_themes:  list[str] = field(default_factory=list)
    price_objections: list[str] = field(default_factory=list)
    competitor_mentions: list[str] = field(default_factory=list)

    error:            str = ""


# ── Tavily Search Client ──────────────────────────────────────────────────────

async def _tavily_search(
    session: aiohttp.ClientSession,
    query:   str,
    max_results: int = 5
) -> list[dict]:
    """
    Execute a Tavily search and return raw results.
    Each result: {title, url, content, score}
    """
    if not config.TAVILY_API_KEY:
        print("[RedditIngestor] No Tavily API key")
        return []

    try:
        async with session.post(
            "https://api.tavily.com/search",
            json={
                "api_key":      config.TAVILY_API_KEY,
                "query":        query,
                "max_results":  max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status != 200:
                print(f"[RedditIngestor] Tavily HTTP {resp.status}")
                return []

            data = await resp.json()
            return data.get("results", [])

    except Exception as e:
        print(f"[RedditIngestor] Tavily error: {e}")
        return []


# ── Signal Parsing ────────────────────────────────────────────────────────────

def _extract_subreddit(url: str) -> str:
    """Extract subreddit name from Reddit URL."""
    try:
        parts = url.split("/")
        if "r" in parts:
            idx = parts.index("r")
            return parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    except Exception:
        pass
    return "unknown"


def _score_result(result: dict, product_name: str, competitors: list[str]) -> float:
    """
    Score a search result for signal quality (0-1).

    High score factors:
    - Is from Reddit (verified community signal)
    - Mentions the product by name (direct signal)
    - Has substantial content (not just a title)
    - From a category-relevant subreddit

    Low score factors:
    - Not from Reddit (web article, sponsored content)
    - Very short content
    - Generic/off-topic
    """
    score = 0.0
    url     = result.get("url", "").lower()
    content = result.get("content", "").lower()
    title   = result.get("title", "").lower()

    # Reddit source is highest quality
    if "reddit.com" in url:
        score += 0.4

    # Direct product mention
    if product_name.lower() in content or product_name.lower() in title:
        score += 0.2

    # Competitor mention (competitive intelligence)
    for comp in competitors:
        if comp.lower() in content:
            score += 0.1
            break

    # Content length proxy for signal richness
    content_len = len(content)
    if content_len > 500:
        score += 0.2
    elif content_len > 200:
        score += 0.1

    # Relevance score from Tavily
    tavily_score = float(result.get("score", 0.5) or 0.5)
    score += tavily_score * 0.1

    return min(1.0, score)


def _detect_sentiment(text: str) -> str:
    """
    Rule-based sentiment detection for Reddit signals.
    Simple but fast — full LLM sentiment done in buyer_persona_generator.py.
    """
    text_lower = text.lower()

    positive_signals = [
        "love", "amazing", "great", "excellent", "best", "recommend",
        "worth it", "works", "effective", "helped", "improved",
        "repurchase", "holy grail", "game changer", "obsessed",
    ]
    negative_signals = [
        "terrible", "awful", "worst", "waste", "disappointed",
        "doesn't work", "don't work", "broke me out", "irritated",
        "returned", "refund", "scam", "overpriced", "not worth",
        "avoid", "regret", "burned", "reaction", "allergic",
    ]

    pos_count = sum(1 for s in positive_signals if s in text_lower)
    neg_count = sum(1 for s in negative_signals if s in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def _parse_results_to_signals(
    results:      list[dict],
    product_name: str,
    competitors:  list[str],
    signal_type:  str
) -> list[RedditSignal]:
    """Convert Tavily results to RedditSignal objects."""
    signals = []

    for result in results:
        url     = result.get("url", "")
        title   = result.get("title", "")
        content = result.get("content", "")

        if not content or len(content) < 30:
            continue

        score      = _score_result(result, product_name, competitors)
        subreddit  = _extract_subreddit(url) if "reddit.com" in url else "web"
        sentiment  = _detect_sentiment(f"{title} {content}")

        signals.append(RedditSignal(
            text=content[:600],
            title=title,
            url=url,
            subreddit=subreddit,
            signal_type=signal_type,
            sentiment=sentiment,
            score=score,
        ))

    # Sort by signal quality
    return sorted(signals, key=lambda s: s.score, reverse=True)


# ── Theme Extraction ──────────────────────────────────────────────────────────

def _extract_themes(signals: list[RedditSignal]) -> tuple[list[str], list[str], list[str]]:
    """
    Extract positive themes, negative themes, and price objections
    from Reddit signal corpus.

    Returns: (positive_themes, negative_themes, price_objections)
    """
    # Theme keyword clusters
    positive_clusters = {
        "results_visible":  ["noticed", "worked", "results", "difference", "improved", "cleared"],
        "good_value":       ["worth it", "affordable", "great price", "value", "can't beat"],
        "texture_feel":     ["lightweight", "absorbs", "non-greasy", "smooth", "silky"],
        "skin_improvement": ["pores", "texture", "glow", "clear", "bright", "hydrated"],
        "repurchase":       ["repurchase", "reorder", "staple", "holy grail", "keep buying"],
    }

    negative_clusters = {
        "skin_reaction":    ["broke out", "reaction", "irritated", "burned", "allergic", "rash"],
        "no_results":       ["doesn't work", "no difference", "didn't help", "useless", "waste"],
        "price_concern":    ["too expensive", "overpriced", "not worth the price", "cheaper"],
        "texture_issue":    ["sticky", "greasy", "pills", "tacky", "heavy", "clogs"],
        "trust_concern":    ["fake", "counterfeit", "different formula", "changed", "watered down"],
    }

    price_keywords = [
        "too expensive", "overpriced", "not worth", "cheaper",
        "price is high", "price dropped", "sale", "discount",
        "compared to price", "better value", "worth the money",
    ]

    positive_themes = []
    negative_themes = []
    price_objections = []

    all_text = " ".join(s.text.lower() for s in signals)

    for theme, keywords in positive_clusters.items():
        if any(kw in all_text for kw in keywords):
            positive_themes.append(theme)

    for theme, keywords in negative_clusters.items():
        if any(kw in all_text for kw in keywords):
            negative_themes.append(theme)

    for signal in signals:
        text_lower = signal.text.lower()
        for kw in price_keywords:
            if kw in text_lower:
                # Extract containing sentence
                for sent in signal.text.split("."):
                    if kw in sent.lower() and len(sent.strip()) > 10:
                        price_objections.append(sent.strip()[:150])
                        break

    return (
        list(set(positive_themes)),
        list(set(negative_themes)),
        list(set(price_objections))[:10],
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def fetch_reddit_intelligence(
    product_name:  str,
    category:      str,
    competitors:   list[str] = None,
    price:         float = 0.0,
) -> RedditIntelligence:
    """
    Fetch Reddit intelligence for a product via Tavily.

    Runs 4 targeted searches:
      1. Product mentions on Reddit
      2. Competitor comparison discussions
      3. Category subreddit discussions
      4. Price/value discussions

    Args:
        product_name: e.g. "CollagenRise Daily Serum"
        category:     e.g. "beauty_skincare"
        competitors:  e.g. ["The Ordinary", "Drunk Elephant"]
        price:        e.g. 49.0

    Returns:
        RedditIntelligence with signals and themes
    """
    if competitors is None:
        competitors = []

    intel = RedditIntelligence(
        product_name=product_name,
        category=category,
    )

    # Select subreddits for this category
    subreddits = CATEGORY_SUBREDDITS.get(category, CATEGORY_SUBREDDITS["general"])
    intel.subreddits_used = subreddits[:3]

    print(f"[RedditIngestor] Fetching Reddit signal for: {product_name}")
    print(f"[RedditIngestor] Category: {category} | Subreddits: {subreddits[:3]}")

    # Build search queries
    sub_str = " OR ".join(f"r/{s}" for s in subreddits[:2])

    queries = [
        # 1. Direct product mentions
        f'site:reddit.com "{product_name}" review',

        # 2. Category discussion in target subreddits
        f'site:reddit.com ({sub_str}) {category.replace("_", " ")} serum recommendation',

        # 3. Competitor comparisons (if competitors provided)
        f'site:reddit.com {" vs ".join(competitors[:2])} {category.replace("_", " ")}' if competitors else
        f'site:reddit.com {subreddits[0]} best {category.replace("_", " ")}',

        # 4. Price sensitivity discussion
        f'site:reddit.com {category.replace("_", " ")} "worth it" price {"$" + str(int(price)) if price else ""}',
    ]

    signal_types = [
        "product_mention",
        "category_discussion",
        "competitor_mention",
        "price_discussion",
    ]

    all_signals = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            _tavily_search(session, q, max_results=4)
            for q in queries
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                print(f"[RedditIngestor] Search {i+1} failed: {results}")
                continue

            signal_type = signal_types[i] if i < len(signal_types) else "general"
            signals = _parse_results_to_signals(results, product_name, competitors, signal_type)
            all_signals.extend(signals)

            print(f"[RedditIngestor] Query {i+1}: {len(signals)} signals "
                  f"({sum(1 for s in signals if 'reddit.com' in s.url)} from Reddit)")

    # Deduplicate by URL
    seen_urls = set()
    unique_signals = []
    for s in all_signals:
        if s.url not in seen_urls:
            seen_urls.add(s.url)
            unique_signals.append(s)

    # Sort by score, keep top 20
    intel.signals = sorted(unique_signals, key=lambda s: s.score, reverse=True)[:20]

    # Aggregate sentiment
    intel.positive_count = sum(1 for s in intel.signals if s.sentiment == "positive")
    intel.negative_count = sum(1 for s in intel.signals if s.sentiment == "negative")
    intel.neutral_count  = sum(1 for s in intel.signals if s.sentiment == "neutral")

    # Extract themes
    intel.positive_themes, intel.negative_themes, intel.price_objections = \
        _extract_themes(intel.signals)

    # Extract competitor mentions
    comp_mentions = []
    for s in intel.signals:
        for comp in competitors:
            if comp.lower() in s.text.lower() and comp not in comp_mentions:
                comp_mentions.append(comp)
    intel.competitor_mentions = comp_mentions

    print(f"[RedditIngestor] ✓ {product_name}: "
          f"{len(intel.signals)} signals | "
          f"+{intel.positive_count} -{intel.negative_count} ~{intel.neutral_count} | "
          f"themes: {intel.positive_themes[:2]} | "
          f"objections: {intel.negative_themes[:2]}")

    return intel


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("Assembly Tier 2 — Reddit Ingestor Test")
        print("=" * 60)
        print(f"Tavily key: {'✓' if config.TAVILY_API_KEY else '✗ MISSING'}")

        intel = await fetch_reddit_intelligence(
            product_name="CollagenRise Daily Serum",
            category="beauty_skincare",
            competitors=["The Ordinary", "Drunk Elephant"],
            price=49.0,
        )

        print("\n── Result ──────────────────────────────────────────")
        print(f"Signals:           {len(intel.signals)}")
        print(f"Reddit signals:    {sum(1 for s in intel.signals if 'reddit.com' in s.url)}")
        print(f"Sentiment:         +{intel.positive_count} -{intel.negative_count} ~{intel.neutral_count}")
        print(f"Positive themes:   {intel.positive_themes}")
        print(f"Negative themes:   {intel.negative_themes}")
        print(f"Price objections:  {len(intel.price_objections)}")
        print(f"Competitor hits:   {intel.competitor_mentions}")

        print("\nTop 3 signals:")
        for i, s in enumerate(intel.signals[:3]):
            print(f"\n  [{i+1}] {s.subreddit} | {s.sentiment} | score={s.score:.2f}")
            print(f"       {s.title[:80]}")
            print(f"       {s.text[:120]}...")

    asyncio.run(test())