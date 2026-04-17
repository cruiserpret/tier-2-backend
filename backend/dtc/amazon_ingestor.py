"""
backend/dtc/amazon_ingestor.py

Amazon data ingestor using Easyparser DETAIL + SEARCH operations.
Rainforest dropped — reviews endpoint permanently unavailable.

EASYPARSER API:
  Base URL:  https://realtime.easyparser.com/v1/request
  SEARCH:    operation=SEARCH, keyword=...
  DETAIL:    operation=DETAIL, asin=...
  Reviews:   Embedded in DETAIL response as top_reviews (8-10 reviews)
             + rating_breakdown (star distribution %)
             + customer_say.review_topics (AI-extracted topic mentions)
             + customer_say.summary_analysis (AI summary)
             + bought_activity (market penetration)

RESEARCH BASIS:
  Chevalier & Mayzlin (2006): Amazon reviews = revealed purchase behavior
  Morwitz et al. (2007): Verified reviews 3.4x more predictive
  Van Westendorp (1976): Price sensitivity from unprompted language
"""

import asyncio
import aiohttp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from dataclasses import dataclass, field
import math


EASYPARSER_URL = "https://realtime.easyparser.com/v1/request"


@dataclass
class ReviewSignal:
    star_rating:   int
    text:          str
    verified:      bool
    helpful_votes: int
    title:         str = ""
    date:          str = ""


@dataclass
class CompetitorProfile:
    name:              str
    asin:              str = ""
    found_on_amazon:   bool = False
    price:             float = 0.0
    avg_rating:        float = 0.0
    total_reviews:     int = 0
    bought_last_month: int = 0
    star_distribution: dict = field(default_factory=dict)
    reviews:           list = field(default_factory=list)
    review_topics:     list = field(default_factory=list)
    ai_summary:        str = ""
    feature_bullets:   list = field(default_factory=list)
    price_sensitivity_signals: list = field(default_factory=list)
    category:          str = ""
    product_title:     str = ""
    error:             str = ""


async def _search_asin(session, query, category=""):
    search_query = f"{query} {category}".strip() if category else query
    params = {
        "api_key":   config.EASYPARSER_API_KEY,
        "platform":  "AMZ",
        "operation": "SEARCH",
        "output":    "json",
        "domain":    ".com",
        "keyword":   search_query,
    }
    try:
        async with session.get(EASYPARSER_URL, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return "", ""
            data = await resp.json()
            if not data.get("request_info", {}).get("success"):
                return "", ""
            results = data.get("result", {}).get("search_results", [])
            if not results:
                return "", ""
            top = results[0]
            asin  = top.get("asin", "")
            title = top.get("title", "")
            print(f"[AmazonIngestor] Found: {asin} — {title[:60]}")
            return asin, title
    except Exception as e:
        print(f"[AmazonIngestor] SEARCH error: {e}")
        return "", ""


async def _fetch_detail(session, asin):
    params = {
        "api_key":   config.EASYPARSER_API_KEY,
        "platform":  "AMZ",
        "operation": "DETAIL",
        "output":    "json",
        "domain":    ".com",
        "asin":      asin,
    }
    try:
        async with session.get(EASYPARSER_URL, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return {}
            data = await resp.json()
            if not data.get("request_info", {}).get("success"):
                return {}
            return data.get("result", {}).get("detail", {})
    except Exception as e:
        print(f"[AmazonIngestor] DETAIL error: {e}")
        return {}


def _parse_star_distribution(rating_breakdown):
    mapping = {"five_star": "5", "four_star": "4", "three_star": "3", "two_star": "2", "one_star": "1"}
    dist = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0}
    for key, star in mapping.items():
        entry = rating_breakdown.get(key, {})
        pct = entry.get("percentage", 0) or 0
        dist[star] = round(pct / 100, 3)
    return dist


def _parse_reviews(top_reviews):
    reviews = []
    for r in top_reviews:
        try:
            rating_raw = r.get("rating", "3")
            star = int(float(str(rating_raw).replace(" out of 5 stars", "")))
            text = r.get("body", "") or ""
            if len(text) < 10:
                continue
            verified      = r.get("verified_purchase", False) or False
            helpful_votes = int(r.get("helpful_votes", 0) or 0)
            title         = r.get("title", "") or ""
            date_raw      = r.get("date", {})
            date          = date_raw.get("raw", "") if isinstance(date_raw, dict) else ""
            reviews.append(ReviewSignal(
                star_rating=max(1, min(5, star)),
                text=text[:800],
                verified=verified,
                helpful_votes=helpful_votes,
                title=title,
                date=date,
            ))
        except Exception:
            continue
    return reviews


def _parse_review_topics(customer_say):
    topics = []
    seen = set()
    for item in customer_say.get("review_topics", []):
        count = item.get("mention_count", 0)
        key   = item.get("topic_key", "")
        label = item.get("display_label", "")
        if count > 0 and key and key not in seen:
            seen.add(key)
            clean_label = label.split("(")[0].strip() if "(" in label else label
            topics.append({"topic": key, "label": clean_label, "count": count})
    return sorted(topics, key=lambda x: x["count"], reverse=True)


def compute_weighted_signal(reviews, star_distribution):
    # Primary: full star distribution (all reviews)
    if star_distribution and any(v > 0 for v in star_distribution.values()):
        p_for     = star_distribution.get("5", 0) + star_distribution.get("4", 0)
        p_against = star_distribution.get("1", 0) + star_distribution.get("2", 0)
        p_neutral = star_distribution.get("3", 0)
    else:
        p_for = p_against = p_neutral = 0

    # Secondary: Morwitz-weighted top reviews
    if reviews:
        fw = aw = nw = tw = 0.0
        for r in reviews:
            w = (2.0 if r.verified else 1.0) * min(1.5, 1.0 + math.log1p(r.helpful_votes) / 10)
            tw += w
            if r.star_rating >= 4:   fw += w
            elif r.star_rating <= 2: aw += w
            else:                    nw += w
        if tw > 0:
            s_for, s_against, s_neutral = fw/tw, aw/tw, nw/tw
        else:
            s_for = s_against = s_neutral = 0
    else:
        s_for = s_against = s_neutral = 0

    has_primary   = (p_for + p_against + p_neutral) > 0
    has_secondary = (s_for + s_against + s_neutral) > 0

    if has_primary and has_secondary:
        f = 0.70*p_for     + 0.30*s_for
        a = 0.70*p_against + 0.30*s_against
        n = 0.70*p_neutral + 0.30*s_neutral
    elif has_primary:
        f, a, n = p_for, p_against, p_neutral
    elif has_secondary:
        f, a, n = s_for, s_against, s_neutral
    else:
        f = a = n = 1/3

    return {
        "for": round(f, 3), "against": round(a, 3), "neutral": round(n, 3),
        "total_signal": len(reviews),
        "verified_count": sum(1 for r in reviews if r.verified),
    }


def extract_price_sensitivity_signals(reviews):
    keywords = [
        "too expensive", "overpriced", "not worth", "too much",
        "worth the price", "worth every penny", "great value",
        "reasonably priced", "good price", "affordable", "worth it",
        "can't beat the price", "half the price", "price point",
        "excellent value", "bang for your buck",
    ]
    signals = []
    seen = set()
    for r in reviews:
        text_lower = r.text.lower()
        for kw in keywords:
            if kw in text_lower:
                for sent in r.text.split("."):
                    if kw in sent.lower() and len(sent.strip()) > 10:
                        key = sent.strip()[:50]
                        if key not in seen:
                            seen.add(key)
                            signals.append(sent.strip()[:200])
                        break
    return signals[:20]


async def fetch_competitor_profile(competitor_name, category="", asin=""):
    print(f"[AmazonIngestor] Analyzing: {competitor_name}")
    profile = CompetitorProfile(name=competitor_name, asin=asin)

    async with aiohttp.ClientSession() as session:
        if not asin:
            asin, _ = await _search_asin(session, competitor_name, category)
            profile.asin = asin

        if not asin:
            profile.error = f"No Amazon listing found for {competitor_name}"
            return profile

        profile.found_on_amazon = True
        detail = await _fetch_detail(session, asin)

        if not detail:
            profile.error = "DETAIL fetch failed"
            return profile

        profile.product_title = detail.get("title", "")
        profile.avg_rating    = float(detail.get("rating", 0) or 0)
        profile.total_reviews = int(detail.get("ratings_total", 0) or 0)

        price_raw = detail.get("buybox_winner", {}).get("price", {}).get("value", 0)
        try:
            profile.price = float(str(price_raw).replace("$", "").replace(",", "").strip())
        except Exception:
            profile.price = 0.0

        bought = detail.get("bought_activity", {})
        profile.bought_last_month = int(bought.get("value", 0) or 0)

        profile.star_distribution = _parse_star_distribution(detail.get("rating_breakdown", {}))
        profile.reviews           = _parse_reviews(detail.get("top_reviews", []))

        customer_say = detail.get("customer_say", {})
        profile.review_topics = _parse_review_topics(customer_say)
        profile.ai_summary    = customer_say.get("summary_analysis", {}).get("overall_summary_text", "")

        profile.feature_bullets            = detail.get("feature_bullets", [])
        profile.price_sensitivity_signals  = extract_price_sensitivity_signals(profile.reviews)

        print(f"[AmazonIngestor] ✓ {competitor_name}: "
              f"{profile.total_reviews:,} reviews | {profile.avg_rating}★ | "
              f"${profile.price} | {profile.bought_last_month:,}/month | "
              f"5★={profile.star_distribution.get('5',0)*100:.0f}% "
              f"1★={profile.star_distribution.get('1',0)*100:.0f}% | "
              f"{len(profile.reviews)} top reviews")

    return profile


async def fetch_all_competitors(competitors, category=""):
    tasks = [
        fetch_competitor_profile(c.get("name", ""), category, c.get("asin", ""))
        for c in competitors if c.get("name", "").strip()
    ]
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    profiles = [r for r in results if not isinstance(r, Exception)]
    print(f"[AmazonIngestor] {len(profiles)}/{len(tasks)} competitors fetched")
    return profiles


if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("Assembly Tier 2 — Amazon Ingestor Test")
        print("=" * 60)
        print(f"Easyparser key: {'✓' if config.EASYPARSER_API_KEY else '✗ MISSING'}")

        profile = await fetch_competitor_profile(
            competitor_name="The Ordinary Niacinamide",
            category="skincare serum"
        )

        print("\n── Result ──────────────────────────────────────────")
        print(f"ASIN:          {profile.asin}")
        print(f"Title:         {profile.product_title[:60]}")
        print(f"Price:         ${profile.price}")
        print(f"Avg rating:    {profile.avg_rating}★")
        print(f"Total reviews: {profile.total_reviews:,}")
        print(f"Bought/month:  {profile.bought_last_month:,}")
        print(f"Star dist:     {profile.star_distribution}")
        print(f"Top reviews:   {len(profile.reviews)}")
        print(f"Topics:        {profile.review_topics[:3]}")
        print(f"AI summary:    {profile.ai_summary[:120]}...")
        print(f"Price signals: {len(profile.price_sensitivity_signals)}")

        signal = compute_weighted_signal(profile.reviews, profile.star_distribution)
        print(f"\nMarket signal:")
        print(f"  FOR:     {signal['for']*100:.1f}%")
        print(f"  AGAINST: {signal['against']*100:.1f}%")
        print(f"  NEUTRAL: {signal['neutral']*100:.1f}%")

    asyncio.run(test())