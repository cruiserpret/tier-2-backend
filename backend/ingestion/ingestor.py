import asyncio
import aiohttp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from backend.utils.text_utils import chunk_text, clean_text

async def search_web(query: str, num_results: int = 10, domain_boost: list = None) -> list[dict]:
    """Search web using Tavily and return raw results."""
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "num_results": num_results,
        "search_depth": "advanced",
        "include_raw_content": True,
    }

    if domain_boost:
        payload["include_domains"] = domain_boost

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            data = await response.json()
            return data.get("results", [])

async def parse_pdf(filepath: str) -> list[dict]:
    """Parse a PDF and return chunked text with source attribution."""
    import pymupdf

    chunks = []
    doc = pymupdf.open(filepath)

    for page_num, page in enumerate(doc):
        raw_text = page.get_text()
        content = clean_text(raw_text)
        if content:
            for chunk in chunk_text(content, mode="institutional"):
                chunks.append({
                    "text": chunk,
                    "source": filepath,
                    "title": f"Page {page_num + 1}",
                    "type": "pdf",
                    "chunk_type": "institutional"
                })

    doc.close()
    return chunks

# ── Domain routing ────────────────────────────────────────────────
# Routes Tavily to the right sources based on topic type.
# UCSD questions → r/UCSD, ucsdguardian.org
# General campus questions → college subreddits, higher ed press
# Default → no domain restriction

UCSD_KEYWORDS = [
    "ucsd", "triton", "uc san diego", "rady",
    "price center", "geisel", "revelle", "muir", "warren college"
]

UCSD_DOMAINS = [
    "reddit.com/r/UCSD",
    "ucsdguardian.org",
    "as.ucsd.edu",
    "triton.news",
]

CAMPUS_KEYWORDS = [
    "campus", "university", "college", "student government",
    "library hours", "dormitory", "dorm", "dining hall",
    "tuition", "financial aid", "student loan", "major", "graduation"
]

CAMPUS_DOMAINS = [
    "reddit.com/r/college",
    "reddit.com/r/StudentLoans",
    "chronicle.com",
    "insidehighered.com",
]

def get_domain_boost(topic: str, context: str = "") -> list:
    """
    Returns domain boost list for Tavily based on topic and context.
    UCSD-specific → UCSD domains
    General campus → campus education domains
    Default → no restriction (full web)
    """
    combined = (topic + " " + context).lower()

    if any(kw in combined for kw in UCSD_KEYWORDS):
        print(f"[Ingestor] UCSD topic detected — boosting UCSD domains")
        return UCSD_DOMAINS

    if any(kw in combined for kw in CAMPUS_KEYWORDS):
        print(f"[Ingestor] Campus topic detected — boosting education domains")
        return CAMPUS_DOMAINS

    return []

def build_institutional_queries(topic: str, context: str = "") -> list[str]:
    """
    Queries targeting news, policy, expert analysis.
    Context enriches queries for product/startup questions.
    """
    base = topic.rstrip("?").strip()

    queries = [
        topic,
        f"arguments for {base}",
        f"arguments against {base}",
        f"{base} policy regulation industry",
        f"{base} expert analysis",
    ]

    # If context provided, add context-enriched queries
    if context:
        context_short = context[:100].strip()
        queries.append(f"{base} {context_short}")
        queries.append(f"{context_short} market research analysis")

    return queries

def build_public_queries(topic: str, context: str = "") -> list[str]:
    """
    Queries targeting lived experience and public discourse.
    Context enriches queries for product/startup questions.
    """
    base = topic.rstrip("?").strip()

    queries = [
        f'site:reddit.com "{base}"',
        f'site:reddit.com {base} personal experience',
        f'site:reddit.com {base} affected my life',
        f'site:quora.com {base}',
        f'site:quora.com how has {base} affected you',
        f'{base} "I lost" OR "I gained" OR "it affected me" personal story',
        f'{base} workers creators families impact lived experience',
        f'{base} "I feel" OR "I think" OR "in my experience" opinion',
        f'{base} community response people affected',
    ]

    # If context provided (startup use case), add product-specific queries
    if context:
        context_short = context[:100].strip()
        queries.append(f'site:reddit.com {context_short} would you pay')
        queries.append(f'site:reddit.com {context_short} opinion review')
        queries.append(f'{context_short} user reviews reddit')

    return queries

def process_institutional_results(
    results: list[list[dict]],
    seen: set
) -> list[dict]:
    """Process and deduplicate institutional search results."""
    chunks = []
    for result_set in results:
        for r in result_set:
            content = clean_text(
                r.get("raw_content") or r.get("content", ""),
                preserve_emotion=False
            )
            if not content:
                continue
            for chunk in chunk_text(content, mode="institutional"):
                key = f"{r.get('url', '')}_{chunk[:50]}"
                if key not in seen:
                    seen.add(key)
                    chunks.append({
                        "text": chunk,
                        "source": r.get("url", ""),
                        "title": r.get("title", ""),
                        "type": "institutional",
                        "chunk_type": "institutional"
                    })
    return chunks

def process_public_results(
    results: list[list[dict]],
    seen_inst: set
) -> list[dict]:
    """
    Process public sentiment results.
    Preserves emotional language, tags sentiment strength.
    Caps at MAX_PUBLIC_CHUNKS sorted by signal strength.
    """
    from backend.utils.text_utils import detect_sentiment_strength

    seen_pub = set()
    chunks = []

    for result_set in results:
        for r in result_set:
            content = clean_text(
                r.get("raw_content") or r.get("content", ""),
                preserve_emotion=True
            )
            if not content:
                continue

            source_url = r.get("url", "")

            source_boost = 1.0
            if "reddit.com" in source_url:
                source_boost = 1.5
            elif "quora.com" in source_url:
                source_boost = 1.3

            for chunk in chunk_text(content, mode="public"):
                key = f"{source_url}_{chunk[:50]}"

                if key in seen_pub or key in seen_inst:
                    continue

                seen_pub.add(key)
                sentiment_strength = detect_sentiment_strength(chunk) * source_boost

                chunks.append({
                    "text": chunk,
                    "source": source_url,
                    "title": r.get("title", ""),
                    "type": "public_sentiment",
                    "chunk_type": "public",
                    "sentiment_strength": min(sentiment_strength, 1.0),
                    "is_forum": "reddit.com" in source_url or "quora.com" in source_url
                })

    chunks.sort(key=lambda c: c.get("sentiment_strength", 0), reverse=True)

    MAX_PUBLIC_CHUNKS = 600
    if len(chunks) > MAX_PUBLIC_CHUNKS:
        chunks = chunks[:MAX_PUBLIC_CHUNKS]
        print(f"[Ingestor] Capped public chunks to {MAX_PUBLIC_CHUNKS} (sorted by sentiment strength)")

    return chunks

async def ingest(
    topic: str,
    pdf_paths: list[str] = [],
    context: str = ""
) -> tuple[list[dict], list[dict]]:
    """
    Main ingestion function.
    context — optional additional information about the topic.
              For UCSD questions: "This is a UCSD campus policy question"
              For startups: product description, price point, target user
    Returns (institutional_chunks, public_chunks).
    """
    inst_queries = build_institutional_queries(topic, context)
    pub_queries = build_public_queries(topic, context)
    domain_boost = get_domain_boost(topic, context)

    # Domain boost applies to public queries — that's where local discourse lives
    inst_tasks = [search_web(q, num_results=8) for q in inst_queries]
    pub_tasks = [
        search_web(q, num_results=10, domain_boost=domain_boost if domain_boost else None)
        for q in pub_queries
    ]
    pdf_tasks = [parse_pdf(path) for path in pdf_paths]

    all_results = await asyncio.gather(*inst_tasks, *pub_tasks, *pdf_tasks)

    num_inst = len(inst_tasks)
    num_pub = len(pub_tasks)

    inst_raw = all_results[:num_inst]
    pub_raw = all_results[num_inst:num_inst + num_pub]
    pdf_results = all_results[num_inst + num_pub:]

    seen_inst = set()
    inst_chunks = process_institutional_results(inst_raw, seen_inst)

    for pdf_chunk_list in pdf_results:
        inst_chunks.extend(pdf_chunk_list)

    pub_chunks = process_public_results(pub_raw, seen_inst)

    print(f"[Ingestor] {len(inst_chunks)} institutional chunks | {len(pub_chunks)} public chunks for: {topic}")
    if context:
        print(f"[Ingestor] Context used: {context[:80]}...")

    forum_chunks = sum(1 for c in pub_chunks if c.get("is_forum", False))
    high_sentiment = sum(1 for c in pub_chunks if c.get("sentiment_strength", 0) > 0.3)
    print(f"[Ingestor] Forum chunks (Reddit/Quora): {forum_chunks}/{len(pub_chunks)}")
    print(f"[Ingestor] High sentiment signal chunks: {high_sentiment}/{len(pub_chunks)}")

    return inst_chunks, pub_chunks