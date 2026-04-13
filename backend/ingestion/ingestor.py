import asyncio
import aiohttp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from backend.utils.text_utils import chunk_text, clean_text

async def search_web(query: str, num_results: int = 10) -> list[dict]:
    """Search web using Tavily and return raw results."""
    url = "https://api.tavily.com/search"

    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "num_results": num_results,
        "search_depth": "advanced",
        "include_raw_content": True,
    }

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

def build_institutional_queries(topic: str) -> list[str]:
    """
    Queries targeting news, policy, corporate statements.
    Captures what institutions, executives, and policy makers say.
    """
    base = topic.rstrip("?").strip()
    return [
        topic,
        f"arguments for {base}",
        f"arguments against {base}",
        f"{base} policy regulation industry",
        f"{base} expert analysis",
    ]

def build_public_queries(topic: str) -> list[str]:
    """
    Queries targeting actual public discourse — personal experiences,
    emotional reactions, lived impact. NOT articles about public opinion.
    Designed to find the voices of directly affected people.
    """
    base = topic.rstrip("?").strip()
    return [
        f"{base} reddit",
        f"{base} personal experience story",
        f"{base} affected workers artists creators",
        f"how {base} affected my life",
        f"{base} forum discussion debate opinions",
        f"{base} community response backlash support",
    ]

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
    Uses smaller chunks and preserves emotional language.
    Tags chunks with sentiment strength for graph weighting.
    """
    from backend.utils.text_utils import detect_sentiment_strength

    seen_pub = set()
    chunks = []

    for result_set in results:
        for r in result_set:
            content = clean_text(
                r.get("raw_content") or r.get("content", ""),
                preserve_emotion=True  # keep emotional signal
            )
            if not content:
                continue

            # Use smaller chunks for public discourse
            for chunk in chunk_text(content, mode="public"):
                key = f"{r.get('url', '')}_{chunk[:50]}"

                # Skip if already in institutional or public
                if key in seen_pub or key in seen_inst:
                    continue

                seen_pub.add(key)
                sentiment_strength = detect_sentiment_strength(chunk)

                chunks.append({
                    "text": chunk,
                    "source": r.get("url", ""),
                    "title": r.get("title", ""),
                    "type": "public_sentiment",
                    "chunk_type": "public",
                    "sentiment_strength": sentiment_strength
                })

    return chunks

async def ingest(
    topic: str,
    pdf_paths: list[str] = []
) -> tuple[list[dict], list[dict]]:
    """
    Main ingestion function.

    Returns (institutional_chunks, public_chunks).
    Institutional: news, policy, corporate content — 500 word chunks
    Public: forum posts, personal stories, reactions — 150 word chunks

    Each population only sees their own chunks during debate.
    """
    inst_queries = build_institutional_queries(topic)
    pub_queries = build_public_queries(topic)

    # All searches fire in parallel
    inst_tasks = [search_web(q, num_results=8) for q in inst_queries]
    pub_tasks = [search_web(q, num_results=8) for q in pub_queries]
    pdf_tasks = [parse_pdf(path) for path in pdf_paths]

    all_results = await asyncio.gather(*inst_tasks, *pub_tasks, *pdf_tasks)

    num_inst = len(inst_tasks)
    num_pub = len(pub_tasks)

    inst_raw = all_results[:num_inst]
    pub_raw = all_results[num_inst:num_inst + num_pub]
    pdf_results = all_results[num_inst + num_pub:]

    # Process institutional
    seen_inst = set()
    inst_chunks = process_institutional_results(inst_raw, seen_inst)

    # Add PDFs to institutional
    for pdf_chunk_list in pdf_results:
        inst_chunks.extend(pdf_chunk_list)

    # Process public — with emotion preservation and sentiment scoring
    pub_chunks = process_public_results(pub_raw, seen_inst)

    print(f"[Ingestor] {len(inst_chunks)} institutional chunks | {len(pub_chunks)} public chunks for: {topic}")

    # Log sentiment distribution
    high_sentiment = sum(1 for c in pub_chunks if c.get("sentiment_strength", 0) > 0.3)
    print(f"[Ingestor] Public chunks with strong sentiment signal: {high_sentiment}/{len(pub_chunks)}")

    return inst_chunks, pub_chunks