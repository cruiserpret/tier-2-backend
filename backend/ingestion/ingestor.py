import asyncio
import aiohttp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from backend.utils.text_utils import chunk_text, clean_text

async def search_web(query: str, num_results: int = 10) -> list[dict]:
    """Search web using Tavily and return chunked results."""
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
            results = data.get("results", [])

            chunks = []
            for r in results:
                content = clean_text(r.get("raw_content") or r.get("content", ""))
                if content:
                    for chunk in chunk_text(content):
                        chunks.append({
                            "text": chunk,
                            "source": r.get("url", ""),
                            "title": r.get("title", ""),
                            "type": "web"
                        })
            return chunks

async def parse_pdf(filepath: str) -> list[dict]:
    """Parse a PDF and return chunked text with source attribution."""
    import pymupdf

    chunks = []
    doc = pymupdf.open(filepath)

    for page_num, page in enumerate(doc):
        raw_text = page.get_text()
        content = clean_text(raw_text)
        if content:
            for chunk in chunk_text(content):
                chunks.append({
                    "text": chunk,
                    "source": filepath,
                    "title": f"Page {page_num + 1}",
                    "type": "pdf"
                })

    doc.close()
    return chunks

def build_institutional_queries(topic: str) -> list[str]:
    """
    Queries targeting news, policy, corporate statements.
    Captures institutional actor positions.
    """
    base = topic.rstrip("?").strip()
    return [
        topic,
        f"arguments for {base}",
        f"arguments against {base}",
        f"{base} stakeholders impact",
        f"{base} policy regulation industry",
    ]

def build_public_queries(topic: str) -> list[str]:
    """
    Queries targeting public discourse — forums, employees, consumers.
    Captures everyday people's lived experience and sentiment.
    """
    base = topic.rstrip("?").strip()
    return [
        f"{base} employee opinion",
        f"{base} workers perspective",
        f"{base} people reaction",
        f"how people feel about {base}",
    ]

async def ingest(topic: str, pdf_paths: list[str] = []) -> tuple[list[dict], list[dict]]:
    """
    Main ingestion function.
    Runs institutional and public sentiment searches in parallel.
    Returns (institutional_chunks, public_chunks) as separate lists.
    Each agent population will only query their own chunk set.
    """
    inst_queries = build_institutional_queries(topic)
    pub_queries = build_public_queries(topic)

    # All searches run in parallel
    inst_tasks = [search_web(q, num_results=8) for q in inst_queries]
    pub_tasks = [search_web(q, num_results=6) for q in pub_queries]
    pdf_tasks = [parse_pdf(path) for path in pdf_paths]

    all_results = await asyncio.gather(*inst_tasks, *pub_tasks, *pdf_tasks)

    num_inst = len(inst_tasks)
    num_pub = len(pub_tasks)

    inst_results = all_results[:num_inst]
    pub_results = all_results[num_inst:num_inst + num_pub]
    pdf_results = all_results[num_inst + num_pub:]

    # Deduplicate institutional chunks
    seen_inst = set()
    inst_chunks = []
    for result in inst_results:
        for chunk in result:
            key = f"{chunk.get('source', '')}_{chunk['text'][:50]}"
            if key not in seen_inst:
                seen_inst.add(key)
                chunk["type"] = "institutional"
                inst_chunks.append(chunk)

    # Add PDFs to institutional
    for result in pdf_results:
        for chunk in result:
            inst_chunks.append(chunk)

    # Deduplicate public chunks
    # Also exclude anything already in institutional to avoid overlap
    seen_pub = set()
    pub_chunks = []
    for result in pub_results:
        for chunk in result:
            key = f"{chunk.get('source', '')}_{chunk['text'][:50]}"
            if key not in seen_pub and key not in seen_inst:
                seen_pub.add(key)
                chunk["type"] = "public_sentiment"
                pub_chunks.append(chunk)

    print(f"[Ingestor] {len(inst_chunks)} institutional chunks collected for: {topic}")
    print(f"[Ingestor] {len(pub_chunks)} public sentiment chunks collected for: {topic}")

    return inst_chunks, pub_chunks