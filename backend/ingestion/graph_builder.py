import json
import asyncio
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import add_node, add_edge, get_most_influential

VALID_ENTITY_TYPES = {
    "person", "organization", "government",
    "concept", "event", "claim", "policy", "product", "experience"
}

def normalize_type(raw_type: str) -> str:
    raw = raw_type.lower().strip()
    mapping = {
        "org": "organization",
        "org.": "organization",
        "company": "organization",
        "corporation": "organization",
        "institution": "organization",
        "ngo": "organization",
        "govt": "government",
        "gov": "government",
        "gov.": "government",
        "nation": "government",
        "country": "government",
        "state": "government",
        "law": "policy",
        "regulation": "policy",
        "bill": "policy",
        "act": "policy",
        "tech": "product",
        "technology": "product",
        "platform": "product",
        "app": "product",
        "tool": "product",
        "idea": "concept",
        "issue": "concept",
        "topic": "concept",
        "opinion": "experience",
        "feeling": "experience",
        "testimony": "experience",
    }
    return mapping.get(raw, raw if raw in VALID_ENTITY_TYPES else "concept")

def find_similar_node(G: nx.DiGraph, name: str) -> str | None:
    """
    Find an existing node that likely refers to the same entity.
    Minimum length of 7 chars to avoid false matches.
    """
    name_lower = name.lower().strip()
    name_clean = name_lower.replace(".", "").replace(",", "").replace("-", " ")

    if len(name_clean) < 7:
        return None

    for existing in G.nodes:
        existing_lower = existing.lower().strip()
        existing_clean = existing_lower.replace(".", "").replace(",", "").replace("-", " ")

        if len(existing_clean) < 7:
            continue

        if name_clean == existing_clean:
            return existing

        if len(name_clean) > 8 and len(existing_clean) > 8:
            if name_clean in existing_clean or existing_clean in name_clean:
                return existing

    return None

async def extract_institutional_entities(chunk: dict) -> dict:
    """
    Extract structured entities from institutional content.
    Focuses on organizations, people, policies, and factual claims.
    """
    system = """You are an expert at extracting structured information from news and policy text.
Extract named entities, factual claims, and relationships.
Use ONLY these entity types: person, organization, government, concept, event, claim, policy, product
Respond in valid JSON only."""

    prompt = f"""Extract structured information from this institutional text.

Text: {chunk['text']}
Source: {chunk['source']}

Respond in this exact JSON format:
{{
    "entities": [
        {{
            "name": "exact entity name",
            "type": "person/organization/government/concept/event/claim/policy/product",
            "description": "what this entity is in 1 sentence"
        }}
    ],
    "claims": [
        {{
            "text": "specific factual claim made",
            "sentiment": "positive/negative/neutral",
            "entity_refs": ["entity names this claim is about"]
        }}
    ],
    "relationships": [
        {{
            "from": "entity name",
            "to": "entity name",
            "relation": "specific relationship",
            "weight": 0.8
        }}
    ]
}}

Rules:
- Only extract NAMED real entities
- Maximum 8 entities, 5 claims, 5 relationships
- Claims must be factual statements, not opinions"""

    try:
        result = await call_llm_json(prompt, system)
        import re
        result_clean = re.sub(r'[\x00-\x1f\x7f]', ' ', result)
        parsed = json.loads(result_clean)
        parsed["source"] = chunk["source"]
        parsed["title"] = chunk.get("title", "")
        parsed["chunk_type"] = "institutional"
        return parsed
    except:
        return {
            "entities": [], "claims": [], "relationships": [],
            "source": chunk["source"],
            "title": chunk.get("title", ""),
            "chunk_type": "institutional"
        }

async def extract_public_entities(chunk: dict) -> dict:
    """
    Extract opinion patterns from public discourse.
    Focuses on personal experiences, emotional positions, and lived impact.
    Different from institutional extraction — captures HOW PEOPLE FEEL not just WHAT HAPPENED.
    """
    system = """You are an expert at understanding public sentiment and lived experience.
Extract the core opinions, experiences, and emotional positions from this public discourse.
Focus on WHAT PEOPLE FEEL and WHY, not just factual information.
Respond in valid JSON only."""

    sentiment_strength = chunk.get("sentiment_strength", 0.0)

    prompt = f"""Extract opinion patterns from this public discourse.

Text: {chunk['text']}
Source: {chunk['source']}
Detected emotional intensity: {sentiment_strength}

Respond in this exact JSON format:
{{
    "entities": [
        {{
            "name": "specific opinion or experience expressed",
            "type": "experience/concept/person/organization",
            "description": "what this person feels or experienced in 1 sentence — use their actual words and emotions"
        }}
    ],
    "claims": [
        {{
            "text": "specific opinion or personal experience claim",
            "sentiment": "positive/negative/neutral",
            "emotional_intensity": "high/medium/low",
            "entity_refs": []
        }}
    ],
    "relationships": []
}}

Rules:
- Focus on OPINIONS and EXPERIENCES not just facts
- Capture emotional language and personal impact
- "I lost clients because of AI" is more valuable than "AI affects artists"
- Maximum 5 entities, 5 claims
- Preserve the emotional specificity — don't sanitize into corporate language"""

    try:
        result = await call_llm_json(prompt, system)
        import re
        result_clean = re.sub(r'[\x00-\x1f\x7f]', ' ', result)
        parsed = json.loads(result_clean)
        parsed["source"] = chunk["source"]
        parsed["title"] = chunk.get("title", "")
        parsed["chunk_type"] = "public"
        parsed["sentiment_strength"] = sentiment_strength
        return parsed
    except:
        return {
            "entities": [], "claims": [], "relationships": [],
            "source": chunk["source"],
            "title": chunk.get("title", ""),
            "chunk_type": "public",
            "sentiment_strength": sentiment_strength
        }

async def build_graph(
    chunks: list[dict],
    graph_source: str = "institutional"
) -> nx.DiGraph:
    """
    Build a knowledge graph from ingested chunks.
    graph_source: "institutional" or "public" — determines extraction strategy.
    """
    print(f"[GraphBuilder] Extracting entities from {len(chunks)} {graph_source} chunks...")

    # Choose extraction strategy based on source
    if graph_source == "public":
        tasks = [extract_public_entities(chunk) for chunk in chunks]
    else:
        tasks = [extract_institutional_entities(chunk) for chunk in chunks]

    extractions = await asyncio.gather(*tasks)

    G = nx.DiGraph()

    for extraction in extractions:
        source = extraction.get("source", "")
        title = extraction.get("title", "")
        chunk_type = extraction.get("chunk_type", graph_source)
        sentiment_strength = extraction.get("sentiment_strength", 0.0)

        # Add entity nodes
        for entity in extraction.get("entities", []):
            if not entity.get("name"):
                continue

            name = entity["name"].strip()
            if len(name) < 3:
                continue

            normalized_type = normalize_type(entity.get("type", "concept"))
            existing = find_similar_node(G, name)

            if existing:
                G.nodes[existing]["citations"] = G.nodes[existing].get("citations", 1) + 1
                if not G.nodes[existing].get("description") and entity.get("description"):
                    G.nodes[existing]["description"] = entity.get("description", "")
                sources = G.nodes[existing].get("sources", [])
                if source not in sources:
                    sources.append(source)
                G.nodes[existing]["sources"] = sources
                # Keep highest sentiment strength
                existing_strength = G.nodes[existing].get("sentiment_strength", 0.0)
                G.nodes[existing]["sentiment_strength"] = max(existing_strength, sentiment_strength)
            else:
                G.add_node(
                    name,
                    type=normalized_type,
                    description=entity.get("description", ""),
                    citations=1,
                    sources=[source],
                    title=title,
                    graph_source=graph_source,
                    sentiment_strength=sentiment_strength
                )

        # Add claim nodes — full text preserved for public discourse
        for claim in extraction.get("claims", []):
            if not claim.get("text"):
                continue

            # Public claims get full text, institutional get truncated
            if graph_source == "public":
                claim_name = claim["text"][:200]  # was 80 — too short
            else:
                claim_name = claim["text"][:80]

            if len(claim_name) < 10:
                continue

            existing = find_similar_node(G, claim_name)
            if not existing:
                G.add_node(
                    claim_name,
                    type="claim",
                    description=claim["text"],
                    sentiment=claim.get("sentiment", "neutral"),
                    emotional_intensity=claim.get("emotional_intensity", "low"),
                    citations=1,
                    sources=[source],
                    entity_refs=claim.get("entity_refs", []),
                    graph_source=graph_source,
                    sentiment_strength=sentiment_strength
                )
            else:
                G.nodes[existing]["citations"] = G.nodes[existing].get("citations", 1) + 1

        # Add relationships (institutional only — public discourse rarely has clean relationships)
        if graph_source == "institutional":
            for rel in extraction.get("relationships", []):
                if not rel.get("from") or not rel.get("to") or not rel.get("relation"):
                    continue

                from_node = find_similar_node(G, rel["from"]) or rel["from"]
                to_node = find_similar_node(G, rel["to"]) or rel["to"]

                if from_node in G.nodes and to_node in G.nodes:
                    if G.has_edge(from_node, to_node):
                        G[from_node][to_node]["weight"] = min(
                            1.0,
                            G[from_node][to_node].get("weight", 0.5) + 0.1
                        )
                        G[from_node][to_node]["citations"] = G[from_node][to_node].get("citations", 1) + 1
                    else:
                        G.add_edge(
                            from_node,
                            to_node,
                            relation=rel["relation"],
                            weight=float(rel.get("weight", 0.5)),
                            source=source,
                            citations=1
                        )

    # PageRank for influence scores
    if len(G.nodes) > 0:
        try:
            pagerank = nx.pagerank(G, alpha=0.85, weight="weight")
        except:
            pagerank = nx.pagerank(G, alpha=0.85)

        for node in G.nodes:
            G.nodes[node]["influence_score"] = round(pagerank.get(node, 0.0), 4)

    # Remove noise nodes — but never remove public sentiment nodes
    # even if they have low citations (they might be the only testimony we have)
    nodes_to_remove = [
        n for n in G.nodes
        if G.nodes[n].get("citations", 1) == 1
        and G.degree(n) == 0
        and len(n) < 5
        and G.nodes[n].get("graph_source") != "public"
    ]
    G.remove_nodes_from(nodes_to_remove)

    print(f"[GraphBuilder] {graph_source} graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

def get_graph_summary(G: nx.DiGraph) -> dict:
    type_counts = {}
    for n, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_nodes": len(G.nodes),
        "total_edges": len(G.edges),
        "node_types": type_counts,
        "top_entities": get_most_influential(G, top_n=10),
    }