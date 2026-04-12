import json
import asyncio
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import add_node, add_edge, get_most_influential

# Strict type whitelist — no more inconsistent types
VALID_ENTITY_TYPES = {
    "person", "organization", "government",
    "concept", "event", "claim", "policy", "product"
}

def normalize_type(raw_type: str) -> str:
    """Normalize inconsistent LLM entity types to our whitelist."""
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
    }
    return mapping.get(raw, raw if raw in VALID_ENTITY_TYPES else "concept")

def find_similar_node(G: nx.DiGraph, name: str) -> str | None:
    """
    Find an existing node that likely refers to the same entity.
    Handles common variations like 'U.S.' vs 'US' vs 'United States'.
    """
    name_lower = name.lower().strip()
    name_clean = name_lower.replace(".", "").replace(",", "").replace("-", " ")

    for existing in G.nodes:
        existing_lower = existing.lower().strip()
        existing_clean = existing_lower.replace(".", "").replace(",", "").replace("-", " ")

        # Exact match after cleaning
        if name_clean == existing_clean:
            return existing

        # One contains the other (handles "US Government" vs "United States Government")
        if len(name_clean) > 4 and len(existing_clean) > 4:
            if name_clean in existing_clean or existing_clean in name_clean:
                return existing

    return None

async def extract_entities(chunk: dict) -> dict:
    """Use LLM to extract entities and claims from a chunk."""

    system = """You are an expert at extracting structured information from text.
Extract entities, claims, and relationships with precision.
Use ONLY these entity types: person, organization, government, concept, event, claim, policy, product
Always respond in valid JSON."""

    prompt = f"""Extract structured information from this text.

Text: {chunk['text']}
Source: {chunk['source']}

Extract:
1. Named entities (real people, organizations, governments, policies, products)
2. Key claims or factual statements
3. Relationships between entities

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
            "text": "the specific claim made",
            "sentiment": "positive/negative/neutral",
            "entity_refs": ["entity names this claim is about"]
        }}
    ],
    "relationships": [
        {{
            "from": "entity name",
            "to": "entity name",
            "relation": "specific relationship description",
            "weight": 0.8
        }}
    ]
}}

Rules:
- Only extract NAMED real entities — no vague terms like "the company" or "experts"
- entity type must be one of: person, organization, government, concept, event, claim, policy, product
- relationships must reference entities you extracted above
- weight is 0.0-1.0 based on how strong/direct the relationship is
- maximum 8 entities, 5 claims, 5 relationships per chunk"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        parsed["source"] = chunk["source"]
        parsed["title"] = chunk["title"]
        return parsed
    except:
        return {"entities": [], "claims": [], "relationships": [], "source": chunk["source"], "title": chunk["title"]}

async def build_graph(chunks: list[dict]) -> nx.DiGraph:
    """
    Build a dense knowledge graph from ingested chunks.
    Features: entity deduplication, type normalization, weighted relationships.
    """
    print(f"[GraphBuilder] Extracting entities from {len(chunks)} chunks...")

    # Run all extractions in parallel
    tasks = [extract_entities(chunk) for chunk in chunks]
    extractions = await asyncio.gather(*tasks)

    G = nx.DiGraph()

    for extraction in extractions:
        source = extraction.get("source", "")
        title = extraction.get("title", "")

        # Add entity nodes with deduplication
        for entity in extraction.get("entities", []):
            if not entity.get("name"):
                continue

            name = entity["name"].strip()
            normalized_type = normalize_type(entity.get("type", "concept"))

            # Check if similar node already exists
            existing = find_similar_node(G, name)

            if existing:
                # Merge — increment citations, keep best description
                G.nodes[existing]["citations"] = G.nodes[existing].get("citations", 1) + 1
                if not G.nodes[existing].get("description") and entity.get("description"):
                    G.nodes[existing]["description"] = entity.get("description", "")
                # Add source to sources list
                sources = G.nodes[existing].get("sources", [])
                if source not in sources:
                    sources.append(source)
                G.nodes[existing]["sources"] = sources
            else:
                # New node
                G.add_node(
                    name,
                    type=normalized_type,
                    description=entity.get("description", ""),
                    citations=1,
                    sources=[source],
                    title=title
                )

        # Add claim nodes
        for claim in extraction.get("claims", []):
            if not claim.get("text"):
                continue
            claim_name = claim["text"][:80]
            existing = find_similar_node(G, claim_name)
            if not existing:
                G.add_node(
                    claim_name,
                    type="claim",
                    description=claim["text"],
                    sentiment=claim.get("sentiment", "neutral"),
                    citations=1,
                    sources=[source],
                    entity_refs=claim.get("entity_refs", [])
                )
            else:
                G.nodes[existing]["citations"] = G.nodes[existing].get("citations", 1) + 1

        # Add weighted relationship edges
        for rel in extraction.get("relationships", []):
            if not rel.get("from") or not rel.get("to") or not rel.get("relation"):
                continue

            from_node = find_similar_node(G, rel["from"]) or rel["from"]
            to_node = find_similar_node(G, rel["to"]) or rel["to"]

            if from_node in G.nodes and to_node in G.nodes:
                if G.has_edge(from_node, to_node):
                    # Edge exists — strengthen it
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

    # Calculate influence scores using weighted PageRank
    if len(G.nodes) > 0:
        try:
            pagerank = nx.pagerank(G, alpha=0.85, weight="weight")
        except:
            pagerank = nx.pagerank(G, alpha=0.85)

        for node in G.nodes:
            G.nodes[node]["influence_score"] = round(pagerank.get(node, 0.0), 4)

    # Filter out low-quality nodes (only 1 citation and no connections)
    # Filter out low-quality nodes (only 1 citation and no connections)
    # Only remove truly isolated noise nodes
    nodes_to_remove = [
        n for n in G.nodes
        if G.nodes[n].get("citations", 1) == 1
        and G.degree(n) == 0
        and len(n) < 5
    ]
    G.remove_nodes_from(nodes_to_remove)

    print(f"[GraphBuilder] Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

def get_graph_summary(G: nx.DiGraph) -> dict:
    """Return a summary of the graph for debugging."""
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