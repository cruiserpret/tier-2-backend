import json
import asyncio
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import add_node, add_edge, get_most_influential

async def extract_entities(chunk: dict) -> dict:
    """Use LLM to extract entities and claims from a chunk."""
    
    system = """You are an expert at extracting structured information.
Extract entities and claims from the given text.
Always respond in valid JSON."""

    prompt = f"""From this text, extract:
1. entities (people, organizations, concepts, events)
2. claims (factual statements or opinions)
3. relationships between entities

Text: {chunk['text']}
Source: {chunk['source']}

Respond in this exact JSON format:
{{
    "entities": [
        {{"name": "entity name", "type": "person/org/concept/event", "description": "brief description"}}
    ],
    "claims": [
        {{"text": "the claim", "sentiment": "positive/negative/neutral"}}
    ],
    "relationships": [
        {{"from": "entity1", "to": "entity2", "relation": "relationship description"}}
    ]
}}"""

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
    Build a knowledge graph from ingested chunks.
    Runs entity extraction in parallel across all chunks.
    Returns a NetworkX directed graph.
    """
    print(f"[GraphBuilder] Extracting entities from {len(chunks)} chunks...")
    
    # Run all extractions in parallel
    tasks = [extract_entities(chunk) for chunk in chunks]
    extractions = await asyncio.gather(*tasks)
    
    # Build the graph
    G = nx.DiGraph()
    
    for extraction in extractions:
        source = extraction.get("source", "")
        title = extraction.get("title", "")
        
        # Add entity nodes
        for entity in extraction.get("entities", []):
            add_node(G, 
                name=entity["name"],
                type=entity["type"],
                description=entity.get("description", ""),
                source=source,
                title=title
            )
        
        # Add claim nodes
        for claim in extraction.get("claims", []):
            add_node(G,
                name=claim["text"][:80],
                type="claim",
                description=claim["text"],
                sentiment=claim.get("sentiment", "neutral"),
                source=source,
                title=title
            )
        
        # Add relationship edges
        for rel in extraction.get("relationships", []):
            if not rel.get("from") or not rel.get("to") or not rel.get("relation"):
                continue
            add_edge(G,
                from_node=rel["from"],
                to_node=rel["to"],
                relation=rel["relation"],
                source=source
            )
    
    # Calculate influence scores using PageRank
    if len(G.nodes) > 0:
        pagerank = nx.pagerank(G, alpha=0.85)
        for node in G.nodes:
            G.nodes[node]["influence_score"] = pagerank.get(node, 0.0)
    
    print(f"[GraphBuilder] Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

def get_graph_summary(G: nx.DiGraph) -> dict:
    """Return a summary of the graph for debugging."""
    return {
        "total_nodes": len(G.nodes),
        "total_edges": len(G.edges),
        "top_entities": get_most_influential(G, top_n=10),
    }