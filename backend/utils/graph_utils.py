import networkx as nx

def add_node(G: nx.DiGraph, name: str, **attributes):
    """Add a node to the graph, merging if it already exists."""
    if name in G.nodes:
        G.nodes[name]["citations"] = G.nodes[name].get("citations", 1) + 1
    else:
        G.add_node(name, citations=1, **attributes)

def add_edge(G: nx.DiGraph, from_node: str, to_node: str, **attributes):
    """Add a directed edge between two nodes."""
    if from_node in G.nodes and to_node in G.nodes:
        G.add_edge(from_node, to_node, **attributes)

def get_most_influential(G: nx.DiGraph, top_n: int = 10) -> list[dict]:
    """Return top N nodes by influence score (PageRank)."""
    nodes = []
    for name, data in G.nodes(data=True):
        nodes.append({
            "name": name,
            "type": data.get("type", "unknown"),
            "influence_score": data.get("influence_score", 0.0),
            "citations": data.get("citations", 1),
            "source": data.get("sources", [""])[0] if data.get("sources") else data.get("source", ""),
            "description": data.get("description", ""),
        })

    return sorted(nodes, key=lambda x: x["influence_score"], reverse=True)[:top_n]

def get_nodes_by_type(G: nx.DiGraph, node_type: str) -> list[dict]:
    """Return all nodes of a specific type."""
    return [
        {"name": n, **data}
        for n, data in G.nodes(data=True)
        if data.get("type") == node_type
    ]

def get_neighbors(G: nx.DiGraph, node_name: str) -> list[dict]:
    """Return all neighboring nodes of a given node."""
    if node_name not in G.nodes:
        return []
    return [
        {"name": n, **G.nodes[n]}
        for n in G.neighbors(node_name)
    ]

def query_graph(G: nx.DiGraph, keywords: list[str], top_n: int = 5) -> list[dict]:
    """
    Find the most relevant nodes for a given set of keywords.
    Combines keyword matching with influence score and sentiment weight.
    Public sentiment nodes get a boost to ensure they surface alongside institutional nodes.
    """
    if not keywords:
        return get_most_influential(G, top_n)

    results = []
    keywords_lower = [k.lower().strip() for k in keywords if k]

    for name, data in G.nodes(data=True):
        name_lower = name.lower()
        description_lower = data.get("description", "").lower()

        keyword_score = 0
        for keyword in keywords_lower:
            if not keyword:
                continue
            if keyword in name_lower:
                keyword_score += 2
            if keyword in description_lower:
                keyword_score += 1

        if keyword_score > 0:
            # Boost public sentiment nodes so they surface alongside institutional
            sentiment_boost = 1.5 if data.get("graph_source") == "public" else 1.0
            sentiment_weight = data.get("sentiment_strength", 0.0)

            final_score = (
                keyword_score *
                (data.get("influence_score", 0.01) + 0.01) *
                sentiment_boost *
                (1 + sentiment_weight)
            )

            results.append({
                "name": name,
                "score": final_score,
                "description": data.get("description", ""),
                "source": data.get("sources", [""])[0] if data.get("sources") else data.get("source", ""),
                "citations": data.get("citations", 1),
                "type": data.get("type", "unknown"),
                "graph_source": data.get("graph_source", "institutional"),
                "sentiment_strength": sentiment_weight
            })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

def get_public_sentiment_nodes(G: nx.DiGraph, top_n: int = 10) -> list[dict]:
    """
    Return top public sentiment nodes by emotional intensity.
    Used by public_agent_generator to find the strongest opinion signals.
    """
    nodes = []
    for name, data in G.nodes(data=True):
        if data.get("graph_source") == "public":
            nodes.append({
                "name": name,
                "type": data.get("type", "unknown"),
                "description": data.get("description", ""),
                "sentiment_strength": data.get("sentiment_strength", 0.0),
                "citations": data.get("citations", 1),
                "source": data.get("sources", [""])[0] if data.get("sources") else "",
            })

    return sorted(nodes, key=lambda x: x["sentiment_strength"], reverse=True)[:top_n]