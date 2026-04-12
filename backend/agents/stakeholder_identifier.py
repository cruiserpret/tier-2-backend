import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type
import networkx as nx

# Category defaults — only resistance and influence, never stance
STAKEHOLDER_CATEGORIES = {
    "tech_company":       {"persuasion_resistance": 0.75, "influence_weight": 0.90},
    "government":         {"persuasion_resistance": 0.85, "influence_weight": 0.95},
    "civil_society":      {"persuasion_resistance": 0.60, "influence_weight": 0.65},
    "academic":           {"persuasion_resistance": 0.50, "influence_weight": 0.70},
    "labor_union":        {"persuasion_resistance": 0.70, "influence_weight": 0.60},
    "consumer":           {"persuasion_resistance": 0.30, "influence_weight": 0.40},
    "media":              {"persuasion_resistance": 0.60, "influence_weight": 0.55},
    "investor":           {"persuasion_resistance": 0.65, "influence_weight": 0.75},
    "affected_community": {"persuasion_resistance": 0.55, "influence_weight": 0.50},
    "international_body": {"persuasion_resistance": 0.80, "influence_weight": 0.85},
}

MAX_CATEGORY_SHARE = 0.30  # No single category can exceed 30% of agents
MIN_AGENTS = 10
async def classify_and_position_entities(
    topic: str,
    entities: list[dict],
    graph_context: str
) -> list[dict]:
    """
    Classify entities into stakeholder categories and determine
    their stance based on INTERESTS, not current public positions.
    """

    entity_list = "\n".join([
        f"- {e['name']} (influence: {e.get('influence_score', 0):.3f}, citations: {e.get('citations', 1)})"
        for e in entities[:35]
    ])

    system = """You are an expert at identifying stakeholder interests and how they drive positions.
Your job is NOT to report what entities currently say publicly.
Your job is to analyze what their FUNDAMENTAL INTERESTS are and what stance those interests logically produce.
A company may publicly support regulation while privately opposing it — analyze interests, not PR statements.
Respond in valid JSON only."""

    prompt = f"""Proposition: {topic}

Entities from knowledge graph:
{entity_list}

Knowledge graph context:
{graph_context}

For each real stakeholder, analyze their fundamental interests and derive their logical stance.

Think through each stakeholder like this:
1. What does this entity fundamentally want? (profit, power, safety, freedom, etc.)
2. How does this proposition affect those interests?
3. What stance do those interests logically push them toward?

IMPORTANT: Do NOT default everyone to the same stance. Real debates have genuine disagreement.
For any regulation topic — some entities genuinely benefit from regulation, others genuinely lose.
Derive that from interests, not from what today's news says they support.

Respond in this exact JSON format:
{{
    "stakeholders": [
        {{
            "name": "entity name exactly as given",
            "category": "tech_company/government/civil_society/academic/labor_union/consumer/media/investor/affected_community/international_body",
            "fundamental_interests": "what this entity fundamentally wants in 1 sentence",
            "interest_analysis": "how this proposition affects their interests in 1-2 sentences",
            "real_position": "their logical position derived from interests, NOT their public PR stance",
            "stance": "for/against/neutral",
            "stake": "why this outcome matters to them",
            "relevance_score": 0.85
        }}
    ]
}}

Rules:
- Stance comes from INTEREST ANALYSIS, never from public statements or news sentiment
- Ensure genuine diversity — not everyone can have the same stance
- A tech company might be AGAINST regulation (threatens profits) OR FOR it (legitimizes industry, hurts competitors)
- Derive which one from their specific situation in the graph context
- Maximum 15 stakeholders
- Only include entities with a genuine stake"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        return parsed.get("stakeholders", [])
    except Exception as e:
        print(f"[StakeholderIdentifier] Classification error: {e}")
        return []
    
def enforce_diversity(stakeholders: list[dict], num_agents: int) -> list[dict]:
    """
    Ensure no single category dominates.
    Cap any category at MAX_CATEGORY_SHARE of total agents.
    Returns ranked diverse list.
    """
    max_per_category = max(1, int(num_agents * MAX_CATEGORY_SHARE))

    # Sort by relevance score descending
    sorted_s = sorted(stakeholders, key=lambda x: x.get("relevance_score", 0.5), reverse=True)

    category_counts = {}
    selected = []

    for s in sorted_s:
        cat = s.get("category", "civil_society")
        count = category_counts.get(cat, 0)
        if count < max_per_category:
            selected.append(s)
            category_counts[cat] = count + 1

    return selected

def fill_to_count(stakeholders: list[dict], num_agents: int) -> list[dict]:
    """
    If we have fewer stakeholders than agents needed,
    create representatives of the same stakeholders with slightly
    different roles to fill the count.
    """
    if len(stakeholders) >= num_agents:
        return stakeholders[:num_agents]

    filled = stakeholders.copy()
    i = 0
    while len(filled) < num_agents:
        base = stakeholders[i % len(stakeholders)]
        filled.append({
            **base,
            "name": f"{base['name']} (representative {i // len(stakeholders) + 2})",
            "real_position": f"A representative perspective aligned with {base['name']}'s position: {base['real_position']}"
        })
        i += 1

    return filled

async def identify_stakeholders(
    topic: str,
    G: nx.DiGraph,
    num_agents: int = 20
) -> list[dict]:
    """
    Main function — identifies real stakeholders from knowledge graph.
    Returns enriched stakeholder list ready for persona generation.
    """
    # Enforce minimum
    num_agents = max(num_agents, MIN_AGENTS)
    print(f"[StakeholderIdentifier] Identifying stakeholders for: {topic} ({num_agents} agents)")

    # Pull entities from graph
    influential = get_most_influential(G, top_n=40)
    orgs = get_nodes_by_type(G, "org")
    people = get_nodes_by_type(G, "person")

    all_entities = influential.copy()
    for e in orgs + people:
        if not any(x["name"] == e["name"] for x in all_entities):
            all_entities.append(e)

    # Build graph context string for LLM to read real positions from
    graph_context = "\n".join([
        f"- {n['name']}: {n.get('description', '')[:120]} [cited {n.get('citations', 1)}x]"
        for n in influential[:20]
    ])

    # Classify entities and get real positions
    raw_stakeholders = await classify_and_position_entities(topic, all_entities, graph_context)

    if not raw_stakeholders:
        print("[StakeholderIdentifier] No stakeholders identified, using graph entities as fallback")
        raw_stakeholders = [
            {
                "name": e["name"],
                "category": "civil_society",
                "real_position": f"Has a stake in {topic}",
                "stance": "neutral",
                "stake": "Identified from knowledge graph",
                "relevance_score": e.get("influence_score", 0.5)
            }
            for e in influential[:15]
        ]

    # Enforce diversity — no category dominates
    diverse = enforce_diversity(raw_stakeholders, num_agents)

    # Fill to required count if needed
    filled = fill_to_count(diverse, num_agents)

    # Enrich with category weights
    enriched = []
    for s in filled:
        category = s.get("category", "civil_society")
        defaults = STAKEHOLDER_CATEGORIES.get(category, STAKEHOLDER_CATEGORIES["civil_society"])
        enriched.append({
            **s,
            "persuasion_resistance": defaults["persuasion_resistance"],
            "influence_weight": defaults["influence_weight"],
        })

    print(f"[StakeholderIdentifier] Identified {len(enriched)} stakeholders across {len(set(s['category'] for s in enriched))} categories")
    for s in enriched:
        print(f"  → {s['name']} [{s['category']}] stance: {s['stance']}")

    return enriched