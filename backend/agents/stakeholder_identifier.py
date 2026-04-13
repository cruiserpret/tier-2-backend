import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type
import networkx as nx

STAKEHOLDER_CATEGORIES = {
    "tech_company":       {"persuasion_resistance": 0.55, "influence_weight": 0.90},
    "government":         {"persuasion_resistance": 0.65, "influence_weight": 0.95},
    "civil_society":      {"persuasion_resistance": 0.40, "influence_weight": 0.65},
    "academic":           {"persuasion_resistance": 0.30, "influence_weight": 0.70},
    "labor_union":        {"persuasion_resistance": 0.50, "influence_weight": 0.60},
    "consumer":           {"persuasion_resistance": 0.20, "influence_weight": 0.40},
    "media":              {"persuasion_resistance": 0.40, "influence_weight": 0.55},
    "investor":           {"persuasion_resistance": 0.45, "influence_weight": 0.75},
    "affected_community": {"persuasion_resistance": 0.35, "influence_weight": 0.50},
    "international_body": {"persuasion_resistance": 0.60, "influence_weight": 0.85},
}

CATEGORY_NORMALIZER = {
    "tech_leader": "tech_company",
    "tech_executive": "tech_company",
    "tech_entrepreneur": "tech_company",
    "tech_industry": "tech_company",
    "technology": "tech_company",
    "corporation": "tech_company",
    "business": "tech_company",
    "company": "tech_company",
    "ngo": "civil_society",
    "nonprofit": "civil_society",
    "advocacy": "civil_society",
    "university": "academic",
    "research": "academic",
    "think_tank": "academic",
    "defense_technology": "government",
    "military": "government",
    "defense": "government",
    "nation": "government",
    "country": "government",
    "finance": "investor",
    "fund": "investor",
    "bank": "investor",
    "venture_capital": "investor",
    "vc": "investor",
    "press": "media",
    "news": "media",
    "journalist": "media",
    "union": "labor_union",
    "workers": "labor_union",
    "people": "affected_community",
    "community": "affected_community",
    "citizens": "consumer",
    "public": "consumer",
    "un": "international_body",
    "nato": "international_body",
    "who": "international_body",
    "wto": "international_body",
}

MAX_CATEGORY_SHARE = 0.30
MIN_AGENTS = 5

def normalize_category(raw: str) -> str:
    clean = raw.lower().strip().replace(" ", "_").replace("-", "_")
    if clean in STAKEHOLDER_CATEGORIES:
        return clean
    if clean in CATEGORY_NORMALIZER:
        return CATEGORY_NORMALIZER[clean]
    for key in STAKEHOLDER_CATEGORIES:
        if key in clean or clean in key:
            return key
    return "civil_society"

async def classify_and_position_entities(
    topic: str,
    entities: list[dict],
    graph_context: str
) -> list[dict]:
    entity_list = "\n".join([
        f"- {e['name']} (influence: {e.get('influence_score', 0):.3f}, citations: {e.get('citations', 1)})"
        for e in entities[:35]
    ])

    system = """You are an expert at identifying stakeholders and their real-world positions.
Derive stance from INTERESTS, not public statements or news sentiment.
Respond in valid JSON only."""



    proposition = topic if "?" in topic else f"{topic}?"
    prompt = f"""Proposition being debated: {proposition}

CRITICAL: stance must always be relative to the proposition itself.
- "for" = this entity SUPPORTS what the proposition is suggesting
- "against" = this entity OPPOSES what the proposition is suggesting  
- "neutral" = this entity has genuinely mixed or unclear position

Think carefully about what the proposition is actually asking before assigning stance.
Derive stance from the entity's fundamental interests relative to THIS specific proposition.

Entities from knowledge graph:
{entity_list}

Knowledge graph context:
{graph_context}

For each real stakeholder, analyze their fundamental interests and derive their logical stance.

Think through each stakeholder:
1. What do they fundamentally want?
2. How does this proposition affect those interests?
3. What stance do those interests logically produce?

IMPORTANT: Ensure genuine diversity — not everyone can have the same stance.

Respond in this exact JSON format:
{{
    "stakeholders": [
        {{
            "name": "entity name exactly as given",
            "category": "tech_company/government/civil_society/academic/labor_union/consumer/media/investor/affected_community/international_body",
            "fundamental_interests": "what this entity fundamentally wants",
            "real_position": "their logical position derived from interests",
            "stance": "for/against/neutral",
            "stake": "why this outcome matters to them",
            "relevance_score": 0.85
        }}
    ]
}}

Rules:
- Stance from INTEREST ANALYSIS only — never from public PR statements
- Ensure genuine diversity — not everyone the same stance
- Maximum 15 stakeholders
- Only real organizations with genuine stakes"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        return parsed.get("stakeholders", [])
    except Exception as e:
        print(f"[StakeholderIdentifier] Classification error: {e}")
        return []

def enforce_diversity(stakeholders: list[dict], num_agents: int) -> list[dict]:
    max_per_category = max(1, int(num_agents * MAX_CATEGORY_SHARE))
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
    Fill to required count with representatives.
    No cap — representatives fill until we hit num_agents.
    """
    if len(stakeholders) >= num_agents:
        return stakeholders[:num_agents]

    filled = stakeholders.copy()
    unique_count = len(stakeholders)
    i = 0

    while len(filled) < num_agents:
        base = stakeholders[i % unique_count]
        rep_num = i // unique_count + 2
        filled.append({
            **base,
            "name": f"{base['name']} (representative {rep_num})",
            "real_position": f"A secondary perspective aligned with {base['name']}'s position."
        })
        i += 1

    return filled

async def identify_stakeholders(
    topic: str,
    G: nx.DiGraph,
    num_agents: int = 10
) -> list[dict]:
    num_agents = max(num_agents, MIN_AGENTS)
    print(f"[StakeholderIdentifier] Identifying stakeholders for: {topic} ({num_agents} agents)")

    influential = get_most_influential(G, top_n=40)
    orgs = get_nodes_by_type(G, "org")
    people = get_nodes_by_type(G, "person")

    all_entities = influential.copy()
    for e in orgs + people:
        if not any(x["name"] == e["name"] for x in all_entities):
            all_entities.append(e)

    graph_context = "\n".join([
        f"- {n['name']}: {n.get('description', '')[:120]} [cited {n.get('citations', 1)}x]"
        for n in influential[:20]
    ])

    raw_stakeholders = await classify_and_position_entities(topic, all_entities, graph_context)

    if not raw_stakeholders:
        raw_stakeholders = [
            {
                "name": e["name"],
                "category": "civil_society",
                "fundamental_interests": f"Has a stake in {topic}",
                "real_position": f"Has a stake in {topic}",
                "stance": "neutral",
                "stake": "Identified from knowledge graph",
                "relevance_score": e.get("influence_score", 0.5)
            }
            for e in influential[:10]
        ]

    # Normalize categories
    for s in raw_stakeholders:
        s["category"] = normalize_category(s.get("category", "civil_society"))

    # Enforce diversity
    diverse = enforce_diversity(raw_stakeholders, num_agents)

    # Fill to count with capped representatives
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