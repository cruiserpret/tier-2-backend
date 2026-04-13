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
CALIBRATION_THRESHOLD = 0.15  # lowered from 0.20 for more aggressive correction


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


def extract_public_sentiment_distribution(G_pub: nx.DiGraph) -> dict:
    """Extract FOR/AGAINST/NEUTRAL distribution from public sentiment graph."""
    if G_pub is None:
        return None

    claims = get_nodes_by_type(G_pub, "claim")
    if not claims:
        all_nodes = [
            data for _, data in G_pub.nodes(data=True)
            if data.get("sentiment") in ("positive", "negative", "neutral")
        ]
        claims = all_nodes

    if not claims:
        return None

    pos = sum(1 for c in claims if c.get("sentiment") == "positive")
    neg = sum(1 for c in claims if c.get("sentiment") == "negative")
    neu = sum(1 for c in claims if c.get("sentiment") == "neutral")
    total = pos + neg + neu

    if total == 0:
        return None

    distribution = {
        "for":     round(pos / total, 2),
        "against": round(neg / total, 2),
        "neutral": round(neu / total, 2),
        "total_signal": total
    }

    print(f"[StakeholderIdentifier] Public sentiment signal — "
          f"for: {distribution['for']*100:.0f}% / "
          f"against: {distribution['against']*100:.0f}% / "
          f"neutral: {distribution['neutral']*100:.0f}% "
          f"(from {total} claim nodes)")

    return distribution


def get_current_distribution(stakeholders: list[dict]) -> dict:
    total = len(stakeholders)
    if total == 0:
        return {"for": 0, "against": 0, "neutral": 0}

    for_count     = sum(1 for s in stakeholders if s.get("stance") == "for")
    against_count = sum(1 for s in stakeholders if s.get("stance") == "against")
    neutral_count = sum(1 for s in stakeholders if s.get("stance") == "neutral")

    return {
        "for":     round(for_count / total, 2),
        "against": round(against_count / total, 2),
        "neutral": round(neutral_count / total, 2),
    }


async def request_missing_stakeholders(
    topic: str,
    missing_stances: list[str],
    num_needed: int,
    graph_context: str,
    existing_names: list[str]
) -> list[dict]:
    """Ask LLM to generate additional stakeholders for underrepresented stances."""
    system = """You are identifying additional stakeholders whose interests are being overlooked.
These must be REAL, SPECIFIC entities with genuine stakes in this topic.
Respond in valid JSON only."""

    existing_str = ", ".join(existing_names) if existing_names else "none"

    prompt = f"""Topic: {topic}

The current stakeholder list is missing perspectives from these stances: {', '.join(missing_stances)}
Already identified stakeholders: {existing_str}

Context from knowledge graph:
{graph_context}

Identify {num_needed} additional UNIQUE real stakeholders who would hold these stances.
Do NOT repeat any stakeholders already listed above.
These should be stakeholders whose voices are underrepresented.

For stance "for" — who genuinely BENEFITS from what the proposition suggests?
For stance "against" — who genuinely LOSES from what the proposition suggests?
For stance "neutral" — who has genuinely mixed interests?

Respond in this exact JSON format:
{{
    "stakeholders": [
        {{
            "name": "specific real entity name — must not be in the existing list",
            "category": "tech_company/government/civil_society/academic/labor_union/consumer/media/investor/affected_community/international_body",
            "fundamental_interests": "what this entity fundamentally wants",
            "real_position": "their logical position derived from interests",
            "stance": "for/against/neutral",
            "stake": "why this outcome matters to them",
            "relevance_score": 0.75
        }}
    ]
}}"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        additional = parsed.get("stakeholders", [])
        for s in additional:
            s["category"] = normalize_category(s.get("category", "civil_society"))
        print(f"[StakeholderIdentifier] Calibration added {len(additional)} stakeholders for: {missing_stances}")
        return additional
    except Exception as e:
        print(f"[StakeholderIdentifier] Calibration request error: {e}")
        return []


async def request_more_unique_stakeholders(
    topic: str,
    existing: list[dict],
    num_needed: int,
    graph_context: str
) -> list[dict]:
    """
    Request more UNIQUE stakeholders when we don't have enough.
    Replaces the fill_to_count duplication approach.
    """
    existing_names = [s["name"] for s in existing]
    existing_str = ", ".join(existing_names) if existing_names else "none"

    system = """You are identifying additional stakeholders for a debate simulation.
Find real, distinct entities not already represented.
Respond in valid JSON only."""

    prompt = f"""Topic: {topic}

Already identified stakeholders: {existing_str}

We need {num_needed} more UNIQUE stakeholders for this debate.
Do NOT repeat any stakeholders already listed.
Find stakeholders from different categories and perspectives.

Context:
{graph_context}

Respond in this exact JSON format:
{{
    "stakeholders": [
        {{
            "name": "specific real entity — must not already be listed",
            "category": "tech_company/government/civil_society/academic/labor_union/consumer/media/investor/affected_community/international_body",
            "fundamental_interests": "what this entity fundamentally wants",
            "real_position": "their logical position on this topic",
            "stance": "for/against/neutral",
            "stake": "why this outcome matters to them",
            "relevance_score": 0.70
        }}
    ]
}}

Rules:
- Every stakeholder must be genuinely unique and not in the existing list
- Ensure stance diversity — spread across for/against/neutral
- Must be real organizations or people with genuine stakes"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        additional = parsed.get("stakeholders", [])
        for s in additional:
            s["category"] = normalize_category(s.get("category", "civil_society"))
        print(f"[StakeholderIdentifier] Added {len(additional)} unique stakeholders")
        return additional
    except Exception as e:
        print(f"[StakeholderIdentifier] Error requesting more stakeholders: {e}")
        return []


async def calibrate_distribution(
    stakeholders: list[dict],
    topic: str,
    G_pub: nx.DiGraph,
    graph_context: str
) -> list[dict]:
    """
    Compare current distribution to public sentiment signal.
    Correct underrepresented stances by requesting additional stakeholders.
    """
    public_dist = extract_public_sentiment_distribution(G_pub)

    if public_dist is None or public_dist["total_signal"] < 10:
        print("[StakeholderIdentifier] Insufficient public signal — skipping calibration")
        return stakeholders

    current_dist = get_current_distribution(stakeholders)

    print(f"[StakeholderIdentifier] Current distribution — "
          f"for: {current_dist['for']*100:.0f}% / "
          f"against: {current_dist['against']*100:.0f}% / "
          f"neutral: {current_dist['neutral']*100:.0f}%")
    print(f"[StakeholderIdentifier] Target distribution — "
          f"for: {public_dist['for']*100:.0f}% / "
          f"against: {public_dist['against']*100:.0f}% / "
          f"neutral: {public_dist['neutral']*100:.0f}%")

    missing_stances = []
    for stance in ["for", "against", "neutral"]:
        gap = public_dist[stance] - current_dist[stance]
        if gap > CALIBRATION_THRESHOLD:
            missing_stances.append(stance)
            print(f"[StakeholderIdentifier] '{stance}' underrepresented by "
                  f"{gap*100:.0f}% — flagging for correction")

    if not missing_stances:
        print("[StakeholderIdentifier] Distribution within acceptable range — no calibration needed")
        return stakeholders

    total = len(stakeholders)
    num_needed = max(2, round(max(
        public_dist[s] - current_dist[s]
        for s in missing_stances
    ) * total))

    existing_names = [s["name"] for s in stakeholders]
    additional = await request_missing_stakeholders(
        topic, missing_stances, num_needed, graph_context, existing_names
    )

    if additional:
        calibrated = stakeholders + additional
        new_dist = get_current_distribution(calibrated)
        print(f"[StakeholderIdentifier] Post-calibration distribution — "
              f"for: {new_dist['for']*100:.0f}% / "
              f"against: {new_dist['against']*100:.0f}% / "
              f"neutral: {new_dist['neutral']*100:.0f}%")
        return calibrated

    return stakeholders


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

Entities from knowledge graph:
{entity_list}

Knowledge graph context:
{graph_context}

For each real stakeholder, analyze their fundamental interests and derive their logical stance.

IMPORTANT: Ensure genuine diversity — not everyone can have the same stance.
Find at least one FOR, one AGAINST, and one NEUTRAL stakeholder minimum.

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
- Stance from INTEREST ANALYSIS only
- Ensure genuine diversity across for/against/neutral
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


async def fill_to_count(
    stakeholders: list[dict],
    num_agents: int,
    topic: str,
    graph_context: str
) -> list[dict]:
    """
    Fill to required count by requesting MORE UNIQUE stakeholders from LLM.
    Replaces the old approach of duplicating agents as 'representative 2/3/etc'
    which was producing 5 unique agents repeated 3x.
    """
    if len(stakeholders) >= num_agents:
        return stakeholders[:num_agents]

    needed = num_agents - len(stakeholders)
    print(f"[StakeholderIdentifier] Need {needed} more unique stakeholders — requesting from LLM")

    additional = await request_more_unique_stakeholders(
        topic, stakeholders, needed, graph_context
    )

    filled = stakeholders + additional

    # Only fall back to duplication if LLM couldn't provide enough
    if len(filled) < num_agents:
        print(f"[StakeholderIdentifier] LLM provided {len(additional)}, "
              f"still need {num_agents - len(filled)} — using representatives as last resort")
        i = 0
        unique_count = len(stakeholders)
        while len(filled) < num_agents and unique_count > 0:
            base = stakeholders[i % unique_count]
            rep_num = i // unique_count + 2
            filled.append({
                **base,
                "name": f"{base['name']} (representative {rep_num})",
                "real_position": f"A secondary perspective aligned with {base['name']}'s position."
            })
            i += 1

    return filled[:num_agents]


async def identify_stakeholders(
    topic: str,
    G: nx.DiGraph,
    num_agents: int = 10,
    G_pub: nx.DiGraph = None
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

    for s in raw_stakeholders:
        s["category"] = normalize_category(s.get("category", "civil_society"))

    diverse = enforce_diversity(raw_stakeholders, num_agents)

    # Calibrate distribution against public sentiment signal
    if G_pub is not None:
        diverse = await calibrate_distribution(diverse, topic, G_pub, graph_context)
    else:
        print("[StakeholderIdentifier] No public graph — skipping calibration")

    # Fill to count with LLM-generated unique stakeholders (not duplicates)
    filled = await fill_to_count(diverse, num_agents, topic, graph_context)

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

    final_dist = get_current_distribution(enriched)
    print(f"[StakeholderIdentifier] Identified {len(enriched)} stakeholders across "
          f"{len(set(s['category'] for s in enriched))} categories")
    print(f"[StakeholderIdentifier] Final stance distribution — "
          f"for: {final_dist['for']*100:.0f}% / "
          f"against: {final_dist['against']*100:.0f}% / "
          f"neutral: {final_dist['neutral']*100:.0f}%")
    for s in enriched:
        print(f"  -> {s['name']} [{s['category']}] stance: {s['stance']}")

    return enriched