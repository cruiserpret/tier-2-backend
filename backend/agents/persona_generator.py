import json
import asyncio
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type
import networkx as nx

STANCE_MAP = {0: "strongly against", 1: "strongly for", 2: "neutral"}

SCORE_RANGE_BASE = {
    "for":     {"min": 7.0, "max": 8.5},
    "against": {"min": 1.5, "max": 3.0},
    "neutral": {"min": 4.2, "max": 5.8},
}


def get_score_range(stance: str, relevance_score: float = 0.75) -> tuple[float, float]:
    base = SCORE_RANGE_BASE.get(stance, SCORE_RANGE_BASE["neutral"])
    if stance == "for":
        center = base["min"] + (relevance_score * (base["max"] - base["min"]))
        return (max(base["min"], center - 0.4), min(base["max"], center + 0.4))
    elif stance == "against":
        center = base["max"] - (relevance_score * (base["max"] - base["min"]))
        return (max(base["min"], center - 0.4), min(base["max"], center + 0.4))
    else:
        return (base["min"], base["max"])


def is_product_entity(stakeholder: dict) -> bool:
    """
    Detect if a stakeholder represents an inanimate product rather than
    a human actor. Products cannot hold opinions or have interests.

    Bug 1 fix: Added human institution indicators that override product detection.
    Previously "UCSD Information Technology Services" was flagged because it
    contains "service" — but it's a human institution not a product.
    "Community College System Administrators" was flagged because of "system".

    Logic: only flag as product if product signal present AND no human
    institution signal present. Human institution indicators take priority.

    Prediction: eliminates false positives on government/academic bodies
    while still catching genuine products like "Mobile Order app".
    """
    name = stakeholder.get("name", "").lower()

    product_indicators = [
        " app", "mobile order", " platform ", " software",
        " tool ", " website", " bot ", " api ", " algorithm"
    ]

    # Human institution indicators — override product detection
    # These indicate a real organization with people, not a product
    human_indicators = [
        "administrator", "director", "department", "office",
        "council", "committee", "board", "union", "association",
        "staff", "faculty", "services", "division", "bureau",
        "agency", "institute", "center", "programme", "program",
        "authority", "ministry", "commission", "coalition",
        "foundation", "organization", "organisation", "network",
        "alliance", "partnership", "task force", "working group"
    ]

    has_product_signal = any(ind in f" {name} " for ind in product_indicators)
    has_human_signal   = any(ind in name for ind in human_indicators)

    # Only flag as product if product signal AND no human institution signal
    return has_product_signal and not has_human_signal


async def generate_single_persona(
    topic: str,
    G: nx.DiGraph,
    agent_index: int,
    existing_names: list[str],
    stakeholder: dict = None,
    context: str = ""
) -> dict:
    """Generate a single agent persona grounded in a real stakeholder."""
    influential_nodes = get_most_influential(G, top_n=15)
    claims = get_nodes_by_type(G, "claim")

    graph_context = "\n".join([
        f"- {n['name']} (cited {n['citations']}x): {n.get('description', '')[:100]}"
        for n in influential_nodes
    ])

    claims_context = "\n".join([
        f"- {c['name'][:100]} [{c.get('sentiment', 'neutral')}]"
        for c in claims[:10]
    ])

    existing_names_str = ", ".join(existing_names) if existing_names else "none"
    context_line = f"\nAdditional context: {context}" if context else ""

    if stakeholder:
        if is_product_entity(stakeholder):
            original_name = stakeholder["name"]
            stakeholder_context = f"""The entity "{original_name}" is a product or platform.
Do NOT generate a persona for the product itself.
Instead, generate a persona for either:
- The COMPANY or ORGANIZATION that created/operates it (if the company has clear interests)
- The PRIMARY USER GROUP of this product (if users are the real stakeholders here)
Choose whichever makes more sense for this specific topic.
Their stance is: {stakeholder.get('stance', 'neutral')}
Their interests: {stakeholder.get('stake', 'Direct stake in this outcome')}"""
            stakeholder_name = f"Users/Operators of {original_name}"
            stakeholder_category = stakeholder.get("category", "affected_community")
            print(f"[PersonaGenerator] Entity discrimination: '{original_name}' → "
                  f"generating user/company representative")
        else:
            stance_tendency = stakeholder.get("stance", "neutral")
            persuasion_resistance = stakeholder.get("persuasion_resistance", 0.5)
            stakeholder_context = f"""You are generating an agent representing: {stakeholder['name']}
Category: {stakeholder['category']}
Their real-world position: {stakeholder['real_position']}
Why they care: {stakeholder['stake']}
Their stance: {stance_tendency}
Persuasion resistance: {persuasion_resistance} (0=easily convinced, 1=never)"""
            stakeholder_name = stakeholder["name"]
            stakeholder_category = stakeholder["category"]

        stance_tendency = stakeholder.get("stance", "neutral")
        relevance_score = stakeholder.get("relevance_score", 0.75)
        persuasion_resistance = stakeholder.get("persuasion_resistance", 0.5)
    else:
        forced = STANCE_MAP[agent_index % 3]
        stance_tendency = forced
        relevance_score = 0.70
        persuasion_resistance = 0.5
        stakeholder_context = f"Generate a unique persona with stance: {forced}"
        stakeholder_name = None
        stakeholder_category = "individual"

    if stance_tendency in ["strongly against", "against"]:
        stance = "against"
    elif stance_tendency in ["strongly for", "for"]:
        stance = "for"
    else:
        stance = "neutral"

    score_min, score_max = get_score_range(stance, relevance_score)

    system = """You are designing a realistic human persona for a debate simulation.
The persona must represent the given stakeholder grounded in real knowledge graph data.
Always respond in valid JSON."""

    prompt = f"""Create a realistic debate persona for agent {agent_index + 1}.

Topic: {topic}{context_line}

Stakeholder context:
{stakeholder_context}

Real-world knowledge graph (use these facts to ground the persona):
{graph_context}

Key claims circulating about this topic:
{claims_context}

Names already used (do not repeat): {existing_names_str}

Respond in this exact JSON format:
{{
    "name": "full name of a specific person representing this stakeholder",
    "age": 45,
    "profession": "specific job title reflecting the stakeholder",
    "location": "city, country",
    "persona": "2-3 sentences about their background and why they hold their position",
    "initial_opinion": "their specific opinion citing real facts from the knowledge graph above",
    "key_beliefs": ["specific belief grounded in graph data", "specific belief grounded in graph data"],
    "known_entities": ["entity1 from the graph", "entity2 from the graph"]
}}

Rules:
- If the stakeholder is a well-known organization, use their ACTUAL real-world representative
  (e.g. ByteDance → Shou Zi Chew, European Commission → Ursula von der Leyen)
- Only invent a name if the stakeholder has no single known representative
- initial_opinion MUST reference specific facts or entities from the knowledge graph above
- key_beliefs must be grounded in real graph data, not generic positions
- Name must be unique — not in: {existing_names_str}
- No "Dr." prefix unless the real person actually uses it"""

    try:
        import re
        result = await call_llm_json(prompt, system)
        result_clean = re.sub(r'[\x00-\x1f\x7f]', ' ', result)
        persona = json.loads(result_clean)

        import random
        score = round(random.uniform(score_min, score_max), 1)

        return {
            "id": f"agent_{uuid.uuid4().hex[:8]}",
            "name": persona.get("name", f"Agent {agent_index + 1}"),
            "age": persona.get("age", 40),
            "profession": persona.get("profession", ""),
            "location": persona.get("location", ""),
            "persona": persona.get("persona", ""),
            "stakeholder_name": stakeholder_name,
            "stakeholder_category": stakeholder_category,
            "stance": stance,
            "opinion": persona.get("initial_opinion", ""),
            "score": score,
            "opinion_delta": 0.0,
            "key_beliefs": persona.get("key_beliefs", []),
            "persuasion_resistance": persuasion_resistance,
            "known_entities": persona.get("known_entities", []),
            "memory": []
        }
    except Exception as e:
        import traceback
        print(f"[PersonaGenerator] Error generating persona {agent_index}: {e}")
        traceback.print_exc()
        return None


async def generate_personas(
    topic: str,
    G,
    num_agents: int,
    stakeholders: list,
    context: str = ""
) -> list[dict]:
    print(f"[PersonaGenerator] Generating {num_agents} personas for topic: {topic}")

    existing_names = []
    valid_personas = []

    for i in range(num_agents):
        await asyncio.sleep(0.5)
        stakeholder = stakeholders[i] if stakeholders and i < len(stakeholders) else None
        persona = await generate_single_persona(
            topic, G, i, existing_names.copy(), stakeholder, context=context
        )
        if persona:
            existing_names.append(persona["name"])
            valid_personas.append(persona)
            print(f"[PersonaGenerator] Agent {i+1}: {persona['name']} | "
                  f"{persona.get('stakeholder_name', 'individual')} | "
                  f"stance: {persona['stance']} | score: {persona['score']}")

    print(f"[PersonaGenerator] Successfully generated {len(valid_personas)} personas")
    return valid_personas