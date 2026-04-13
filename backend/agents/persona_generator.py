import json
import asyncio
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type
import networkx as nx

# Force stance diversity across agents
STANCE_MAP = {0: "strongly against", 1: "strongly for", 2: "neutral"}
SCORE_RANGE = {
    "strongly against": (1.0, 2.5),
    "against":          (1.5, 3.5),   # was (2.0, 5.5)
    "neutral":          (4.0, 6.0),
    "for":              (6.5, 8.5),   # was (5.5, 8.0)
    "strongly for":     (7.5, 9.5),
}

async def generate_single_persona(
    topic: str,
    G: nx.DiGraph,
    agent_index: int,
    existing_names: list[str],
    stakeholder: dict = None
) -> dict:
    """Generate a single agent persona grounded in a real stakeholder."""

    influential_nodes = get_most_influential(G, top_n=15)
    claims = get_nodes_by_type(G, "claim")

    graph_context = "\n".join([
        f"- {n['name']} (cited {n['citations']}x, influence: {n['influence_score']:.3f}): {n.get('description', '')[:100]}"
        for n in influential_nodes
    ])

    claims_context = "\n".join([
        f"- {c['name'][:100]} [{c.get('sentiment', 'neutral')}]"
        for c in claims[:10]
    ])

    existing_names_str = ", ".join(existing_names) if existing_names else "none"

    # Use stakeholder if available, otherwise fall back to forced stance
    if stakeholder:
        stance_tendency = stakeholder.get("stance", "neutral")
        persuasion_resistance = stakeholder.get("persuasion_resistance", 0.5)
        stakeholder_context = f"""You are generating an agent representing: {stakeholder['name']}
Category: {stakeholder['category']}
Their real-world position: {stakeholder['real_position']}
Why they care: {stakeholder['stake']}
Their known stance on this topic: {stance_tendency}
Persuasion resistance: {persuasion_resistance} (0=easily convinced, 1=never convinced)"""
        stakeholder_name = stakeholder["name"]
        stakeholder_category = stakeholder["category"]
    else:
        forced = STANCE_MAP[agent_index % 3]
        stance_tendency = forced
        persuasion_resistance = 0.5
        stakeholder_context = f"Generate a unique persona with stance: {forced}"
        stakeholder_name = None
        stakeholder_category = "individual"

    score_range = SCORE_RANGE.get(stance_tendency, (4.0, 6.0))
    score_min, score_max = score_range

    system = """You are designing a realistic human persona for a debate simulation.
The persona must represent the given stakeholder grounded in real knowledge graph data.
Always respond in valid JSON."""

    prompt = f"""Create a realistic debate persona for agent {agent_index + 1}.

Topic: {topic}

Stakeholder context:
{stakeholder_context}

Real-world knowledge graph:
{graph_context}

Key claims:
{claims_context}

Names already used: {existing_names_str}

Respond in this exact JSON format:
{{
    "name": "full name of a specific person representing this stakeholder",
    "age": 45,
    "profession": "specific job title reflecting the stakeholder",
    "location": "city, country",
    "persona": "2-3 sentences about their background and why they hold their position",
    "initial_opinion": "their specific opinion citing real facts from the knowledge graph",
    "key_beliefs": ["belief grounded in graph", "belief grounded in graph"],
    "known_entities": ["entity1", "entity2", "entity3"]
}}

Rules:
- If the stakeholder is a well-known organization, use their ACTUAL real-world representative
  (e.g. Meta → Mark Zuckerberg, European Union → Ursula von der Leyen, OpenAI → Sam Altman)
- Only invent a name if the stakeholder has no single known representative
- No "Dr." prefix unless the real person actually uses it
- Name must be unique — not in: {existing_names_str}
- The following names are already taken, use someone else: {existing_names_str}
- This is a hard rule — generating a duplicate name will break the simulation"""

    try:
        import re
        result = await call_llm_json(prompt, system)
        result_clean = re.sub(r'[\x00-\x1f\x7f]', ' ', result)
        persona = json.loads(result_clean)

        import random
        score = round(random.uniform(score_min, score_max), 1)

        # Clean up stance string
        stance = stance_tendency
        if stance in ["strongly against", "against"]:
            stance = "against"
        elif stance in ["strongly for", "for"]:
            stance = "for"
        else:
            stance = "neutral"

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
    G: nx.DiGraph,
    num_agents: int = 20,
    stakeholders: list[dict] = None
) -> list[dict]:
    print(f"[PersonaGenerator] Generating {num_agents} personas for topic: {topic}")

    existing_names = []
    valid_personas = []

    # Run sequentially to guarantee unique names and correct stance cycling
    for i in range(num_agents):
        await asyncio.sleep(0.5)
        stakeholder = stakeholders[i] if stakeholders and i < len(stakeholders) else None
        persona = await generate_single_persona(topic, G, i, existing_names.copy(), stakeholder)
        if persona:
            existing_names.append(persona["name"])
            valid_personas.append(persona)
            print(f"[PersonaGenerator] Agent {i+1}: {persona['name']} | {persona.get('stakeholder_name', 'individual')} | stance: {persona['stance']} | score: {persona['score']}")

    print(f"[PersonaGenerator] Successfully generated {len(valid_personas)} personas")
    return valid_personas