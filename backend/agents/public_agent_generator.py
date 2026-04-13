import json
import asyncio
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type, query_graph
import networkx as nx

SCORE_RANGE = {
    "for":     (5.5, 8.5),
    "against": (1.5, 4.5),
    "neutral": (4.0, 6.0),
}

async def extract_opinion_patterns(
    topic: str,
    G: nx.DiGraph
) -> list[dict]:
    """
    Extract recurring public opinion patterns from the public sentiment graph.
    These become the seeds for demographic agent personas.
    """
    claims = get_nodes_by_type(G, "claim")
    influential = get_most_influential(G, top_n=15)

    claims_context = "\n".join([
        f"- {c['name'][:120]} [{c.get('sentiment', 'neutral')}]"
        for c in claims[:20]
    ])

    entities_context = "\n".join([
        f"- {n['name']}: {n.get('description', '')[:100]}"
        for n in influential[:10]
    ])

    system = """You are analyzing public discourse to identify recurring opinion patterns.
Extract distinct viewpoint clusters that represent how different types of people think about this topic.
Respond in valid JSON only."""

    prompt = f"""Topic: {topic}

Public discourse content:
Claims and opinions found:
{claims_context}

Key themes:
{entities_context}

Identify 4-6 distinct opinion patterns from this public discourse.
Each pattern should represent a real demographic segment with a specific viewpoint.

Respond in this exact JSON format:
{{
    "patterns": [
        {{
            "demographic": "who holds this view (e.g. 'remote software engineers', 'working parents', 'middle managers')",
            "core_belief": "their central argument in 1 sentence",
            "emotional_driver": "what emotionally drives this position (e.g. 'fear of lost flexibility', 'concern about productivity')",
            "stance": "for/against/neutral",
            "prevalence": "high/medium/low based on how common this view appears in the discourse",
            "sample_argument": "a realistic argument this person would make, grounded in the discourse above"
        }}
    ]
}}

Rules:
- Each pattern must be genuinely distinct
- Ground each pattern in the actual discourse content above
- Ensure diversity — include for, against, and neutral patterns
- Demographic must be specific, not vague (not "people" but "remote tech workers")"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        patterns = parsed.get("patterns", [])
        print(f"[PublicAgentGenerator] Extracted {len(patterns)} opinion patterns")
        return patterns
    except Exception as e:
        print(f"[PublicAgentGenerator] Pattern extraction error: {e}")
        return []

async def generate_public_agent(
    topic: str,
    pattern: dict,
    agent_index: int,
    existing_names: list[str],
    G: nx.DiGraph
) -> dict:
    """
    Generate a demographic agent grounded in a real public opinion pattern.
    """
    stance = pattern.get("stance", "neutral")
    score_range = SCORE_RANGE.get(stance, (4.0, 6.0))

    existing_names_str = ", ".join(existing_names) if existing_names else "none"

    # Query graph for relevant evidence this demographic would know about
    keywords = pattern.get("core_belief", "").split()[:5]
    evidence = query_graph(G, keywords, top_n=3)
    evidence_context = "\n".join([
        f"- {e['name']}: {e['description'][:100]}"
        for e in evidence
    ])

    system = """You are generating a realistic everyday person for a debate simulation.
This is NOT a CEO or executive — this is a regular person with lived experience.
Their opinions come from personal experience, not corporate strategy.
Respond in valid JSON only."""

    prompt = f"""Generate a realistic public persona for this debate.

Topic: {topic}

This person represents: {pattern['demographic']}
Their core belief: {pattern['core_belief']}
Their emotional driver: {pattern['emotional_driver']}
Their stance: {stance}
Sample argument they would make: {pattern['sample_argument']}

Relevant public discourse context:
{evidence_context}

Names already used: {existing_names_str}

Generate a specific, realistic person from this demographic.

Respond in this exact JSON format:
{{
    "name": "realistic full name",
    "age": 34,
    "profession": "specific job title",
    "location": "city, state/country",
    "persona": "2-3 sentences about their background and why they hold this position based on lived experience",
    "initial_opinion": "their specific opinion in 2 sentences — personal, emotional, grounded in lived experience NOT corporate speak",
    "key_beliefs": ["personal belief 1", "personal belief 2"],
    "known_entities": ["relevant topic or entity they know about"]
}}

Rules:
- This is a regular person, not a public figure or executive
- Their opinion comes from personal experience, not policy analysis
- Name must be unique — not in: {existing_names_str}
- Make them feel real and specific, not generic"""

    try:
        result = await call_llm_json(prompt, system)
        import re
        result_clean = re.sub(r'[\x00-\x1f\x7f]', ' ', result)
        persona = json.loads(result_clean)

        import random
        score = round(random.uniform(score_range[0], score_range[1]), 1)

        return {
            "id": f"agent_{uuid.uuid4().hex[:8]}",
            "name": persona.get("name", f"Public Agent {agent_index}"),
            "age": persona.get("age", 32),
            "profession": persona.get("profession", ""),
            "location": persona.get("location", ""),
            "persona": persona.get("persona", ""),
            "stakeholder_name": pattern["demographic"],
            "stakeholder_category": "affected_community",
            "agent_type": "public",
            "stance": stance,
            "opinion": persona.get("initial_opinion", ""),
            "score": score,
            "opinion_delta": 0.0,
            "key_beliefs": persona.get("key_beliefs", []),
            "persuasion_resistance": 0.35,
            "influence_weight": 0.40,
            "known_entities": persona.get("known_entities", []),
            "confirmation_bias": 0.45,
            "attacks_received": 0,
            "memory": []
        }

    except Exception as e:
        print(f"[PublicAgentGenerator] Error generating public agent {agent_index}: {e}")
        return None

async def generate_public_agents(
    topic: str,
    G: nx.DiGraph,
    num_agents: int
) -> list[dict]:
    """
    Main function — extract opinion patterns and generate public demographic agents.
    """
    print(f"[PublicAgentGenerator] Generating {num_agents} public agents for: {topic}")

    # Step 1 — Extract opinion patterns from public graph
    patterns = await extract_opinion_patterns(topic, G)

    if not patterns:
        print("[PublicAgentGenerator] No patterns found — skipping public agents")
        return []

    # Step 2 — Generate agents from patterns
    # Distribute agents across patterns weighted by prevalence
    prevalence_weight = {"high": 3, "medium": 2, "low": 1}
    weights = [prevalence_weight.get(p.get("prevalence", "medium"), 2) for p in patterns]
    total_weight = sum(weights)

    agent_counts = [
        max(1, round(num_agents * w / total_weight))
        for w in weights
    ]

    # Adjust to hit exact num_agents
    while sum(agent_counts) > num_agents:
        agent_counts[agent_counts.index(max(agent_counts))] -= 1
    while sum(agent_counts) < num_agents:
        agent_counts[agent_counts.index(min(agent_counts))] += 1

    # Step 3 — Generate agents
    existing_names = []
    all_agents = []

    for pattern, count in zip(patterns, agent_counts):
        for i in range(count):
            agent = await generate_public_agent(
                topic, pattern,
                len(all_agents),
                existing_names.copy(),
                G
            )
            if agent:
                existing_names.append(agent["name"])
                all_agents.append(agent)
                print(f"[PublicAgentGenerator] {agent['name']} | {agent['stakeholder_name']} | {agent['stance']} | {agent['score']}")

    print(f"[PublicAgentGenerator] Generated {len(all_agents)} public agents")
    return all_agents