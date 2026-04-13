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
Extract distinct viewpoint clusters that represent how different types of real people think about this topic.
These must be everyday people with lived experience — NOT executives, academics, or policy officials.
Respond in valid JSON only."""

    prompt = f"""Topic: {topic}

Public discourse content:
Claims and opinions found:
{claims_context}

Key themes:
{entities_context}

Identify 6-8 distinct opinion patterns from this public discourse.
Each pattern must represent a SPECIFIC real demographic segment with a specific viewpoint.

CRITICAL RULES:
1. Demographic must be hyper-specific — not "workers" but "32-year-old freelance graphic designer in Chicago who lost 40% of income"
2. You MUST include patterns across for, against, AND neutral stances — no stance should have more than 50% of patterns
3. Emotional driver must be personal and visceral — fear, anger, hope, relief — not abstract policy concerns
4. Sample argument must sound like a real Reddit/Quora comment — personal, specific, emotional — NOT a policy brief
5. Ground every pattern in the actual discourse content above

Respond in this exact JSON format:
{{
    "patterns": [
        {{
            "demographic": "hyper-specific description of who holds this view with age, profession, location context",
            "core_belief": "their central argument in 1 personal sentence — first person voice",
            "emotional_driver": "the specific personal fear, anger, hope or relief driving this position",
            "emotional_intensity": "high/medium/low — how emotionally charged their arguments are",
            "stance": "for/against/neutral",
            "prevalence": "high/medium/low based on how common this view appears in the discourse",
            "sample_argument": "a realistic Reddit-style comment this specific person would write — personal, emotional, specific to their situation"
        }}
    ]
}}"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        patterns = parsed.get("patterns", [])

        # ── Ideological balance enforcement ──────────────────────────
        # Check distribution and flag if imbalanced
        for_count = sum(1 for p in patterns if p.get("stance") == "for")
        against_count = sum(1 for p in patterns if p.get("stance") == "against")
        neutral_count = sum(1 for p in patterns if p.get("stance") == "neutral")
        total = len(patterns)

        print(f"[PublicAgentGenerator] Extracted {total} patterns: {for_count} for / {against_count} against / {neutral_count} neutral")

        # If any stance has more than 50% of patterns, request rebalancing
        if total > 0 and (for_count / total > 0.55 or against_count / total > 0.55):
            print(f"[PublicAgentGenerator] Imbalance detected — requesting missing perspectives")
            missing = []
            if for_count == 0: missing.append("for")
            if against_count == 0: missing.append("against")
            if neutral_count == 0: missing.append("neutral")

            if missing:
                balance_prompt = f"""The following stances are missing from the opinion patterns for topic: "{topic}"
Missing stances: {', '.join(missing)}

Generate 1-2 additional opinion patterns for each missing stance.
These must be grounded in real public discourse about this topic.
Same format as before — hyper-specific demographic, personal emotional driver, Reddit-style sample argument.

Respond with just the additional patterns in this format:
{{"patterns": [...]}}"""

                try:
                    balance_result = await call_llm_json(balance_prompt, system)
                    balance_parsed = json.loads(balance_result)
                    additional = balance_parsed.get("patterns", [])
                    patterns.extend(additional)
                    print(f"[PublicAgentGenerator] Added {len(additional)} balancing patterns")
                except Exception as e:
                    print(f"[PublicAgentGenerator] Balance correction error: {e}")

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
    emotional_intensity = pattern.get("emotional_intensity", "medium")

    existing_names_str = ", ".join(existing_names) if existing_names else "none"

    # Query graph for relevant evidence this demographic would know about
    keywords = pattern.get("core_belief", "").split()[:5]
    evidence = query_graph(G, keywords, top_n=3)
    evidence_context = "\n".join([
        f"- {e['name']}: {e['description'][:100]}"
        for e in evidence
    ])

    system = """You are generating a realistic everyday person for a debate simulation.
This is NOT a CEO, academic, or policy official.
This is a regular person whose opinion comes entirely from lived personal experience.
Their arguments must sound like real forum posts — specific, emotional, personal.
Respond in valid JSON only."""

    prompt = f"""Generate a realistic public persona for this debate.

Topic: {topic}

This person represents: {pattern['demographic']}
Their core belief: {pattern['core_belief']}
Their emotional driver: {pattern['emotional_driver']}
Emotional intensity: {emotional_intensity}
Their stance: {stance}
How they actually talk about this: {pattern['sample_argument']}

Relevant context:
{evidence_context}

Names already used: {existing_names_str}

Generate a SPECIFIC, REAL-FEELING person from this demographic.

Respond in this exact JSON format:
{{
    "name": "realistic full name",
    "age": 34,
    "profession": "specific job title — not executive or academic",
    "location": "city, state/country",
    "persona": "2-3 sentences — their specific life situation and the personal experience that formed their opinion. Name a specific event or moment that shaped their view.",
    "initial_opinion": "their opinion in 2 sentences — first person, emotional, specific to their life situation. Must sound like a real person talking, NOT a policy statement.",
    "key_beliefs": ["personal belief from lived experience 1", "personal belief from lived experience 2"],
    "known_entities": ["relevant topic or entity they personally encountered"]
}}

Rules:
- Regular person only — no executives, academics, politicians, or policy officials
- Opinion must come from personal experience, not analysis
- initial_opinion must be emotional and specific — include a personal detail
- Name must not be in: {existing_names_str}
- Location must be realistic and specific"""

    try:
        result = await call_llm_json(prompt, system)
        import re
        result_clean = re.sub(r'[\x00-\x1f\x7f]', ' ', result)
        persona = json.loads(result_clean)

        import random
        score = round(random.uniform(score_range[0], score_range[1]), 1)

        # Emotional intensity affects persuasion resistance
        # High emotion = more resistant to logic, lower resistance to emotional arguments
        intensity_resistance = {
            "high": 0.45,
            "medium": 0.35,
            "low": 0.25,
        }
        persuasion_resistance = intensity_resistance.get(emotional_intensity, 0.35)

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
            "graph_type": "public",
            "stance": stance,
            "opinion": persona.get("initial_opinion", ""),
            "score": score,
            "opinion_delta": 0.0,
            "key_beliefs": persona.get("key_beliefs", []),
            "persuasion_resistance": persuasion_resistance,
            "influence_weight": 0.40,
            "known_entities": persona.get("known_entities", []),
            "confirmation_bias": 0.45,
            "emotional_intensity": emotional_intensity,
            "attacks_received": 0,
            "last_argument": "",
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