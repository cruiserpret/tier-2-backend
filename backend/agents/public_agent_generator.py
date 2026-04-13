import json
import asyncio
import uuid
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type, query_graph
import networkx as nx

# ── Score ranges with clear separation from neutral zone ──────────
# derive_stance in debate_engine: ≤3.5 = against, ≥6.5 = for, else neutral
# Old ranges caused against agents to score 4.4/4.5 → instantly flipped neutral
# New ranges create a buffer zone so agents hold their stance across rounds
SCORE_RANGE = {
    "for":     (7.0, 8.8),  # well above 6.5 threshold — holds stance under pressure
    "against": (1.5, 3.0),  # well below 3.5 threshold — holds stance under pressure
    "neutral": (4.2, 5.8),  # clearly in middle
}

def safe_parse_json(raw: str) -> dict:
    """Robust JSON parser with multiple fallback strategies."""
    if not raw:
        return {}

    try:
        clean = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
        return json.loads(clean)
    except Exception:
        pass

    try:
        clean = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end != -1:
            return json.loads(clean[start:end + 1])
    except Exception:
        pass

    try:
        clean = re.sub(r'```json|```', '', raw).strip()
        clean = re.sub(r'[\x00-\x1f\x7f]', ' ', clean)
        return json.loads(clean)
    except Exception:
        pass

    try:
        clean = raw.replace('\u201c', '"').replace('\u201d', '"')
        clean = clean.replace('\u2018', "'").replace('\u2019', "'")
        clean = re.sub(r'[\x00-\x1f\x7f]', ' ', clean)
        return json.loads(clean)
    except Exception:
        pass

    return {}


async def extract_opinion_patterns(topic: str, G) -> list[dict]:
    """Extract recurring public opinion patterns from the public sentiment graph."""
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
Extract distinct viewpoint clusters representing how different types of real people think.
These must be everyday people with lived experience — NOT executives or policy officials.
Respond in valid JSON only."""

    prompt = f"""Topic: {topic}

Public discourse content:
Claims and opinions found:
{claims_context}

Key themes:
{entities_context}

Identify 6-8 distinct opinion patterns from this public discourse.
Each must represent a SPECIFIC real demographic with a specific viewpoint.

CRITICAL RULES:
1. Demographic must be hyper-specific — not "workers" but "32-year-old freelance graphic designer in Chicago"
2. You MUST include patterns across for, against, AND neutral — no stance should exceed 50%
3. Each demographic must be genuinely different — different age, profession, location, life situation
4. Sample argument must sound like a real Reddit comment — personal, specific, emotional

Respond in this exact JSON format:
{{
    "patterns": [
        {{
            "demographic": "hyper-specific description with age, profession, location",
            "core_belief": "their central argument in 1 personal sentence",
            "emotional_driver": "specific personal fear, anger, hope driving this",
            "emotional_intensity": "high/medium/low",
            "stance": "for/against/neutral",
            "prevalence": "high/medium/low",
            "sample_argument": "realistic Reddit-style comment this specific person would write"
        }}
    ]
}}"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = safe_parse_json(result)
        patterns = parsed.get("patterns", [])

        for_count = sum(1 for p in patterns if p.get("stance") == "for")
        against_count = sum(1 for p in patterns if p.get("stance") == "against")
        neutral_count = sum(1 for p in patterns if p.get("stance") == "neutral")
        total = len(patterns)

        print(f"[PublicAgentGenerator] Extracted {total} patterns: "
              f"{for_count} for / {against_count} against / {neutral_count} neutral")

        if total > 0 and (for_count / total > 0.55 or against_count / total > 0.55):
            missing = []
            if for_count == 0: missing.append("for")
            if against_count == 0: missing.append("against")
            if neutral_count == 0: missing.append("neutral")

            if missing:
                balance_prompt = f"""Missing stances for topic "{topic}": {', '.join(missing)}
Generate 1-2 additional patterns for each missing stance.
Same format — hyper-specific demographic, personal emotional driver, Reddit-style argument.
{{"patterns": [...]}}"""

                try:
                    balance_result = await call_llm_json(balance_prompt, system)
                    balance_parsed = safe_parse_json(balance_result)
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
    existing_personas: list[dict],  # ← NEW: track full demographics to prevent duplication
    G
) -> dict:
    """Generate a demographic agent grounded in a real public opinion pattern."""
    stance = pattern.get("stance", "neutral")
    score_range = SCORE_RANGE.get(stance, (4.2, 5.8))
    emotional_intensity = pattern.get("emotional_intensity", "medium")

    existing_names_str = ", ".join(existing_names) if existing_names else "none"

    # ── Demographic duplication prevention ───────────────────────
    # Pass full persona descriptions so LLM can't generate the same
    # "29-year-old Black software engineer in Atlanta" twice
    existing_demographics_str = "\n".join([
        f"- {p.get('age', '?')}-year-old {p.get('profession', '?')} in {p.get('location', '?')}"
        for p in existing_personas
    ]) if existing_personas else "none"

    keywords = pattern.get("core_belief", "").split()[:5]
    evidence = query_graph(G, keywords, top_n=3)
    evidence_context = "\n".join([
        f"- {e['name']}: {e['description'][:100]}"
        for e in evidence
    ])

    system = """You are generating a realistic everyday person for a debate simulation.
This is NOT a CEO, academic, or policy official.
This person's opinion comes entirely from lived personal experience.
Arguments must sound like real forum posts — specific, emotional, personal.
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

Names already used (do not repeat): {existing_names_str}

CRITICAL — Demographics already used (you MUST generate someone genuinely different):
{existing_demographics_str}

Generate someone with a different age, profession, AND location from all of the above.
Do not generate another "tech startup founder in San Francisco" or another "software engineer in Atlanta"
if those already exist above. Be creative — different city, different profession, different life stage.

Respond in this exact JSON format:
{{
    "name": "realistic full name",
    "age": 34,
    "profession": "specific job title — different from all existing personas above",
    "location": "city, state/country — different from all existing personas above",
    "persona": "2-3 sentences — their specific life situation and experience that formed their opinion",
    "initial_opinion": "their opinion in 2 sentences — first person, emotional, specific to their life",
    "key_beliefs": ["personal belief from lived experience 1", "personal belief from lived experience 2"],
    "known_entities": ["relevant topic they personally encountered"]
}}

Rules:
- Regular person only — no executives, academics, politicians
- Must be genuinely different from all existing demographics listed above
- initial_opinion must sound like a real person talking, NOT a policy statement
- Name must not be in: {existing_names_str}"""

    try:
        result = await call_llm_json(prompt, system)
        persona = safe_parse_json(result)

        if not persona or not persona.get("name"):
            print(f"[PublicAgentGenerator] Empty persona for agent {agent_index} — skipping")
            return None

        import random
        score = round(random.uniform(score_range[0], score_range[1]), 1)

        intensity_resistance = {"high": 0.45, "medium": 0.35, "low": 0.25}
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
    G,
    num_agents: int
) -> list[dict]:
    """Main function — extract opinion patterns and generate public demographic agents."""
    print(f"[PublicAgentGenerator] Generating {num_agents} public agents for: {topic}")

    patterns = await extract_opinion_patterns(topic, G)

    if not patterns:
        print("[PublicAgentGenerator] No patterns found — skipping public agents")
        return []

    prevalence_weight = {"high": 3, "medium": 2, "low": 1}
    weights = [prevalence_weight.get(p.get("prevalence", "medium"), 2) for p in patterns]
    total_weight = sum(weights)

    agent_counts = [
        max(1, round(num_agents * w / total_weight))
        for w in weights
    ]

    while sum(agent_counts) > num_agents:
        agent_counts[agent_counts.index(max(agent_counts))] -= 1
    while sum(agent_counts) < num_agents:
        agent_counts[agent_counts.index(min(agent_counts))] += 1

    existing_names = []
    existing_personas = []  # ← track full demographics
    all_agents = []

    for pattern, count in zip(patterns, agent_counts):
        for i in range(count):
            agent = await generate_public_agent(
                topic=topic,
                pattern=pattern,
                agent_index=len(all_agents),
                existing_names=existing_names.copy(),
                existing_personas=existing_personas.copy(),  # ← pass demographics
                G=G
            )
            if agent:
                existing_names.append(agent["name"])
                existing_personas.append({
                    "age": agent["age"],
                    "profession": agent["profession"],
                    "location": agent["location"]
                })
                all_agents.append(agent)
                print(f"[PublicAgentGenerator] {agent['name']} | "
                      f"{agent['profession']} in {agent['location']} | "
                      f"{agent['stance']} | {agent['score']}")

    print(f"[PublicAgentGenerator] Generated {len(all_agents)} public agents")
    return all_agents