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

SCORE_RANGE = {
    "for":     (6.5, 8.0),
    "against": (2.0, 3.5),
    "neutral": (4.2, 5.8),
}

MAX_NEUTRAL_RATIO = 0.15


def safe_parse_json(raw: str) -> dict:
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


def enforce_neutral_cap(
    patterns: list[dict],
    keyword_signal: dict = None
) -> list[dict]:
    """
    Cap neutral patterns at MAX_NEUTRAL_RATIO of total.
    
    Fix 1: Convert excess neutrals toward the direction the keyword
    signal supports — not toward whichever side currently has fewer
    patterns. This prevents neutral cap from adding AGAINST agents
    on topics where the signal clearly shows FOR dominance.
    
    Research basis:
    - Converse 1964: genuine neutrals rare on salient topics
    - Perez et al. 2022: LLMs over-generate neutrals due to RLHF bias
    - Fix: use real data signal to determine conversion direction
    """
    total = len(patterns)
    if total == 0:
        return patterns

    neutral_patterns = [p for p in patterns if p.get("stance") == "neutral"]
    for_patterns     = [p for p in patterns if p.get("stance") == "for"]
    against_patterns = [p for p in patterns if p.get("stance") == "against"]

    max_allowed_neutral = max(1, int(total * MAX_NEUTRAL_RATIO))

    if len(neutral_patterns) <= max_allowed_neutral:
        return patterns

    excess = len(neutral_patterns) - max_allowed_neutral
    print(f"[PublicAgentGenerator] Neutral cap: {len(neutral_patterns)} neutral patterns "
          f"exceeds {MAX_NEUTRAL_RATIO*100:.0f}% cap — converting {excess} to for/against")

    to_convert         = neutral_patterns[:excess]
    remaining_neutrals = neutral_patterns[excess:]

    # Fix 1 — use keyword signal to determine conversion direction
    # If signal shows FOR > AGAINST → convert toward FOR
    # If signal shows AGAINST > FOR → convert toward AGAINST
    # If no signal → fall back to balancing pattern counts
    if keyword_signal:
        signal_for     = keyword_signal.get("for", 0.33)
        signal_against = keyword_signal.get("against", 0.33)
        convert_to_for = signal_for >= signal_against
        print(f"[PublicAgentGenerator] Neutral conversion direction: "
              f"{'FOR' if convert_to_for else 'AGAINST'} "
              f"(signal: {signal_for*100:.0f}% for / {signal_against*100:.0f}% against)")
    else:
        convert_to_for = len(for_patterns) <= len(against_patterns)

    for pattern in to_convert:
        if convert_to_for:
            pattern["stance"] = "for"
            pattern["emotional_driver"] = (
                pattern.get("emotional_driver", "") +
                " — personal stake in gaining/maintaining access"
            )
            for_patterns.append(pattern)
        else:
            pattern["stance"] = "against"
            pattern["emotional_driver"] = (
                pattern.get("emotional_driver", "") +
                " — personal stake in restricting/opposing change"
            )
            against_patterns.append(pattern)

    converted = for_patterns + against_patterns + remaining_neutrals
    print(f"[PublicAgentGenerator] After cap: "
          f"{len(for_patterns)} for / {len(against_patterns)} against / {len(remaining_neutrals)} neutral")
    return converted


async def extract_opinion_patterns(
    topic: str,
    G,
    keyword_signal: dict = None,
    context: str = ""
) -> list[dict]:
    """
    Extract recurring public opinion patterns from the public sentiment graph.
    Three-tier enforcement based on keyword signal strength.
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

    # ── Tier classification ───────────────────────────────────────
    tier = 2
    min_for = 2
    min_against = 2

    if keyword_signal and keyword_signal.get("keyword_hits", 0) >= 100:
        for_pct     = keyword_signal.get("for", 0.33)
        against_pct = keyword_signal.get("against", 0.33)
        dominant    = max(for_pct, against_pct)
        minority    = min(for_pct, against_pct)

        if dominant > 0.90:
            tier = 1
            min_for = 0
            min_against = 0
            print(f"[PublicAgentGenerator] Tier 1 — moral consensus "
                  f"({dominant*100:.0f}% dominant) — no balance enforcement")

        elif dominant > 0.70 or minority < 0.15:
            tier = 3
            min_for = 1
            min_against = 1
            reason = "dominant >70%" if dominant > 0.70 else f"minority {minority*100:.0f}% < 15% threshold"
            print(f"[PublicAgentGenerator] Tier 3 — moderate consensus "
                  f"({for_pct*100:.0f}% for / {against_pct*100:.0f}% against) "
                  f"— {reason} — gentle correction")

        else:
            tier = 2
            min_for = 2
            min_against = 2
            print(f"[PublicAgentGenerator] Tier 2 — genuine split "
                  f"({for_pct*100:.0f}% for / {against_pct*100:.0f}% against) — full enforcement")
    else:
        print(f"[PublicAgentGenerator] Tier 2 — insufficient signal — defaulting to full enforcement")

    if tier == 1:
        balance_instruction = """
CONSENSUS DETECTED: This topic has near-universal agreement in real public opinion.
Do NOT generate artificial opposition — it would be factually wrong.
Generate patterns that reflect the genuine overwhelming majority position."""

    elif tier == 3:
        balance_instruction = """
MODERATE CONSENSUS: A clear majority position exists but a genuine minority also exists.
Generate at least 1 pattern representing the minority position.
Do not force equal balance — reflect the actual distribution."""

    else:
        balance_instruction = """
GENUINE SPLIT DETECTED: Real public opinion is divided on this topic.

SILENT MAJORITY RULE (Noelle-Neumann 1974):
Online discourse oversamples the vocal, emotionally activated minority.
You MUST generate at least 2 AGAINST patterns representing the silent majority.
These are people who exist in Gallup/Pew polling but not on Reddit.

Common silent majorities:
- Healthcare single-payer: people satisfied with employer insurance
- Student loans: people who paid theirs off
- Minimum wage: small business owners worried about margins
- Drug legalization: suburban parents, religious communities
- UBI: middle-class taxpayers who don't want to fund it
- 4-day work week: employers in retail, healthcare, manufacturing

Do NOT let online discourse bias you toward generating only FOR patterns."""

    context_line = f"\nAdditional context: {context}" if context else ""

    system = """You are analyzing public discourse to identify recurring opinion patterns.
Extract distinct viewpoint clusters representing how different types of real people think.
These must be everyday people with lived experience — NOT executives or policy officials.
Respond in valid JSON only."""

    prompt = f"""Topic: {topic}{context_line}

Public discourse content:
Claims and opinions found:
{claims_context}

Key themes:
{entities_context}

{balance_instruction}

Identify 6-8 distinct opinion patterns from this public discourse.
Each must represent a SPECIFIC real demographic with a specific viewpoint.

CORE RULES:
1. Demographic must be hyper-specific — not "workers" but
   "32-year-old freelance graphic designer in Chicago who lost 40% of income"
2. Each demographic must be genuinely different — different age, profession, location
3. Sample argument must sound like a real Reddit comment — personal, specific, emotional

NEUTRAL PATTERNS — STRICT DEFINITION (Converse 1964):
A neutral pattern is ONLY valid if the person has GENUINELY CONFLICTING personal interests.
Valid: A pharmacist who supports access but fears losing prescription revenue.
Invalid: Someone who "sees both sides" or wants "a balanced approach".

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

        for_count     = sum(1 for p in patterns if p.get("stance") == "for")
        against_count = sum(1 for p in patterns if p.get("stance") == "against")
        neutral_count = sum(1 for p in patterns if p.get("stance") == "neutral")
        total = len(patterns)

        print(f"[PublicAgentGenerator] Extracted {total} patterns: "
              f"{for_count} for / {against_count} against / {neutral_count} neutral")

        needs_more_for     = for_count < min_for
        needs_more_against = against_count < min_against

        if needs_more_for or needs_more_against:
            missing = []
            if needs_more_for:
                missing.append(f"for (need {min_for}, have {for_count})")
            if needs_more_against:
                missing.append(f"against (need {min_against}, have {against_count})")

            print(f"[PublicAgentGenerator] Tier {tier} minimum not met: {missing} — requesting additional patterns")

            balance_prompt = f"""Topic: "{topic}"
Current patterns: {for_count} for / {against_count} against / {neutral_count} neutral
Need additional: {', '.join(missing)}

Generate the missing patterns. Focus on demographics UNDERREPRESENTED in online discourse.

{{"patterns": [
    {{
        "demographic": "...",
        "core_belief": "...",
        "emotional_driver": "...",
        "emotional_intensity": "high/medium/low",
        "stance": "for/against",
        "prevalence": "medium",
        "sample_argument": "..."
    }}
]}}"""

            try:
                balance_result = await call_llm_json(balance_prompt, system)
                balance_parsed = safe_parse_json(balance_result)
                additional = balance_parsed.get("patterns", [])
                patterns.extend(additional)
                print(f"[PublicAgentGenerator] Added {len(additional)} patterns to meet Tier {tier} minimums")
            except Exception as e:
                print(f"[PublicAgentGenerator] Balance correction error: {e}")

        # Fix 1 — pass keyword_signal so neutral cap converts in correct direction
        patterns = enforce_neutral_cap(patterns, keyword_signal=keyword_signal)

        final_for     = sum(1 for p in patterns if p.get("stance") == "for")
        final_against = sum(1 for p in patterns if p.get("stance") == "against")
        final_neutral = sum(1 for p in patterns if p.get("stance") == "neutral")
        print(f"[PublicAgentGenerator] Final patterns: "
              f"{final_for} for / {final_against} against / {final_neutral} neutral")

        return patterns

    except Exception as e:
        print(f"[PublicAgentGenerator] Pattern extraction error: {e}")
        return []


async def generate_public_agent(
    topic: str,
    pattern: dict,
    agent_index: int,
    existing_names: list[str],
    existing_personas: list[dict],
    G,
    context: str = ""
) -> dict:
    """Generate a demographic agent grounded in a real public opinion pattern."""
    stance = pattern.get("stance", "neutral")
    score_range = SCORE_RANGE.get(stance, (4.2, 5.8))
    emotional_intensity = pattern.get("emotional_intensity", "medium")

    existing_names_str = ", ".join(existing_names) if existing_names else "none"

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

    context_line = f"\nAdditional context about this topic: {context}" if context else ""

    system = """You are generating a realistic everyday person for a debate simulation.
This is NOT a CEO, academic, or policy official.
This person's opinion comes entirely from lived personal experience.
Arguments must sound like real forum posts — specific, emotional, personal.
Respond in valid JSON only."""

    prompt = f"""Generate a realistic public persona for this debate.

Topic: {topic}{context_line}

This person represents: {pattern['demographic']}
Their core belief: {pattern['core_belief']}
Their emotional driver: {pattern['emotional_driver']}
Emotional intensity: {emotional_intensity}
Their stance: {stance}
How they actually talk about this: {pattern['sample_argument']}

Relevant context:
{evidence_context}

Names already used (do not repeat): {existing_names_str}

CRITICAL — Demographics already used (generate someone genuinely different):
{existing_demographics_str}

Respond in this exact JSON format:
{{
    "name": "realistic full name",
    "age": 34,
    "profession": "specific job title — different from all existing personas above",
    "location": "city, state/country — different from all existing personas above",
    "persona": "2-3 sentences — their specific life situation and experience",
    "initial_opinion": "their opinion in 2 sentences — first person, emotional, specific",
    "key_beliefs": ["personal belief from lived experience 1", "personal belief 2"],
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
    num_agents: int,
    keyword_signal: dict = None,
    context: str = ""
) -> list[dict]:
    """Main function — extract opinion patterns and generate public demographic agents."""
    print(f"[PublicAgentGenerator] Generating {num_agents} public agents for: {topic}")

    patterns = await extract_opinion_patterns(
        topic, G,
        keyword_signal=keyword_signal,
        context=context
    )

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
    existing_personas = []
    all_agents = []

    for pattern, count in zip(patterns, agent_counts):
        for i in range(count):
            agent = await generate_public_agent(
                topic=topic,
                pattern=pattern,
                agent_index=len(all_agents),
                existing_names=existing_names.copy(),
                existing_personas=existing_personas.copy(),
                G=G,
                context=context
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