import json
import asyncio
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import query_graph
import networkx as nx

# ── Deffuant Model Parameters ─────────────────────────────────────
BASE_MU = 0.3
CONFIDENCE_THRESHOLD = 2.0          # Fix E: lowered from 3.0
MAX_EVIDENCE_MULTIPLIER = 1.5

# ── Emotional Contagion Parameters ────────────────────────────────
# Kelman 1958 — identification-based influence operates independently
# of rational persuasion. Public agents influence through personal
# testimony. Institutional agents cannot counter this with logic alone.
EMOTIONAL_CONTAGION_REACH = 4.0
EMOTIONAL_CONTAGION_MU = 0.08
EMOTIONAL_INTENSITY_MULTIPLIER = {
    "high":   1.5,
    "medium": 1.0,
    "low":    0.5,
}

# ── Confirmation Bias Parameters ──────────────────────────────────
CONFIRMATION_BIAS_BY_CATEGORY = {
    "government":         0.65,
    "international_body": 0.60,
    "tech_company":       0.55,
    "labor_union":        0.50,
    "investor":           0.40,
    "civil_society":      0.45,
    "media":              0.35,
    "affected_community": 0.40,
    "academic":           0.20,
    "consumer":           0.15,
}

# ── Backfire Effect Parameters ────────────────────────────────────
BACKFIRE_THRESHOLD = 2
BACKFIRE_RESISTANCE_BOOST = 0.05
MAX_RESISTANCE = 0.95


def safe_parse_json(raw: str) -> dict:
    """
    Robust JSON parser with 4 fallback strategies.
    Grounded in Ouyang et al. 2022 — systematic LLM output failure modes.
    """
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


def get_confirmation_bias(agent: dict) -> float:
    category = agent.get("stakeholder_category", "civil_society")
    return CONFIRMATION_BIAS_BY_CATEGORY.get(category, 0.35)


def apply_confirmation_bias(
    evidence_multiplier: float,
    agent: dict,
    weighted_avg: float
) -> float:
    """
    Discount evidence multiplier when opponents push against agent's position.
    Grounded in Chuang et al. NAACL 2024.
    """
    agent_score = agent.get("score", 5.0)
    bias = get_confirmation_bias(agent)
    contradicting = (agent_score > 5.0 and weighted_avg < agent_score) or \
                    (agent_score < 5.0 and weighted_avg > agent_score)
    if contradicting:
        discounted = evidence_multiplier * (1.0 - bias)
        return round(max(discounted, 1.0), 3)
    return evidence_multiplier


def apply_backfire_effect(agent: dict) -> float:
    """
    Repeated attacks without shifting increases resistance.
    Grounded in Nyhan & Reifler 2010.
    """
    attacks_received = agent.get("attacks_received", 0)
    if attacks_received <= BACKFIRE_THRESHOLD:
        return agent.get("persuasion_resistance", 0.5)
    extra_attacks = attacks_received - BACKFIRE_THRESHOLD
    boost = extra_attacks * BACKFIRE_RESISTANCE_BOOST
    new_resistance = agent.get("persuasion_resistance", 0.5) + boost
    return round(min(new_resistance, MAX_RESISTANCE), 3)


def apply_emotional_contagion(
    agent: dict,
    public_opponents: list[dict],
    old_score: float
) -> tuple[float, float, str | None]:
    """
    Fix D — Asymmetric emotional contagion.
    Kelman 1958 — identification-based influence.
    Public agents CAN shift institutional agents emotionally.
    Institutional agents CANNOT shift public agents through logic.
    """
    agent_type = agent.get("agent_type", "institutional")
    if agent_type == "public":
        return old_score, 0.0, None

    if not public_opponents:
        return old_score, 0.0, None

    reachable = [
        opp for opp in public_opponents
        if abs(old_score - opp["score"]) < EMOTIONAL_CONTAGION_REACH
    ]

    if not reachable:
        return old_score, 0.0, None

    best_contagion = None
    best_pull = 0.0

    for opp in reachable:
        intensity = opp.get("emotional_intensity", "medium")
        intensity_mult = EMOTIONAL_INTENSITY_MULTIPLIER.get(intensity, 1.0)
        gap = abs(old_score - opp["score"])
        pull = intensity_mult / (gap + 0.1)
        if pull > best_pull:
            best_pull = pull
            best_contagion = opp

    if not best_contagion:
        return old_score, 0.0, None

    emotional_delta = EMOTIONAL_CONTAGION_MU * (best_contagion["score"] - old_score)
    new_score = round(max(1.0, min(10.0, old_score + emotional_delta)), 2)
    actual_delta = abs(new_score - old_score)

    if actual_delta > 0.01:
        print(f"[DebateEngine] Emotional contagion: {agent['name']} nudged by "
              f"{best_contagion['name']} ({best_contagion.get('emotional_intensity', 'medium')} intensity) "
              f"delta={actual_delta:.3f}")

    return new_score, actual_delta, best_contagion["name"]


def calculate_evidence_multiplier(evidence: list[dict]) -> float:
    if not evidence:
        return 1.0
    avg_citations = sum(e.get("citations", 1) for e in evidence) / len(evidence)
    multiplier = 1.0 + (min(avg_citations, 10) / 10) * (MAX_EVIDENCE_MULTIPLIER - 1.0)
    return round(min(multiplier, MAX_EVIDENCE_MULTIPLIER), 3)


def deffuant_update(
    opinion_i: float,
    opinion_j: float,
    resistance_i: float,
    evidence_multiplier: float
) -> tuple[float, float]:
    gap = abs(opinion_i - opinion_j)
    if gap >= CONFIDENCE_THRESHOLD:
        return opinion_i, 0.0
    effective_mu = BASE_MU * (1 - resistance_i) * evidence_multiplier
    delta = effective_mu * (opinion_j - opinion_i)
    new_opinion = round(max(1.0, min(10.0, opinion_i + delta)), 2)
    actual_delta = abs(new_opinion - opinion_i)
    return new_opinion, actual_delta


def derive_stance(score: float) -> str:
    if score <= 3.5:
        return "against"
    elif score >= 6.5:
        return "for"
    else:
        return "neutral"


async def run_single_agent_round(
    agent: dict,
    all_agents: list[dict],
    topic: str,
    G_inst: nx.DiGraph,
    G_pub: nx.DiGraph,
    round_num: int
) -> dict:
    G = G_pub if agent.get("graph_type") == "public" else G_inst
    keywords = agent.get("key_beliefs", []) + agent.get("known_entities", [])
    evidence = query_graph(G, keywords, top_n=5)

    evidence_context = "\n".join([
        f"- {e['name']}: {e['description'][:150]} [source: {e['source'][:60]}] (cited {e['citations']}x)"
        for e in evidence
    ])

    opponents = [a for a in all_agents if a["id"] != agent["id"]]
    opponent_context = "\n".join([
        f"- {o['name']} ({o['stance']}, score {o['score']}): {o['opinion']}"
        + (f"\n  → Last argument: \"{o['last_argument']}\"" if o.get('last_argument') else "")
        for o in opponents[:6]
    ])

    within_threshold = [
        opp for opp in opponents
        if abs(agent["score"] - opp["score"]) < CONFIDENCE_THRESHOLD
    ]

    public_opponents = [
        opp for opp in opponents
        if opp.get("agent_type") == "public"
    ]

    target_opponent = None
    if within_threshold:
        target_opponent = max(within_threshold, key=lambda o: o.get("influence_weight", 0.5))
    elif opponents:
        target_opponent = max(opponents, key=lambda o: abs(agent["score"] - o["score"]))

    target_argument = ""
    if target_opponent and target_opponent.get("last_argument"):
        target_argument = (
            f"\n\nYou MUST directly respond to this specific argument from {target_opponent['name']}:\n"
            f"\"{target_opponent['last_argument']}\""
        )

    base_evidence_multiplier = calculate_evidence_multiplier(evidence)
    old_score = agent["score"]

    if within_threshold:
        weights = [
            (1.0 / (abs(agent["score"] - opp["score"]) + 0.1)) * opp.get("influence_weight", 0.5)
            for opp in within_threshold
        ]
        total_weight = sum(weights)
        weighted_avg = sum(
            opp["score"] * w for opp, w in zip(within_threshold, weights)
        ) / total_weight

        biased_multiplier = apply_confirmation_bias(
            base_evidence_multiplier, agent, weighted_avg
        )
        effective_resistance = apply_backfire_effect(agent)

        new_score, delta = deffuant_update(
            opinion_i=old_score,
            opinion_j=weighted_avg,
            resistance_i=effective_resistance,
            evidence_multiplier=biased_multiplier
        )

        influential_opponent = max(
            within_threshold,
            key=lambda o: o.get("influence_weight", 0.5)
        )
        min_gap = abs(agent["score"] - influential_opponent["score"])

        being_attacked = (old_score > 5.0 and weighted_avg < old_score) or \
                         (old_score < 5.0 and weighted_avg > old_score)

    else:
        new_score = old_score
        delta = 0.0
        influential_opponent = None
        min_gap = float('inf')
        biased_multiplier = base_evidence_multiplier
        effective_resistance = agent.get("persuasion_resistance", 0.5)
        being_attacked = False

    emotional_score, emotional_delta, contagion_source = apply_emotional_contagion(
        agent, public_opponents, new_score
    )

    if emotional_delta > 0.01:
        new_score = emotional_score
        delta = max(delta, emotional_delta)

    new_stance = derive_stance(new_score)
    stance_changed = new_stance != agent.get("stance", "neutral")
    shifted = stance_changed or (delta > 0.10)

    attacks_received = agent.get("attacks_received", 0)
    if being_attacked and not shifted:
        attacks_received += 1
    elif shifted:
        attacks_received = 0

    system = """You are simulating a realistic human debater grounded in real evidence.
Your opinion shift has already been mathematically calculated.
Your job is to generate realistic argument text that reflects this outcome.
You must directly engage with what specific opponents said — not generic statements.
Respond in valid JSON only."""

    opponent_name = influential_opponent['name'] if influential_opponent else 'the group'
    shift_direction = (
        f"You moved toward {opponent_name}'s position"
        if influential_opponent and shifted
        else "You held your position firm"
    )

    emotional_note = ""
    if contagion_source:
        emotional_note = (
            f"\nIMPORTANT: You were emotionally affected by {contagion_source}'s "
            f"personal testimony. Acknowledge this in your argument — not as a full "
            f"position change, but as a moment of genuine human recognition."
        )

    prompt = f"""You are {agent['name']}, representing {agent.get('stakeholder_name', 'yourself')}.
Background: {agent['persona']}
Persuasion resistance: {effective_resistance} (0=easily convinced, 1=never)
Confirmation bias: {get_confirmation_bias(agent)} (how much you discount opposing evidence)

Topic: {topic}
Round: {round_num}

Your current position:
- Opinion: {agent['opinion']}
- Score: {old_score}/10
- Stance: {agent['stance']}

Evidence available from real sources:
{evidence_context}

What other debaters said:
{opponent_context}
{target_argument}

Mathematical outcome (already decided):
- Your new score: {new_score}/10
- Opinion shifted: {shifted}
- {shift_direction}
- You {"are being repeatedly challenged and digging in harder" if attacks_received > BACKFIRE_THRESHOLD else "are engaging with the debate openly"}
{emotional_note}

Generate argument text that reflects this outcome naturally.
If you have a target argument above, your response MUST explicitly reference it by name.

Respond in this exact JSON format:
{{
    "argument": "your argument this round citing specific evidence and directly referencing the target opponent if provided",
    "responding_to": "{target_opponent['name'] if target_opponent else opponent_name}",
    "new_opinion": "your updated opinion in 2 sentences reflecting score {new_score}",
    "shift_reason": "why you {'moved slightly' if shifted else 'held firm'} on this issue",
    "key_evidence_used": ["evidence point 1", "evidence point 2"]
}}"""

    try:
        result = await call_llm_json(prompt, system)
        response = safe_parse_json(result)

        if not response:
            raise ValueError("safe_parse_json returned empty — LLM response unparseable")

        updated_agent = agent.copy()
        updated_agent["opinion"] = response.get("new_opinion", agent["opinion"])
        updated_agent["score"] = new_score
        updated_agent["stance"] = new_stance
        updated_agent["opinion_delta"] = delta
        updated_agent["last_argument"] = response.get("argument", "")
        updated_agent["shifted"] = shifted
        updated_agent["shift_reason"] = response.get("shift_reason", "")
        updated_agent["key_evidence_used"] = response.get("key_evidence_used", [])
        updated_agent["responding_to"] = response.get("responding_to", "")
        updated_agent["shift_caused_by"] = (
            target_opponent["name"] if (shifted and target_opponent) else None
        )
        updated_agent["evidence_multiplier"] = biased_multiplier
        updated_agent["deffuant_gap"] = round(min_gap, 2) if influential_opponent else None
        updated_agent["attacks_received"] = attacks_received
        updated_agent["effective_resistance"] = effective_resistance
        updated_agent["confirmation_bias"] = get_confirmation_bias(agent)
        updated_agent["emotional_contagion_source"] = contagion_source

        return updated_agent

    except Exception as e:
        print(f"[DebateEngine] Error in round {round_num} for {agent['name']}: {e}")
        updated_agent = agent.copy()
        updated_agent["score"] = new_score
        updated_agent["stance"] = new_stance
        updated_agent["opinion_delta"] = delta
        updated_agent["shifted"] = shifted
        updated_agent["attacks_received"] = attacks_received
        return updated_agent


async def run_debate_round(
    agents: list[dict],
    topic: str,
    G_inst: nx.DiGraph,
    G_pub: nx.DiGraph,
    round_num: int
) -> list[dict]:
    print(f"[DebateEngine] Running round {round_num} with {len(agents)} agents...")
    tasks = [
        run_single_agent_round(agent, agents, topic, G_inst, G_pub, round_num)
        for agent in agents
    ]
    updated_agents = await asyncio.gather(*tasks)
    return list(updated_agents)


async def run_debate(
    topic: str,
    agents: list[dict],
    G_inst: nx.DiGraph,
    num_rounds: int = 3,
    G_pub: nx.DiGraph = None
) -> dict:
    if G_pub is None:
        G_pub = G_inst

    # ── Guard: abort cleanly if no agents generated ───────────────
    # Prevents ZeroDivisionError crash when LLM refuses harmful topics
    # or when agent generation fails completely.
    if not agents:
        print("[DebateEngine] No agents provided — aborting debate cleanly")
        return {"rounds": [], "final_agents": []}

    print(f"[DebateEngine] Starting debate on: {topic}")
    print(f"[DebateEngine] {len(agents)} agents, {num_rounds} rounds")
    print(f"[DebateEngine] Deffuant params: mu={BASE_MU}, threshold={CONFIDENCE_THRESHOLD}")
    print(f"[DebateEngine] Emotional contagion: reach={EMOTIONAL_CONTAGION_REACH}, mu={EMOTIONAL_CONTAGION_MU}")

    all_rounds = []
    current_agents = agents.copy()

    for round_num in range(1, num_rounds + 1):
        updated_agents = await run_debate_round(
            current_agents, topic, G_inst, G_pub, round_num
        )

        # Guard against empty round result
        if not updated_agents:
            print(f"[DebateEngine] Round {round_num} produced no agents — stopping")
            break

        round_result = {
            "round": round_num,
            "agents": [
                {
                    "id": a["id"],
                    "name": a["name"],
                    "persona": a["persona"],
                    "opinion": a["opinion"],
                    "score": a["score"],
                    "opinion_delta": a["opinion_delta"],
                    "stance": a["stance"],
                    "stakeholder_name": a.get("stakeholder_name"),
                    "stakeholder_category": a.get("stakeholder_category"),
                    "agent_type": a.get("agent_type", "institutional"),
                    "deffuant_gap": a.get("deffuant_gap"),
                    "evidence_multiplier": a.get("evidence_multiplier"),
                    "attacks_received": a.get("attacks_received", 0),
                    "effective_resistance": a.get("effective_resistance"),
                    "confirmation_bias": a.get("confirmation_bias"),
                    "shift_caused_by": a.get("shift_caused_by"),
                    "emotional_contagion_source": a.get("emotional_contagion_source"),
                    "last_argument": a.get("last_argument", ""),
                    "responding_to": a.get("responding_to", ""),
                }
                for a in updated_agents
            ]
        }

        all_rounds.append(round_result)
        current_agents = updated_agents

        shifts = sum(1 for a in updated_agents if a.get("shifted", False))
        avg_delta = sum(a.get("opinion_delta", 0) for a in updated_agents) / len(updated_agents)
        emotional_nudges = sum(1 for a in updated_agents if a.get("emotional_contagion_source"))
        print(f"[DebateEngine] Round {round_num} complete. Shifts: {shifts}/{len(updated_agents)} | "
              f"Avg delta: {avg_delta:.3f} | Emotional nudges: {emotional_nudges}")

    return {
        "rounds": all_rounds,
        "final_agents": current_agents
    }