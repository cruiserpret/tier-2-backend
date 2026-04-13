import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import query_graph
import networkx as nx

# ── Deffuant Model Parameters ─────────────────────────────────────
BASE_MU = 0.3
CONFIDENCE_THRESHOLD = 3.0
MAX_EVIDENCE_MULTIPLIER = 1.5

# ── Confirmation Bias Parameters ──────────────────────────────────
# How much contradicting evidence is discounted per category
# 0 = no bias (accepts all evidence), 1 = full bias (ignores contradicting evidence)
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
BACKFIRE_THRESHOLD = 2        # attacks before resistance starts increasing
BACKFIRE_RESISTANCE_BOOST = 0.05  # resistance increase per attack above threshold
MAX_RESISTANCE = 0.95         # resistance can never exceed this

def get_confirmation_bias(agent: dict) -> float:
    category = agent.get("stakeholder_category", "civil_society")
    return CONFIRMATION_BIAS_BY_CATEGORY.get(category, 0.35)

def apply_confirmation_bias(
    evidence_multiplier: float,
    agent: dict,
    weighted_avg: float
) -> float:
    """
    Discount evidence multiplier when opponents are pushing against agent's position.
    Grounded in Chuang et al. NAACL 2024 — LLMs need confirmation bias injection
    to produce realistic opinion fragmentation.
    """
    agent_score = agent.get("score", 5.0)
    bias = get_confirmation_bias(agent)

    # Is the weighted average pushing against the agent's position?
    contradicting = (agent_score > 5.0 and weighted_avg < agent_score) or \
                    (agent_score < 5.0 and weighted_avg > agent_score)

    if contradicting:
        # Discount the evidence multiplier by confirmation bias
        discounted = evidence_multiplier * (1.0 - bias)
        return round(max(discounted, 1.0), 3)

    return evidence_multiplier

def apply_backfire_effect(agent: dict) -> float:
    """
    If agent has been repeatedly attacked without shifting,
    increase their resistance — they dig in harder.
    Grounded in Nyhan & Reifler 2010 backfire effect research.
    """
    attacks_received = agent.get("attacks_received", 0)

    if attacks_received <= BACKFIRE_THRESHOLD:
        return agent.get("persuasion_resistance", 0.5)

    extra_attacks = attacks_received - BACKFIRE_THRESHOLD
    boost = extra_attacks * BACKFIRE_RESISTANCE_BOOST
    new_resistance = agent.get("persuasion_resistance", 0.5) + boost
    return round(min(new_resistance, MAX_RESISTANCE), 3)

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
    # Step 1 — Pull evidence from knowledge graph
    # Each agent queries their own graph — no cross-contamination
    G = G_pub if agent.get("graph_type") == "public" else G_inst
    keywords = agent.get("key_beliefs", []) + agent.get("known_entities", [])
    evidence = query_graph(G, keywords, top_n=5)

    evidence_context = "\n".join([
        f"- {e['name']}: {e['description'][:150]} [source: {e['source'][:60]}] (cited {e['citations']}x)"
        for e in evidence
    ])

    # Step 2 — Read opponents
    opponents = [a for a in all_agents if a["id"] != agent["id"]]
    opponent_context = "\n".join([
        f"- {o['name']} ({o['stance']}, score {o['score']}): {o['opinion']}"
        for o in opponents[:6]
    ])

    # Step 3 — Find all opponents within threshold
    within_threshold = [
        opp for opp in opponents
        if abs(agent["score"] - opp["score"]) < CONFIDENCE_THRESHOLD
    ]

    # Step 4 — Calculate base evidence multiplier
    base_evidence_multiplier = calculate_evidence_multiplier(evidence)
    old_score = agent["score"]

    if within_threshold:
        # Fix 4 — Influence-weighted Deffuant
        # High influence agents pull others harder — not everyone has equal pull
        weights = [
            (1.0 / (abs(agent["score"] - opp["score"]) + 0.1)) * opp.get("influence_weight", 0.5)
            for opp in within_threshold
        ]
        total_weight = sum(weights)
        weighted_avg = sum(
            opp["score"] * w for opp, w in zip(within_threshold, weights)
        ) / total_weight

        # Fix 2 — Confirmation bias
        # Discount evidence when opponents are pushing against agent's worldview
        biased_multiplier = apply_confirmation_bias(
            base_evidence_multiplier, agent, weighted_avg
        )

        # Fix 3 — Backfire effect
        # Repeated attacks without shifting increases resistance
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

        # Track attacks for backfire effect
        # Agent is being "attacked" if weighted avg is against their position
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

    # Step 5 — Derive stance and shift
    new_stance = derive_stance(new_score)
    stance_changed = new_stance != agent.get("stance", "neutral")
    shifted = stance_changed or (delta > 0.10)

    # Update attack counter for backfire effect tracking
    attacks_received = agent.get("attacks_received", 0)
    if being_attacked and not shifted:
        attacks_received += 1
    elif shifted:
        attacks_received = 0  # reset if they actually shifted

    # Step 6 — LLM generates argument text
    system = """You are simulating a realistic human debater grounded in real evidence.
Your opinion shift has already been mathematically calculated.
Your job is to generate realistic argument text that reflects this outcome.
Respond in valid JSON only."""

    opponent_name = influential_opponent['name'] if influential_opponent else 'the group'
    shift_direction = f"You moved toward {opponent_name}'s position" if influential_opponent and shifted else "You held your position firm"

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

Mathematical outcome (already decided):
- Your new score: {new_score}/10
- Opinion shifted: {shifted}
- {shift_direction}
- You {"are being repeatedly challenged and digging in harder" if attacks_received > BACKFIRE_THRESHOLD else "are engaging with the debate openly"}

Generate argument text that reflects this outcome naturally.

Respond in this exact JSON format:
{{
    "argument": "your argument this round citing specific evidence",
    "responding_to": "{opponent_name}",
    "new_opinion": "your updated opinion in 2 sentences reflecting score {new_score}",
    "shift_reason": "why you {'moved slightly' if shifted else 'held firm'} on this issue",
    "key_evidence_used": ["evidence point 1", "evidence point 2"]
}}"""

    try:
        result = await call_llm_json(prompt, system)
        response = json.loads(result)

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
        updated_agent["evidence_multiplier"] = biased_multiplier
        updated_agent["deffuant_gap"] = round(min_gap, 2) if influential_opponent else None
        updated_agent["attacks_received"] = attacks_received
        updated_agent["effective_resistance"] = effective_resistance
        updated_agent["confirmation_bias"] = get_confirmation_bias(agent)

        return updated_agent

    except Exception as e:
        print(f"[DebateEngine] Error in round {round_num} for {agent['name']}: {e}")
        return agent

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
        G_pub = G_inst  # fallback — use same graph if no public graph

    print(f"[DebateEngine] Starting debate on: {topic}")
    print(f"[DebateEngine] {len(agents)} agents, {num_rounds} rounds")
    print(f"[DebateEngine] Deffuant params: mu={BASE_MU}, threshold={CONFIDENCE_THRESHOLD}")

    all_rounds = []
    current_agents = agents.copy()

    for round_num in range(1, num_rounds + 1):
        updated_agents = await run_debate_round(
            current_agents, topic, G_inst, G_pub, round_num
        )
        # rest stays exactly the same

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
                    "deffuant_gap": a.get("deffuant_gap"),
                    "evidence_multiplier": a.get("evidence_multiplier"),
                    "attacks_received": a.get("attacks_received", 0),
                    "effective_resistance": a.get("effective_resistance"),
                    "confirmation_bias": a.get("confirmation_bias"),
                }
                for a in updated_agents
            ]
        }

        all_rounds.append(round_result)
        current_agents = updated_agents

        shifts = sum(1 for a in updated_agents if a.get("shifted", False))
        avg_delta = sum(a.get("opinion_delta", 0) for a in updated_agents) / len(updated_agents)
        print(f"[DebateEngine] Round {round_num} complete. Shifts: {shifts}/{len(updated_agents)} | Avg delta: {avg_delta:.3f}")

    return {
        "rounds": all_rounds,
        "final_agents": current_agents
    }