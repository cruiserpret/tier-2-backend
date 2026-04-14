import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential
import networkx as nx

# ── Round Summarizer ──────────────────────────────────────────────

async def summarize_round(round_data: dict, round_num: int) -> dict:
    """
    Compress a debate round into a structured summary.
    Used to give the report agent full context without hitting token limits.
    """
    agents = round_data["agents"]
    total = len(agents)
    
    shifted = [a for a in agents if a.get("opinion_delta", 0) > 0.3]
    held = [a for a in agents if a.get("opinion_delta", 0) <= 0.3]
    
    avg_delta = sum(a.get("opinion_delta", 0) for a in agents) / total if total > 0 else 0
    
    stance_dist = {"for": 0, "against": 0, "neutral": 0}
    for a in agents:
        stance = a.get("stance", "neutral")
        stance_dist[stance] = stance_dist.get(stance, 0) + 1

    agent_summaries = "\n".join([
        f"- {a['name']} ({a.get('stakeholder_name', 'individual')}): score={a['score']} stance={a['stance']} delta={a.get('opinion_delta', 0):.2f}"
        for a in agents
    ])
    shifted_names = [a["name"] for a in shifted]
    held_names = [a["name"] for a in held]

    system = """You are summarizing a debate round for a simulation report.
Be precise and factual. Reference specific agents and arguments.
Respond in valid JSON only."""

    prompt = f"""Summarize debate round {round_num}.

Agent states this round:
{agent_summaries}

Agents who shifted: {shifted_names}
Agents who held firm: {held_names}
Average opinion delta: {avg_delta:.3f}
Stance distribution: {stance_dist}

Respond in this exact JSON format:
{{
    "round": {round_num},
    "key_development": "the single most important thing that happened this round in 1 sentence",
    "dominant_argument": "the argument that had the most influence this round",
    "who_shifted": {json.dumps(shifted_names)},
    "why_they_shifted": "brief explanation of what caused shifts",
    "who_held": {json.dumps(held_names)},
    "stance_distribution": {json.dumps(stance_dist)},
    "avg_delta": {round(avg_delta, 3)}
}}"""
    try:
        result = await call_llm_json(prompt, system)
        return json.loads(result)
    except Exception as e:
        print(f"[ReportAgent] Round {round_num} summarizer error: {e}")
        return {
            "round": round_num,
            "key_development": f"Round {round_num} completed",
            "dominant_argument": "",
            "who_shifted": [a["name"] for a in shifted],
            "why_they_shifted": "",
            "who_held": [a["name"] for a in held],
            "stance_distribution": stance_dist,
            "avg_delta": round(avg_delta, 3)
        }

async def summarize_all_rounds(rounds: list[dict]) -> list[dict]:
    """Summarize all rounds in parallel."""
    tasks = [summarize_round(r, r["round"]) for r in rounds]
    summaries = await asyncio.gather(*tasks)
    return list(summaries)

# ── Verdict Calculator ────────────────────────────────────────────

def calculate_verdict(final_agents: list[dict], rounds: list[dict]) -> dict:
    """
    Mathematically derive verdict confidence.

    confidence = (stance_concentration × 0.50) +
                 (score_separation     × 0.35) +
                 (shift_convergence    × 0.15)

    Research basis for weight change (Pransh, April 2026):
    Old weights gave shift_convergence 30%. On entrenched topics
    (abortion, gun control) agents correctly hold firm — near-zero
    shifts is realistic human behavior, not a simulation failure.
    Penalising it with 30% produced artificially low confidence
    (25-35%) on topics with genuine strong consensus.

    New weights: stance_concentration dominates at 50%.
    A simulation where 12 FOR agents hold at score 8.0 across 3
    rounds now gets high confidence — correctly.
    shift_convergence demoted to 15% — it still matters but doesn't
    tank the score when polarisation is the real-world truth.

    score_separation — how far dominant group's avg score is from
    neutral (5.0). A group at 8.0 is more decided than one at 6.0.
    """
    total = len(final_agents)
    if total == 0:
        return {"confidence": 0, "dominant_stance": "neutral", "verdict_strength": "inconclusive"}

    # Stance counts
    stance_counts = {"for": 0, "against": 0, "neutral": 0}
    for a in final_agents:
        s = a.get("stance", "neutral")
        stance_counts[s] = stance_counts.get(s, 0) + 1

    dominant_stance = max(stance_counts, key=stance_counts.get)
    dominant_count = stance_counts[dominant_stance]
    stance_concentration = dominant_count / total

    # Score separation — how far dominant group is from neutral (5.0)
    dominant_agents = [a for a in final_agents if a.get("stance") == dominant_stance]
    if dominant_agents:
        avg_dominant_score = sum(a["score"] for a in dominant_agents) / len(dominant_agents)
        score_separation = min(abs(avg_dominant_score - 5.0) / 5.0, 1.0)
    else:
        score_separation = 0.0

    # Shift convergence — agents who shifted toward dominant stance
    total_convergent_shifts = 0
    total_agent_rounds = total * len(rounds) if rounds else 1

    for r in rounds:
        for a in r["agents"]:
            delta = a.get("opinion_delta", 0)
            stance = a.get("stance", "neutral")
            if delta > 0.10 and stance == dominant_stance:
                total_convergent_shifts += 1

    shift_convergence = min(total_convergent_shifts / total_agent_rounds, 1.0)

    # Final confidence — updated weights (Pransh April 2026)
    confidence = (
        stance_concentration * 0.50 +
        score_separation     * 0.35 +
        shift_convergence    * 0.15
    )
    confidence_pct = round(confidence * 100, 1)

    # Strength label
    if confidence_pct >= 65:
        strength = "strong"
    elif confidence_pct >= 40:
        strength = "moderate"
    else:
        strength = "contested"

    # Minority
    minority_stances = {k: v for k, v in stance_counts.items() if k != dominant_stance}
    minority_stance = max(minority_stances, key=minority_stances.get)

    return {
        "dominant_stance": dominant_stance,
        "dominant_count": dominant_count,
        "minority_stance": minority_stance,
        "minority_count": stance_counts[minority_stance],
        "neutral_count": stance_counts["neutral"],
        "confidence_pct": confidence_pct,
        "verdict_strength": strength,
        "stance_concentration": round(stance_concentration, 3),
        "shift_convergence": round(shift_convergence, 3),
        "score_separation": round(score_separation, 3),
    }

# ── Main Report Generator ─────────────────────────────────────────

async def generate_report(
    topic: str,
    simulation_id: str,
    rounds: list[dict],
    final_agents: list[dict],
    G: nx.DiGraph
) -> dict:
    """
    God's Eye View report with round summaries and mathematically-backed verdict.
    """
    print(f"[ReportAgent] Generating God's Eye View for simulation {simulation_id}")

    # Step 1 — Summarize all rounds
    print(f"[ReportAgent] Summarizing {len(rounds)} rounds...")
    round_summaries = await summarize_all_rounds(rounds)

    # Step 2 — Calculate verdict mathematically
    verdict_data = calculate_verdict(final_agents, rounds)

    # Step 3 — Find who shifted and who held
    first_round_agents = {a["id"]: a for a in rounds[0]["agents"]} if rounds else {}

    agents_shifted = []
    agents_held = []

    for agent in final_agents:
        first_state = first_round_agents.get(agent["id"])
        if not first_state:
            shifted = agent.get("shifted", False)
            initial_stance = agent["stance"]
        else:
            initial_stance = first_state["stance"]
            final_stance = agent["stance"]
            shifted = (initial_stance != final_stance) or agent.get("shifted", False)

        summary = {
            "agent_id": agent["id"],
            "name": agent["name"],
            "stakeholder": agent.get("stakeholder_name", "individual"),
            "shifted": shifted,
            "initial_stance": initial_stance,
            "final_stance": agent["stance"],
            "final_score": agent["score"],
            "key_moment": agent.get("shift_reason", "held firm throughout")
        }

        if shifted:
            agents_shifted.append(summary)
        else:
            agents_held.append(summary)

    # Step 4 — Graph insights
    top_entities = get_most_influential(G, top_n=3)
    graph_context = "\n".join([
        f"- {e['name']} (influence: {e['influence_score']:.3f}, cited {e['citations']}x)"
        for e in top_entities
    ])

    # Step 5 — Build round summary context for LLM
    round_context = "\n".join([
        f"Round {s['round']}: {s['key_development']} | shifts: {len(s['who_shifted'])} | avg delta: {s['avg_delta']}"
        for s in round_summaries
    ])

    # Step 6 — LLM generates verdict text + trajectory
    system = """You are the God's Eye View analyst for a multi-agent debate simulation.
You have seen the complete debate through structured round summaries.
Generate a precise, evidence-backed analysis.
Respond in valid JSON only."""

    prompt = f"""Analyze this complete debate simulation.

Topic: {topic}
Total agents: {len(final_agents)}
Agents shifted: {len(agents_shifted)}
Agents held: {len(agents_held)}

Verdict data (mathematically calculated):
- Dominant stance: {verdict_data['dominant_stance']} ({verdict_data['dominant_count']} agents)
- Minority stance: {verdict_data['minority_stance']} ({verdict_data['minority_count']} agents)
- Confidence: {verdict_data['confidence_pct']}%
- Verdict strength: {verdict_data['verdict_strength']}

Round summaries:
{round_context}

Most influential knowledge graph entities:
{graph_context}

Generate the God's Eye View in this exact JSON format:
{{
    "summary": "2-3 sentence executive summary of what happened in the debate",
    "predicted_trajectory": "specific prediction of real-world outcome based on debate dynamics, cite dominant arguments",
    "verdict_statement": "one clear declarative sentence verdict on the proposition",
    "decisive_factor": "the single argument or piece of evidence that most shaped the outcome",
    "minority_position": "why the minority held firm and what that means for real-world implementation",
    "real_world_implication": "what this debate result means for the real world in 1-2 sentences",
    "actionable_insight": "one specific actionable recommendation for a decision-maker reading this",
    "consensus_level": "high/medium/low"
}}"""

    try:
        result = await call_llm_json(prompt, system)
        synthesis = json.loads(result)
    except Exception as e:
        print(f"[ReportAgent] LLM synthesis error: {e}")
        synthesis = {
            "summary": "Debate completed.",
            "predicted_trajectory": "Unable to generate trajectory.",
            "verdict_statement": "Outcome inconclusive.",
            "decisive_factor": "",
            "minority_position": "",
            "real_world_implication": "",
            "actionable_insight": "",
            "consensus_level": "medium"
        }

    # Step 7 — Sentiment per round
    sentiment_ticks = []
    for round_data in rounds:
        agents_in_round = round_data["agents"]
        total = len(agents_in_round)
        if total == 0:
            continue
        positive = sum(1 for a in agents_in_round if a["stance"] == "for") / total
        negative = sum(1 for a in agents_in_round if a["stance"] == "against") / total
        neutral = sum(1 for a in agents_in_round if a["stance"] == "neutral") / total
        sentiment_ticks.append({
            "tick": round_data["round"],
            "positive": round(positive, 2),
            "neutral": round(neutral, 2),
            "negative": round(negative, 2)
        })

    # Step 8 — Decisive arguments
    argument_influence = {}
    for agent in final_agents:
        if agent.get("shifted") and agent.get("last_argument"):
            key = agent.get("last_argument", "")[:100]
            if key not in argument_influence:
                argument_influence[key] = {
                    "argument": agent.get("last_argument", ""),
                    "influenced_agents": [],
                    "evidence_used": agent.get("key_evidence_used", [])
                }
            argument_influence[key]["influenced_agents"].append(agent["id"])

    decisive_arguments = [
        {
            "agent_id": final_agents[0]["id"] if final_agents else "unknown",
            "argument": v["argument"],
            "influenced_agents": v["influenced_agents"],
            "evidence_used": v["evidence_used"]
        }
        for v in argument_influence.values()
    ][:5]

    # Step 9 — Assemble final report
    report = {
        "simulation_id": simulation_id,
        "topic": topic,

        "summary": synthesis.get("summary", ""),
        "predicted_trajectory": synthesis.get("predicted_trajectory", ""),
        "verdict": {
            "statement": synthesis.get("verdict_statement", ""),
            "confidence_pct": verdict_data["confidence_pct"],
            "strength": verdict_data["verdict_strength"],
            "dominant_stance": verdict_data["dominant_stance"],
            "dominant_count": verdict_data["dominant_count"],
            "minority_stance": verdict_data["minority_stance"],
            "minority_count": verdict_data["minority_count"],
            "decisive_factor": synthesis.get("decisive_factor", ""),
            "minority_position": synthesis.get("minority_position", ""),
            "real_world_implication": synthesis.get("real_world_implication", ""),
        },

        "actionable_insight": synthesis.get("actionable_insight", ""),
        "consensus_level": synthesis.get("consensus_level", "medium"),
        "agents_shifted": len(agents_shifted),
        "agents_held": len(agents_held),
        "decisive_arguments": decisive_arguments,
        "agent_summaries": [
            {
                "agent_id": a["agent_id"],
                "name": a["name"],
                "stakeholder": a["stakeholder"],
                "shifted": a["shifted"],
                "final_stance": a["final_stance"],
                "key_moment": a["key_moment"]
            }
            for a in agents_shifted + agents_held
        ],
        "round_summaries": round_summaries,
        "sentiment_history": {
            "simulation_id": simulation_id,
            "ticks": sentiment_ticks
        }
    }

    print(f"[ReportAgent] Report complete. "
          f"{len(agents_shifted)} shifted, {len(agents_held)} held. "
          f"Verdict: {verdict_data['dominant_stance']} ({verdict_data['confidence_pct']}% confidence)")

    return report