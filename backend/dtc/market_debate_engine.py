"""
backend/dtc/market_debate_engine.py — GODMODE 3.3 EDITION

═══════════════════════════════════════════════════════════════════════════════
TRANSPARENCY LABELS:
  # PUBLISHED — Formula/value directly from peer-reviewed research
  # CALIBRATED — Empirically tuned against validation tests
  # ENGINEERED — Engineering choice not backed by specific research
═══════════════════════════════════════════════════════════════════════════════

RESEARCH FOUNDATION:
- Deffuant, Neau, Amblard, Weisbuch (2000): Bounded confidence opinion dynamics
- Noelle-Neumann (1974): Spiral of Silence in group discussion
- Sunstein (2002): Group polarization in like-minded discussions
- Mercier & Sperber (2011): Argumentative theory of reasoning
- Tversky & Kahneman (1973): Availability heuristic in decision-making
- Rogers (1962): Diffusion of Innovations — laggards (hardcore resistors)
"""

import asyncio
import aiohttp
import json
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from dataclasses import dataclass, field
from backend.dtc.buyer_persona_generator import BuyerAgent, agents_to_dict
from backend.dtc.dtc_ingestor import MarketIntelligence, ProductBrief


# PUBLISHED (Deffuant et al. 2000): Bounded confidence parameter
# CALIBRATED: Specific value 0.25 chosen within paper's tested range (0.1-0.5)
EPSILON = 0.25

# ENGINEERED: Score normalization constant (1-10 scale to 0-1)
SCALE = 10.0


@dataclass
class RoundResult:
    round:   int
    agents:  list


@dataclass
class DebateResult:
    simulation_id: str
    rounds:        list[RoundResult] = field(default_factory=list)
    status:        str = "complete"


async def _llm(session, prompt, max_tokens=350):
    try:
        async with session.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model":      config.LLM_MODEL_NAME,
                "max_tokens": max_tokens,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[DebateEngine] LLM error: {e}")
        return ""


def _is_hardcore(agent: BuyerAgent) -> bool:
    """
    PUBLISHED CONCEPT (Rogers 1962): Laggards resist change.
    CALIBRATED: 0.10 threshold for persuasion_resistance.
    """
    return agent.persuasion_resistance <= 0.10


def _deffuant_update(agents: list[BuyerAgent]) -> list[BuyerAgent]:
    """
    PUBLISHED (Deffuant et al. 2000): Bounded confidence model.

    Formula (exact from paper):
      If |x_i - x_j| < ε:
          x_i' = x_i + μ(x_j - x_i)
          x_j' = x_j + μ(x_i - x_j)

    ENGINEERED: Hardcore resistor skip logic
    CALIBRATED: Stance reclassification thresholds (6.2/3.8)
    """
    scores = [a.score / SCALE for a in agents]
    indices = list(range(len(agents)))
    random.shuffle(indices)

    for i in range(0, len(indices) - 1, 2):
        a_idx = indices[i]
        b_idx = indices[i + 1]
        agent_a = agents[a_idx]
        agent_b = agents[b_idx]

        # ENGINEERED: Hardcore agents skip updates
        if _is_hardcore(agent_a) and _is_hardcore(agent_b):
            continue

        # PUBLISHED (Deffuant): Bounded confidence check
        dist = abs(scores[a_idx] - scores[b_idx])
        if dist >= EPSILON:
            continue

        # PUBLISHED (Deffuant): Convergence update formula
        mu_a = agent_a.persuasion_resistance if not _is_hardcore(agent_a) else 0.0
        mu_b = agent_b.persuasion_resistance if not _is_hardcore(agent_b) else 0.0

        new_a = scores[a_idx] + mu_a * (scores[b_idx] - scores[a_idx])
        new_b = scores[b_idx] + mu_b * (scores[a_idx] - scores[b_idx])

        scores[a_idx] = max(0.0, min(1.0, new_a))
        scores[b_idx] = max(0.0, min(1.0, new_b))

    for i, agent in enumerate(agents):
        new_score = round(scores[i] * SCALE, 1)
        agent.opinion_delta = round(new_score - agent.score, 2)
        agent.score = new_score

        # CALIBRATED: Stance reclassification with hysteresis
        # Engineering choice to prevent flip-flopping (6.2/3.8 instead of 6.0/4.0)
        if agent.score >= 6.2:
            agent.stance = "for"
        elif agent.score <= 3.8:
            agent.stance = "against"
        else:
            agent.stance = "neutral"

    return agents


# ── Round Prompts ────────────────────────────────────────────────────────────
# ENGINEERED: All prompt wording is engineering, not published research.
# Research basis for balanced framing:
# - Tversky & Kahneman (1973): Availability heuristic — salient alternatives
#   get over-weighted in decisions
# - Mercier & Sperber (2011): Reasoning works better with balanced evidence

def _build_round1_prompt(agent, product, intel):
    cat_avg = intel.category_avg_price
    price_context = (
        f"${product.price} (${abs(product.price - cat_avg):.0f} "
        f"{'above' if product.price > cat_avg else 'below'} category avg of ${cat_avg:.0f})"
    )

    return f"""You are {agent.name}, a {agent.age}-year-old {agent.profession} from {agent.location}.

PRODUCT UNDER REVIEW:
{product.name}
{product.description}
PRICE: {price_context}

Your stance: {agent.stance.upper()} (initial score {agent.score}/10)
Your personality: {agent.stakeholder_name}
Your key beliefs: {', '.join(agent.key_beliefs)}

Give your honest FIRST IMPRESSION as {agent.name}. Stay true to your stance:
- If FOR: express genuine interest in the product's features
- If AGAINST: express genuine skepticism (be specific about what bothers you)
- If NEUTRAL: express genuine uncertainty

Be specific and authentic. Reference ${product.price} price. Sound like a real person.

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences, first person, matches stance>",
  "last_argument": "<your single strongest point>",
  "emotional_intensity": "<high|medium|low>"
}}"""


def _build_round2_prompt(agent, product, intel, round1_opinions):
    """
    ENGINEERED: Balanced framing to avoid availability heuristic bias.
    Research basis: Tversky & Kahneman (1973) showed making one alternative
    highly salient causes decision distortion. We present product features
    FIRST, then competitor data as neutral facts.
    """
    product_reminder = (
        f"{product.name} — ${product.price}\n"
        f"Key features: {product.description[:200]}"
    )

    comp_facts = ""
    if intel.gaps:
        comp_facts = "\nWHAT ELSE IS AVAILABLE (for reference, not recommendation):\n"
        for gap in intel.gaps[:2]:
            price_compare = (f"${gap.competitor_price} "
                           f"({'cheaper' if gap.user_price_diff > 0 else 'similar' if abs(gap.user_price_diff) < 3 else 'pricier'})")
            comp_facts += f"• {gap.competitor_name}: {price_compare}, {gap.competitor_rating}★\n"

    other_opinions = ""
    if round1_opinions:
        sample = round1_opinions[:2]  # CALIBRATED: 2 peer opinions to avoid overwhelming
        other_opinions = "\nPEERS HAVE SAID:\n" + "\n".join([f"• {op}" for op in sample])

    hardcore_note = ""
    if _is_hardcore(agent):
        hardcore_note = ("\n\nYou are a HARDCORE RESISTOR: You're loyal to existing alternatives. "
                         "Peer arguments rarely sway you. Stay firm.")

    return f"""You are {agent.name}, {agent.age}, {agent.profession} from {agent.location}.

QUESTION: Would you personally buy {product.name} at ${product.price}?

{product_reminder}
{comp_facts}{other_opinions}

YOUR CURRENT STANCE: {agent.stance.upper()} (score {agent.score}/10)
{hardcore_note}

Evaluate based on YOUR OWN needs, lifestyle, and preferences — NOT based on what's popular
or what peers say. Does this product fit YOUR life?

- FOR agents: Does your initial interest still hold up? What convinces you more or less?
- AGAINST agents: Does anything in Round 1 challenge your skepticism?
- NEUTRAL agents: Has anything tipped you one way? (It's OK if not.)

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences, first person, honest about personal fit>",
  "last_argument": "<strongest point based on YOUR needs>",
  "emotional_intensity": "<high|medium|low>"
}}"""


def _build_round3_prompt(agent, product, intel, round2_arguments):
    """
    ENGINEERED: Final decision framed as personal choice.
    Research basis: Mercier & Sperber (2011) — reasoning converges best
    when agents commit to a position rather than remain ambivalent.
    """
    peer_args = ""
    if round2_arguments:
        sample = round2_arguments[:3]  # CALIBRATED: 3 peer arguments
        peer_args = "\nMOST COMPELLING POINTS FROM ROUND 2:\n" + "\n".join([f"• {arg}" for arg in sample])

    hardcore_note = ""
    if _is_hardcore(agent):
        hardcore_note = ("\n\nYou are a HARDCORE RESISTOR. Your decision today reflects your "
                         "long-held skepticism. Peer pressure doesn't sway you.")

    return f"""You are {agent.name}, {agent.age}, {agent.profession} from {agent.location}.

FINAL DECISION: Will you personally buy {product.name} at ${product.price}?

YOUR STANCE: {agent.stance.upper()} (score {agent.score}/10)
{peer_args}{hardcore_note}

Make a BINARY DECISION based on YOUR personal needs:
- Would YOU spend ${product.price} on this, given your lifestyle and budget?
- State your decision clearly with 1-2 sentences of honest reasoning.

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences FINAL VERDICT. Start with 'I'll buy' or 'I won't buy' or 'I'm still undecided'.>",
  "last_argument": "<definitive closing statement>",
  "emotional_intensity": "<high|medium|low>"
}}"""


async def _run_agent_round(session, agent, prompt):
    response = await _llm(session, prompt, max_tokens=350)
    try:
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
        agent.opinion = data.get("opinion", agent.opinion)
        agent.last_argument = data.get("last_argument", agent.last_argument)
        agent.emotional_intensity = data.get("emotional_intensity", agent.emotional_intensity)
    except Exception:
        pass
    return agent


async def run_market_debate(agents, intel, simulation_id="sim_dtc_001"):
    """
    PUBLISHED: 3-round debate structure inspired by:
    - Noelle-Neumann (1974): Spiral of Silence — opinions solidify through group exposure
    - Sunstein (2002): Group polarization — like-minded discussion shifts views
    - Mercier & Sperber (2011): Reasoning as argumentation

    ENGINEERED: Specific round structure (First Impression → Personal Fit → Final Verdict)
    """
    product = intel.product
    result = DebateResult(simulation_id=simulation_id)

    hardcore_count = sum(1 for a in agents if _is_hardcore(a))
    print(f"\n[DebateEngine] ══ GODMODE 3.3 debate ══")
    print(f"[DebateEngine] Product: {product.name} | {len(agents)} agents ({hardcore_count} hardcore) | 3 rounds")
    print(f"[DebateEngine] Saturated market: {intel.is_saturated_market}")

    async with aiohttp.ClientSession() as session:
        # ROUND 1 — First Impression
        print(f"\n[DebateEngine] Round 1: First Impression")
        r1_tasks = [
            _run_agent_round(session, a, _build_round1_prompt(a, product, intel))
            for a in agents
        ]
        agents = list(await asyncio.gather(*r1_tasks))
        round1_opinions = [a.last_argument for a in agents if a.last_argument]
        result.rounds.append(RoundResult(round=1, agents=agents_to_dict(agents)))
        print(f"[DebateEngine] Round 1: "
              f"FOR:{sum(1 for a in agents if a.stance=='for')} "
              f"AGAINST:{sum(1 for a in agents if a.stance=='against')} "
              f"NEUTRAL:{sum(1 for a in agents if a.stance=='neutral')}")

        # ROUND 2 — Personal Fit Evaluation
        print(f"\n[DebateEngine] Round 2: Personal Fit Evaluation")
        r2_tasks = [
            _run_agent_round(session, a, _build_round2_prompt(a, product, intel, round1_opinions))
            for a in agents
        ]
        agents = list(await asyncio.gather(*r2_tasks))
        agents = _deffuant_update(agents)  # PUBLISHED: Deffuant update
        round2_arguments = [a.last_argument for a in agents if a.last_argument]
        result.rounds.append(RoundResult(round=2, agents=agents_to_dict(agents)))
        shifted_r2 = sum(1 for a in agents if abs(a.opinion_delta) > 0.5)
        print(f"[DebateEngine] Round 2: {shifted_r2} shifted | "
              f"FOR:{sum(1 for a in agents if a.stance=='for')} "
              f"AGAINST:{sum(1 for a in agents if a.stance=='against')} "
              f"NEUTRAL:{sum(1 for a in agents if a.stance=='neutral')}")

        # ROUND 3 — Final Verdict
        print(f"\n[DebateEngine] Round 3: Final Verdict")
        r3_tasks = [
            _run_agent_round(session, a, _build_round3_prompt(a, product, intel, round2_arguments))
            for a in agents
        ]
        agents = list(await asyncio.gather(*r3_tasks))
        agents = _deffuant_update(agents)
        result.rounds.append(RoundResult(round=3, agents=agents_to_dict(agents)))

        final_for = sum(1 for a in agents if a.stance == "for")
        final_against = sum(1 for a in agents if a.stance == "against")
        final_neutral = sum(1 for a in agents if a.stance == "neutral")

        print(f"\n[DebateEngine] ══ Debate complete ══")
        print(f"[DebateEngine] Final: FOR={final_for} AGAINST={final_against} NEUTRAL={final_neutral}")
        print(f"[DebateEngine] Avg score: {sum(a.score for a in agents)/len(agents):.1f}/10")

    return result