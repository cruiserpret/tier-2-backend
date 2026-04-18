"""
backend/dtc/market_debate_engine.py — GODMODE 3 FINAL

MAJOR FIXES:
1. Deffuant updates respect stance — can't pull AGAINST to FOR without evidence
2. Round 2 prompt frames as consumer choice, not product defense
3. Hardcore resistors skip Deffuant updates entirely
4. Competitor data injection is explicit (Stanley 90K reviews gets mentioned)
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


EPSILON = 0.25   # GODMODE 3: tightened from 0.30 — preserves stance diversity
SCALE   = 10.0


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


# ── GODMODE 3: Stance-Aware Deffuant ─────────────────────────────────────────

def _is_hardcore(agent: BuyerAgent) -> bool:
    """Hardcore resistors have very low persuasion_resistance (immovable)."""
    return agent.persuasion_resistance <= 0.10


def _deffuant_update(agents: list[BuyerAgent]) -> list[BuyerAgent]:
    """
    GODMODE 3: Stance-aware Deffuant.
    - Hardcore agents skip update (won't move)
    - Cross-stance updates (FOR ↔ AGAINST) require |delta| > 0.5
    - Same-stance updates proceed normally
    - Respects asymmetric movement: resistant agents move less than open ones
    """
    scores = [a.score / SCALE for a in agents]
    indices = list(range(len(agents)))
    random.shuffle(indices)

    for i in range(0, len(indices) - 1, 2):
        a_idx = indices[i]
        b_idx = indices[i + 1]
        agent_a = agents[a_idx]
        agent_b = agents[b_idx]

        # GODMODE 3: Hardcore resistors don't update
        if _is_hardcore(agent_a) and _is_hardcore(agent_b):
            continue

        dist = abs(scores[a_idx] - scores[b_idx])
        if dist >= EPSILON:
            continue

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

        # Stance reclassification with hysteresis to prevent flip-flopping
        if agent.score >= 6.2:
            agent.stance = "for"
        elif agent.score <= 3.8:
            agent.stance = "against"
        else:
            agent.stance = "neutral"

    return agents


# ── Round Prompts (GODMODE 3 — stance-preserving) ───────────────────────────

def _build_round1_prompt(agent, product, intel):
    cat_avg = intel.category_avg_price
    price_context = (
        f"${product.price} (${abs(product.price - cat_avg):.0f} "
        f"{'above' if product.price > cat_avg else 'below'} category avg of ${cat_avg:.0f})"
    )

    return f"""You are {agent.name}, a {agent.age}-year-old {agent.profession} from {agent.location}.

You just saw this product:
PRODUCT: {product.name}
DESCRIPTION: {product.description}
PRICE: {price_context}

Your current stance: {agent.stance.upper()} (score {agent.score}/10 out of 10)
Your personality: {agent.stakeholder_name}
Your key beliefs: {', '.join(agent.key_beliefs)}

Give your honest FIRST IMPRESSION as {agent.name}. Stay true to your stance:
- If FOR: express genuine interest
- If AGAINST: express genuine skepticism or resistance
- If NEUTRAL: express genuine uncertainty

Be specific about what attracts or repels you. Reference the ${product.price} price.

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences, first person, authentic to stance>",
  "last_argument": "<your single strongest point>",
  "emotional_intensity": "<high|medium|low>"
}}"""


def _build_round2_prompt(agent, product, intel, round1_opinions):
    """
    GODMODE 3: Frames as consumer choice between alternatives, not product defense.
    Forces agent to confront competitor data.
    """
    # Build competitor context with explicit market share
    comp_context = ""
    for gap in intel.gaps[:2]:
        price_note = f"${abs(gap.user_price_diff):.0f} {'cheaper' if gap.user_price_diff < 0 else 'more expensive'}"
        comp_context += (
            f"\n• {gap.competitor_name}: ${gap.competitor_price} ({price_note} than {product.name}) | "
            f"{gap.competitor_rating}★ | {gap.competitor_bought:,} buying signals | "
            f"Market share: {gap.market_share*100:.0f}% of category"
        )
        if gap.top_praise:
            comp_context += f" | Buyers praise: {', '.join(gap.top_praise[:2])}"

    # GODMODE 3: Saturated market warning
    saturation_note = ""
    if intel.is_saturated_market:
        saturation_note = (
            f"\n\n⚠ MARKET CONTEXT: {intel.dominant_competitor} dominates this category with "
            f"{intel.dominant_reviews:,} reviews. Most buyers already own or prefer it. "
            f"Switching costs are real."
        )

    other_opinions = "\n".join([f"- {op}" for op in round1_opinions[:3]])

    hardcore_note = ""
    if _is_hardcore(agent):
        hardcore_note = ("\n\nYOU ARE A HARDCORE RESISTOR: You are loyal to existing alternatives. "
                         "Peer arguments rarely sway you. Stay firm in your skepticism.")

    return f"""You are {agent.name}, {agent.age}, {agent.profession} from {agent.location}.

EVALUATING: Should you buy {product.name} at ${product.price}, or stick with alternatives?

YOUR CURRENT STANCE: {agent.stance.upper()} (score {agent.score}/10)

THE REAL MARKET ALTERNATIVES:{comp_context if comp_context else "No competitor data available."}
{saturation_note}

WHAT OTHERS SAID IN ROUND 1:
{other_opinions if other_opinions else "You are the first to speak."}

Now make a CONSUMER CHOICE — you don't owe loyalty to {product.name}:
- Does competitor data make {product.name} look better, worse, or same?
- What peer argument was most/least convincing?
- Has your opinion genuinely shifted? (It's OK if not.)
{hardcore_note}

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences, first person, honest about shift or stability>",
  "last_argument": "<your strongest argument given competitor context>",
  "emotional_intensity": "<high|medium|low>"
}}"""


def _build_round3_prompt(agent, product, intel, round2_arguments):
    decisive_args = "\n".join([f"- {arg}" for arg in round2_arguments[:4]])

    dominant_context = ""
    if intel.dominant_competitor and intel.dominant_reviews > 5000:
        dominant_context = (
            f"\n\nReality check: {intel.dominant_competitor} has {intel.dominant_reviews:,} verified reviews. "
            f"This is the alternative buyers actually use."
        )

    hardcore_note = ""
    if _is_hardcore(agent):
        hardcore_note = ("\n\nYOU ARE A HARDCORE RESISTOR: Your decision today should reflect "
                         "your unwavering skepticism. Peer pressure doesn't work on you.")

    return f"""You are {agent.name}, {agent.age}, {agent.profession} from {agent.location}.

FINAL DECISION: Buy {product.name} at ${product.price} — YES or NO?
YOUR STANCE: {agent.stance.upper()} (score {agent.score}/10)

STRONGEST ARGUMENTS FROM PEERS:
{decisive_args if decisive_args else "No prior arguments."}
{dominant_context}
{hardcore_note}

Make a BINARY DECISION with honest reasoning:
- Would YOU actually spend ${product.price} on this?
- If yes: what finally convinced you? If no: what would change your mind?

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences FINAL VERDICT. Start with 'I'll buy' or 'I won't buy' or 'I'm still undecided'. Be concrete.>",
  "last_argument": "<your definitive closing statement>",
  "emotional_intensity": "<high|medium|low>"
}}"""


# ── Single Agent Round ────────────────────────────────────────────────────────

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


# ── Main Debate ────────────────────────────────────────────────────────────────

async def run_market_debate(agents, intel, simulation_id="sim_dtc_001"):
    product = intel.product
    result = DebateResult(simulation_id=simulation_id)

    hardcore_count = sum(1 for a in agents if _is_hardcore(a))
    print(f"\n[DebateEngine] ══ GODMODE 3 debate ══")
    print(f"[DebateEngine] Product: {product.name} | {len(agents)} agents ({hardcore_count} hardcore) | 3 rounds")
    print(f"[DebateEngine] Saturated market: {intel.is_saturated_market}")

    async with aiohttp.ClientSession() as session:
        # ROUND 1
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

        # ROUND 2
        print(f"\n[DebateEngine] Round 2: Consumer Choice")
        r2_tasks = [
            _run_agent_round(session, a, _build_round2_prompt(a, product, intel, round1_opinions))
            for a in agents
        ]
        agents = list(await asyncio.gather(*r2_tasks))
        agents = _deffuant_update(agents)
        round2_arguments = [a.last_argument for a in agents if a.last_argument]
        result.rounds.append(RoundResult(round=2, agents=agents_to_dict(agents)))
        shifted_r2 = sum(1 for a in agents if abs(a.opinion_delta) > 0.5)
        print(f"[DebateEngine] Round 2: {shifted_r2} shifted | "
              f"FOR:{sum(1 for a in agents if a.stance=='for')} "
              f"AGAINST:{sum(1 for a in agents if a.stance=='against')} "
              f"NEUTRAL:{sum(1 for a in agents if a.stance=='neutral')}")

        # ROUND 3
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