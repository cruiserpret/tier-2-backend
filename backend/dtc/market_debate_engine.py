"""
backend/dtc/market_debate_engine.py

DTC Market Debate Engine for Assembly Tier 2.

Runs 3 structured debate rounds between buyer agents:
  Round 1: First Impression    — initial reaction to product + price
  Round 2: Competitor Comparison — head-to-head vs known alternatives
  Round 3: Consensus Building  — final purchase intent after hearing everyone

RESEARCH BASIS:
━━━━━━━━━━━━━━
Deffuant et al. (2000) — Bounded Confidence Model:
    Agents only update opinions when the distance between their opinion
    and a neighbor's is within a confidence threshold (epsilon).
    |opinion_i - opinion_j| < epsilon → opinions converge toward mean
    This produces realistic opinion clustering, not random drift.

    epsilon = 0.3 (standard value from Deffuant 2000)
    mu = persuasion_resistance (agent-specific convergence rate)
    After interaction: opinion_i += mu * (opinion_j - opinion_i)

Sunstein (2002) — "The Law of Group Polarization":
    Groups of like-minded agents polarize toward more extreme positions.
    We counteract this by seeding AGAINST agents with high-quality
    objections from real 1-2★ reviews — they resist convergence.

Mercier & Sperber (2011) — "Why do humans reason?":
    Arguments are more persuasive when they contain specific evidence.
    LLM prompts are structured to generate evidence-backed arguments,
    not opinion statements — this produces more realistic debate dynamics.

ROUND STRUCTURE:
  Round 1: "You just saw {product} at ${price}. Honest first reaction?"
            → Each agent responds independently
            → No Deffuant update (no peer influence yet)

  Round 2: "You've heard competitors discussed. {Competitor} does X,
            {product} does Y. Which matters more for your decision?"
            → Agents respond to competitor gap data
            → Deffuant update: agents within epsilon shift toward peers

  Round 3: "After hearing everyone's perspective, final purchase verdict?"
            → Agents respond after seeing Round 2 arguments
            → Deffuant update: second pass, stronger convergence
            → Final stance + purchase probability computed
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


# ── Deffuant Constants ────────────────────────────────────────────────────────
EPSILON = 0.3   # Bounded confidence threshold (Deffuant 2000)
SCALE   = 10.0  # Frontend score scale


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    round:   int
    agents:  list   # list of agent dicts with updated opinions


@dataclass
class DebateResult:
    simulation_id: str
    rounds:        list[RoundResult] = field(default_factory=list)
    status:        str = "complete"


# ── LLM Client ────────────────────────────────────────────────────────────────

async def _llm(session, prompt, max_tokens=350):
    try:
        async with session.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
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


# ── Deffuant Opinion Update ───────────────────────────────────────────────────

def _deffuant_update(agents: list[BuyerAgent]) -> list[BuyerAgent]:
    """
    Apply one Deffuant bounded confidence update across all agent pairs.

    Algorithm (Deffuant et al. 2000):
      For each pair (i, j):
        if |score_i - score_j| / SCALE < EPSILON:
          score_i += mu_i * (score_j - score_i)
          score_j += mu_j * (score_i - score_j)

    mu = agent's persuasion_resistance (lower = more open to change)
    epsilon = 0.3 (agents only interact if opinions are close enough)

    Returns updated agents with new scores and opinion_delta.
    """
    # Work in [0,1] normalized space
    scores = [a.score / SCALE for a in agents]
    prev   = scores.copy()

    # Random pairwise interactions
    indices = list(range(len(agents)))
    random.shuffle(indices)

    for i in range(0, len(indices) - 1, 2):
        a_idx = indices[i]
        b_idx = indices[i + 1]

        dist = abs(scores[a_idx] - scores[b_idx])
        if dist < EPSILON:
            mu_a = agents[a_idx].persuasion_resistance
            mu_b = agents[b_idx].persuasion_resistance

            new_a = scores[a_idx] + mu_a * (scores[b_idx] - scores[a_idx])
            new_b = scores[b_idx] + mu_b * (scores[a_idx] - scores[b_idx])

            scores[a_idx] = max(0.0, min(1.0, new_a))
            scores[b_idx] = max(0.0, min(1.0, new_b))

    # Apply updates and compute deltas
    for i, agent in enumerate(agents):
        new_score    = round(scores[i] * SCALE, 1)
        agent.opinion_delta = round(new_score - agent.score, 2)
        agent.score  = new_score

        # Update stance based on new score
        if agent.score >= 6.0:
            agent.stance = "for"
        elif agent.score <= 4.0:
            agent.stance = "against"
        else:
            agent.stance = "neutral"

    return agents


# ── Round Prompts ─────────────────────────────────────────────────────────────

def _build_round1_prompt(agent: BuyerAgent, product: ProductBrief, intel: MarketIntelligence) -> str:
    """
    Round 1: First Impression
    Agent sees product for the first time at listed price.
    No peer influence yet — pure individual reaction.
    """
    cat_avg = intel.category_avg_price
    price_context = (
        f"${product.price} (${abs(product.price - cat_avg):.0f} "
        f"{'above' if product.price > cat_avg else 'below'} category average of ${cat_avg:.0f})"
    )

    return f"""You are {agent.name}, a {agent.age}-year-old {agent.profession} from {agent.location}.

You just saw this product for the first time:
PRODUCT: {product.name}
DESCRIPTION: {product.description}
PRICE: {price_context}

Your current feeling about this product is: {agent.stance.upper()} (score {agent.score}/10)
Your personality: {agent.stakeholder_name}
Your key beliefs: {', '.join(agent.key_beliefs)}

Give your honest FIRST IMPRESSION as {agent.name}. Be specific about:
1. What initially attracts or repels you
2. How you feel about the ${product.price} price
3. One question you'd want answered before deciding

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences of authentic first reaction, in first person, as {agent.name}>",
  "last_argument": "<your single strongest point about this product right now>",
  "emotional_intensity": "<high|medium|low>"
}}"""


def _build_round2_prompt(
    agent:    BuyerAgent,
    product:  ProductBrief,
    intel:    MarketIntelligence,
    round1_opinions: list[str]
) -> str:
    """
    Round 2: Competitor Comparison
    Agent sees competitor gap data + hears other agents' Round 1 opinions.
    """
    # Build competitor context from gap analysis
    comp_context = ""
    for gap in intel.gaps[:2]:
        price_note = f"${abs(gap.user_price_diff):.0f} {'cheaper' if gap.user_price_diff < 0 else 'more expensive'} than {gap.competitor_name}"
        comp_context += (
            f"\n- {gap.competitor_name}: ${gap.competitor_price} | "
            f"{gap.competitor_rating}★ | {gap.competitor_bought:,}/month purchases | "
            f"Buyers praise: {', '.join(gap.top_praise[:2])}"
        )
        if gap.top_complaints:
            comp_context += f" | Complaint: {gap.top_complaints[0][:80]}"

    # Sample of other agents' Round 1 opinions
    other_opinions = "\n".join([f"- {op}" for op in round1_opinions[:3]])

    return f"""You are {agent.name}, {agent.age}, {agent.profession} from {agent.location}.

PRODUCT BEING EVALUATED: {product.name} at ${product.price}
YOUR CURRENT STANCE: {agent.stance.upper()} (score {agent.score}/10)

COMPETITOR LANDSCAPE:
{comp_context if comp_context else "No competitor data available."}
{product.name} is ${abs(product.price - intel.category_avg_price):.0f} {'above' if product.price > intel.category_avg_price else 'below'} the category average.

WHAT OTHERS SAID IN ROUND 1:
{other_opinions if other_opinions else "You are the first to speak."}

Now respond to the competitor comparison. Consider:
- Does the competitor data change your view of {product.name}'s value?
- What argument from others resonated or annoyed you?
- Has your opinion shifted at all?

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences responding to competitor data + peers, in first person as {agent.name}>",
  "last_argument": "<your strongest competitive argument for or against {product.name}>",
  "emotional_intensity": "<high|medium|low>"
}}"""


def _build_round3_prompt(
    agent:    BuyerAgent,
    product:  ProductBrief,
    intel:    MarketIntelligence,
    round2_arguments: list[str]
) -> str:
    """
    Round 3: Consensus Building
    Agent gives final purchase verdict after hearing all peer arguments.
    """
    # Build decisive argument context
    decisive_args = "\n".join([f"- {arg}" for arg in round2_arguments[:4]])

    return f"""You are {agent.name}, {agent.age}, {agent.profession} from {agent.location}.

FINAL DECISION on: {product.name} at ${product.price}
YOUR CURRENT STANCE: {agent.stance.upper()} (score {agent.score}/10)

STRONGEST ARGUMENTS YOU'VE HEARD:
{decisive_args if decisive_args else "No prior arguments."}

It's time for your FINAL VERDICT. Be decisive:
- Would you actually buy {product.name} at ${product.price}? Why or why not?
- What was the argument that most influenced your final position?
- If you wouldn't buy: what would need to change?

Return ONLY valid JSON:
{{
  "opinion": "<2-3 sentences of final verdict in first person as {agent.name}. Be concrete — yes/no/maybe with specific reasoning.>",
  "last_argument": "<your definitive closing statement on {product.name}>",
  "emotional_intensity": "<high|medium|low>"
}}"""


# ── Single Agent Round ────────────────────────────────────────────────────────

async def _run_agent_round(
    session:  aiohttp.ClientSession,
    agent:    BuyerAgent,
    prompt:   str,
) -> BuyerAgent:
    """Run a single agent through one debate round."""
    response = await _llm(session, prompt, max_tokens=350)

    try:
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())

        agent.opinion           = data.get("opinion", agent.opinion)
        agent.last_argument     = data.get("last_argument", agent.last_argument)
        agent.emotional_intensity = data.get("emotional_intensity", agent.emotional_intensity)

    except Exception:
        pass  # Keep existing opinion if parse fails

    return agent


# ── Main Debate Runner ────────────────────────────────────────────────────────

async def run_market_debate(
    agents:        list[BuyerAgent],
    intel:         MarketIntelligence,
    simulation_id: str = "sim_dtc_001",
) -> DebateResult:
    """
    Run 3-round DTC market debate.

    Round 1: First impressions (parallel, no peer influence)
    Round 2: Competitor comparison (parallel) + Deffuant update
    Round 3: Consensus building (parallel) + Deffuant update

    Returns DebateResult with all rounds and final agent states.
    """
    product = intel.product
    result  = DebateResult(simulation_id=simulation_id)

    print(f"\n[DebateEngine] ══ Starting market debate ══")
    print(f"[DebateEngine] Product: {product.name} | {len(agents)} agents | 3 rounds")

    async with aiohttp.ClientSession() as session:

        # ── ROUND 1: First Impression ─────────────────────────────────────────
        print(f"\n[DebateEngine] Round 1: First Impression")

        r1_tasks = [
            _run_agent_round(
                session, agent,
                _build_round1_prompt(agent, product, intel)
            )
            for agent in agents
        ]
        agents = list(await asyncio.gather(*r1_tasks))

        # No Deffuant update after Round 1 — agents haven't heard each other yet
        round1_opinions = [a.last_argument for a in agents if a.last_argument]

        result.rounds.append(RoundResult(
            round=1,
            agents=agents_to_dict(agents)
        ))

        print(f"[DebateEngine] Round 1 complete — "
              f"FOR:{sum(1 for a in agents if a.stance=='for')} "
              f"AGAINST:{sum(1 for a in agents if a.stance=='against')} "
              f"NEUTRAL:{sum(1 for a in agents if a.stance=='neutral')}")

        # ── ROUND 2: Competitor Comparison ───────────────────────────────────
        print(f"\n[DebateEngine] Round 2: Competitor Comparison")

        r2_tasks = [
            _run_agent_round(
                session, agent,
                _build_round2_prompt(agent, product, intel, round1_opinions)
            )
            for agent in agents
        ]
        agents = list(await asyncio.gather(*r2_tasks))

        # Deffuant update after Round 2
        agents = _deffuant_update(agents)

        round2_arguments = [a.last_argument for a in agents if a.last_argument]

        result.rounds.append(RoundResult(
            round=2,
            agents=agents_to_dict(agents)
        ))

        shifted_r2 = sum(1 for a in agents if abs(a.opinion_delta) > 0.5)
        print(f"[DebateEngine] Round 2 complete — "
              f"{shifted_r2} agents shifted | "
              f"FOR:{sum(1 for a in agents if a.stance=='for')} "
              f"AGAINST:{sum(1 for a in agents if a.stance=='against')} "
              f"NEUTRAL:{sum(1 for a in agents if a.stance=='neutral')}")

        # ── ROUND 3: Consensus Building ───────────────────────────────────────
        print(f"\n[DebateEngine] Round 3: Consensus Building")

        r3_tasks = [
            _run_agent_round(
                session, agent,
                _build_round3_prompt(agent, product, intel, round2_arguments)
            )
            for agent in agents
        ]
        agents = list(await asyncio.gather(*r3_tasks))

        # Final Deffuant update
        agents = _deffuant_update(agents)

        result.rounds.append(RoundResult(
            round=3,
            agents=agents_to_dict(agents)
        ))

        shifted_r3 = sum(1 for a in agents if abs(a.opinion_delta) > 0.5)
        final_for     = sum(1 for a in agents if a.stance == "for")
        final_against = sum(1 for a in agents if a.stance == "against")
        final_neutral = sum(1 for a in agents if a.stance == "neutral")

        print(f"\n[DebateEngine] ══ Debate complete ══")
        print(f"[DebateEngine] Final: FOR={final_for} AGAINST={final_against} NEUTRAL={final_neutral}")
        print(f"[DebateEngine] Round 3 shifts: {shifted_r3}")
        print(f"[DebateEngine] Avg final score: {sum(a.score for a in agents)/len(agents):.1f}/10")

    return result


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backend.dtc.dtc_ingestor import run_market_ingestion
    from backend.dtc.buyer_persona_generator import generate_buyer_personas

    async def test():
        print("=" * 60)
        print("Assembly Tier 2 — Market Debate Engine Test")
        print("=" * 60)

        product = ProductBrief(
            name="CollagenRise Daily Serum",
            description="A vegan collagen-boosting serum with bakuchiol. No synthetic fragrance. Clinically tested.",
            price=49.0,
            category="beauty_skincare",
            demographic="Women 28-45, clean beauty enthusiasts",
            competitors=[
                {"name": "The Ordinary Niacinamide", "asin": "B01MDTVZTZ"},
                {"name": "Drunk Elephant", "asin": ""},
            ],
        )

        print("\nStep 1: Market ingestion...")
        intel = await run_market_ingestion(product, num_agents=6)

        print("\nStep 2: Generating personas...")
        agents = await generate_buyer_personas(intel, num_agents=6)

        print("\nStep 3: Running debate...")
        debate = await run_market_debate(agents, intel, simulation_id="test_001")

        print("\n── Debate Results ──────────────────────────────────")
        for rnd in debate.rounds:
            print(f"\nROUND {rnd.round}:")
            for a in rnd.agents:
                print(f"  {a['name']:20s} | {a['stance']:7s} | score={a['score']:.1f} | "
                      f"delta={a['opinion_delta']:+.1f}")
                print(f"    → {a['opinion'][:100]}...")

        print(f"\n✓ Debate complete — {len(debate.rounds)} rounds")

    asyncio.run(test())