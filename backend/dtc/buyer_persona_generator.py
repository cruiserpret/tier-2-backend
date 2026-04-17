"""
backend/dtc/buyer_persona_generator.py

Buyer Persona Generator for Assembly Tier 2 DTC Market Simulator.
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
from backend.dtc.dtc_ingestor import MarketIntelligence, ProductBrief


# ── Demographic Pool ──────────────────────────────────────────────────────────
# Pre-assigned to prevent LLM convergence on same names/professions

DEMOGRAPHIC_POOL = [
    {"name": "Maya Chen",       "age": 32, "profession": "UX Designer",             "location": "San Francisco, CA"},
    {"name": "Sarah Okonkwo",   "age": 38, "profession": "Elementary School Teacher","location": "Austin, TX"},
    {"name": "Rachel Torres",   "age": 41, "profession": "Marketing Director",       "location": "New York, NY"},
    {"name": "Priya Sharma",    "age": 34, "profession": "Dermatology PA",           "location": "Chicago, IL"},
    {"name": "Jennifer Walsh",  "age": 29, "profession": "Freelance Photographer",   "location": "Portland, OR"},
    {"name": "Diana Reeves",    "age": 58, "profession": "Retired Accountant",       "location": "Scottsdale, AZ"},
    {"name": "Marcus Webb",     "age": 45, "profession": "Software Engineer",        "location": "Seattle, WA"},
    {"name": "Chloe Bernard",   "age": 27, "profession": "Graduate Student",         "location": "Boston, MA"},
    {"name": "James Kim",       "age": 52, "profession": "Restaurant Owner",         "location": "Los Angeles, CA"},
    {"name": "Fatima Hassan",   "age": 36, "profession": "Nurse Practitioner",       "location": "Houston, TX"},
    {"name": "Tom Gallagher",   "age": 44, "profession": "Financial Advisor",        "location": "Denver, CO"},
    {"name": "Aisha Johnson",   "age": 31, "profession": "Graphic Designer",         "location": "Atlanta, GA"},
]


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class BuyerAgent:
    id:                   str
    name:                 str
    age:                  int
    profession:           str
    location:             str
    stakeholder_name:     str
    stakeholder_category: str
    agent_type:           str
    stance:               str
    score:                float
    opinion:              str
    last_argument:        str
    emotional_intensity:  str
    key_beliefs:          list  = field(default_factory=list)
    confirmation_bias:    float = 0.5
    persuasion_resistance: float = 0.5
    influence_weight:     float = 0.5
    opinion_delta:        float = 0.0
    source_review:        str   = ""


# ── LLM Client ────────────────────────────────────────────────────────────────

async def _llm_generate(session, prompt, max_tokens=600):
    try:
        async with session.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": config.LLM_MODEL_NAME,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[PersonaGenerator] LLM error: {e}")
        return ""


# ── Review Sampling ───────────────────────────────────────────────────────────

def _sample_review_for_stance(intel, stance):
    all_reviews = []
    for comp in intel.competitors:
        all_reviews.extend(comp.reviews)

    if stance == "for":
        matching = [r for r in all_reviews if r.star_rating >= 4]
    elif stance == "against":
        matching = [r for r in all_reviews if r.star_rating <= 2]
    else:
        matching = [r for r in all_reviews if r.star_rating == 3]

    if matching:
        return random.choice(matching).text[:300]

    if intel.reddit and intel.reddit.signals:
        sentiment_map = {"for": "positive", "against": "negative", "neutral": "neutral"}
        reddit_matches = [s for s in intel.reddit.signals if s.sentiment == sentiment_map[stance]]
        if reddit_matches:
            return random.choice(reddit_matches).text[:300]

    return ""


# ── Deffuant Parameters ───────────────────────────────────────────────────────

def _derive_deffuant_params(stance, score):
    extremity = abs(score - 0.5) * 2
    noise = random.uniform(-0.05, 0.05)
    cb  = max(0.1, min(0.9, 0.25 + 0.50 * extremity + noise))
    pr  = max(0.1, min(0.9, 0.20 + 0.45 * extremity + noise))
    iw  = max(0.2, min(0.9, 0.30 + 0.55 * extremity + noise))
    return round(cb, 2), round(pr, 2), round(iw, 2)


def _compute_initial_score(stance, intel):
    cat_sat = intel.category_avg_rating / 5.0 if intel.category_avg_rating > 0 else 0.75
    if stance == "for":
        score = max(0.55, min(0.95, 0.55 + 0.35 * cat_sat + random.uniform(-0.08, 0.12)))
    elif stance == "against":
        score = max(0.05, min(0.45, 0.45 - 0.35 * cat_sat + random.uniform(-0.08, 0.08)))
    else:
        score = max(0.38, min(0.62, 0.45 + random.uniform(0.0, 0.15)))
    return round(score, 3)


# ── Persona Generation ────────────────────────────────────────────────────────

async def _generate_single_persona(session, agent_index, stance, intel, product, demographic):
    source_review = _sample_review_for_stance(intel, stance)
    score         = _compute_initial_score(stance, intel)
    frontend_score = round(1.0 + score * 9.0, 1)
    cb, pr, iw    = _derive_deffuant_params(stance, score)

    cat_avg_price  = intel.category_avg_price
    price_position = "premium" if product.price > cat_avg_price * 1.3 else \
                     "budget"  if product.price < cat_avg_price * 0.7 else "mid-range"

    comp_names = [c.name for c in intel.competitors if c.found_on_amazon]
    comp_context  = f"They are familiar with: {', '.join(comp_names)}." if comp_names else ""

    stance_instruction = {
        "for":     f"This person is GENUINELY INTERESTED in {product.name} and likely to try it.",
        "against": f"This person is SKEPTICAL or RESISTANT — they prefer existing solutions.",
        "neutral": f"This person is ON THE FENCE — could go either way.",
    }[stance]

    prompt = f"""You are generating a real buyer persona for a DTC market simulation.

PRODUCT: {product.name} — {product.description}
PRICE: ${product.price} ({price_position} vs category avg ${cat_avg_price:.0f})
CATEGORY: {product.category.replace('_', ' ')}
TARGET DEMOGRAPHIC: {product.demographic or 'general consumers'}

ASSIGNED PERSONA (use EXACTLY as given — do not change name, age, profession, or location):
Name: {demographic['name']}
Age: {demographic['age']}
Profession: {demographic['profession']}
Location: {demographic['location']}

STANCE: {stance.upper()}
{stance_instruction}
{comp_context}

REAL MARKET SIGNAL:
{f'Source review: "{source_review}"' if source_review else 'Use Reddit community context.'}

Return ONLY valid JSON, no markdown:

{{
  "stakeholder_name": "<brief descriptor e.g. 'Price-conscious skincare enthusiast, 38-year-old teacher'>",
  "emotional_intensity": "<high|medium|low>",
  "opinion": "<2-3 sentence authentic first-person reaction. Reference price, ingredients, or specific competitors. Sound like a real person.>",
  "last_argument": "<1-2 sentence argument they would make in group discussion>",
  "key_beliefs": ["<belief 1>", "<belief 2>"]
}}

Rules:
- Opinion MUST reflect {stance} stance
- Reference ${product.price} price specifically
- If against: mention a competitor they prefer by name
- If neutral: express real uncertainty, not vague ambivalence
- Sound like {demographic['name']}, a {demographic['age']}-year-old {demographic['profession']} from {demographic['location']}"""

    response = await _llm_generate(session, prompt, max_tokens=400)

    try:
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
    except Exception:
        data = {
            "stakeholder_name":  f"{demographic['profession']}, {demographic['age']}",
            "emotional_intensity": "medium",
            "opinion":           f"I {'support' if stance == 'for' else 'question' if stance == 'neutral' else 'doubt'} {product.name} at ${product.price}.",
            "last_argument":     f"The ${product.price} price point needs {'justification' if stance != 'for' else 'context'}.",
            "key_beliefs":       ["Quality matters", "Value is important"],
        }

    agent = BuyerAgent(
        id=f"agent_{random.randbytes(4).hex()}",
        name=demographic["name"],
        age=demographic["age"],
        profession=demographic["profession"],
        location=demographic["location"],
        stakeholder_name=data.get("stakeholder_name", f"{demographic['profession']}, {demographic['age']}"),
        stakeholder_category="affected_community",
        agent_type="public",
        stance=stance,
        score=frontend_score,
        opinion=data.get("opinion", ""),
        last_argument=data.get("last_argument", ""),
        emotional_intensity=data.get("emotional_intensity", "medium"),
        key_beliefs=data.get("key_beliefs", []),
        confirmation_bias=cb,
        persuasion_resistance=pr,
        influence_weight=iw,
        opinion_delta=0.0,
        source_review=source_review[:100] if source_review else "",
    )

    print(f"[PersonaGenerator] ✓ {agent.name} ({stance}, score={frontend_score}, "
          f"age={agent.age}, {agent.profession})")

    return agent


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def generate_buyer_personas(intel, num_agents=6):
    product    = intel.product
    num_agents = max(4, min(12, num_agents))

    n_for     = max(1, round(intel.agent_for_ratio     * num_agents))
    n_against = max(1, round(intel.agent_against_ratio * num_agents))
    n_neutral = num_agents - n_for - n_against

    if n_neutral < 1:
        n_neutral = 1
        if n_for > n_against: n_for -= 1
        else: n_against -= 1

    print(f"\n[PersonaGenerator] Generating {num_agents} buyer personas")
    print(f"[PersonaGenerator] Distribution: {n_for} FOR | {n_against} AGAINST | {n_neutral} NEUTRAL")

    stances = (["for"] * n_for) + (["against"] * n_against) + (["neutral"] * n_neutral)
    random.shuffle(stances)

    pool = DEMOGRAPHIC_POOL.copy()
    random.shuffle(pool)
    demographics = pool[:num_agents]

    async with aiohttp.ClientSession() as session:
        tasks = [
            _generate_single_persona(session, i, stances[i], intel, product, demographics[i])
            for i in range(len(stances))
        ]
        agents = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [a for a in agents if not isinstance(a, Exception)]
    print(f"\n[PersonaGenerator] ✓ Generated {len(valid)}/{num_agents} personas")
    return valid


def agents_to_dict(agents):
    return [{
        "id": a.id, "name": a.name, "age": a.age,
        "profession": a.profession, "location": a.location,
        "stakeholder_name": a.stakeholder_name,
        "stakeholder_category": a.stakeholder_category,
        "agent_type": a.agent_type, "stance": a.stance,
        "score": a.score, "opinion": a.opinion,
        "last_argument": a.last_argument,
        "emotional_intensity": a.emotional_intensity,
        "key_beliefs": a.key_beliefs,
        "confirmation_bias": a.confirmation_bias,
        "persuasion_resistance": a.persuasion_resistance,
        "influence_weight": a.influence_weight,
        "opinion_delta": a.opinion_delta,
    } for a in agents]


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backend.dtc.dtc_ingestor import run_market_ingestion

    async def test():
        print("=" * 60)
        print("Assembly Tier 2 — Buyer Persona Generator Test")
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

        intel  = await run_market_ingestion(product, num_agents=6)
        agents = await generate_buyer_personas(intel, num_agents=6)

        print("\n── Generated Personas ──────────────────────────────")
        for a in agents:
            print(f"\n{'='*50}")
            print(f"Name:      {a.name}, {a.age} — {a.profession}, {a.location}")
            print(f"Stance:    {a.stance.upper()} | Score: {a.score}/10")
            print(f"Opinion:   {a.opinion[:150]}...")
            print(f"Argument:  {a.last_argument[:100]}...")
            print(f"Deffuant:  bias={a.confirmation_bias} resist={a.persuasion_resistance} influence={a.influence_weight}")

        print(f"\n✓ {len(agents)} personas ready for debate engine")

    asyncio.run(test())