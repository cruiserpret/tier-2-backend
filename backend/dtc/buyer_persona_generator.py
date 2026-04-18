"""
backend/dtc/buyer_persona_generator.py — GODMODE FINAL

All fixes:
 - 50 agent cap (was 12)
 - 49 unique demographics (no assertion crash)
 - Hardcore resistor support
 - Pool cycling for >49 agents
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


DEMOGRAPHIC_POOL = [
    {"name": "Priya Sharma",      "age": 34, "profession": "Dermatology PA",              "location": "Chicago, IL"},
    {"name": "Fatima Hassan",     "age": 36, "profession": "Nurse Practitioner",          "location": "Houston, TX"},
    {"name": "David Goldberg",    "age": 47, "profession": "Family Physician",            "location": "Minneapolis, MN"},
    {"name": "Nia Williams",      "age": 29, "profession": "Registered Dietitian",        "location": "Atlanta, GA"},
    {"name": "Sanjay Patel",      "age": 52, "profession": "Cardiologist",                "location": "Philadelphia, PA"},
    {"name": "Maya Chen",         "age": 32, "profession": "UX Designer",                 "location": "San Francisco, CA"},
    {"name": "Marcus Webb",       "age": 45, "profession": "Software Engineer",           "location": "Seattle, WA"},
    {"name": "Aisha Johnson",     "age": 31, "profession": "Graphic Designer",            "location": "Atlanta, GA"},
    {"name": "Kevin Park",        "age": 38, "profession": "Product Manager",             "location": "Austin, TX"},
    {"name": "Lena Kowalski",     "age": 26, "profession": "Junior Developer",            "location": "Brooklyn, NY"},
    {"name": "Sarah Okonkwo",     "age": 38, "profession": "Elementary School Teacher",   "location": "Austin, TX"},
    {"name": "Robert Chen",       "age": 54, "profession": "High School Principal",       "location": "Sacramento, CA"},
    {"name": "Amara Diallo",      "age": 33, "profession": "University Professor",        "location": "Boston, MA"},
    {"name": "Rachel Torres",     "age": 41, "profession": "Marketing Director",          "location": "New York, NY"},
    {"name": "Tom Gallagher",     "age": 44, "profession": "Financial Advisor",           "location": "Denver, CO"},
    {"name": "Yuki Tanaka",       "age": 36, "profession": "Management Consultant",       "location": "Chicago, IL"},
    {"name": "Carlos Mendoza",    "age": 48, "profession": "Real Estate Agent",           "location": "Miami, FL"},
    {"name": "Lisa Andersson",    "age": 39, "profession": "CPA",                         "location": "Portland, OR"},
    {"name": "Jennifer Walsh",    "age": 29, "profession": "Freelance Photographer",      "location": "Portland, OR"},
    {"name": "Diego Ramirez",     "age": 34, "profession": "Video Editor",                "location": "Los Angeles, CA"},
    {"name": "Brittany Moore",    "age": 27, "profession": "Content Creator",             "location": "Nashville, TN"},
    {"name": "Elijah Foster",     "age": 42, "profession": "Musician",                    "location": "Austin, TX"},
    {"name": "James Kim",         "age": 52, "profession": "Restaurant Owner",            "location": "Los Angeles, CA"},
    {"name": "Gabrielle Rousseau","age": 33, "profession": "Sommelier",                   "location": "San Francisco, CA"},
    {"name": "Anthony Russo",     "age": 46, "profession": "Contractor",                  "location": "Long Island, NY"},
    {"name": "Destiny Jackson",   "age": 28, "profession": "Yoga Instructor",             "location": "Boulder, CO"},
    {"name": "Chloe Bernard",     "age": 27, "profession": "Graduate Student",            "location": "Boston, MA"},
    {"name": "Tyler Hendricks",   "age": 23, "profession": "College Senior",              "location": "Ann Arbor, MI"},
    {"name": "Fernanda Castro",   "age": 25, "profession": "Nursing Student",             "location": "Miami, FL"},
    {"name": "Jake Whitfield",    "age": 24, "profession": "Sales Associate",             "location": "Dallas, TX"},
    {"name": "Rebecca Hill",      "age": 37, "profession": "Stay-at-home Parent",         "location": "Cincinnati, OH"},
    {"name": "Darius Washington", "age": 40, "profession": "Youth Coach",                 "location": "Memphis, TN"},
    {"name": "Mei Lin Zhao",      "age": 35, "profession": "Part-time Accountant",        "location": "San Jose, CA"},
    {"name": "Diana Reeves",      "age": 58, "profession": "Retired Accountant",          "location": "Scottsdale, AZ"},
    {"name": "Harold Brennan",    "age": 64, "profession": "Retired Engineer",            "location": "Raleigh, NC"},
    {"name": "Patricia O'Malley", "age": 61, "profession": "Retired Teacher",             "location": "Sarasota, FL"},
    {"name": "Walter Kimura",     "age": 67, "profession": "Retired Pharmacist",          "location": "Honolulu, HI"},
    {"name": "Luis Salazar",      "age": 43, "profession": "HVAC Technician",             "location": "Phoenix, AZ"},
    {"name": "Megan Sullivan",    "age": 30, "profession": "Paramedic",                   "location": "Cleveland, OH"},
    {"name": "Trey Coleman",      "age": 35, "profession": "Firefighter",                 "location": "Saint Louis, MO"},
    {"name": "Vanessa Nguyen",    "age": 32, "profession": "Dental Hygienist",            "location": "San Diego, CA"},
    {"name": "Amir Khan",         "age": 45, "profession": "City Planner",                "location": "Washington, DC"},
    {"name": "Olivia Hartman",    "age": 39, "profession": "Social Worker",               "location": "Madison, WI"},
    {"name": "Theo Branson",      "age": 28, "profession": "Policy Analyst",              "location": "Washington, DC"},
    {"name": "Brooke Stephens",   "age": 31, "profession": "Personal Trainer",            "location": "Austin, TX"},
    {"name": "Rafael Santos",     "age": 33, "profession": "Physical Therapist",          "location": "San Antonio, TX"},
    {"name": "Zara Ahmad",        "age": 29, "profession": "Startup Founder",             "location": "Brooklyn, NY"},
    {"name": "Nathan Goldsmith",  "age": 41, "profession": "E-commerce Owner",            "location": "Denver, CO"},
    {"name": "Imani Okoye",       "age": 36, "profession": "Consulting Firm Owner",       "location": "Atlanta, GA"},
]
# NO ASSERT — lets us add/remove entries freely


@dataclass
class BuyerAgent:
    id:                    str
    name:                  str
    age:                   int
    profession:            str
    location:              str
    stakeholder_name:      str
    stakeholder_category:  str
    agent_type:            str
    stance:                str
    score:                 float
    opinion:               str
    last_argument:         str
    emotional_intensity:   str
    key_beliefs:           list  = field(default_factory=list)
    confirmation_bias:     float = 0.5
    persuasion_resistance: float = 0.5
    influence_weight:      float = 0.5
    opinion_delta:         float = 0.0
    source_review:         str   = ""


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


def _derive_deffuant_params(stance, score, hardcore=False):
    if hardcore:
        noise = random.uniform(-0.03, 0.03)
        return (
            round(max(0.85, min(0.98, 0.92 + noise)), 2),
            round(max(0.02, min(0.10, 0.05 + noise)), 2),
            round(max(0.60, min(0.85, 0.75 + noise)), 2),
        )
    extremity = abs(score - 0.5) * 2
    noise = random.uniform(-0.05, 0.05)
    cb = max(0.1, min(0.9, 0.25 + 0.50 * extremity + noise))
    pr = max(0.1, min(0.9, 0.20 + 0.45 * extremity + noise))
    iw = max(0.2, min(0.9, 0.30 + 0.55 * extremity + noise))
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


async def _generate_single_persona(session, agent_index, stance, intel, product, demographic, is_hardcore=False):
    source_review = _sample_review_for_stance(intel, stance)
    score = _compute_initial_score(stance, intel)
    frontend_score = round(1.0 + score * 9.0, 1)
    cb, pr, iw = _derive_deffuant_params(stance, score, hardcore=is_hardcore)

    cat_avg_price = intel.category_avg_price
    price_position = "premium" if product.price > cat_avg_price * 1.3 else \
                     "budget"  if product.price < cat_avg_price * 0.7 else "mid-range"

    comp_names = [c.name for c in intel.competitors if c.found_on_amazon]
    comp_context = f"They are familiar with: {', '.join(comp_names)}." if comp_names else ""

    stance_instruction = {
        "for":     f"This person is GENUINELY INTERESTED in {product.name} and likely to try it.",
        "against": f"This person is SKEPTICAL or RESISTANT — they prefer existing solutions.",
        "neutral": f"This person is ON THE FENCE — could go either way.",
    }[stance]

    hardcore_instruction = ""
    if is_hardcore:
        hardcore_instruction = ("\nIMPORTANT: This person is a HARDCORE RESISTOR — extremely skeptical, "
                                "hard to convince, loyal to existing alternatives. Their opinion is strongly held "
                                "and they rarely change their mind.")

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
{stance_instruction}{hardcore_instruction}
{comp_context}

REAL MARKET SIGNAL:
{f'Source review: "{source_review}"' if source_review else 'Use Reddit community context.'}

Return ONLY valid JSON, no markdown:

{{
  "stakeholder_name": "<brief descriptor>",
  "emotional_intensity": "<high|medium|low>",
  "opinion": "<2-3 sentence authentic first-person reaction. Reference price, ingredients, or specific competitors.>",
  "last_argument": "<1-2 sentence argument they would make>",
  "key_beliefs": ["<belief 1>", "<belief 2>"]
}}

Rules:
- Opinion MUST reflect {stance} stance
- Reference ${product.price} price specifically
- If against: mention a specific competitor they prefer by name
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
            "stakeholder_name":    f"{demographic['profession']}, {demographic['age']}",
            "emotional_intensity": "medium",
            "opinion":             f"I {'support' if stance == 'for' else 'question' if stance == 'neutral' else 'doubt'} {product.name} at ${product.price}.",
            "last_argument":       f"The ${product.price} price point needs {'justification' if stance != 'for' else 'context'}.",
            "key_beliefs":         ["Quality matters", "Value is important"],
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

    hardcore_tag = " [HARDCORE]" if is_hardcore else ""
    print(f"[PersonaGenerator] ✓ {agent.name} ({stance}{hardcore_tag}, score={frontend_score}, "
          f"age={agent.age}, {agent.profession})")
    return agent


async def generate_buyer_personas(intel, num_agents=6):
    """
    GODMODE: Support 4-50 agents.
    """
    product = intel.product
    num_agents = max(4, min(50, num_agents))  # ✅ RAISED CAP TO 50

    n_for     = max(1, round(intel.agent_for_ratio     * num_agents))
    n_against = max(1, round(intel.agent_against_ratio * num_agents))
    n_neutral = num_agents - n_for - n_against

    if n_neutral < 1:
        n_neutral = 1
        if n_for > n_against: n_for -= 1
        else: n_against -= 1

    n_hardcore = getattr(intel, 'hardcore_resistor_count', 1)
    n_hardcore = min(n_hardcore, n_against)

    print(f"\n[PersonaGenerator] Generating {num_agents} buyer personas")
    print(f"[PersonaGenerator] Distribution: {n_for} FOR | {n_against} AGAINST ({n_hardcore} hardcore) | {n_neutral} NEUTRAL")

    stances = (["for"] * n_for) + (["against"] * n_against) + (["neutral"] * n_neutral)
    random.shuffle(stances)

    pool = DEMOGRAPHIC_POOL.copy()
    random.shuffle(pool)

    if num_agents <= len(pool):
        demographics = pool[:num_agents]
    else:
        # Pool cycling with variations
        demographics = []
        for i in range(num_agents):
            base = pool[i % len(pool)]
            cycle = i // len(pool)
            demographics.append({
                "name":       base["name"] + (f" Jr." if cycle == 1 else f" III" if cycle >= 2 else ""),
                "age":        max(22, min(72, base["age"] + cycle * 3 - 5)),
                "profession": base["profession"],
                "location":   base["location"],
            })

    against_indices = [i for i, s in enumerate(stances) if s == "against"]
    random.shuffle(against_indices)
    hardcore_indices = set(against_indices[:n_hardcore])

    async with aiohttp.ClientSession() as session:
        tasks = [
            _generate_single_persona(
                session, i, stances[i], intel, product, demographics[i],
                is_hardcore=(i in hardcore_indices)
            )
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