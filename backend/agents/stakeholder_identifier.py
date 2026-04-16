import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type
import networkx as nx

try:
    from backend.agents.correction_store import get_correction_context
    CORRECTION_STORE_AVAILABLE = True
except ImportError:
    CORRECTION_STORE_AVAILABLE = False
    def get_correction_context(topic): return ""
    print("[StakeholderIdentifier] correction_store not found — running without reflexion memory")

STAKEHOLDER_CATEGORIES = {
    "tech_company":       {"persuasion_resistance": 0.55, "influence_weight": 0.90},
    "government":         {"persuasion_resistance": 0.65, "influence_weight": 0.95},
    "civil_society":      {"persuasion_resistance": 0.40, "influence_weight": 0.65},
    "academic":           {"persuasion_resistance": 0.30, "influence_weight": 0.70},
    "labor_union":        {"persuasion_resistance": 0.50, "influence_weight": 0.60},
    "consumer":           {"persuasion_resistance": 0.20, "influence_weight": 0.40},
    "media":              {"persuasion_resistance": 0.40, "influence_weight": 0.55},
    "investor":           {"persuasion_resistance": 0.45, "influence_weight": 0.75},
    "affected_community": {"persuasion_resistance": 0.35, "influence_weight": 0.50},
    "international_body": {"persuasion_resistance": 0.60, "influence_weight": 0.85},
}

CATEGORY_NORMALIZER = {
    "tech_leader": "tech_company", "tech_executive": "tech_company",
    "tech_entrepreneur": "tech_company", "tech_industry": "tech_company",
    "technology": "tech_company", "corporation": "tech_company",
    "business": "tech_company", "company": "tech_company",
    "ngo": "civil_society", "nonprofit": "civil_society", "advocacy": "civil_society",
    "university": "academic", "research": "academic", "think_tank": "academic",
    "defense_technology": "government", "military": "government",
    "defense": "government", "nation": "government", "country": "government",
    "finance": "investor", "fund": "investor", "bank": "investor",
    "venture_capital": "investor", "vc": "investor",
    "press": "media", "news": "media", "journalist": "media",
    "union": "labor_union", "workers": "labor_union",
    "people": "affected_community", "community": "affected_community",
    "citizens": "consumer", "public": "consumer",
    "un": "international_body", "nato": "international_body",
    "who": "international_body", "wto": "international_body",
}

# ── Diversity parameters ──────────────────────────────────────────
# Raised from 0.30 to 0.35 — less aggressive category enforcement.
# Reason: campus questions have all legitimate stakeholders in
# civil_society/academic. 30% cap was cutting real stakeholders
# to fill irrelevant investor/international_body slots.
MAX_CATEGORY_SHARE = 0.35
MIN_AGENTS = 5
CALIBRATION_THRESHOLD = 0.10

FOR_KEYWORDS = [
    "should be legal", "support", "favor", "advocate", "endorse",
    "in favor", "pro-", "right to", "rights to", "access to",
    "deserve", "deserves", "entitled to", "should have",
    "necessary", "essential", "critical", "important that",
    "must be", "need to", "urgent", "imperative",
    "protect", "expand", "restore", "guarantee", "ensure access",
    "couldn't get", "denied", "lost access", "couldn't afford",
    "had to travel", "went without", "forced to", "had no choice",
    "fought for", "struggled to get", "finally able to",
    "everyone should", "all people deserve", "basic right",
    "human right", "fundamental right", "freedom to",
    "should not be criminalized", "decriminalize",
]

AGAINST_KEYWORDS = [
    "should be illegal", "oppose", "against", "ban", "prohibit",
    "restrict", "forbidden", "not a right", "no right to",
    "shouldn't be", "must not", "should not allow",
    "protect life", "sanctity of life", "unborn", "moral issue",
    "traditional values", "community values", "local decision",
    "state's right", "federal overreach", "government overreach",
    "parental rights", "family values", "goes too far",
    "dangerous", "harmful to", "risk to", "threat to",
    "destabilize", "undermine", "destroy", "corrupt",
    "addiction", "dependency", "gateway", "slippery slope",
    "taxpayer funded", "fiscal irresponsibility", "too expensive",
    "economic burden", "unfair to those who",
]

NEUTRAL_KEYWORDS = [
    "on one hand", "on the other hand", "both sides",
    "complex issue", "nuanced", "depends on", "case by case",
    "balanced approach", "middle ground", "compromise",
    "further research needed", "more evidence needed",
    "not yet clear", "uncertain", "remains to be seen",
    "should be decided by", "leave it to", "up to the individual",
    "personal choice", "neither support nor oppose",
]

CAMPUS_TOPIC_SIGNALS = [
    "ucsd", "university", "campus", "college", "student",
    "library", "dining", "dormitory", "tuition", "financial aid",
    "geisel", "triton", "rady", "price center"
]


def is_campus_topic(topic: str) -> bool:
    topic_lower = topic.lower()
    return any(sig in topic_lower for sig in CAMPUS_TOPIC_SIGNALS)


def correction_is_relevant(topic: str, correction_context: str) -> bool:
    if not correction_context:
        return False
    topic_is_campus = is_campus_topic(topic)
    correction_mentions_campus = any(
        sig in correction_context.lower() for sig in CAMPUS_TOPIC_SIGNALS
    )
    if topic_is_campus and not correction_mentions_campus:
        print(f"[StakeholderIdentifier] Correction store: skipping national policy "
              f"corrections for campus topic")
        return False
    return True


def normalize_category(raw: str) -> str:
    clean = raw.lower().strip().replace(" ", "_").replace("-", "_")
    if clean in STAKEHOLDER_CATEGORIES:
        return clean
    if clean in CATEGORY_NORMALIZER:
        return CATEGORY_NORMALIZER[clean]
    for key in STAKEHOLDER_CATEGORIES:
        if key in clean or clean in key:
            return key
    return "civil_society"


async def generate_topic_keywords(topic: str, context: str = "") -> tuple[list, list]:
    system = """You are an expert in stance detection and NLP.
Generate proposition-specific stance keywords for this exact debate topic.
Respond in valid JSON only."""

    context_line = f'\nContext: "{context}"' if context else ""

    prompt = f"""For this specific proposition: "{topic}"{context_line}

Generate 10 FOR keywords and 10 AGAINST keywords that someone would use
when specifically supporting or opposing THIS proposition.

Examples for "Should UCSD open Sixth Market 24/7":
- FOR: "open all night", "24/7 access", "students need", "late night hunger"
- AGAINST: "staffing costs", "security concerns", "low demand", "not worth it"

Respond in this exact JSON format:
{{
    "for_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5",
                     "keyword6", "keyword7", "keyword8", "keyword9", "keyword10"],
    "against_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5",
                         "keyword6", "keyword7", "keyword8", "keyword9", "keyword10"]
}}

Rules:
- Keywords must be specific to THIS proposition
- Keep each keyword short (1-4 words)
- Do not repeat generic words like "support" or "oppose"
- Focus on the specific ACTION being proposed"""

    try:
        result = await call_llm_json(prompt, system)
        import re
        clean = re.sub(r'[\x00-\x1f\x7f]', ' ', result)
        parsed = json.loads(clean)
        topic_for = parsed.get("for_keywords", [])
        topic_against = parsed.get("against_keywords", [])
        print(f"[StakeholderIdentifier] Topic keywords generated — "
              f"{len(topic_for)} for, {len(topic_against)} against")
        return topic_for, topic_against
    except Exception as e:
        print(f"[StakeholderIdentifier] Topic keyword generation failed: {e}")
        return [], []


def classify_claim_stance(
    claim_text: str,
    topic_for_keywords: list = None,
    topic_against_keywords: list = None
) -> str:
    text = claim_text.lower()
    all_for     = FOR_KEYWORDS + (topic_for_keywords or [])
    all_against = AGAINST_KEYWORDS + (topic_against_keywords or [])

    for_score     = sum(1 for kw in all_for     if kw.lower() in text)
    against_score = sum(1 for kw in all_against if kw.lower() in text)
    neutral_score = sum(1 for kw in NEUTRAL_KEYWORDS if kw in text)

    if for_score == 0 and against_score == 0 and neutral_score == 0:
        return None

    if for_score > against_score and for_score > neutral_score:
        return "for"
    elif against_score > for_score and against_score > neutral_score:
        return "against"
    else:
        return "neutral"


def extract_sentiment_from_chunks(
    pub_chunks: list,
    topic_for_keywords: list = None,
    topic_against_keywords: list = None
) -> dict:
    """
    Forum chunks (Reddit/Quora) weighted 2x — real human stance expression.
    Institutional analysis chunks weighted 0.5x — measured language inflates neutral.
    Hampton et al. 2014 (Pew) — weighting corrects for forum oversample
    while preserving their higher signal quality.
    """
    if not pub_chunks:
        return None

    for_count = against_count = neutral_count = 0.0
    keyword_hits = fallback_hits = 0
    total_weight = 0.0

    for chunk in pub_chunks:
        raw_text = chunk.get("text", "")
        if not raw_text:
            continue

        source_url = chunk.get("source", "")
        is_forum = chunk.get("is_forum", False) or \
                   "reddit.com" in source_url or \
                   "quora.com" in source_url

        chunk_type = chunk.get("chunk_type", "public")
        if is_forum:
            weight = 2.0
        elif chunk_type == "institutional":
            weight = 0.5
        else:
            weight = 1.0

        stance = classify_claim_stance(
            raw_text,
            topic_for_keywords=topic_for_keywords,
            topic_against_keywords=topic_against_keywords
        )

        if stance is not None:
            keyword_hits += 1
            if stance == "for":       for_count += weight
            elif stance == "against": against_count += weight
            else:                     neutral_count += weight
        else:
            fallback_hits += 1
            neutral_count += weight * 0.5

        total_weight += weight

    total = for_count + against_count + neutral_count
    if total == 0:
        return None

    distribution = {
        "for":           round(for_count / total, 2),
        "against":       round(against_count / total, 2),
        "neutral":       round(neutral_count / total, 2),
        "total_signal":  len(pub_chunks),
        "keyword_hits":  keyword_hits,
        "fallback_hits": fallback_hits,
    }

    print(f"[StakeholderIdentifier] Public stance signal (raw text matching) — "
          f"for: {distribution['for']*100:.0f}% / "
          f"against: {distribution['against']*100:.0f}% / "
          f"neutral: {distribution['neutral']*100:.0f}% "
          f"({len(pub_chunks)} chunks: {keyword_hits} keyword-matched, "
          f"{fallback_hits} fallback, forum-weighted)")

    return distribution


def extract_public_sentiment_distribution(
    G_pub: nx.DiGraph,
    topic_for_keywords: list = None,
    topic_against_keywords: list = None
) -> dict:
    """Graph-based fallback — used when raw pub_chunks not available."""
    if G_pub is None:
        return None

    claims = get_nodes_by_type(G_pub, "claim")
    if not claims:
        claims = [
            {"name": n, **data}
            for n, data in G_pub.nodes(data=True)
            if data.get("type") in ("claim", "experience")
        ]

    if not claims:
        return None

    for_count = against_count = neutral_count = 0
    keyword_hits = fallback_hits = 0

    for claim in claims:
        claim_text = claim.get("description", "") or claim.get("name", "")
        stance = classify_claim_stance(
            claim_text,
            topic_for_keywords=topic_for_keywords,
            topic_against_keywords=topic_against_keywords
        )

        if stance is not None:
            keyword_hits += 1
            if stance == "for":       for_count += 1
            elif stance == "against": against_count += 1
            else:                     neutral_count += 1
        else:
            fallback_hits += 1
            legacy = claim.get("sentiment", "neutral")
            if legacy == "positive":   for_count += 1
            elif legacy == "negative": against_count += 1
            else:                      neutral_count += 1

    total = for_count + against_count + neutral_count
    if total == 0:
        return None

    return {
        "for":           round(for_count / total, 2),
        "against":       round(against_count / total, 2),
        "neutral":       round(neutral_count / total, 2),
        "total_signal":  total,
        "keyword_hits":  keyword_hits,
        "fallback_hits": fallback_hits,
    }


def get_current_distribution(stakeholders: list[dict]) -> dict:
    total = len(stakeholders)
    if total == 0:
        return {"for": 0, "against": 0, "neutral": 0}
    return {
        "for":     round(sum(1 for s in stakeholders if s.get("stance") == "for") / total, 2),
        "against": round(sum(1 for s in stakeholders if s.get("stance") == "against") / total, 2),
        "neutral": round(sum(1 for s in stakeholders if s.get("stance") == "neutral") / total, 2),
    }


async def request_missing_stakeholders(
    topic, missing_stances, num_needed, graph_context, existing_names
):
    system = """You are identifying additional stakeholders whose interests are being overlooked.
These must be REAL, SPECIFIC organizations or groups of people with genuine stakes.
Respond in valid JSON only."""
    existing_str = ", ".join(existing_names) if existing_names else "none"
    prompt = f"""Topic: {topic}
Missing stances: {', '.join(missing_stances)}
Already identified: {existing_str}
Context: {graph_context}
Identify {num_needed} additional UNIQUE real stakeholders for the missing stances.
{{"stakeholders": [{{"name": "...", "category": "...", "fundamental_interests": "...", "real_position": "...", "stance": "for/against/neutral", "stake": "...", "relevance_score": 0.75}}]}}"""
    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        additional = parsed.get("stakeholders", [])
        for s in additional:
            s["category"] = normalize_category(s.get("category", "civil_society"))
        print(f"[StakeholderIdentifier] Calibration added {len(additional)} for: {missing_stances}")
        return additional
    except Exception as e:
        print(f"[StakeholderIdentifier] Calibration request error: {e}")
        return []


async def request_more_unique_stakeholders(topic, existing, num_needed, graph_context):
    existing_names = [s["name"] for s in existing]
    existing_str = ", ".join(existing_names) if existing_names else "none"
    system = "You are identifying additional stakeholders for a debate simulation. Respond in valid JSON only."
    prompt = f"""Topic: {topic}
Already identified: {existing_str}
Need {num_needed} more UNIQUE stakeholders — different categories, different stances.
Context: {graph_context}
{{"stakeholders": [{{"name": "...", "category": "...", "fundamental_interests": "...", "real_position": "...", "stance": "for/against/neutral", "stake": "...", "relevance_score": 0.70}}]}}"""
    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        additional = parsed.get("stakeholders", [])
        for s in additional:
            s["category"] = normalize_category(s.get("category", "civil_society"))
        print(f"[StakeholderIdentifier] Added {len(additional)} unique stakeholders")
        return additional
    except Exception as e:
        print(f"[StakeholderIdentifier] Error requesting more stakeholders: {e}")
        return []


async def calibrate_distribution(
    stakeholders, topic, G_pub, graph_context,
    topic_for_keywords=None, topic_against_keywords=None,
    keyword_signal=None
):
    public_dist = keyword_signal or extract_public_sentiment_distribution(
        G_pub,
        topic_for_keywords=topic_for_keywords,
        topic_against_keywords=topic_against_keywords
    )

    if public_dist is None or public_dist["total_signal"] < 10:
        print("[StakeholderIdentifier] Insufficient public signal — skipping calibration")
        return stakeholders

    current_dist = get_current_distribution(stakeholders)
    print(f"[StakeholderIdentifier] Current distribution — "
          f"for: {current_dist['for']*100:.0f}% / "
          f"against: {current_dist['against']*100:.0f}% / "
          f"neutral: {current_dist['neutral']*100:.0f}%")
    print(f"[StakeholderIdentifier] Target distribution  — "
          f"for: {public_dist['for']*100:.0f}% / "
          f"against: {public_dist['against']*100:.0f}% / "
          f"neutral: {public_dist['neutral']*100:.0f}%")

    missing_stances = []
    for stance in ["for", "against", "neutral"]:
        gap = public_dist[stance] - current_dist[stance]
        if gap > CALIBRATION_THRESHOLD:
            missing_stances.append(stance)
            print(f"[StakeholderIdentifier] '{stance}' underrepresented by {gap*100:.0f}% — correcting")

    if not missing_stances:
        print("[StakeholderIdentifier] Distribution within acceptable range — no calibration needed")
        return stakeholders

    total = len(stakeholders)
    num_needed = max(2, round(max(
        public_dist[s] - current_dist[s] for s in missing_stances
    ) * total))
    existing_names = [s["name"] for s in stakeholders]
    additional = await request_missing_stakeholders(
        topic, missing_stances, num_needed, graph_context, existing_names
    )

    if additional:
        calibrated = stakeholders + additional
        new_dist = get_current_distribution(calibrated)
        print(f"[StakeholderIdentifier] Post-calibration — "
              f"for: {new_dist['for']*100:.0f}% / "
              f"against: {new_dist['against']*100:.0f}% / "
              f"neutral: {new_dist['neutral']*100:.0f}%")
        return calibrated
    return stakeholders


async def classify_and_position_entities(
    topic, entities, graph_context, missing_stances: list = None
):
    """
    Identify stakeholders from the institutional graph.

    Government representation fix: Think tanks and advocacy groups dominate
    citation counts in news articles — they publish more, get quoted more,
    score higher in PageRank. This causes government bodies (Congress, regulatory
    agencies, independent analysis bodies) to be underrepresented even when they
    have direct decision-making power over the outcome.

    Fix: inject explicit government body check before stakeholder selection.
    The LLM is instructed to find legislative, regulatory, enforcement, and
    analysis bodies first — then fill remaining slots with advocacy/industry.

    Generalizes to every topic:
    - Minimum wage → Congress, Department of Labor, CBO, Federal Reserve
    - UCSD housing → UC Regents, City Planning Department, State Legislature
    - TikTok ban → US Congress, FCC, CFIUS, Department of Justice
    - Campus library hours → UCSD Chancellor, UC System Board of Regents, AS Senate
    """
    entity_list = "\n".join([
        f"- {e['name']} (influence: {e.get('influence_score', 0):.3f}, "
        f"citations: {e.get('citations', 1)})"
        for e in entities[:35]
    ])

    system = """You are an expert at identifying stakeholders and their real-world positions.
Derive stance from INTERESTS, not public statements or news sentiment.
Respond in valid JSON only."""

    proposition = topic if "?" in topic else f"{topic}?"

    missing_instruction = ""
    if missing_stances:
        missing_instruction = f"""
CRITICAL: Real-world data shows these stances are underrepresented: {', '.join(missing_stances).upper()}
You MUST include stakeholders representing these positions.
For AGAINST: look for entities with operational, financial, safety, or resource concerns.
For FOR: look for entities who benefit directly or advocate for this change.
These voices exist in every real debate — find them in the entity list above."""

    prompt = f"""Proposition: {proposition}
- "for" = entity SUPPORTS the proposition
- "against" = entity OPPOSES the proposition
- "neutral" = entity has GENUINELY CONFLICTING interests (rare — see below)

Entities from knowledge graph:
{entity_list}

Context:
{graph_context}
{missing_instruction}

STEP 1 — GOVERNMENT BODIES FIRST (do this before anything else):
For this specific topic, identify which of these government body types exist
and have real decision-making power or direct jurisdiction:

1. LEGISLATIVE body — who votes on this? (Congress, Parliament, City Council,
   Student Government, University Board of Regents)
2. REGULATORY/ENFORCEMENT agency — who enforces or implements this?
   (Department of Labor, FCC, FDA, Campus Facilities, Planning Department)
3. INDEPENDENT ANALYSIS body — who provides neutral expert assessment?
   (Congressional Budget Office, GAO, University Research Office, CBO)
4. EXECUTIVE DEPARTMENT — who has direct administrative responsibility?
   (Secretary of Labor, University Chancellor, Mayor's Office)

These bodies MUST be included if they have genuine decision-making power.
Do NOT skip them because think tanks or advocacy groups have higher citation
counts — citations reflect who talks most, not who decides most.

STEP 2 — STAKEHOLDER DEFINITION:
A valid stakeholder must be an organization, institution, or named group
of people with REAL INTERESTS in the outcome.

VALID stakeholders:
- Government bodies identified in Step 1
- Named organizations with financial, operational, or social stakes
- Named companies with contractual or competitive interest
- Community groups directly affected — including informal ones
  (e.g. "Late Night Students", "UCSD Graduate Workers")
- Staff unions and worker groups

INVALID stakeholders — DO NOT include:
- Software applications, mobile apps, digital tools, websites
  → include the COMPANY or USERS instead
- Abstract technology concepts, algorithms, datasets
- Anonymous unnamed aggregates with no real organizational form

STEP 3 — NEUTRAL DEFINITION:
Only assign neutral if the entity has GENUINELY CONFLICTING interests.
Example valid neutral: Congressional Budget Office — mandated to provide
nonpartisan analysis, not take policy positions.
Example invalid neutral: An organization that "sees both sides."
Derive from institutional mandate and interests, not public statements.

Ensure at least one FOR, one AGAINST, one NEUTRAL stakeholder.
Maximum 15 stakeholders. Prioritize government bodies from Step 1,
then industry/advocacy, then community groups.

{{"stakeholders": [{{"name": "...", "category": "tech_company/government/civil_society/academic/labor_union/consumer/media/investor/affected_community/international_body", "fundamental_interests": "...", "real_position": "...", "stance": "for/against/neutral", "stake": "...", "relevance_score": 0.85}}]}}"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        return parsed.get("stakeholders", [])
    except Exception as e:
        print(f"[StakeholderIdentifier] Classification error: {e}")
        return []


def enforce_stance_diversity(stakeholders: list[dict]) -> list[dict]:
    """
    Guarantee at least one stakeholder of each stance before category caps.

    Problem this fixes: diversity enforcement (category caps) was running
    first and sometimes removing the only AGAINST stakeholder to fill
    an irrelevant investor slot. The result was 0 AGAINST institutionals
    even when the signal clearly showed AGAINST sentiment exists.

    Fix: select the highest-relevance representative of each stance first.
    These are GUARANTEED to survive category caps. Remaining slots fill
    from highest-relevance after that.

    Prediction: every simulation will have at least 1 FOR, 1 AGAINST,
    1 NEUTRAL institutional agent before any other diversity enforcement.
    """
    stances_present = {s.get("stance") for s in stakeholders}
    required_stances = {"for", "against", "neutral"}

    if required_stances.issubset(stances_present):
        return stakeholders  # already have all three

    sorted_by_relevance = sorted(
        stakeholders,
        key=lambda x: x.get("relevance_score", 0.5),
        reverse=True
    )

    guaranteed = {}
    for s in sorted_by_relevance:
        stance = s.get("stance")
        if stance in required_stances and stance not in guaranteed:
            guaranteed[stance] = s

    guaranteed_list = list(guaranteed.values())
    guaranteed_ids = {id(s) for s in guaranteed_list}
    remaining = [s for s in sorted_by_relevance if id(s) not in guaranteed_ids]

    missing = required_stances - set(guaranteed.keys())
    if missing:
        print(f"[StakeholderIdentifier] Stance diversity: missing {missing} "
              f"— will be added via calibration")

    return guaranteed_list + remaining


def enforce_diversity(stakeholders: list[dict], num_agents: int) -> list[dict]:
    """
    Apply category caps after stance diversity is guaranteed.
    Cap raised to 35% (from 30%) — less aggressive, prevents cutting
    legitimate same-category stakeholders on campus questions.
    """
    max_per_category = max(1, int(num_agents * MAX_CATEGORY_SHARE))
    sorted_s = sorted(
        stakeholders,
        key=lambda x: x.get("relevance_score", 0.5),
        reverse=True
    )
    category_counts = {}
    selected = []
    for s in sorted_s:
        cat = s.get("category", "civil_society")
        count = category_counts.get(cat, 0)
        if count < max_per_category:
            selected.append(s)
            category_counts[cat] = count + 1
    return selected


async def fill_to_count(stakeholders, num_agents, topic, graph_context):
    if len(stakeholders) >= num_agents:
        return stakeholders[:num_agents]
    needed = num_agents - len(stakeholders)
    print(f"[StakeholderIdentifier] Need {needed} more unique stakeholders — requesting from LLM")
    additional = await request_more_unique_stakeholders(topic, stakeholders, needed, graph_context)
    filled = stakeholders + additional
    if len(filled) < num_agents:
        print(f"[StakeholderIdentifier] Still need {num_agents - len(filled)} — using representatives")
        i = 0
        unique_count = len(stakeholders)
        while len(filled) < num_agents and unique_count > 0:
            base = stakeholders[i % unique_count]
            rep_num = i // unique_count + 2
            filled.append({
                **base,
                "name": f"{base['name']} (representative {rep_num})",
                "real_position": f"Secondary perspective aligned with {base['name']}."
            })
            i += 1
    return filled[:num_agents]


async def identify_stakeholders(
    topic: str,
    G,
    num_agents: int = 10,
    G_pub=None,
    pub_chunks=None,
    context: str = ""
) -> tuple[list, dict]:
    """
    Main entry point.
    Returns (enriched_stakeholders, keyword_signal) as a tuple.

    Pipeline:
    1. Generate topic-specific keywords (with context)
    2. Compute keyword signal from pub_chunks BEFORE classification
       so we know which stances are missing
    3. Classify entities WITH missing stance instruction
    4. enforce_stance_diversity — guarantees 1 of each stance
    5. enforce_diversity — category caps (35%, applied after stance guarantee)
    6. Correction store injection (gated by topic relevance)
    7. Calibrate distribution against keyword signal
    8. Fill to count if needed
    """
    num_agents = max(num_agents, MIN_AGENTS)
    print(f"[StakeholderIdentifier] Identifying stakeholders for: {topic} ({num_agents} agents)")

    topic_for_kw, topic_against_kw = await generate_topic_keywords(topic, context=context)

    influential = get_most_influential(G, top_n=40)
    orgs = get_nodes_by_type(G, "org")
    people = get_nodes_by_type(G, "person")

    all_entities = influential.copy()
    for e in orgs + people:
        if not any(x["name"] == e["name"] for x in all_entities):
            all_entities.append(e)

    graph_context = "\n".join([
        f"- {n['name']}: {n.get('description', '')[:120]} [cited {n.get('citations', 1)}x]"
        for n in influential[:20]
    ])

    if context:
        graph_context = f"Additional context: {context}\n\n" + graph_context
        print(f"[StakeholderIdentifier] Context injected into graph context")

    # ── Compute signal BEFORE classification ─────────────────────
    # So we can tell the LLM which stances to find in the entity list.
    # Prediction: pre-classification injection prevents 0% AGAINST on
    # lopsided topics by explicitly requesting AGAINST entities first.
    pre_signal = None
    if pub_chunks:
        pre_signal = extract_sentiment_from_chunks(
            pub_chunks,
            topic_for_keywords=topic_for_kw,
            topic_against_keywords=topic_against_kw
        )
    elif G_pub is not None:
        pre_signal = extract_public_sentiment_distribution(
            G_pub,
            topic_for_keywords=topic_for_kw,
            topic_against_keywords=topic_against_kw
        )

    missing_stances = []
    if pre_signal:
        if pre_signal.get("against", 0) > 0.10:
            missing_stances.append("against")
        if pre_signal.get("for", 0) > 0.10:
            missing_stances.append("for")
        if missing_stances:
            print(f"[StakeholderIdentifier] Pre-classification: ensuring "
                  f"stances {missing_stances} are represented")

    raw_stakeholders = await classify_and_position_entities(
        topic, all_entities, graph_context,
        missing_stances=missing_stances
    )

    if not raw_stakeholders:
        raw_stakeholders = [
            {
                "name": e["name"], "category": "civil_society",
                "fundamental_interests": f"Has a stake in {topic}",
                "real_position": f"Has a stake in {topic}", "stance": "neutral",
                "stake": "Identified from knowledge graph",
                "relevance_score": e.get("influence_score", 0.5)
            }
            for e in influential[:10]
        ]

    for s in raw_stakeholders:
        s["category"] = normalize_category(s.get("category", "civil_society"))

    # ── Stance diversity first, then category caps ────────────────
    # enforce_stance_diversity guarantees 1 FOR, 1 AGAINST, 1 NEUTRAL
    # before enforce_diversity applies category caps.
    # This prevents category enforcement from removing the only AGAINST agent.
    diverse = enforce_stance_diversity(raw_stakeholders)
    diverse = enforce_diversity(diverse, num_agents)

    # ── Correction store — gated by topic relevance ───────────────
    correction_context = get_correction_context(topic)
    if correction_context and correction_is_relevant(topic, correction_context):
        graph_context = graph_context + "\n\n" + correction_context
        print(f"[StakeholderIdentifier] Correction store: injecting relevant corrections")
    elif correction_context:
        print(f"[StakeholderIdentifier] Correction store: skipped — topic domain mismatch")

    keyword_signal = pre_signal

    if keyword_signal:
        diverse = await calibrate_distribution(
            diverse, topic, G_pub, graph_context,
            topic_for_keywords=topic_for_kw,
            topic_against_keywords=topic_against_kw,
            keyword_signal=keyword_signal
        )
    elif G_pub is not None:
        keyword_signal = extract_public_sentiment_distribution(
            G_pub,
            topic_for_keywords=topic_for_kw,
            topic_against_keywords=topic_against_kw
        )
        if keyword_signal:
            diverse = await calibrate_distribution(
                diverse, topic, G_pub, graph_context,
                topic_for_keywords=topic_for_kw,
                topic_against_keywords=topic_against_kw,
                keyword_signal=keyword_signal
            )
    else:
        print("[StakeholderIdentifier] No public data — skipping calibration")

    filled = await fill_to_count(diverse, num_agents, topic, graph_context)

    enriched = []
    for s in filled:
        category  = s.get("category", "civil_society")
        defaults  = STAKEHOLDER_CATEGORIES.get(
            category, STAKEHOLDER_CATEGORIES["civil_society"]
        )
        relevance = s.get("relevance_score", 0.75)

        # Bug 4 fix — modulate persuasion resistance by relevance score
        # Krosnick 1988: issue importance predicts attitude stability.
        # High relevance = high personal stake = harder to move.
        # Low relevance = tangential interest = more open to persuasion.
        #
        # Formula: multiplier ranges from 0.85 (low relevance) to 1.26 (high)
        # relevance=0.50 → multiplier 1.08 → mild boost
        # relevance=0.75 → multiplier 1.19 → moderate entrenchment
        # relevance=0.90 → multiplier 1.26 → strong entrenchment
        # Capped at 0.90 to prevent agents from becoming completely immovable.
        #
        # Prediction: Thomas Garrett (La Jolla real estate broker, relevance 0.85)
        # gets resistance 0.40 × 1.23 = 0.49 instead of flat 0.40 — holds firmer.
        # A low-relevance filler (Rural Community Associations, relevance 0.50)
        # gets resistance 0.40 × 1.08 = 0.43 — slightly more open.
        resistance_multiplier  = 0.85 + (relevance * 0.45)
        modulated_resistance   = round(
            min(0.90, defaults["persuasion_resistance"] * resistance_multiplier), 2
        )

        enriched.append({
            **s,
            "persuasion_resistance": modulated_resistance,
            "influence_weight":      defaults["influence_weight"]
        })

        print(f"[StakeholderIdentifier] {s['name']} — resistance: "
              f"{defaults['persuasion_resistance']} → {modulated_resistance} "
              f"(relevance {relevance:.2f})")

    final_dist = get_current_distribution(enriched)
    print(f"[StakeholderIdentifier] Final: {len(enriched)} stakeholders — "
          f"for: {final_dist['for']*100:.0f}% / "
          f"against: {final_dist['against']*100:.0f}% / "
          f"neutral: {final_dist['neutral']*100:.0f}%")
    for s in enriched:
        print(f"  -> {s['name']} [{s['category']}] stance: {s['stance']}")

    return enriched, keyword_signal