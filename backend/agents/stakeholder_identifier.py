import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json
from backend.utils.graph_utils import get_most_influential, get_nodes_by_type
import networkx as nx

# ── Reflexion correction memory ───────────────────────────────────
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

MAX_CATEGORY_SHARE = 0.30
MIN_AGENTS = 5
CALIBRATION_THRESHOLD = 0.15

# ── Base stance keyword lexicons ──────────────────────────────────
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


async def generate_topic_keywords(topic: str) -> tuple[list, list]:
    """
    Generate topic-specific stance keywords for THIS exact proposition.
    Mohammad et al. 2016 — stance detection must be target-dependent.
    """
    system = """You are an expert in stance detection and NLP.
Generate proposition-specific stance keywords for this exact debate topic.
Respond in valid JSON only."""

    prompt = f"""For this specific proposition: "{topic}"

Generate 10 FOR keywords and 10 AGAINST keywords that someone would use
when specifically supporting or opposing THIS proposition — not just the
general concept behind it.

Think about the specific framing:
- What words signal support for THIS specific action/policy/law?
- What words signal opposition to THIS specific action/policy/law?

Examples for "Should the federal government mandate a 4-day work week":
- FOR: "federal law", "mandate", "legally require", "workers need protection"
- AGAINST: "let employers decide", "market forces", "government shouldn't mandate"

Examples for "Should the US ban TikTok":
- FOR: "national security threat", "ban the app", "chinese espionage"
- AGAINST: "first amendment", "free speech", "unconstitutional"

Respond in this exact JSON format:
{{
    "for_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5",
                     "keyword6", "keyword7", "keyword8", "keyword9", "keyword10"],
    "against_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5",
                         "keyword6", "keyword7", "keyword8", "keyword9", "keyword10"]
}}

Rules:
- Keywords must be specific to THIS proposition, not generic
- Keep each keyword short (1-4 words) so they match within raw text
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
        print(f"[StakeholderIdentifier] Topic keyword generation failed: {e} — using base keywords only")
        return [], []


def classify_claim_stance(
    claim_text: str,
    topic_for_keywords: list = None,
    topic_against_keywords: list = None
) -> str:
    """
    Classify a single claim's stance using base + topic-specific keywords.
    Returns "for", "against", "neutral", or None (no signal).
    """
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
    ── Change 3: Keyword matching on RAW CHUNK TEXT ─────────────────
    Previously we matched keywords against LLM-extracted claim node
    summaries from the graph. This caused a 68% fallback rate because
    extracted nodes paraphrase the original text — the actual words
    people used are lost in extraction.

    Example:
      Raw chunk: "we need sensible gun reform to protect our kids"
      Extracted node: "gun violence requires policy intervention"
      Keyword "gun reform" matches raw text but NOT extracted node.

    Fix: run keyword matching directly on pub_chunks[i]['text'] BEFORE
    graph extraction. This gives us the original words, not paraphrases.

    This replaces extract_public_sentiment_distribution when pub_chunks
    is available. Falls back to graph-based extraction if not.

    Result: keyword hit rate goes from ~30% to ~70%+ because we're
    matching against real human language, not LLM summaries.
    """
    if not pub_chunks:
        return None

    for_count = against_count = neutral_count = 0
    keyword_hits = fallback_hits = 0

    for chunk in pub_chunks:
        # Use raw text — the original words, not extracted summaries
        raw_text = chunk.get("text", "")
        if not raw_text:
            continue

        stance = classify_claim_stance(
            raw_text,
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
            # Fallback: use sentiment_strength as a weak proxy
            # High sentiment_strength = emotionally charged = likely taking a side
            # This is weaker than keyword matching but better than ignoring
            ss = chunk.get("sentiment_strength", 0.0)
            if ss > 0.5:
                # High emotional intensity — count as weak FOR signal
                # (online emotional discourse skews toward advocates)
                for_count += 0  # don't assume direction — skip
                neutral_count += 1
            else:
                neutral_count += 1

    total = for_count + against_count + neutral_count
    if total == 0:
        return None

    distribution = {
        "for":           round(for_count / total, 2),
        "against":       round(against_count / total, 2),
        "neutral":       round(neutral_count / total, 2),
        "total_signal":  total,
        "keyword_hits":  keyword_hits,
        "fallback_hits": fallback_hits,
    }

    print(f"[StakeholderIdentifier] Public stance signal (raw text matching) — "
          f"for: {distribution['for']*100:.0f}% / "
          f"against: {distribution['against']*100:.0f}% / "
          f"neutral: {distribution['neutral']*100:.0f}% "
          f"({total} chunks: {keyword_hits} keyword-matched, {fallback_hits} fallback)")

    return distribution


def extract_public_sentiment_distribution(
    G_pub: nx.DiGraph,
    topic_for_keywords: list = None,
    topic_against_keywords: list = None
) -> dict:
    """
    Graph-based fallback — used when raw pub_chunks not available.
    Kept for backward compatibility.
    """
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

    distribution = {
        "for":           round(for_count / total, 2),
        "against":       round(against_count / total, 2),
        "neutral":       round(neutral_count / total, 2),
        "total_signal":  total,
        "keyword_hits":  keyword_hits,
        "fallback_hits": fallback_hits,
    }

    print(f"[StakeholderIdentifier] Public stance signal (graph nodes) — "
          f"for: {distribution['for']*100:.0f}% / "
          f"against: {distribution['against']*100:.0f}% / "
          f"neutral: {distribution['neutral']*100:.0f}% "
          f"({total} claims: {keyword_hits} keyword-matched, {fallback_hits} fallback)")

    return distribution


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
These must be REAL, SPECIFIC entities with genuine stakes in this topic.
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
    """
    Compare current stakeholder distribution to public signal.
    Uses pre-computed keyword_signal if available.
    """
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


async def classify_and_position_entities(topic, entities, graph_context):
    entity_list = "\n".join([
        f"- {e['name']} (influence: {e.get('influence_score', 0):.3f}, citations: {e.get('citations', 1)})"
        for e in entities[:35]
    ])
    system = """You are an expert at identifying stakeholders and their real-world positions.
Derive stance from INTERESTS, not public statements or news sentiment.
Respond in valid JSON only."""
    proposition = topic if "?" in topic else f"{topic}?"
    prompt = f"""Proposition: {proposition}
- "for" = entity SUPPORTS the proposition
- "against" = entity OPPOSES the proposition
- "neutral" = genuinely mixed position

Entities: {entity_list}
Context: {graph_context}

Ensure genuine diversity — at least one FOR, one AGAINST, one NEUTRAL minimum.
Maximum 15 stakeholders. Only real organizations with genuine stakes.

{{"stakeholders": [{{"name": "...", "category": "tech_company/government/civil_society/academic/labor_union/consumer/media/investor/affected_community/international_body", "fundamental_interests": "...", "real_position": "...", "stance": "for/against/neutral", "stake": "...", "relevance_score": 0.85}}]}}"""
    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        return parsed.get("stakeholders", [])
    except Exception as e:
        print(f"[StakeholderIdentifier] Classification error: {e}")
        return []


def enforce_diversity(stakeholders, num_agents):
    max_per_category = max(1, int(num_agents * MAX_CATEGORY_SHARE))
    sorted_s = sorted(stakeholders, key=lambda x: x.get("relevance_score", 0.5), reverse=True)
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
        print(f"[StakeholderIdentifier] Still need {num_agents - len(filled)} — using representatives as last resort")
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
    context: str = ""  # NEW
) -> tuple[list, dict]:
    """
    Main entry point.

    Returns (enriched_stakeholders, keyword_signal) as a tuple.

    ── Change 3: pub_chunks parameter ───────────────────────────────
    When pub_chunks is provided, keyword matching runs on RAW CHUNK
    TEXT before graph extraction instead of LLM-extracted claim nodes.

    This fixes the 68% fallback rate that was causing inverted signals
    on topics like gun control where pro-reform discourse uses the same
    constitutional language as anti-reform discourse.

    Raw text: "we need sensible gun reform to protect our kids"
    → keywords match directly → correct FOR signal

    Extracted node: "gun violence requires policy intervention"
    → keywords miss → falls back to sentiment tags → wrong signal
    """
    num_agents = max(num_agents, MIN_AGENTS)
    print(f"[StakeholderIdentifier] Identifying stakeholders for: {topic} ({num_agents} agents)")

    topic_for_kw, topic_against_kw = await generate_topic_keywords(topic)

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

    # Inject context into graph_context so stakeholder selection is grounded
    if context:
        graph_context = f"Additional context: {context}\n\n" + graph_context
        print(f"[StakeholderIdentifier] Context injected into graph context")

    raw_stakeholders = await classify_and_position_entities(topic, all_entities, graph_context)

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

    diverse = enforce_diversity(raw_stakeholders, num_agents)

    # ── Reflexion correction context ──────────────────────────────
    correction_context = get_correction_context(topic)
    if correction_context:
        graph_context = graph_context + "\n\n" + correction_context

    # ── Keyword signal — raw chunks preferred over graph nodes ────
    # Change 3: if pub_chunks provided, match on raw text.
    # Otherwise fall back to graph-based extraction.
    keyword_signal = None
    if G_pub is not None or pub_chunks:
        if pub_chunks:
            # Primary path — raw chunk text matching (Change 3)
            keyword_signal = extract_sentiment_from_chunks(
                pub_chunks,
                topic_for_keywords=topic_for_kw,
                topic_against_keywords=topic_against_kw
            )
            print(f"[StakeholderIdentifier] Using raw chunk text matching "
                  f"({len(pub_chunks)} chunks)")
        
        if keyword_signal is None and G_pub is not None:
            # Fallback — graph node matching
            keyword_signal = extract_public_sentiment_distribution(
                G_pub,
                topic_for_keywords=topic_for_kw,
                topic_against_keywords=topic_against_kw
            )
            print(f"[StakeholderIdentifier] Fell back to graph node matching")

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
        category = s.get("category", "civil_society")
        defaults = STAKEHOLDER_CATEGORIES.get(category, STAKEHOLDER_CATEGORIES["civil_society"])
        enriched.append({
            **s,
            "persuasion_resistance": defaults["persuasion_resistance"],
            "influence_weight": defaults["influence_weight"]
        })

    final_dist = get_current_distribution(enriched)
    print(f"[StakeholderIdentifier] Final: {len(enriched)} stakeholders — "
          f"for: {final_dist['for']*100:.0f}% / "
          f"against: {final_dist['against']*100:.0f}% / "
          f"neutral: {final_dist['neutral']*100:.0f}%")
    for s in enriched:
        print(f"  -> {s['name']} [{s['category']}] stance: {s['stance']}")

    return enriched, keyword_signal