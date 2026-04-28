"""
backend/dtc_v3/discussion.py — AI buyer panel generator for v3-lite.

CRITICAL INVARIANTS:
  1. Discussion MUST NOT change the forecast number.
  2. Same input -> same output (deterministic via seed + cache).
  3. If LLM fails or returns invalid JSON, fallback to deterministic template.
  4. Never claim agents are "real buyers" — they are AI buyer personas
     explaining a comparable-brand forecast.
  5. agent_count must be in {20, 50}.
  6. Default mode is "template" (fast, free, diverse).
     LLM is only used when mode="llm" is explicitly passed.
"""

from __future__ import annotations
import os
import json
import hashlib
from typing import Any

from backend.dtc_v3.bucket_allocator import allocate_buckets
from backend.dtc_v3.persona_generator import select_personas_for_product
from backend.dtc_v3.llm_dialogue_enricher import enrich_with_llm_dialogue

DISCUSSION_VERSION = "discussion_v1"
ALLOWED_AGENT_COUNTS = (20, 50)
DEFAULT_AGENT_COUNT = 20
ALLOWED_MODES = ("template", "llm")
DEFAULT_MODE = "template"

_DISCUSSION_CACHE: dict[str, dict] = {}


# ────────────────────────────────────────────────────────────────────
# Seed
# ────────────────────────────────────────────────────────────────────
def generate_seed(product: dict, forecast: dict, agent_count: int, mode: str = DEFAULT_MODE) -> str:
    """Deterministic seed for caching + LLM reproducibility."""
    canonical_payload = {
        "product": product,
        "forecast_core": {
            "trial_rate_median": forecast.get("trial_rate", {}).get("median"),
            "confidence": forecast.get("confidence"),
            "verdict": forecast.get("verdict"),
            "prior_source": forecast.get("diagnostics", {}).get("prior_source"),
            "version": forecast.get("version", "v3-lite"),
        },
        "agent_count": agent_count,
        "mode": mode,
        "discussion_version": DISCUSSION_VERSION,
    }
    canonical = json.dumps(canonical_payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ────────────────────────────────────────────────────────────────────
# Public entrypoint
# ────────────────────────────────────────────────────────────────────
def generate_discussion(
    product: dict,
    forecast: dict,
    agent_count: int = DEFAULT_AGENT_COUNT,
    mode: str = DEFAULT_MODE,
) -> dict:
    """
    Returns {"agent_panel": {...}}.

    mode="template" (default): deterministic, fast, diverse template path.
    mode="llm": one structured LLM call. Falls back to template on any failure.

    Never raises (other than invalid input).
    """
    if agent_count not in ALLOWED_AGENT_COUNTS:
        raise ValueError(f"agent_count must be in {ALLOWED_AGENT_COUNTS}, got {agent_count}")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be in {ALLOWED_MODES}, got {mode}")
    if not isinstance(product, dict) or not product:
        raise ValueError("product must be a non-empty dict")
    if not isinstance(forecast, dict) or not forecast:
        raise ValueError("forecast must be a non-empty dict")

    seed = generate_seed(product, forecast, agent_count, mode)

    cached = _DISCUSSION_CACHE.get(seed)
    if cached is not None:
        return cached

    # Build template panel first — structural ground truth.
    panel = _template_panel(product, forecast, agent_count, seed)
    panel_source = "template"

    # If mode=llm, overlay LLM-generated narrative onto the template panel.
    # On any LLM failure, enricher returns None and we keep the template panel.
    if mode == "llm":
        enriched = enrich_with_llm_dialogue(
            panel=panel, product=product, forecast=forecast, seed=seed,
        )
        if enriched is not None:
            panel = enriched
            panel_source = "llm"

    panel["seed"] = seed
    panel["agent_count"] = agent_count
    panel["mode"] = panel_source
    panel["coverage_warning"] = _coverage_warning(forecast)

    response = {"agent_panel": panel}
    _DISCUSSION_CACHE[seed] = response
    return response


def clear_cache() -> None:
    """Test helper."""
    _DISCUSSION_CACHE.clear()


# ────────────────────────────────────────────────────────────────────
# Coverage warning copy
# ────────────────────────────────────────────────────────────────────
def _coverage_warning(forecast: dict) -> str:
    confidence = (forecast.get("confidence") or "").lower()
    coverage_tier = (forecast.get("diagnostics", {}).get("coverage_tier") or "").lower()
    prior_source = (forecast.get("diagnostics", {}).get("prior_source") or "").lower()

    is_low_or_fallback = (
        confidence == "low"
        or coverage_tier in ("weak", "thin")
        or prior_source.startswith("fallback")
    )
    if is_low_or_fallback:
        return (
            "Comparable coverage is weak for this product subtype, so this buyer-panel "
            "discussion is directional — it explains the forecast's logic, not validated buyer behavior."
        )
    return ""


# ────────────────────────────────────────────────────────────────────
# LLM path (only fires when mode="llm")
# ────────────────────────────────────────────────────────────────────
def _try_llm_panel(product: dict, forecast: dict, agent_count: int, seed: str) -> dict | None:
    """One structured LLM call, low temperature, strict JSON. Returns None on any failure."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        client = OpenAI(api_key=api_key, max_retries=0, timeout=20)
        system_prompt, user_prompt = _build_prompts(product, forecast, agent_count)
        seed_int = int(seed[:15], 16)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            seed=seed_int,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=20,
        )

        raw = response.choices[0].message.content
        if not raw:
            return None

        parsed = json.loads(raw)
        return _validate_llm_panel(parsed, agent_count)

    except Exception:
        return None


def _build_prompts(product: dict, forecast: dict, agent_count: int) -> tuple[str, str]:
    system = (
        "You are generating a panel of AI buyer personas to explain a comparable-brand "
        "trial-rate forecast for a DTC product. Strict rules:\n"
        "1. DO NOT change or recompute the forecast number — it is fixed.\n"
        "2. DO NOT claim these are real buyers. They are AI personas explaining the forecast.\n"
        "3. DO NOT make causal guarantees ('will lift trial', 'guarantees adoption').\n"
        "4. Use the comparable-brand anchors as evidence for stances.\n"
        "5. Stances are 'for', 'against', or 'neutral'.\n"
        "6. Score is 0.0-1.0 (purchase intent). 'for' agents score >= 0.55, 'against' <= 0.4, 'neutral' between.\n"
        "7. Round summaries describe what the panel discussed, not what they decided about the rate.\n"
        "8. top_drivers and top_objections are short phrases (3-7 words each).\n"
        "9. consensus is one sentence summarizing where the panel landed.\n"
        "10. Return strict JSON matching the schema. No prose outside JSON.\n"
    )

    schema_text = (
        '{\n'
        '  "rounds": [\n'
        '    {"round": 1, "title": "First Reaction", "summary": "...", "for_count": int, "neutral_count": int, "against_count": int, "avg_score": 0.0-1.0},\n'
        '    {"round": 2, "title": "Comparable Comparison", "summary": "...", "for_count": int, "neutral_count": int, "against_count": int, "avg_score": 0.0-1.0},\n'
        '    {"round": 3, "title": "Consensus", "summary": "...", "for_count": int, "neutral_count": int, "against_count": int, "avg_score": 0.0-1.0}\n'
        '  ],\n'
        f'  "agents": [exactly {agent_count} agents, each with: id (agent_01..agent_{agent_count:02d}), segment, profile, stance, score, reason, top_objection],\n'
        '  "top_drivers": [4-6 short phrases],\n'
        '  "top_objections": [4-6 short phrases],\n'
        '  "most_receptive_segment": "short phrase",\n'
        '  "winning_message": "one sentence",\n'
        '  "risk_factors": [3-5 short phrases],\n'
        '  "consensus": "one sentence"\n'
        '}'
    )

    forecast_summary = {
        "trial_rate_pct": forecast.get("trial_rate", {}).get("percentage"),
        "confidence": forecast.get("confidence"),
        "verdict": forecast.get("verdict"),
        "anchored_on": forecast.get("anchored_on", [])[:6],
        "top_drivers_seed": forecast.get("top_drivers", []),
        "top_objections_seed": forecast.get("top_objections", []),
        "why_might_be_wrong": forecast.get("why_might_be_wrong", []),
    }

    product_summary = {
        "name": product.get("product_name") or product.get("name"),
        "description": product.get("description"),
        "price": product.get("price"),
        "category": product.get("category"),
        "demographic": product.get("demographic"),
        "competitors": product.get("competitors", []),
    }

    user = (
        f"PRODUCT:\n{json.dumps(product_summary, indent=2)}\n\n"
        f"FORECAST (do not change):\n{json.dumps(forecast_summary, indent=2)}\n\n"
        f"Generate exactly {agent_count} agents across diverse buyer segments. "
        f"Stance distribution should reflect the forecast's trial rate and confidence. "
        f"Return JSON matching this schema:\n{schema_text}"
    )

    return system, user


def _validate_llm_panel(parsed: Any, agent_count: int) -> dict | None:
    if not isinstance(parsed, dict):
        return None
    required = ("rounds", "agents", "top_drivers", "top_objections",
                "most_receptive_segment", "winning_message", "risk_factors", "consensus")
    if not all(k in parsed for k in required):
        return None
    if not isinstance(parsed["rounds"], list) or len(parsed["rounds"]) != 3:
        return None
    if not isinstance(parsed["agents"], list):
        return None
    if len(parsed["agents"]) != agent_count:
        return None
    for a in parsed["agents"]:
        if not isinstance(a, dict):
            return None
        for k in ("id", "segment", "profile", "stance", "score", "reason", "top_objection"):
            if k not in a:
                return None
        if a["stance"] not in ("for", "against", "neutral"):
            return None
    return parsed


# ────────────────────────────────────────────────────────────────────
# Deterministic template (default mode)
# ────────────────────────────────────────────────────────────────────
SEGMENT_TEMPLATES = [
    ("Health optimizer", "Tracks sleep, supplements, recovery. Reads labels."),
    ("Cost-conscious skeptic", "Compares unit price. Questions DTC premiums."),
    ("Subscription veteran", "Has 4+ active DTC subs. Cancels often."),
    ("Retail-first shopper", "Trusts what's on Costco / Target shelves."),
    ("Early adopter", "Tries new launches within 60 days."),
    ("Wellness routine builder", "Stacks 3+ daily supplements."),
    ("Influencer-led buyer", "Buys what creators they trust use."),
    ("Skeptic with a need", "Has the problem, doubts the solution."),
    ("Brand-loyal incumbent user", "Currently uses a comparable brand."),
    ("Convenience prioritizer", "Pays for fast shipping, hates lock-in."),
    ("Functional ingredient nerd", "Reads studies. Knows actives."),
    ("Gift / household buyer", "Buys for spouse / family."),
    ("Reddit researcher", "Lurks niche subreddits before buying."),
    ("Aesthetic-driven buyer", "Cares how it looks on the shelf / counter."),
    ("Trial-then-bulk buyer", "Buys small to test, large if it works."),
    ("Price-anchored", "Has a strong reference price in their head."),
    ("Performance-focused athlete", "Buys for measurable outcomes."),
    ("Time-pressed parent", "Buys what works fast, no learning curve."),
    ("Eco / sustainable buyer", "Reads packaging. Cares about waste."),
    ("Wait-and-see buyer", "Lets others try first, buys after reviews."),
    ("Urban professional", "Lifestyle fit > price."),
    ("Suburban household", "Buys family-size, retail-first."),
    ("Beginner / first-timer", "Doesn't know category norms yet."),
    ("Aging into category", "New life stage triggered the search."),
    ("Quitting another category", "Replacing an old habit (alcohol, soda, coffee)."),
    ("Performance plateau", "Tried 2+ alternatives, looking for next."),
    ("Recovering from injury / setback", "Buying for a specific bounce-back."),
    ("Lifestyle adjacent", "Doesn't need it, but it fits the identity."),
    ("Flash-sale watcher", "Only buys on promo."),
    ("Bundle hunter", "Wants stacked discounts, never single-item."),
    ("Routine experimenter", "Cycles products every 30-60 days."),
    ("Loyalist of incumbent", "Won't switch unless 2x value."),
    ("Brand-agnostic value buyer", "Whatever's cheapest with decent reviews."),
    ("Hardcore optimizer", "Tracks everything; looking for marginal gain."),
    ("Casual curious", "Saw an ad, half-interested."),
    ("Lapsed user", "Used something similar 2 yrs ago, dropped it."),
    ("New parent / lifestage shift", "Recently changed needs."),
    ("Returning to category", "Was out, now back in."),
    ("Aspirational buyer", "Wants the lifestyle the brand sells."),
    ("Guilt buyer", "Knows they should, hasn't yet."),
    ("DIY-er", "Could make it themselves, weighing convenience."),
    ("Group buyer", "Coordinates with friends / partner."),
    ("Brand-new to wellness", "First-time category entrant."),
    ("Switching from prescription / Rx-adjacent", "OTC alternative seeker."),
    ("Skeptical journalist type", "Will read all the fine print."),
    ("Practical experimenter", "Buys 1, tests, then commits or drops."),
    ("Late-night impulse buyer", "Mobile, social-driven."),
    ("Workplace gifter", "Buys for team / coworkers."),
    ("Returns-and-refunds savvy", "Tries everything, sends back what doesn't fit."),
    ("Price-tier downgrader", "Was premium, now shopping mid-tier."),
]

FOR_REASONS = [
    "Anchor brands like {anchor} prove the category has real pull at this price.",
    "Distribution and category fit match what worked for {anchor} — feels grounded.",
    "Price is in line with what I already pay for {anchor} or its peers.",
    "Comparable-brand evidence is strong; this is the type of launch I'd back.",
    "If {anchor} hits the trial rate it does, this product is in striking range.",
    "I'd try this once: the comparable-brand math is reasonable and the price isn't crazy.",
    "Category timing works — buyers are already paying for {anchor}-style products.",
    "Differentiation is small but real, and the price gives it room to breathe.",
    "Solid bet given the comparable-brand anchors. I'd put it in my next order.",
    "The retail/DTC distribution lines up; that's usually what makes or breaks trial.",
]

NEUTRAL_REASONS = [
    "Interested but waiting on independent reviews before pulling the trigger.",
    "Comparable to {anchor}, but I'd want a clearer differentiation story first.",
    "Price is fine; the question is whether it earns its spot in my routine.",
    "Could go either way. I'd watch how the first buyer cohort reacts.",
    "Not against it — just need a reason to switch from what already works.",
    "Decent fit with {anchor}-style adoption, but I haven't seen enough of it yet.",
    "Open to it, but I'd want a sample or money-back guarantee before committing.",
    "Comparable-brand anchors look reasonable. Real test is whether it sticks.",
    "I'd consider it if the brand showed up consistently in my feed for 3 months.",
    "On the fence — the category is right but the moment isn't urgent for me.",
]

AGAINST_REASONS = [
    "I already use {anchor}; switching cost is real and the upside isn't obvious.",
    "Price feels high for what looks like another {anchor} variant.",
    "Subscription fatigue — I've cancelled enough of these to know my pattern.",
    "Differentiation is too thin against incumbent {anchor}-style brands.",
    "Retail dominance of {anchor} makes a DTC-first launch hard for me to bother with.",
    "Marketing claim doesn't match what comparable brands actually deliver.",
    "I'd rather wait for {anchor} to release a similar product than try this one.",
    "Friction outweighs upside; one more thing to manage in my routine.",
    "Existing alternatives serve me — I'd need a 2x reason, not a 1.1x reason.",
    "The category is crowded; I don't have shelf-space (mental or physical) for another.",
]

FOR_OBJECTIONS = [
    "Want to see one trial cycle before fully committing.",
    "Need confirmation it lasts longer than {anchor} promised.",
    "Would prefer single-purchase before subscription.",
    "Hope shipping doesn't slip during peak season.",
    "Want at least 60 days of consistent supply.",
    "Need to verify retail pricing matches DTC.",
    "Want a clear cancellation policy upfront.",
    "Hope the formulation matches the marketing claims.",
    "Want a money-back guarantee for first order.",
    "Need to confirm it's compatible with my current routine.",
]

NEUTRAL_OBJECTIONS = [
    "Not sure it's better than what I'm already using.",
    "Need more independent reviews before deciding.",
    "Differentiation from {anchor} isn't yet clear to me.",
    "Want to see longer-term user data first.",
    "Price-to-value ratio is borderline — could go either way.",
    "Would prefer to see it on a retail shelf before committing.",
    "Need a clearer use-case for my specific situation.",
    "Brand authenticity is hard to judge this early.",
    "Want to see how customer service handles complaints.",
    "Need more time to evaluate vs. existing alternatives.",
]

AGAINST_OBJECTIONS = [
    "Subscription fatigue / no clear reason to switch.",
    "Loyal to {anchor}; switching cost outweighs benefit.",
    "Price is too high for marginal differentiation.",
    "Crowded category — too many lookalikes already.",
    "No retail presence yet — DTC-only is friction.",
    "Marketing claims feel exaggerated vs. comparable brands.",
    "I'd rather stick with what's proven over what's new.",
    "Doesn't solve a problem I have right now.",
    "Reviews aren't strong enough to justify a try.",
    "Brand story doesn't differentiate from {anchor}.",
]


WHAT_CHANGE_MIND_FOR = [
    "A price increase or shrink in pack size could push me out.",
    "If repeat-purchase numbers stay weak, I'd reconsider.",
    "Bad reviews from buyers I trust would make me pause.",
    "Differentiation versus {anchor} starts to blur and I'm out.",
    "If a better incumbent option launches, I'd switch.",
    "Cancellation friction would sour me fast.",
    "Stockouts and supply issues would erode trust.",
    "Quality slippage between batches would lose me.",
    "If shipping or unboxing degrades, I'd churn.",
    "Marketing claims that don't match the actual product would do it.",
    "Switching cost from existing routine creeping up.",
    "Customer-service response times getting worse.",
    "Subscription pricing changes without warning.",
    "Better-tasting or better-feeling competitor at same price.",
    "Loss of the original team-led brand voice.",
]

WHAT_CHANGE_MIND_NEUTRAL = [
    "A genuine starter offer or trial pack would push me to buy.",
    "More proof from real long-term users would tip me.",
    "Trusted reviews from someone in my world would matter.",
    "A clearer head-to-head comparison with {anchor} would help.",
    "Lower-risk trial like money-back or first-month-free.",
    "A friend with similar needs giving a real recommendation.",
    "Clinical or independent third-party data would be persuasive.",
    "A smaller starter SKU at a lower entry price.",
    "Stronger differentiation story versus {anchor} on a single dimension.",
    "Retail availability so I can pick it up without committing online.",
    "A clearer cancellation/return policy upfront.",
    "Bundle pricing with something else I already buy.",
    "Editorial reviews from outlets I read.",
    "A creator I follow doing a real (not paid) review.",
    "Clearer evidence the use case matches my actual day.",
]

WHAT_CHANGE_MIND_AGAINST = [
    "Materially stronger proof of effect, not just claims.",
    "A meaningfully lower price would force me to relook.",
    "A trusted brand partnership or distribution deal.",
    "Clearer differentiation on something I actually care about.",
    "A genuine risk-free trial — full money back, no friction.",
    "Multiple buyers in my circle reporting it actually works.",
    "Independent testing or peer-reviewed data.",
    "Removing the part that turns me off (price/format/claim).",
    "Distribution into a channel I already shop.",
    "A version sized or priced for someone in my situation.",
    "Direct-comparison test against {anchor} I can verify myself.",
    "Significant change in formulation or feature set.",
    "Strong return policy I can use without arguing.",
    "Real long-tail customer voice, not just early hype.",
    "Better fit with my actual routine, not the marketing one.",
]



def _pick_variant(variants: list[str], seed_int: int, agent_index: int, anchor: str) -> str:
    """Deterministic pick by (seed_int + index) % len; substitutes {anchor}."""
    template = variants[(seed_int + agent_index) % len(variants)]
    return template.replace("{anchor}", anchor)


# ────────────────────────────────────────────────────────────────────
# Phase 3b — Per-agent enrichment (score arcs, journeys, round responses)
# ────────────────────────────────────────────────────────────────────


def _score_to_verdict(score_10: float) -> str:
    """7.0-10 = BUY, 4.6-6.9 = CONSIDERING, 1.0-4.5 = WON\'T BUY."""
    if score_10 >= 7.0:
        return "BUY"
    if score_10 >= 4.6:
        return "CONSIDERING"
    return "WON\'T BUY"


def _verdict_to_stance(verdict: str) -> str:
    return {"BUY": "for", "CONSIDERING": "neutral", "WON\'T BUY": "against"}.get(
        verdict, "neutral"
    )


ROUND_FRAMES = {
    1: {
        "for":     "First impression: {body}",
        "neutral": "First impression: {body}",
        "against": "First impression: {body}",
    },
    2: {
        "for":     "Compared with {anchor} and similar brands: {body}",
        "neutral": "Stacked against {anchor}: {body}",
        "against": "Versus {anchor}: {body}",
    },
    3: {
        "for":     "Final read: {body} I\'d be willing to try it.",
        "neutral": "Final read: {body} Still on the fence.",
        "against": "Final read: {body} Not for me right now.",
    },
}


KEY_MOMENT_TEMPLATES = {
    "steady_buy":         "Round 1: immediately saw strong fit with their existing buying habits.",
    "considering_to_buy": "Round 2: competitor comparison made the product feel more differentiated.",
    "steady_considering": "Round 3: remained interested but wanted more proof before committing.",
    "drifted_into_considering_from_buy":  "Round 2: price or trust concerns reduced initial enthusiasm.",
    "drifted_into_considering_from_wont": "Round 3: softened slightly after seeing a clearer use case.",
    "considering_to_wont": "Round 2: trust or differentiation concerns became stronger.",
    "steady_wont":         "Round 1: did not see enough reason to switch from current alternatives.",
    "hardcore":            "Round 1: strongly preferred existing solutions and resisted later arguments.",
}


_ARC_SCORE_RANGES = {
    "steady_buy":                          (7.0, 8.5, 7.2, 9.5),
    "considering_to_buy":                  (5.0, 6.5, 7.0, 8.8),
    "steady_considering":                  (4.8, 6.7, 4.6, 6.9),
    "drifted_into_considering_from_buy":   (7.0, 7.8, 5.0, 6.8),
    "drifted_into_considering_from_wont":  (3.8, 4.5, 5.0, 6.8),
    "steady_wont":                         (2.5, 4.5, 1.8, 4.5),
    "considering_to_wont":                 (4.8, 6.2, 2.5, 4.5),
    "hardcore":                            (1.0, 3.0, 1.0, 3.0),
}


_ARC_INITIAL_VERDICT = {
    "steady_buy":                          "BUY",
    "considering_to_buy":                  "CONSIDERING",
    "steady_considering":                  "CONSIDERING",
    "drifted_into_considering_from_buy":   "BUY",
    "drifted_into_considering_from_wont":  "WON\'T BUY",
    "steady_wont":                         "WON\'T BUY",
    "considering_to_wont":                 "CONSIDERING",
    "hardcore":                            "WON\'T BUY",
}


_ARC_FINAL_VERDICT = {
    "steady_buy":                          "BUY",
    "considering_to_buy":                  "BUY",
    "steady_considering":                  "CONSIDERING",
    "drifted_into_considering_from_buy":   "CONSIDERING",
    "drifted_into_considering_from_wont":  "CONSIDERING",
    "steady_wont":                         "WON\'T BUY",
    "considering_to_wont":                 "WON\'T BUY",
    "hardcore":                            "WON\'T BUY",
}


_ARC_SHIFT_REASON = {
    "steady_buy":                          "Read the product as a strong fit from the start.",
    "considering_to_buy":                  "Competitor comparison reduced uncertainty about the value.",
    "steady_considering":                  "Stayed interested but wanted more proof.",
    "drifted_into_considering_from_buy":   "Initial enthusiasm cooled after closer inspection.",
    "drifted_into_considering_from_wont":  "Use case became clearer over the discussion.",
    "steady_wont":                         "Saw no compelling reason to switch from existing alternatives.",
    "considering_to_wont":                 "Trust and differentiation concerns hardened the position.",
    "hardcore":                            "Strongly preferred existing solutions and did not move.",
}


def _pick_arc_type(bucket_label: str, seed_int: int, agent_idx: int) -> str:
    """Per friend's Option B distribution. Deterministic from (seed, idx)."""
    sel = (seed_int + agent_idx * 1009) % 100
    if bucket_label == "buy":
        return "steady_buy" if sel < 50 else "considering_to_buy"
    if bucket_label == "considering":
        if sel < 80:
            return "steady_considering"
        return ("drifted_into_considering_from_buy"
                if sel < 90
                else "drifted_into_considering_from_wont")
    if sel < 60:
        return "steady_wont"
    if sel < 90:
        return "considering_to_wont"
    return "hardcore"


def _deterministic_in_range(low: float, high: float, seed_int: int,
                             agent_idx: int, salt: int) -> float:
    """Pick a float in [low, high] deterministically."""
    span = high - low
    if span <= 0:
        return low
    step = ((seed_int + agent_idx * 31 + salt * 17) % 100) / 100.0
    return low + span * step


def _what_would_change_mind(stance: str, anchor: str, seed_int: int,
                             agent_idx: int) -> str:
    """Stance-specific template (not cross-pool — each stance has its own list)."""
    pool = {
        "for":     WHAT_CHANGE_MIND_FOR,
        "neutral": WHAT_CHANGE_MIND_NEUTRAL,
        "against": WHAT_CHANGE_MIND_AGAINST,
    }.get(stance, WHAT_CHANGE_MIND_NEUTRAL)
    template = pool[(seed_int + agent_idx + 7) % len(pool)]
    return template.replace("{anchor}", anchor)


def _round_response(stance: str, round_num: int, anchor: str, seed_int: int,
                     agent_idx: int) -> dict:
    pool = {
        "for":     FOR_REASONS,
        "neutral": NEUTRAL_REASONS,
        "against": AGAINST_REASONS,
    }[stance]
    raw = pool[(seed_int + agent_idx + round_num * 11) % len(pool)]
    body = raw.replace("{anchor}", anchor)
    frame = ROUND_FRAMES[round_num][stance]
    response = frame.replace("{body}", body).replace("{anchor}", anchor)
    title = {1: "First Impression", 2: "Competitor Comparison", 3: "Final Verdict"}[round_num]
    return {"round": round_num, "title": title, "response": response}


def _build_enriched_agents(*, personas, bucket, seed_int, anchor_for):
    """Build enriched per-agent records. Bucket-ordered: BUY then CONS then WON\'T BUY."""
    agents = []
    n_buy = bucket.n_buy
    n_consid = bucket.n_considering
    agent_count = bucket.total

    for i in range(agent_count):
        persona = personas[i] if i < len(personas) else {
            "name": f"Buyer #{i+1:03d}",
            "age": 30,
            "profession": "Adult Consumer",
            "segment": "Generic Buyer",
            "profile": "Generic profile.",
        }

        if i < n_buy:
            bucket_label = "buy"
        elif i < n_buy + n_consid:
            bucket_label = "considering"
        else:
            bucket_label = "wont_buy"

        arc_type = _pick_arc_type(bucket_label, seed_int, i)
        i_lo, i_hi, f_lo, f_hi = _ARC_SCORE_RANGES[arc_type]
        initial_score_10 = _deterministic_in_range(i_lo, i_hi, seed_int, i, 1)
        current_score_10 = _deterministic_in_range(f_lo, f_hi, seed_int, i, 2)

        initial_verdict = _ARC_INITIAL_VERDICT[arc_type]
        target_final = _ARC_FINAL_VERDICT[arc_type]
        derived_final = _score_to_verdict(current_score_10)

        if derived_final != target_final:
            if target_final == "BUY":
                current_score_10 = max(7.0, min(9.5, current_score_10))
            elif target_final == "CONSIDERING":
                current_score_10 = max(4.6, min(6.9, current_score_10))
            else:
                current_score_10 = max(1.0, min(4.5, current_score_10))

        final_verdict = target_final
        stance = _verdict_to_stance(final_verdict)
        initial_stance = _verdict_to_stance(initial_verdict)
        shifted = (initial_verdict != final_verdict)
        is_hardcore = (arc_type == "hardcore")
        key_moment = KEY_MOMENT_TEMPLATES[arc_type]

        anchor = anchor_for(i)

        objection_pool = {
            "for":     FOR_OBJECTIONS,
            "neutral": NEUTRAL_OBJECTIONS,
            "against": AGAINST_OBJECTIONS,
        }[stance]
        top_objection = _pick_variant(objection_pool, seed_int, i, anchor)

        reason_pool = {
            "for":     FOR_REASONS,
            "neutral": NEUTRAL_REASONS,
            "against": AGAINST_REASONS,
        }[stance]
        reason = _pick_variant(reason_pool, seed_int, i, anchor)

        round_responses = [
            _round_response(stance, r, anchor, seed_int, i)
            for r in (1, 2, 3)
        ]

        journey = {
            "initial_verdict": initial_verdict,
            "final_verdict":   final_verdict,
            "shifted":         shifted,
            "shift_reason":    _ARC_SHIFT_REASON[arc_type],
            "key_moment":      key_moment,
            "key_quote":       round_responses[2]["response"],
        }

        legacy_score = round(current_score_10 / 10.0, 2)

        agents.append({
            "id":            f"agent_{i+1:02d}",
            "segment":       persona["segment"],
            "profile":       persona["profile"],
            "stance":        stance,
            "score":         legacy_score,
            "reason":        reason,
            "top_objection": top_objection,
            "name":             persona["name"],
            "age":              persona.get("age", 30),
            "profession":       persona.get("profession", "Adult Consumer"),
            "verdict":          final_verdict,
            "initial_score_10": round(initial_score_10, 1),
            "current_score_10": round(current_score_10, 1),
            "score_10":         round(current_score_10, 1),
            "initial_stance":   initial_stance,
            "current_stance":   stance,
            "is_hardcore":      is_hardcore,
            "shifted":          shifted,
            "key_moment":       key_moment,
            "what_would_change_mind": _what_would_change_mind(stance, anchor, seed_int, i),
            "round_responses":  round_responses,
            "journey":          journey,
        })

    return agents


def _template_panel(product: dict, forecast: dict, agent_count: int, seed: str) -> dict:
    """Deterministic template panel with diverse reason/objection variants."""
    trial_rate_pct = forecast.get("trial_rate", {}).get("percentage", 10.0) or 10.0
    confidence = (forecast.get("confidence") or "medium").lower()
    verdict = forecast.get("verdict", "")
    anchored = forecast.get("anchored_on", []) or []
    forecast_drivers = forecast.get("top_drivers", []) or []
    forecast_objections = forecast.get("top_objections", []) or []

    seed_int = int(seed[:8], 16)
    trial_rate = trial_rate_pct / 100.0

    # Safe getter for fallback_used (per friend's spec — may live in
    # diagnostics depending on response shape).
    fallback_used = bool(
        forecast.get("fallback_used")
        or forecast.get("diagnostics", {}).get("fallback_used")
    )

    # Forecast-driven bucket counts (forecast leads, agents follow).
    bucket = allocate_buckets(
        trial_rate=trial_rate,
        confidence=confidence,
        fallback_used=fallback_used,
        n_agents=agent_count,
    )
    base_for = bucket.n_buy
    base_neutral = bucket.n_considering
    base_against = bucket.n_resistant

    # Product-aware persona selection (deterministic, no duplicate names).
    personas, routing_info = select_personas_for_product(
        product, agent_count, seed
    )

    # Anchor helpers reused by the agent enrichment.
    top_anchor = anchored[0].get("brand") if anchored else "the category leader"
    secondary_anchors = [a.get("brand") for a in anchored[1:4] if a.get("brand")]

    def anchor_for(idx: int) -> str:
        if not anchored:
            return "the category leader"
        pool = [top_anchor] + secondary_anchors
        return pool[idx % len(pool)] if pool else top_anchor

    # Build enriched agents (verdict + score arc + journey + round_responses).
    agents = _build_enriched_agents(
        personas=personas,
        bucket=bucket,
        seed_int=seed_int,
        anchor_for=anchor_for,
    )

    avg_score = sum(a["score"] for a in agents) / len(agents)

    rounds = [
        {
            "round": 1,
            "title": "First Reaction",
            "summary": (
                f"Panel reacted to the {trial_rate_pct:.1f}% comparable-anchored trial estimate. "
                f"Stance split: {base_for} for / {base_neutral} neutral / {base_against} against. "
                f"Initial enthusiasm tempered by typical DTC friction."
            ),
            "for_count": base_for,
            "neutral_count": base_neutral,
            "against_count": base_against,
            "avg_score": round(avg_score - 0.03, 2),
        },
        {
            "round": 2,
            "title": "Comparable Comparison",
            "summary": (
                f"Panel weighed the product against {len(anchored)} anchored comparables. "
                f"Discussion centered on whether differentiation justifies switching cost from incumbents."
            ),
            "for_count": base_for,
            "neutral_count": base_neutral,
            "against_count": base_against,
            "avg_score": round(avg_score, 2),
        },
        {
            "round": 3,
            "title": "Consensus",
            "summary": (
                f"Panel converged: trial rate of {trial_rate_pct:.1f}% is consistent with comparable-brand "
                f"adoption patterns. Verdict aligns with '{verdict.replace('_', ' ')}'."
            ),
            "for_count": base_for,
            "neutral_count": base_neutral,
            "against_count": base_against,
            "avg_score": round(avg_score + 0.02, 2),
        },
    ]

    template_drivers = forecast_drivers[:4] if forecast_drivers else [
        "Comparable-brand precedent",
        "Reasonable price-to-value",
        "Distribution match",
        "Category timing",
    ]
    template_objections = forecast_objections[:4] if forecast_objections else [
        "Subscription fatigue",
        "Switching cost from incumbent",
        "Differentiation unclear",
        "Category skepticism",
    ]

    # Most receptive: highest-scoring BUY agent's segment (or first agent's segment)
    _buy_agents = [a for a in agents if a.get("verdict") == "BUY"]
    if _buy_agents:
        _top_buy = max(_buy_agents, key=lambda a: a.get("current_score_10", 0))
        most_receptive_seg = _top_buy["segment"]
    else:
        most_receptive_seg = agents[0]["segment"] if agents else "Wellness-Conscious Pro"

    # ── Build top-level panel fields (Commit 3c) ────────────────────────

    # Intent distribution from bucket (forecast-leads-agents-follow)
    intent_distribution = {
        "buy":         round(bucket.n_buy / max(1, bucket.total), 3),
        "considering": round(bucket.n_considering / max(1, bucket.total), 3),
        "resistant":   round(bucket.n_resistant / max(1, bucket.total), 3),
    }

    # Buyer journeys: pick a narrative subset (shifted agents + best BUY + worst RESISTANT)
    _shifted = [a for a in agents if a.get("shifted")]
    _buys = sorted([a for a in agents if a.get("verdict") == "BUY"],
                   key=lambda x: x.get("current_score_10", 0), reverse=True)
    _wonts = sorted([a for a in agents if a.get("verdict") == "WON\'T BUY"],
                    key=lambda x: x.get("current_score_10", 0))
    _journey_pool = _shifted[:6] + _buys[:2] + _wonts[:2]
    # Dedupe by name, preserve order
    _seen = set()
    _journey_picks = []
    for a in _journey_pool:
        if a["name"] not in _seen:
            _seen.add(a["name"])
            _journey_picks.append(a)
    buyer_journeys = [
        {
            "agent_id":        a["id"],
            "name":            a["name"],
            "segment":         a["segment"],
            "initial_verdict": a["journey"]["initial_verdict"],
            "final_verdict":   a["journey"]["final_verdict"],
            "shifted":         a["journey"]["shifted"],
            "shift_reason":    a["journey"]["shift_reason"],
            "key_quote":       a["journey"]["key_quote"],
        }
        for a in _journey_picks[:10]
    ]

    # Representative quotes — round-3 quote from one BUY, one CONS, one WON'T BUY
    representative_quotes = []
    for v_label in ("BUY", "CONSIDERING", "WON\'T BUY"):
        _matches = [a for a in agents if a.get("verdict") == v_label]
        if _matches:
            _pick = _matches[0]
            representative_quotes.append({
                "verdict":  v_label,
                "agent_id": _pick["id"],
                "name":     _pick["name"],
                "segment":  _pick["segment"],
                "quote":    _pick["round_responses"][2]["response"],
            })

    # Hardest-to-convert: lowest-scoring WON'T BUY agent's segment
    _wont_agents = [a for a in agents if a.get("verdict") == "WON\'T BUY"]
    if _wont_agents:
        _hardest = min(_wont_agents, key=lambda a: a.get("current_score_10", 10))
        hardest_to_convert_segment = _hardest["segment"]
    else:
        hardest_to_convert_segment = "Brand-loyal incumbent buyer"

    # Comparable price range — placeholder structure (full anchor-pricing
    # arrives with the evidence-engine work). Frontend hides this section
    # if min/max are None.
    comparable_price_range = {
        "user_price":    product.get("price"),
        "min":           None,
        "max":           None,
        "anchor_brands": [a.get("brand") for a in anchored if a.get("brand")][:5],
    }

    # Actionable insight — derived from coverage tier + verdict distribution
    if fallback_used or confidence == "low":
        actionable_insight = (
            "Coverage is thin. Run a $500-$2,000 validation test before scaling "
            "spend. Lead with proof, reviews, and a low-risk starter offer."
        )
    elif intent_distribution["buy"] >= 0.30 and confidence in ("medium-high", "high"):
        actionable_insight = (
            "Strong launch candidate by comparable-brand evidence. Test the "
            "winning message in a small paid landing-page run, then scale based on "
            "signup/checkout intent versus category benchmarks."
        )
    elif intent_distribution["buy"] >= 0.15:
        actionable_insight = (
            "Moderate signal. Address the top objection with proof or a starter "
            "offer before scaling. Run a small validation test to confirm intent."
        )
    else:
        actionable_insight = (
            "Weak buy signal. Reconsider price, positioning, or category fit. "
            "Do not scale launch spend until objections are addressed."
        )

    # Risk factors — upgrade existing string to object format
    _hardcore_count = sum(1 for a in agents if a.get("is_hardcore"))
    _wont_count = len(_wont_agents)
    risk_factors = {
        "summary": f"{_wont_count} of {len(agents)} agents resisted — top objection: " + (
            agents[0].get("top_objection", "differentiation") if _wont_agents else "n/a"
        ),
        "detail": (
            f"Includes {_hardcore_count} hardcore resistors who did not move "
            f"across rounds. Address differentiation versus comparable brands "
            f"and trust signals."
        ),
        "holdout_agents": [a["id"] for a in _wont_agents[:5]],
    }

    panel = {
        "rounds": rounds,
        "agents": agents,
        "top_drivers": template_drivers,
        "top_objections": template_objections,
        "most_receptive_segment": most_receptive_seg,
        "winning_message": (
            f"Position against the strongest anchor ({top_anchor}) "
            f"by leading with category-specific differentiation."
            if anchored else
            "Lead with category-specific differentiation and comparable-brand evidence."
        ),
        "risk_factors": [
            "Same-category retail dominance by incumbents",
            "DTC subscription fatigue compresses repeat",
            "Differentiation may not survive shelf comparison",
        ],
        "consensus": (
            f"Forecast of {trial_rate_pct:.1f}% trial is grounded in comparable-brand evidence. "
            f"Panel recommends '{verdict.replace('_', ' ')}'. Treat agent reasoning as explanation, not validation."
        ),

        # ── New 3c top-level fields (additive) ──
        "intent_distribution":         intent_distribution,
        "buyer_journeys":              buyer_journeys,
        "representative_quotes":       representative_quotes,
        "hardest_to_convert_segment":  hardest_to_convert_segment,
        "comparable_price_range":      comparable_price_range,
        "actionable_insight":          actionable_insight,
        "risk_factors_v3":             risk_factors,
    }

    return panel
