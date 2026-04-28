"""
backend/dtc_v3/persona_generator.py — Product-aware persona selection.

Per friend's Day 2 spec (Q2 = C + keyword shortcuts):

  1. Keyword override (highest priority — handles edge cases outside DB)
  2. Inferred subtype (from rag_retrieval._infer_query_subtype)
  3. Product category (broad fallback)
  4. GENERIC fallback (last resort)

Returns a list of N unique-named personas relevant to the product.
Determinism preserved: same (product, n_agents, seed) → same agents.
"""

from __future__ import annotations

import hashlib

from backend.dtc_v3.persona_bank import (
    BANKS,
    GENERIC_PERSONAS,
    Persona,
    get_bank,
    persona_to_dict,
)


# ═══════════════════════════════════════════════════════════════════════
# KEYWORD OVERRIDE TABLE (Tier 1 — highest priority)
# ═══════════════════════════════════════════════════════════════════════
# Per friend's spec: handles products outside the current DB subtypes.
# Order: more specific keywords first, generic ones later.
# Each tuple: (keywords_to_match, list_of_bank_names_to_draw_from)
KEYWORD_OVERRIDES: list[tuple[list[str], list[str]]] = [
    # School supplies — friend's primary example
    (
        ["school", "notebook", "classroom", "student supplies", "stationery",
         "binder", "planner", "lecture notebook", "composition book"],
        ["SCHOOL_STUDENTS", "SCHOOL_PARENTS", "SCHOOL_TEACHERS",
         "SCHOOL_COLLEGE", "SCHOOL_AESTHETIC"],
    ),
    # Religious lifestyle — prayer rug case from friend's spec
    (
        ["prayer", "qibla", "salah", "religious lifestyle",
         "prayer rug", "prayer mat", "muslim", "islamic"],
        ["RELIGIOUS_LIFESTYLE_MUSLIM_PROFESSIONALS",
         "RELIGIOUS_LIFESTYLE_MUSLIM_FAMILIES",
         "RELIGIOUS_LIFESTYLE_SMART_HOME"],
    ),
    # Pet products
    (
        ["dog", "cat", "pet", "puppy", "kitten", "feline", "canine"],
        ["PET_DOG_OWNERS", "PET_CAT_OWNERS", "PET_PREMIUM"],
    ),
    # Baby / family
    (
        ["baby", "infant", "toddler", "newborn", "diaper", "stroller"],
        ["BABY_FAMILY_NEW_PARENTS", "BABY_FAMILY_EXPERIENCED",
         "BABY_FAMILY_ORGANIC"],
    ),
    # Personal care — beard / razor
    (
        ["beard", "shave", "razor", "stubble", "shaving cream",
         "aftershave", "trimmer"],
        ["PERSONAL_CARE_MEN_GROOMING", "PERSONAL_CARE_RAZOR_SUBSCRIBERS",
         "PERSONAL_CARE_BEARD"],
    ),
    # Supplements — keyword shortcut for edge cases
    (
        ["protein", "whey", "creatine", "preworkout", "pre-workout",
         "amino", "bcaa", "mass gainer"],
        ["SUPPLEMENTS_GYM_GOERS", "SUPPLEMENTS_OPTIMIZERS",
         "SUPPLEMENTS_GENERAL_HEALTH"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# SUBTYPE → BANK MAP (Tier 2 — uses rag_retrieval inference)
# ═══════════════════════════════════════════════════════════════════════
# Maps subtypes that _infer_query_subtype() can return to bank lists.
# Subtypes not listed here fall through to category-level routing.
SUBTYPE_TO_BANKS: dict[str, list[str]] = {
    "energy_drink": [
        "ENERGY_DRINK_STUDENTS", "ENERGY_DRINK_FITNESS",
        "ENERGY_DRINK_NIGHT_SHIFT", "ENERGY_DRINK_GAMERS",
    ],
    "hydration_supplement": [
        "HYDRATION_ATHLETES", "HYDRATION_WELLNESS_PROFESSIONALS",
        "HYDRATION_BUSY_PARENTS",
    ],
    "skincare_active": [
        "SKINCARE_ENTHUSIASTS", "SKINCARE_ACNE_PRONE",
        "SKINCARE_ANTI_AGING",
    ],
    "razor_subscription": [
        "PERSONAL_CARE_MEN_GROOMING", "PERSONAL_CARE_RAZOR_SUBSCRIBERS",
        "PERSONAL_CARE_BEARD",
    ],
    "nonalcoholic_beer": [
        "NA_BEER_SOBER_CURIOUS", "NA_BEER_CRAFT_BEER_FANS",
        "NA_BEER_FITNESS",
    ],
    "coffee_alternative": [
        "COFFEE_ALT_WELLNESS", "COFFEE_ALT_COFFEE_REDUCERS",
        "COFFEE_ALT_MUSHROOM_CURIOUS",
    ],
    "mattress": [
        "MATTRESS_FIRST_BUYERS", "MATTRESS_UPGRADERS",
        "MATTRESS_BACK_PAIN",
    ],
    "premium_drinkware": [
        "DRINKWARE_OUTDOOR", "DRINKWARE_OFFICE", "DRINKWARE_FITNESS",
    ],
    "premium_basics": [
        "APPAREL_BASICS_MINIMALISTS", "APPAREL_BASICS_QUALITY_SEEKERS",
        "APPAREL_BASICS_SUSTAINABLE",
    ],
    "supplement_protein": [
        "SUPPLEMENTS_GYM_GOERS", "SUPPLEMENTS_OPTIMIZERS",
        "SUPPLEMENTS_GENERAL_HEALTH",
    ],
    "supplement_greens": [
        "SUPPLEMENTS_OPTIMIZERS", "SUPPLEMENTS_GENERAL_HEALTH",
        "HYDRATION_WELLNESS_PROFESSIONALS",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY → BANK MAP (Tier 3 — fallback when subtype is "default")
# ═══════════════════════════════════════════════════════════════════════
CATEGORY_TO_BANKS: dict[str, list[str]] = {
    "food_beverage": ["FOOD_BEVERAGE_GENERIC"],
    "supplements_health": [
        "SUPPLEMENTS_GENERAL_HEALTH", "SUPPLEMENTS_OPTIMIZERS",
    ],
    "beauty_skincare": ["BEAUTY_PERSONAL_GENERIC"],
    "personal_care": ["BEAUTY_PERSONAL_GENERIC"],
    "home_lifestyle": ["HOME_LIFESTYLE_GENERIC"],
    "home_decor": ["HOME_LIFESTYLE_GENERIC"],
    "electronics_tech": ["TECH_WELLNESS_GENERIC"],
    "wellness_tech": ["TECH_WELLNESS_GENERIC"],
    "apparel_basics": [
        "APPAREL_BASICS_MINIMALISTS", "APPAREL_BASICS_QUALITY_SEEKERS",
        "APPAREL_BASICS_SUSTAINABLE",
    ],
    "drinkware": [
        "DRINKWARE_OUTDOOR", "DRINKWARE_OFFICE", "DRINKWARE_FITNESS",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# ROUTING LOGIC
# ═══════════════════════════════════════════════════════════════════════

def _normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace. Used for keyword matching."""
    return " ".join((s or "").lower().split())


def _match_keyword_override(product: dict) -> list[str] | None:
    """Tier 1: check for keyword overrides. Returns bank list or None."""
    text = _normalize_text(
        f"{product.get('product_name') or product.get('name') or ''} "
        f"{product.get('description') or ''}"
    )
    for keywords, banks in KEYWORD_OVERRIDES:
        if any(kw in text for kw in keywords):
            return banks
    return None


def _match_subtype(product: dict) -> list[str] | None:
    """Tier 2: try subtype inference via rag_retrieval. Returns bank list or None."""
    try:
        from backend.dtc_v3.models import ProductBrief
        from backend.dtc_v3.rag_retrieval import _infer_query_subtype

        brief = ProductBrief(
            name=product.get("product_name") or product.get("name") or "",
            description=product.get("description") or "",
            price=float(product.get("price") or 0),
            category=product.get("category") or "default",
            demographic=product.get("demographic") or "",
            competitors=product.get("competitors") or [],
        )
        subtype = _infer_query_subtype(brief)
    except Exception:
        return None

    return SUBTYPE_TO_BANKS.get(subtype)


def _match_category(product: dict) -> list[str] | None:
    """Tier 3: fall back to product.category. Returns bank list or None."""
    cat = (product.get("category") or "").lower()
    return CATEGORY_TO_BANKS.get(cat)


def select_persona_banks(product: dict) -> tuple[list[str], str]:
    """
    Pick which persona banks to draw from for this product.

    Returns (bank_names, routing_tier) where routing_tier is one of:
      "keyword" | "subtype" | "category" | "generic"
    """
    banks = _match_keyword_override(product)
    if banks:
        return banks, "keyword"

    banks = _match_subtype(product)
    if banks:
        return banks, "subtype"

    banks = _match_category(product)
    if banks:
        return banks, "category"

    return ["GENERIC"], "generic"


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC NO-DUPLICATE SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def _seed_int(seed: str) -> int:
    """Convert seed string (could be hex or arbitrary) to integer."""
    if all(c in "0123456789abcdefABCDEF" for c in seed[:8]):
        return int(seed[:8], 16)
    return int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)


def _round_robin_sample(
    bank_names: list[str], n_agents: int, seed_int: int
) -> list[Persona]:
    """
    Deterministically sample n_agents personas across the given banks
    with no duplicate names within the result.

    Strategy:
      - Round-robin across banks (bank[0][0], bank[1][0], bank[2][0], bank[0][1], ...)
      - Skip personas whose name is already selected
      - Offset starting position per-bank using seed_int
      - If we can't fill n_agents from the requested banks, fall back to
        GENERIC_PERSONAS for the remainder
    """
    # Materialize each bank with a per-bank offset so different products
    # get different starting positions even when banks overlap.
    bank_lists: list[list[Persona]] = []
    for i, name in enumerate(bank_names):
        bank = get_bank(name)
        offset = (seed_int + i * 7919) % len(bank)  # 7919 is prime → good spread
        rotated = bank[offset:] + bank[:offset]
        bank_lists.append(rotated)

    seen_names: set[str] = set()
    out: list[Persona] = []
    cursor = [0] * len(bank_lists)  # one cursor per bank

    while len(out) < n_agents:
        # Try one persona from each bank in round-robin
        progress = False
        for b_idx, bank in enumerate(bank_lists):
            if len(out) >= n_agents:
                break
            while cursor[b_idx] < len(bank):
                p = bank[cursor[b_idx]]
                cursor[b_idx] += 1
                if p[0] not in seen_names:
                    seen_names.add(p[0])
                    out.append(p)
                    progress = True
                    break
        if not progress:
            # All requested banks exhausted of fresh names. Fall back to GENERIC.
            for p in GENERIC_PERSONAS:
                if len(out) >= n_agents:
                    break
                if p[0] not in seen_names:
                    seen_names.add(p[0])
                    out.append(p)
            break  # can't continue further

    # Final safety: if still under n_agents, pad with deterministic synthetic names.
    # In practice this is unreachable because BANKS + GENERIC has 700+ unique names.
    while len(out) < n_agents:
        synthetic_name = f"Buyer #{len(out)+1:03d}"
        if synthetic_name in seen_names:
            synthetic_name = f"Buyer #{len(out)+1:03d}-alt"
        seen_names.add(synthetic_name)
        out.append((synthetic_name, 30, "Adult Consumer", "Generic Buyer", "Generic profile."))

    return out


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API — called by discussion.py
# ═══════════════════════════════════════════════════════════════════════

def select_personas_for_product(
    product: dict, n_agents: int, seed: str
) -> tuple[list[dict], dict]:
    """
    Select N product-relevant personas, deterministically, with no duplicate names.

    Args:
        product: dict with keys product_name/name, description, category, etc.
        n_agents: number of personas to return (typically 20 or 50)
        seed: deterministic seed string (typically from generate_seed in discussion.py)

    Returns:
        (personas, routing_info) tuple where:
          - personas: list of N dicts with keys name/age/profession/segment/profile
          - routing_info: dict with keys:
              "tier": "keyword" | "subtype" | "category" | "generic"
              "banks_used": list of bank names selected
              "fallback_to_generic": bool (true if had to dip into GENERIC_PERSONAS)
    """
    banks, tier = select_persona_banks(product)
    seed_int = _seed_int(seed)

    persona_tuples = _round_robin_sample(banks, n_agents, seed_int)

    # Detect whether we had to dip into GENERIC for any persona
    requested_names = set()
    for bank_name in banks:
        for p in get_bank(bank_name):
            requested_names.add(p[0])
    fallback_to_generic = any(p[0] not in requested_names for p in persona_tuples)

    persona_dicts = [persona_to_dict(p) for p in persona_tuples]
    routing_info = {
        "tier": tier,
        "banks_used": banks,
        "fallback_to_generic": fallback_to_generic,
    }
    return persona_dicts, routing_info
