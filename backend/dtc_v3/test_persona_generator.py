"""
backend/dtc_v3/test_persona_generator.py — Routing + sampling tests.

Per friend's Day 2 acceptance criteria for Commit 2:
  - test_school_notebook_uses_school_personas
  - test_energy_drink_uses_energy_personas
  - test_prayer_rug_uses_religious_lifestyle_personas
  - test_pet_product_uses_pet_personas
  - test_same_product_same_agents
  - test_different_products_different_persona_pools
  - test_no_duplicate_names_in_20
  - test_no_duplicate_names_in_50
"""

import pytest

from backend.dtc_v3.persona_generator import (
    select_persona_banks,
    select_personas_for_product,
)


# ═══════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════

def _energy_drink() -> dict:
    return {
        "product_name": "Triton Drinks",
        "name": "Triton Drinks",
        "description": "High-caffeine energy drink for performance",
        "category": "food_beverage",
        "price": 3.99,
        "demographic": "active adults",
        "competitors": [{"name": "Red Bull"}, {"name": "Monster"}],
    }


def _school_notebook() -> dict:
    return {
        "product_name": "Spiral School Notebook",
        "name": "Spiral School Notebook",
        "description": "Lined notebook for high school classroom use",
        "category": "office_supplies",
        "price": 4.99,
        "demographic": "students",
        "competitors": [],
    }


def _prayer_rug() -> dict:
    return {
        "product_name": "Velvet Prayer Rug",
        "name": "Velvet Prayer Rug",
        "description": "Premium qibla-aligned prayer mat for daily salah",
        "category": "home_lifestyle",
        "price": 89,
        "demographic": "Muslim adults",
        "competitors": [],
    }


def _pet_chew() -> dict:
    return {
        "product_name": "Calm Pet Hemp Chew",
        "name": "Calm Pet Hemp Chew",
        "description": "Hemp calming chew for anxious dogs",
        "category": "pet_supplements",
        "price": 29,
        "demographic": "dog owners",
        "competitors": [],
    }


def _mystery_widget() -> dict:
    return {
        "product_name": "Mystery Generic Widget",
        "name": "Mystery Generic Widget",
        "description": "Some kind of consumer product",
        "category": "unknown",
        "price": 50,
        "demographic": "adults",
        "competitors": [],
    }


# ═══════════════════════════════════════════════════════════════════════
# Routing tests (friend's spec)
# ═══════════════════════════════════════════════════════════════════════

def test_school_notebook_uses_school_personas():
    banks, tier = select_persona_banks(_school_notebook())
    assert tier == "keyword"
    assert all(b.startswith("SCHOOL_") for b in banks), f"got banks: {banks}"
    assert "SCHOOL_STUDENTS" in banks


def test_energy_drink_uses_energy_personas():
    banks, tier = select_persona_banks(_energy_drink())
    assert tier == "subtype"
    assert all(b.startswith("ENERGY_DRINK_") for b in banks), f"got banks: {banks}"
    assert "ENERGY_DRINK_STUDENTS" in banks


def test_prayer_rug_uses_religious_lifestyle_personas():
    banks, tier = select_persona_banks(_prayer_rug())
    assert tier == "keyword"
    assert all(b.startswith("RELIGIOUS_LIFESTYLE_") for b in banks)
    assert "RELIGIOUS_LIFESTYLE_MUSLIM_PROFESSIONALS" in banks


def test_pet_product_uses_pet_personas():
    banks, tier = select_persona_banks(_pet_chew())
    assert tier == "keyword"
    assert all(b.startswith("PET_") for b in banks)
    assert "PET_DOG_OWNERS" in banks


def test_unknown_product_falls_back_to_generic():
    banks, tier = select_persona_banks(_mystery_widget())
    assert tier == "generic"
    assert banks == ["GENERIC"]


# ═══════════════════════════════════════════════════════════════════════
# Determinism + uniqueness tests
# ═══════════════════════════════════════════════════════════════════════

def test_same_product_same_agents():
    """Same input must always produce same agents (forecast invariant)."""
    p1, _ = select_personas_for_product(_energy_drink(), 20, "deadbeef1234")
    p2, _ = select_personas_for_product(_energy_drink(), 20, "deadbeef1234")
    n1 = [p["name"] for p in p1]
    n2 = [p["name"] for p in p2]
    assert n1 == n2


def test_different_seeds_different_agents():
    """Different seeds for same product = different orderings (sanity check)."""
    p1, _ = select_personas_for_product(_energy_drink(), 20, "seed_aaa")
    p2, _ = select_personas_for_product(_energy_drink(), 20, "seed_zzz")
    n1 = [p["name"] for p in p1]
    n2 = [p["name"] for p in p2]
    assert n1 != n2, "different seeds should produce different orderings"


def test_different_products_different_persona_pools():
    """School notebook and energy drink use different bank pools."""
    school_banks, _ = select_persona_banks(_school_notebook())
    energy_banks, _ = select_persona_banks(_energy_drink())
    assert set(school_banks).isdisjoint(set(energy_banks)), \
        f"banks should not overlap, got school={school_banks}, energy={energy_banks}"


def test_no_duplicate_names_in_20():
    """20-agent simulation: all 20 names unique."""
    for product in [_energy_drink(), _school_notebook(), _prayer_rug(),
                    _pet_chew(), _mystery_widget()]:
        personas, _ = select_personas_for_product(product, 20, "test_seed")
        names = [p["name"] for p in personas]
        assert len(personas) == 20
        assert len(set(names)) == 20, f"duplicates in {product['name']}: {names}"


def test_no_duplicate_names_in_50():
    """50-agent simulation: all 50 names unique."""
    for product in [_energy_drink(), _school_notebook(), _prayer_rug(),
                    _pet_chew()]:
        personas, _ = select_personas_for_product(product, 50, "test_seed")
        names = [p["name"] for p in personas]
        assert len(personas) == 50
        assert len(set(names)) == 50, f"duplicates in {product['name']}: count={len(set(names))}"


# ═══════════════════════════════════════════════════════════════════════
# Output shape tests
# ═══════════════════════════════════════════════════════════════════════

def test_output_count_matches_n_agents():
    for n in [10, 20, 35, 50]:
        personas, _ = select_personas_for_product(_energy_drink(), n, "test_seed")
        assert len(personas) == n


def test_personas_are_dicts_with_required_keys():
    personas, _ = select_personas_for_product(_energy_drink(), 5, "test_seed")
    required = {"name", "age", "profession", "segment", "profile"}
    for p in personas:
        assert set(p.keys()) >= required, f"missing keys in {p}"


def test_routing_info_is_complete():
    _, info = select_personas_for_product(_energy_drink(), 20, "test_seed")
    assert "tier" in info
    assert "banks_used" in info
    assert "fallback_to_generic" in info
    assert info["tier"] in ("keyword", "subtype", "category", "generic")
    assert isinstance(info["banks_used"], list)
    assert isinstance(info["fallback_to_generic"], bool)


def test_keyword_overrides_subtype():
    """Keyword routing should win over subtype routing when both could apply."""
    # A product with both "school" keyword AND food_beverage category — 
    # keyword should win
    edge_case = {
        "product_name": "School Lunch Energy Drink",
        "name": "School Lunch Energy Drink",
        "description": "energy drink for school students",
        "category": "food_beverage",
        "price": 3.99,
        "demographic": "students",
        "competitors": [],
    }
    banks, tier = select_persona_banks(edge_case)
    assert tier == "keyword"  # NOT "subtype"
    assert any(b.startswith("SCHOOL_") for b in banks)


def test_no_silent_failure_on_missing_fields():
    """Generator must not crash if optional fields are missing."""
    minimal = {"name": "Mystery", "description": "thing", "category": "unknown"}
    personas, _ = select_personas_for_product(minimal, 5, "test_seed")
    assert len(personas) == 5


def test_50_agent_unique_across_routing_tiers():
    """50-agent uniqueness must hold for keyword, subtype, AND generic tiers."""
    for product in [_school_notebook(), _energy_drink(), _mystery_widget()]:
        personas, info = select_personas_for_product(product, 50, "test_seed")
        names = [p["name"] for p in personas]
        n_unique = len(set(names))
        assert n_unique == 50, \
            f"{product['name']} ({info['tier']} tier): only {n_unique}/50 unique names"
