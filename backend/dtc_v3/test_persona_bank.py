"""
backend/dtc_v3/test_persona_bank.py — Structural tests for persona_bank.

Verifies bank shape, registry completeness, helper functions, and
within-bank name uniqueness. Product-relevant routing tests live in
test_persona_generator.py (Commit 2).
"""

from collections import Counter

import pytest

from backend.dtc_v3.persona_bank import (
    BANKS,
    GENERIC_PERSONAS,
    all_bank_names,
    get_bank,
    persona_to_dict,
)


def test_total_bank_count():
    """50 banks expected: 14 specialized × ~3 sub-pools + 5 generic fallbacks."""
    assert len(BANKS) == 50, f"expected 50 banks, got {len(BANKS)}"


def test_minimum_bank_size():
    """Every bank has at least 12 personas (sub-pool minimum from friend's spec)."""
    for name, bank in BANKS.items():
        assert len(bank) >= 12, f"{name} has only {len(bank)} personas"


def test_persona_tuple_shape():
    """Every persona is a 5-tuple of (str, int, str, str, str)."""
    for bank_name, bank in BANKS.items():
        for i, p in enumerate(bank):
            assert isinstance(p, tuple), f"{bank_name}[{i}] not tuple"
            assert len(p) == 5, f"{bank_name}[{i}] not 5-tuple, got len {len(p)}"
            assert isinstance(p[0], str), f"{bank_name}[{i}].name not str"
            assert isinstance(p[1], int), f"{bank_name}[{i}].age not int"
            assert isinstance(p[2], str), f"{bank_name}[{i}].profession not str"
            assert isinstance(p[3], str), f"{bank_name}[{i}].segment not str"
            assert isinstance(p[4], str), f"{bank_name}[{i}].profile not str"


def test_within_bank_name_uniqueness():
    """No duplicate names within any single bank (sim with one bank stays clean)."""
    for bank_name, bank in BANKS.items():
        names = [p[0] for p in bank]
        counter = Counter(names)
        dups = [n for n, c in counter.items() if c > 1]
        assert not dups, f"{bank_name} has duplicates: {dups}"


def test_persona_to_dict_produces_correct_keys():
    sample = persona_to_dict(GENERIC_PERSONAS[0])
    assert set(sample.keys()) == {"name", "age", "profession", "segment", "profile"}


def test_persona_to_dict_preserves_values():
    p = ("Test Name", 30, "Test Job", "Test Segment", "Test profile.")
    d = persona_to_dict(p)
    assert d["name"] == "Test Name"
    assert d["age"] == 30
    assert d["profession"] == "Test Job"
    assert d["segment"] == "Test Segment"
    assert d["profile"] == "Test profile."


def test_get_bank_returns_real_bank():
    bank = get_bank("ENERGY_DRINK_STUDENTS")
    assert len(bank) >= 12
    assert bank[0][0] == "Chloe Bernard"  # First persona stable


def test_get_bank_falls_back_to_generic():
    """Non-existent bank name returns GENERIC_PERSONAS, not raises."""
    bank = get_bank("DOES_NOT_EXIST_XYZ")
    assert bank is GENERIC_PERSONAS


def test_all_bank_names_returns_50():
    names = all_bank_names()
    assert len(names) == 50
    # Spot-check key banks exist
    assert "ENERGY_DRINK_STUDENTS" in names
    assert "SCHOOL_STUDENTS" in names
    assert "RELIGIOUS_LIFESTYLE_MUSLIM_PROFESSIONALS" in names
    assert "GENERIC" in names


def test_required_specialized_banks_exist():
    """Friend's spec called out specific banks. Confirm all present."""
    required = [
        # Energy drinks (4 sub-pools, friend's example)
        "ENERGY_DRINK_STUDENTS",
        "ENERGY_DRINK_FITNESS",
        "ENERGY_DRINK_NIGHT_SHIFT",
        "ENERGY_DRINK_GAMERS",
        # School (5 sub-pools, friend's example)
        "SCHOOL_STUDENTS",
        "SCHOOL_PARENTS",
        "SCHOOL_TEACHERS",
        "SCHOOL_COLLEGE",
        "SCHOOL_AESTHETIC",
        # Religious lifestyle (3 sub-pools, prayer rug case)
        "RELIGIOUS_LIFESTYLE_MUSLIM_PROFESSIONALS",
        "RELIGIOUS_LIFESTYLE_MUSLIM_FAMILIES",
        "RELIGIOUS_LIFESTYLE_SMART_HOME",
        # Pet (friend's example)
        "PET_DOG_OWNERS",
        "PET_CAT_OWNERS",
        "PET_PREMIUM",
        # Baby/family
        "BABY_FAMILY_NEW_PARENTS",
        "BABY_FAMILY_EXPERIENCED",
        "BABY_FAMILY_ORGANIC",
        # Generic fallbacks
        "FOOD_BEVERAGE_GENERIC",
        "BEAUTY_PERSONAL_GENERIC",
        "HOME_LIFESTYLE_GENERIC",
        "TECH_WELLNESS_GENERIC",
        "GENERIC",
    ]
    for bank_name in required:
        assert bank_name in BANKS, f"missing required bank: {bank_name}"


def test_persona_ages_are_reasonable():
    """All ages in 14-70 range — sanity check against typo bugs."""
    for bank_name, bank in BANKS.items():
        for i, p in enumerate(bank):
            age = p[1]
            assert 14 <= age <= 70, f"{bank_name}[{i}] {p[0]} age={age} out of 14-70"


def test_total_persona_pool_size():
    """Confirm we have a sufficient personality pool for 50-agent simulations."""
    total = sum(len(b) for b in BANKS.values())
    assert total >= 500, f"only {total} personas total — too few for 50-agent diversity"
