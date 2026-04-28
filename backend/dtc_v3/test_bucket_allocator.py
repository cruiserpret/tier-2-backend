"""
backend/dtc_v3/test_bucket_allocator.py — Forecast-driven bucket allocation.

Per friend's Q2 spec:
  - BUY share <= 2 * trial_rate, capped at 0.60
  - CONSIDERING share scales with confidence
  - Low-confidence / fallback hard caps BUY at 15%
  - Total always equals n_agents
"""

import pytest

from backend.dtc_v3.bucket_allocator import (
    BucketCounts,
    _normalize_confidence,
    allocate_buckets,
)


# ═══════════════════════════════════════════════════════════════════════
# Total invariant — sum is always n_agents
# ═══════════════════════════════════════════════════════════════════════

def test_total_always_equals_n_agents_50():
    for rate in [0.0, 0.05, 0.10, 0.22, 0.45, 0.80]:
        for conf in ["high", "medium-high", "medium", "medium-low", "low"]:
            for fb in [True, False]:
                b = allocate_buckets(rate, conf, fb, 50)
                assert b.total == 50, f"rate={rate} conf={conf} fb={fb}: total={b.total}"


def test_total_always_equals_n_agents_20():
    for rate in [0.0, 0.05, 0.10, 0.22, 0.45, 0.80]:
        for conf in ["high", "medium-high", "medium", "medium-low", "low"]:
            b = allocate_buckets(rate, conf, False, 20)
            assert b.total == 20


def test_zero_agents_returns_zeros():
    b = allocate_buckets(0.22, "medium-high", False, 0)
    assert b.n_buy == 0 and b.n_considering == 0 and b.n_resistant == 0


# ═══════════════════════════════════════════════════════════════════════
# Friend's Q2 cap rules — BUY share ≤ 2 × trial_rate, ≤ 0.60
# ═══════════════════════════════════════════════════════════════════════

def test_buy_cap_at_2x_trial_rate_low_rate():
    """Trial rate 4% → BUY share ≤ 8%."""
    b = allocate_buckets(0.04, "medium-high", False, 50)
    assert b.n_buy <= 4, f"BUY={b.n_buy} should be ≤ 4 (50 × 0.08)"


def test_buy_cap_at_60pct_high_rate():
    """Trial rate 50% with high conf → BUY share capped at 60%."""
    b = allocate_buckets(0.50, "high", False, 50)
    assert b.n_buy <= 30, f"BUY={b.n_buy} should be ≤ 30 (50 × 0.60)"


def test_buy_cap_at_2x_trial_rate_22pct():
    """Trial rate 22% / medium-high → BUY ≈ 44%."""
    b = allocate_buckets(0.22, "medium-high", False, 50)
    assert 20 <= b.n_buy <= 24, f"BUY={b.n_buy} should be ≈ 22"


# ═══════════════════════════════════════════════════════════════════════
# Friend's Q2 hard cap — fallback / low confidence ≤ 15%
# ═══════════════════════════════════════════════════════════════════════

def test_fallback_caps_buy_at_15pct():
    """Even with high trial rate, fallback_used=True caps BUY at 15%."""
    b = allocate_buckets(0.50, "medium", True, 50)
    assert b.n_buy <= round(50 * 0.15), \
        f"fallback should cap BUY at 7-8 (50 × 0.15), got {b.n_buy}"


def test_low_confidence_caps_buy_at_15pct():
    """Low confidence caps BUY at 15% even without fallback."""
    b = allocate_buckets(0.50, "low", False, 50)
    assert b.n_buy <= round(50 * 0.15), \
        f"low conf should cap BUY at 7-8 (50 × 0.15), got {b.n_buy}"


def test_fallback_yeti_realistic():
    """YETI realistic case: 7.5% rate / low / fallback → BUY ≤ 8."""
    b = allocate_buckets(0.075, "low", True, 50)
    assert b.n_buy <= 8


# ═══════════════════════════════════════════════════════════════════════
# Confidence-driven CONSIDERING shares
# ═══════════════════════════════════════════════════════════════════════

def test_high_confidence_considering_capped_at_35pct():
    """High conf → considering = min(0.35, 1 - buy_share)."""
    b = allocate_buckets(0.22, "medium-high", False, 50)
    assert b.n_considering <= round(50 * 0.35) + 1


def test_medium_confidence_considering_at_35pct():
    """Medium conf → considering ≈ 35%."""
    b = allocate_buckets(0.10, "medium", False, 50)
    expected = round(50 * 0.35)
    assert abs(b.n_considering - expected) <= 1


def test_low_confidence_considering_at_30pct():
    """Low conf → considering ≈ 30%."""
    b = allocate_buckets(0.10, "low", False, 50)
    expected = round(50 * 0.30)
    assert abs(b.n_considering - expected) <= 1


# ═══════════════════════════════════════════════════════════════════════
# Counts non-negative + valid
# ═══════════════════════════════════════════════════════════════════════

def test_counts_never_negative():
    for rate in [0.0, 0.50, 1.0]:
        for conf in ["high", "medium-high", "medium", "medium-low", "low"]:
            for fb in [True, False]:
                for n in [10, 20, 50]:
                    b = allocate_buckets(rate, conf, fb, n)
                    assert b.n_buy >= 0
                    assert b.n_considering >= 0
                    assert b.n_resistant >= 0


def test_extreme_rate_clamped():
    """Rates outside [0,1] are clamped, not crashing."""
    b = allocate_buckets(2.0, "high", False, 50)  # clamps to 1.0
    assert b.total == 50
    b2 = allocate_buckets(-0.5, "high", False, 50)  # clamps to 0.0
    assert b2.total == 50
    assert b2.n_buy == 0


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def test_normalize_confidence_underscore():
    assert _normalize_confidence("medium_high") == "medium-high"
    assert _normalize_confidence("MEDIUM-HIGH") == "medium-high"
    assert _normalize_confidence(None) == "medium"
    assert _normalize_confidence("garbage") == "medium"


def test_as_distribution_sums_to_one():
    b = allocate_buckets(0.22, "medium-high", False, 50)
    dist = b.as_distribution()
    total = dist["buy"] + dist["considering"] + dist["resistant"]
    assert abs(total - 1.0) < 0.01


def test_as_distribution_zero_total():
    """Empty bucket count returns clean zeros."""
    b = BucketCounts(0, 0, 0)
    dist = b.as_distribution()
    assert dist == {"buy": 0.0, "considering": 0.0, "resistant": 0.0}


# ═══════════════════════════════════════════════════════════════════════
# Determinism — same input always returns same counts
# ═══════════════════════════════════════════════════════════════════════

def test_determinism_same_input_same_output():
    b1 = allocate_buckets(0.22, "medium-high", False, 50)
    b2 = allocate_buckets(0.22, "medium-high", False, 50)
    b3 = allocate_buckets(0.22, "medium-high", False, 50)
    assert b1.n_buy == b2.n_buy == b3.n_buy
    assert b1.n_considering == b2.n_considering == b3.n_considering
    assert b1.n_resistant == b2.n_resistant == b3.n_resistant
