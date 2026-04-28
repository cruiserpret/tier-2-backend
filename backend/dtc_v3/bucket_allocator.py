"""
backend/dtc_v3/bucket_allocator.py — Forecast-driven bucket counts.

Per friend's Q2 spec (verdict cap, pre-generation allocation):

  BUY share <= 2 * trial_rate, capped at 0.60
  CONSIDERING share scales with confidence
  RESISTANT share = remainder
  Low-confidence / fallback hard caps BUY at 15%

This module is forecast-leads-agents-follow logic. Pure function.
No persona work, no template work — just integer counts.

Called from discussion.py BEFORE persona selection so personas can
be assigned to their target bucket up front.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BucketCounts:
    """Allocation result. n_buy + n_considering + n_resistant == n_agents."""
    n_buy: int
    n_considering: int
    n_resistant: int

    @property
    def total(self) -> int:
        return self.n_buy + self.n_considering + self.n_resistant

    def as_distribution(self) -> dict[str, float]:
        """Float fractions for the agent_panel.intent_distribution field."""
        if self.total == 0:
            return {"buy": 0.0, "considering": 0.0, "resistant": 0.0}
        return {
            "buy": self.n_buy / self.total,
            "considering": self.n_considering / self.total,
            "resistant": self.n_resistant / self.total,
        }


def _normalize_confidence(c: str | None) -> str:
    """Normalize confidence string. Tolerates underscore vs hyphen, case."""
    if not c:
        return "medium"
    norm = c.replace("_", "-").lower().strip()
    valid = {"high", "medium-high", "medium", "medium-low", "low"}
    return norm if norm in valid else "medium"


def allocate_buckets(
    trial_rate: float,
    confidence: str,
    fallback_used: bool,
    n_agents: int,
) -> BucketCounts:
    """
    Convert forecast outputs into bucket counts.

    Args:
        trial_rate: forecast.trial_rate.median, e.g. 0.18 (NOT 18)
        confidence: "high" | "medium-high" | "medium" | "medium-low" | "low"
        fallback_used: True if forecast hit a fallback prior
        n_agents: total agent count (typically 20 or 50)

    Returns:
        BucketCounts dataclass with n_buy + n_considering + n_resistant == n_agents

    Math (exact from friend's Q2 spec):
        max_buy_share = min(2.0 * trial_rate, 0.60)

        if confidence in (high, medium-high):
            buy_share = max_buy_share
            considering_share = min(0.35, 1 - buy_share)
        elif confidence == medium:
            buy_share = min(max_buy_share, 0.35)
            considering_share = 0.35
        elif confidence == medium-low:
            buy_share = min(max_buy_share, 0.25)
            considering_share = 0.35
        else:  # low
            buy_share = min(max_buy_share, 0.15)
            considering_share = 0.30

        resistant_share = 1 - buy_share - considering_share

        if fallback_used or low confidence:
            n_buy = min(n_buy, round(n_agents * 0.15))   # hard cap

        Final integer counts use round(), with remainder absorbed by
        n_resistant to ensure exact total = n_agents.
    """
    # Clamp inputs defensively
    rate = max(0.0, min(1.0, float(trial_rate or 0.0)))
    conf = _normalize_confidence(confidence)
    n = max(0, int(n_agents))

    if n == 0:
        return BucketCounts(0, 0, 0)

    # Step 1: compute share targets
    max_buy_share = min(2.0 * rate, 0.60)

    if conf in ("high", "medium-high"):
        buy_share = max_buy_share
        considering_share = min(0.35, 1.0 - buy_share)
    elif conf == "medium":
        buy_share = min(max_buy_share, 0.35)
        considering_share = 0.35
    elif conf == "medium-low":
        buy_share = min(max_buy_share, 0.25)
        considering_share = 0.35
    else:  # low
        buy_share = min(max_buy_share, 0.15)
        considering_share = 0.30

    resistant_share = max(0.0, 1.0 - buy_share - considering_share)

    # Step 2: convert to integer counts via rounding
    n_buy = round(n * buy_share)
    n_considering = round(n * considering_share)

    # Step 3: hard cap for fallback / low confidence
    if fallback_used or conf == "low":
        hard_cap = round(n * 0.15)
        n_buy = min(n_buy, hard_cap)

    # Step 4: clamp non-negative + ensure total == n_agents
    n_buy = max(0, n_buy)
    n_considering = max(0, n_considering)
    n_resistant = n - n_buy - n_considering

    # Edge case: if rounding pushed buy + considering > n, trim from buy first
    if n_resistant < 0:
        # Trim considering first, then buy
        excess = -n_resistant
        trim_considering = min(excess, n_considering)
        n_considering -= trim_considering
        excess -= trim_considering
        if excess > 0:
            n_buy = max(0, n_buy - excess)
        n_resistant = max(0, n - n_buy - n_considering)

    return BucketCounts(
        n_buy=n_buy,
        n_considering=n_considering,
        n_resistant=n_resistant,
    )
