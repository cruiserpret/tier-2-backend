"""
backend/dtc_v3/comparison_context.py — Phase 1.8.1 foundation.

Canonical single source of truth for which brands agents may
mention as comparison material.

PURPOSE
─────────
The matcha-gum trust failure (Apr 30) showed that legacy agent
consumers (template path in discussion.py, LLM prompt in
llm_dialogue_enricher.py, ledger anchor count in
confidence_ledger.py) read brand names from raw forecast.neighbors
instead of the typed Phase-1 evidence_buckets. Under fallback,
that meant agents cited brands the evidence layer itself had
labeled "Retrieved but Not Used."

THE HARD INVARIANT
──────────────────
If a brand is not in allowed_comparison_brands, agents must not
mention it as a competitor or proof point. This is non-negotiable.

This module computes the canonical context object. Downstream
consumers (template, LLM, ledger) migrate to read from it in
P1.8.2 / P1.8.3 / P1.8.4. P1.8.1 is foundation only — this
module is built and attached to CustomerReport but no consumer
reads it yet.

ARCHITECTURE
────────────
Pure classifier. No retrieval, no LLM, no IO, no embeddings.
Inputs: forecast (Forecast or dict), product (dict or
ProductBrief), evidence_buckets (dict from evidence.py),
coverage_tier (str passthrough).
Output: dict matching the schema below.

SCHEMA
──────
{
    "forecast_used_brands":        list[str],   # all anchors
    "dialogue_safe_anchor_brands": list[str],   # direct + adjacent only
    "user_competitors":            list[str],   # parsed from product
    "allowed_comparison_brands":   list[str],   # safe + user, deduped
    "forbidden_brand_names":       list[str],   # rejected brands, minus allowed
    "fallback_used":               bool,
    "confidence":                  str,
    "coverage_tier":               str,
    "comparison_mode":             "anchored" | "user_competitor" | "generic_directional",
    "_meta": {"context_version": "v1.0"},
}

BUILD RULES (per friend Apr 30 ruling)
──────────────────────────────────────
forecast_used_brands       = all forecast_anchor brands
dialogue_safe_anchor_brands = anchors where anchor_strength in {"direct","adjacent"}
user_competitors           = robust-parsed from product
allowed_comparison_brands  = dedup(dialogue_safe + user_competitors)
forbidden_brand_names      = dedup(
                                  fallback_neighbors
                                + candidate_comparables
                                + exploratory_comparables
                                + weak forecast anchors
                             ) − allowed_comparison_brands
comparison_mode =
    anchored             if dialogue_safe_anchor_brands non-empty
    user_competitor      if user_competitors non-empty
    generic_directional  otherwise
"""

from __future__ import annotations
from typing import Any


CONTEXT_VERSION = "v1.0"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _extract_user_competitor_names(product: Any) -> list[str]:
    """Robustly extract user-stated competitor brand names from product input.

    Supports:
      - product as dict with "competitors" key, or as ProductBrief object
        with .competitors attribute
      - competitors as list of strings: ["Trident", "Orbit"]
      - competitors as list of dicts: [{"name": "Trident"}, ...]
      - competitors as mixed list: ["Trident", {"name": "Orbit"}, "", None]
      - empty/missing values dropped
      - case-insensitive dedup, preserving first occurrence's casing

    Returns list of clean brand-name strings.
    """
    if product is None:
        return []

    # Pull raw competitors regardless of dict-vs-object shape
    if isinstance(product, dict):
        raw = product.get("competitors", [])
    else:
        raw = getattr(product, "competitors", [])

    if not raw:
        return []

    extracted: list[str] = []
    for item in raw:
        name = None
        if item is None:
            continue
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name")
        else:
            # Object with .name attr (ProductBrief or similar)
            name = getattr(item, "name", None)

        if not name or not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        extracted.append(name)

    return _dedup_preserving_case(extracted)


def _dedup_preserving_case(brands: list[str]) -> list[str]:
    """Case-insensitive dedup. First occurrence's casing wins."""
    seen_lower: set[str] = set()
    out: list[str] = []
    for b in brands:
        if not b:
            continue
        key = b.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(b)
    return out


def _extract_brand_names(items: list[dict] | None) -> list[str]:
    """Extract .brand from a list of evidence-bucket entries.

    Handles None, missing keys, and non-string brand values defensively.
    """
    if not items:
        return []
    out: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        b = it.get("brand")
        if isinstance(b, str) and b.strip():
            out.append(b.strip())
    return out


def _filter_dialogue_safe(forecast_anchors: list[dict] | None) -> list[str]:
    """Return brand names of forecast anchors with anchor_strength
    in {"direct", "adjacent"} — i.e. dialogue-safe.

    Weak anchors are intentionally excluded from dialogue use even
    though they are visible in the evidence panel.
    """
    if not forecast_anchors:
        return []
    safe: list[str] = []
    for a in forecast_anchors:
        if not isinstance(a, dict):
            continue
        strength = a.get("anchor_strength")
        if strength not in ("direct", "adjacent"):
            continue
        b = a.get("brand")
        if isinstance(b, str) and b.strip():
            safe.append(b.strip())
    return safe


def _filter_weak_anchors(forecast_anchors: list[dict] | None) -> list[str]:
    """Return brand names of forecast anchors with anchor_strength == 'weak'.

    These are forbidden from dialogue (they go into forbidden_brand_names)
    even though forecast math used them.
    """
    if not forecast_anchors:
        return []
    weak: list[str] = []
    for a in forecast_anchors:
        if not isinstance(a, dict):
            continue
        if a.get("anchor_strength") == "weak":
            b = a.get("brand")
            if isinstance(b, str) and b.strip():
                weak.append(b.strip())
    return weak


def _subtract_case_insensitive(superset: list[str], subset: list[str]) -> list[str]:
    """Return superset minus subset, case-insensitive comparison.
    Preserves superset ordering and casing."""
    subset_lower = {s.lower() for s in subset if s}
    return [b for b in superset if b.lower() not in subset_lower]


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def build_comparison_context(
    forecast: Any,
    product: Any,
    evidence_buckets: dict | None,
    coverage_tier: str | None = None,
) -> dict:
    """Build the canonical comparison_context object.

    Pure function. No side effects. No retrieval, no LLM, no IO.

    Parameters
    ──────────
    forecast : Forecast object (or dict-like) with .fallback_used, .confidence
    product : dict or ProductBrief (with .competitors)
    evidence_buckets : dict from evidence.classify_evidence_buckets():
        {
          "forecast_anchors": [...],
          "candidate_comparables": [...],
          "fallback_neighbors": [...],
          "exploratory_comparables": [...],
        }
        May be None — treated as all-empty.
    coverage_tier : str passthrough, may be None

    Returns
    ───────
    dict matching the SCHEMA in the module docstring.
    """
    # Defensive: evidence_buckets may be None
    eb = evidence_buckets or {}
    forecast_anchors_list  = eb.get("forecast_anchors", []) or []
    fallback_neighbors_list = eb.get("fallback_neighbors", []) or []
    candidate_comparables_list = eb.get("candidate_comparables", []) or []
    exploratory_comparables_list = eb.get("exploratory_comparables", []) or []

    # 1. forecast_used_brands — all anchors regardless of strength
    forecast_used_brands = _dedup_preserving_case(
        _extract_brand_names(forecast_anchors_list)
    )

    # 2. dialogue_safe_anchor_brands — direct + adjacent only
    dialogue_safe_anchor_brands = _dedup_preserving_case(
        _filter_dialogue_safe(forecast_anchors_list)
    )

    # 3. user_competitors — parsed robustly
    user_competitors = _extract_user_competitor_names(product)

    # 4. allowed_comparison_brands — safe anchors + user, deduped
    allowed_comparison_brands = _dedup_preserving_case(
        dialogue_safe_anchor_brands + user_competitors
    )

    # 5. forbidden_brand_names — rejected pools minus allowed
    forbidden_pool = (
        _extract_brand_names(fallback_neighbors_list)
        + _extract_brand_names(candidate_comparables_list)
        + _extract_brand_names(exploratory_comparables_list)
        + _filter_weak_anchors(forecast_anchors_list)
    )
    forbidden_pool_dedup = _dedup_preserving_case(forbidden_pool)
    forbidden_brand_names = _subtract_case_insensitive(
        forbidden_pool_dedup, allowed_comparison_brands
    )

    # 6. comparison_mode — friend's revised rule:
    # depends on whether SAFE anchors exist, NOT on fallback_used alone
    if dialogue_safe_anchor_brands:
        comparison_mode = "anchored"
    elif user_competitors:
        comparison_mode = "user_competitor"
    else:
        comparison_mode = "generic_directional"

    # 7. Passthrough fields
    fallback_used = bool(getattr(forecast, "fallback_used", False)) \
        if not isinstance(forecast, dict) \
        else bool(forecast.get("fallback_used", False))
    confidence = (getattr(forecast, "confidence", None)
                  if not isinstance(forecast, dict)
                  else forecast.get("confidence")) or ""
    coverage_tier_out = coverage_tier or ""

    return {
        "forecast_used_brands":        forecast_used_brands,
        "dialogue_safe_anchor_brands": dialogue_safe_anchor_brands,
        "user_competitors":            user_competitors,
        "allowed_comparison_brands":   allowed_comparison_brands,
        "forbidden_brand_names":       forbidden_brand_names,
        "fallback_used":               fallback_used,
        "confidence":                  str(confidence),
        "coverage_tier":               str(coverage_tier_out),
        "comparison_mode":             comparison_mode,
        "_meta": {"context_version": CONTEXT_VERSION},
    }
