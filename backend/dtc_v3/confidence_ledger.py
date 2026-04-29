"""
backend/dtc_v3/confidence_ledger.py — Phase 1 Confidence Ledger builder.

Pure-function module. Consumes the outputs of forecast() and
evidence.classify_evidence_buckets() and emits a list of typed
ledger entries explaining what evidence supports the forecast
and what evidence is missing.

Per friend Apr 29 ruling:
  - Q1=A: emit both positive AND negative entries when applicable
  - Q2=A: suppress anchor-derived signals when forecast.fallback_used
          is True ("0 of 0 anchors matched" reads wrong)
  - Q3:   inferred_subtype passes through from api.py — single
          source of truth, do not re-infer here

Each entry is a typed dict:
  {
    "kind":   "positive" | "negative" | "neutral",
    "signal": machine-readable code (stable across releases),
    "text":   customer-facing string
  }

Signal vocabulary (15 codes):

  ANCHOR COUNT
    +  strong_anchor_count        (>=3 eligible anchors)
    -  low_anchor_count           (<3 eligible anchors)

  SUBTYPE MATCH (suppressed when fallback_used)
    +  direct_subtype_match       (>=2 anchors with strength="direct")
    -  category_subtype_mismatch  (any anchor with adjacent/weak strength)

  VARIANCE (suppressed when fallback_used)
    +  low_anchor_variance        (no variance warning in quality_warnings)
    -  high_anchor_variance       (variance warning present)

  SOURCE GRADES (suppressed when fallback_used)
    +  strong_source_grades       (majority A or B)
    -  weak_source_grades         (majority C or D)
    ~  source_grades_unavailable  (cannot compute)

  PRICE SIMILARITY (suppressed when fallback_used)
    +  price_similarity_strong    (>=50% within +/-25%)
    -  price_similarity_weak      (<50% within +/-25%)
    ~  price_similarity_unavailable (cannot compute)

  FALLBACK
    -  fallback_used              (forecast.fallback_used = True)

  EVIDENCE GAPS (always emit until later phases ship)
    -  no_external_evidence       (Phase 3 not shipped)
    -  no_shopify_outcomes        (Phase 7 not shipped)

Ledger length: friend Apr 29 guideline is 3-5 positives + 2-4
negatives in normal cases. This is achieved through deterministic
rule firing, not post-hoc truncation. If the ledger feels noisy,
tune individual rule thresholds rather than truncating output.

This module performs NO retrieval, NO embedding calls, NO forecast
math. It is a pure classifier.
"""

from __future__ import annotations
from typing import Optional

from .models import Forecast, ProductBrief, GroundTruthRecord


# Threshold constants
MIN_STRONG_ANCHOR_COUNT = 3
MIN_DIRECT_ANCHORS_FOR_STRONG_SUBTYPE = 2
PRICE_SIMILARITY_BAND = 0.25            # +/-25% of product price
PRICE_SIMILARITY_STRONG_SHARE = 0.50    # >=50% of anchors within band
STRONG_GRADE_SHARE = 0.50               # >50% A/B = strong source grades


def _normalize_brand_for_lookup(name: str) -> str:
    """Same normalization as evidence.py — keep these in sync."""
    if not name:
        return ""
    return " ".join(name.lower().split())


def _entry(kind: str, signal: str, text: str) -> dict:
    """Build a single typed ledger entry."""
    return {"kind": kind, "signal": signal, "text": text}


def _anchor_records(
    forecast: Forecast,
    record_by_brand: dict,
) -> list[GroundTruthRecord]:
    """Resolve forecast.neighbors back to GroundTruthRecord via lookup map.

    Returns only records that were found; missing brands are skipped
    silently (defensive — shouldn't happen if api.py builds map correctly).
    """
    out: list[GroundTruthRecord] = []
    for n in forecast.neighbors:
        key = _normalize_brand_for_lookup(n.brand)
        record = record_by_brand.get(key)
        if record is not None:
            out.append(record)
    return out


def _has_quality_warning(forecast: Forecast, substring: str) -> bool:
    """Check whether forecast.data_quality.quality_warnings contains a substring."""
    warnings = getattr(forecast.data_quality, "quality_warnings", None) or []
    return any(substring.lower() in (w or "").lower() for w in warnings)


def build_confidence_ledger(
    forecast: Forecast,
    product: ProductBrief,
    evidence_buckets: dict,
    record_by_brand: dict,
    inferred_subtype: Optional[str] = None,
) -> list[dict]:
    """Build the Confidence Ledger from forecast + classified evidence buckets.

    Parameters
    ----------
    forecast : Forecast
        Source forecast object.
    product : ProductBrief
        Source product brief.
    evidence_buckets : dict
        Output of evidence.classify_evidence_buckets(). Used for
        anchor strength counts.
    record_by_brand : dict
        Lookup map: normalized_brand_name -> GroundTruthRecord.
        Same map used by evidence.classify_evidence_buckets().
    inferred_subtype : str | None
        Subtype inferred for the query at the api.py layer. Pass-through
        only; this module does not re-infer subtype.

    Returns
    -------
    list[dict]
        Typed ledger entries: [{kind, signal, text}, ...].
    """
    entries: list[dict] = []
    fallback = bool(getattr(forecast, "fallback_used", False))

    # ── ANCHOR COUNT (always evaluated) ─────────────────────────────
    eligible_count = getattr(forecast, "eligible_neighbor_count", 0) or 0
    if eligible_count >= MIN_STRONG_ANCHOR_COUNT:
        entries.append(_entry(
            "positive",
            "strong_anchor_count",
            f"{eligible_count} forecast anchors were found for this product.",
        ))
    else:
        entries.append(_entry(
            "negative",
            "low_anchor_count",
            f"Fewer than {MIN_STRONG_ANCHOR_COUNT} eligible forecast anchors were found.",
        ))

    # ── FALLBACK FLAG (always emitted when triggered) ───────────────
    if fallback:
        entries.append(_entry(
            "negative",
            "fallback_used",
            "Assembly did not find enough direct comparable anchors, so this forecast uses a cautious fallback prior.",
        ))

    # ── ANCHOR-DERIVED SIGNALS (suppressed under fallback) ──────────
    if not fallback:
        forecast_anchors = (evidence_buckets or {}).get("forecast_anchors", []) or []

        # SUBTYPE MATCH
        direct_count = sum(
            1 for a in forecast_anchors
            if a.get("anchor_strength") == "direct"
        )
        any_mismatch = any(
            a.get("anchor_strength") in ("adjacent", "weak")
            for a in forecast_anchors
        )

        if direct_count >= MIN_DIRECT_ANCHORS_FOR_STRONG_SUBTYPE:
            entries.append(_entry(
                "positive",
                "direct_subtype_match",
                f"{direct_count} of {len(forecast_anchors)} forecast anchors are direct same-subtype matches.",
            ))
        if any_mismatch:
            entries.append(_entry(
                "negative",
                "category_subtype_mismatch",
                "Some forecast anchors are adjacent or weak category matches, "
                "so this estimate should be treated as directional.",
            ))

        # VARIANCE
        if _has_quality_warning(forecast, "variance"):
            entries.append(_entry(
                "negative",
                "high_anchor_variance",
                "Trial rates among forecast anchors vary widely; the true outcome may shift with execution.",
            ))
        else:
            entries.append(_entry(
                "positive",
                "low_anchor_variance",
                "Forecast anchors show consistent trial rates with low variance.",
            ))

        # SOURCE GRADES
        anchor_records = _anchor_records(forecast, record_by_brand)
        if not anchor_records:
            entries.append(_entry(
                "neutral",
                "source_grades_unavailable",
                "Source grades for forecast anchors could not be evaluated.",
            ))
        else:
            ab_count = sum(1 for r in anchor_records if r.confidence in ("A", "B"))
            if ab_count / len(anchor_records) > STRONG_GRADE_SHARE:
                entries.append(_entry(
                    "positive",
                    "strong_source_grades",
                    f"{ab_count} of {len(anchor_records)} forecast anchors are graded A or B (strong evidence).",
                ))
            else:
                entries.append(_entry(
                    "negative",
                    "weak_source_grades",
                    "Most forecast anchors rely on weaker source grades (C or D).",
                ))

        # PRICE SIMILARITY
        product_price = getattr(product, "price", 0) or 0
        if product_price <= 0 or not anchor_records:
            entries.append(_entry(
                "neutral",
                "price_similarity_unavailable",
                "Price similarity could not be evaluated from the available anchor data.",
            ))
        else:
            band_low = product_price * (1 - PRICE_SIMILARITY_BAND)
            band_high = product_price * (1 + PRICE_SIMILARITY_BAND)
            # GroundTruthRecord doesn't carry numeric price, only price_band.
            # Friend Apr 29 spec asked for "% of anchors within +/-25% of price"
            # which requires per-anchor numeric price. Today we only have band
            # buckets ("budget"/"mid"/"premium"/"luxury") on records, so we
            # fall through to "unavailable" rather than compute against a
            # band-mapping that isn't first-class evidence.
            #
            # Phase 2 (DB expansion) will add numeric prices per record. Until
            # then, the neutral signal is the honest answer.
            entries.append(_entry(
                "neutral",
                "price_similarity_unavailable",
                "Price similarity requires per-anchor numeric prices; available data uses band buckets only.",
            ))

    # ── EVIDENCE GAPS (always emit until later phases ship) ─────────
    entries.append(_entry(
        "negative",
        "no_external_evidence",
        "External web or commerce evidence has not been added to this forecast yet.",
    ))
    entries.append(_entry(
        "negative",
        "no_shopify_outcomes",
        "No Shopify outcome calibration exists yet for this category.",
    ))

    return entries
