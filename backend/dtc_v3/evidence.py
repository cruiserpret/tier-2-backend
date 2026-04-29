"""
backend/dtc_v3/evidence.py — Phase 1 Evidence Panel classifier.

Pure-function module. Takes already-retrieved neighbors and
classifies them into three display buckets:

  forecast_anchors        — eligible neighbors used in forecast math
                            (similarity >= SIMILARITY_FLOOR, kept by
                            forecast.py and present in forecast.neighbors).
                            ALWAYS empty when forecast.fallback_used is True
                            because no neighbors were used in math.
  candidate_comparables   — retrieved but below floor (0.30 <= sim < 0.45),
                            NOT used in forecast math, supplied by
                            api.py via a separate broader-retrieval pass.
  fallback_neighbors      — retrieved during search but NOT used in forecast
                            math because forecast.fallback_used is True.
                            Empty when forecast did not fall back.
                            Per friend Apr 30 ruling: retrieved-but-unused
                            neighbors must NOT be presented as "weak forecast
                            anchors" — they are not anchors at all.
  exploratory_comparables — Phase 3 (dynamic discovery). Empty in Phase 1.

Per friend Apr 29 ruling:
  - This module performs NO retrieval, NO embedding calls, NO
    forecast math. It is a pure classifier.
  - api.py owns the broader-retrieval pass for candidates and
    passes them in.
  - evidence.py only classifies what it is given.

Anchor strength rule (forecast_anchors only):
  same_subtype_match=True                       -> "direct"
  same_category_match=True, subtype_match=False -> "adjacent"
  neither match                                 -> "weak"

Candidate / exploratory items do NOT receive an anchor_strength
label (they are not forecast anchors). They use bucket = "candidate_comparable"
or "exploratory_comparable" with display_warning text instead.
"""

from __future__ import annotations
from typing import Optional

from .models import Forecast, ProductBrief, Neighbor, GroundTruthRecord


def _normalize_brand_for_lookup(name: str) -> str:
    """Light brand normalization for record_by_brand dict lookup.

    Lowercases and collapses whitespace. Does NOT strip punctuation
    aggressively — record_by_brand keys are built with the same
    normalization in api.py, so consistency is what matters.
    """
    if not name:
        return ""
    return " ".join(name.lower().split())


def _classify_anchor_strength(
    neighbor: Neighbor,
    product: ProductBrief,
    record_by_brand: dict,
    inferred_subtype: Optional[str],
) -> tuple[bool, bool, str, Optional[str]]:
    """Classify a forecast-eligible neighbor as direct/adjacent/weak.

    Returns: (same_category_match, same_subtype_match,
              anchor_strength, display_warning)

    If neighbor.brand is not found in record_by_brand (defensive
    fallback — shouldn't happen if api.py builds the map correctly),
    treats the anchor as "weak" with a metadata-unavailable warning.
    """
    key = _normalize_brand_for_lookup(neighbor.brand)
    record: Optional[GroundTruthRecord] = record_by_brand.get(key)

    if record is None:
        # Defensive fallback — brand metadata missing
        return (
            False,
            False,
            "weak",
            "Brand metadata unavailable; treated as weak match.",
        )

    same_category_match = (
        bool(product.category)
        and bool(record.category)
        and product.category == record.category
    )
    same_subtype_match = (
        bool(inferred_subtype)
        and bool(record.category_subtype)
        and inferred_subtype == record.category_subtype
    )

    if same_subtype_match:
        return (
            same_category_match,
            True,
            "direct",
            None,
        )

    if same_category_match:
        return (
            True,
            False,
            "adjacent",
            "Used in forecast, but subtype match is adjacent rather than direct.",
        )

    return (
        False,
        False,
        "weak",
        "Used in forecast, but category and subtype both differ. Treat this estimate as directional.",
    )


def classify_evidence_buckets(
    forecast: Forecast,
    product: ProductBrief,
    record_by_brand: dict,
    candidate_neighbors: Optional[list] = None,
    inferred_subtype: Optional[str] = None,
) -> dict:
    """Classify forecast neighbors + candidate pool into evidence buckets.

    Parameters
    ----------
    forecast : Forecast
        The forecast object whose .neighbors will become forecast_anchors.
    product : ProductBrief
        Source product brief (for category comparison).
    record_by_brand : dict
        Lookup map: normalized_brand_name -> GroundTruthRecord. Built
        by api.py from GROUND_TRUTH_RECORDS.
    candidate_neighbors : list | None
        Optional list of Neighbor objects from a broader retrieval pass
        with similarity in [0.30, 0.45). Supplied by api.py. If None,
        candidate_comparables will be empty (Phase 1 incremental shipping).
    inferred_subtype : str | None
        Subtype inferred for the query, e.g. via _infer_query_subtype().
        Required for same_subtype_match classification. If None, all
        forecast anchors will be classified as adjacent or weak.

    Returns
    -------
    dict
        Non-fallback case:
            {
                "forecast_anchors": [...],
                "candidate_comparables": [...],
                "fallback_neighbors": [],
                "exploratory_comparables": []
            }
        Fallback case (forecast.fallback_used = True):
            {
                "forecast_anchors": [],
                "candidate_comparables": [],
                "fallback_neighbors": [...all retrieved neighbors...],
                "exploratory_comparables": []
            }
    """
    # ── Fallback case: forecast did not use these neighbors in math ──
    # Per friend Apr 30 ruling (Option A): when forecast.fallback_used is
    # True, forecast_anchors must be empty by definition. Retrieved
    # neighbors go into a separate fallback_neighbors bucket so the report
    # can show "we looked, found weak matches, did not pretend they were
    # strong evidence." anchor_strength is None for this bucket — these
    # are NOT forecast anchors.
    if getattr(forecast, "fallback_used", False):
        fallback_neighbors_out = []
        for n in forecast.neighbors:
            key = _normalize_brand_for_lookup(n.brand)
            record = record_by_brand.get(key)
            same_cat = False
            same_sub = False
            if record is not None:
                same_cat = (
                    bool(product.category)
                    and bool(record.category)
                    and product.category == record.category
                )
                same_sub = (
                    bool(inferred_subtype)
                    and bool(record.category_subtype)
                    and inferred_subtype == record.category_subtype
                )
            fallback_neighbors_out.append({
                "brand": n.brand,
                "similarity": round(n.similarity, 3),
                "trial_rate": n.trial_rate_mid,
                "bucket": "fallback_neighbor",
                "same_category_match": same_cat,
                "same_subtype_match": same_sub,
                "anchor_strength": None,
                "used_in_forecast": False,
                "display_warning": (
                    "Retrieved during search, but not used as a forecast anchor "
                    "because Assembly did not find enough eligible direct comparables."
                ),
            })
        return {
            "forecast_anchors": [],
            "candidate_comparables": [],
            "fallback_neighbors": fallback_neighbors_out,
            "exploratory_comparables": [],
        }

    # ── Normal case: classify eligible neighbors as forecast anchors ──
    forecast_anchors = []
    for n in forecast.neighbors:
        same_cat, same_sub, strength, warning = _classify_anchor_strength(
            neighbor=n,
            product=product,
            record_by_brand=record_by_brand,
            inferred_subtype=inferred_subtype,
        )
        forecast_anchors.append({
            "brand": n.brand,
            "similarity": round(n.similarity, 3),
            "trial_rate": n.trial_rate_mid,
            "bucket": "forecast_anchor",
            "same_category_match": same_cat,
            "same_subtype_match": same_sub,
            "anchor_strength": strength,
            "used_in_forecast": True,
            "display_warning": warning,
        })

    candidate_comparables = []
    if candidate_neighbors:
        for n in candidate_neighbors:
            # Candidates are below the forecast floor by definition.
            # We still record category/subtype match flags (they may
            # be true even though the similarity score is too low to
            # qualify the record for forecast math). anchor_strength
            # is None — candidates are not forecast anchors.
            key = _normalize_brand_for_lookup(n.brand)
            record = record_by_brand.get(key)
            same_cat = False
            same_sub = False
            if record is not None:
                same_cat = (
                    bool(product.category)
                    and bool(record.category)
                    and product.category == record.category
                )
                same_sub = (
                    bool(inferred_subtype)
                    and bool(record.category_subtype)
                    and inferred_subtype == record.category_subtype
                )
            candidate_comparables.append({
                "brand": n.brand,
                "similarity": round(n.similarity, 3),
                "trial_rate": n.trial_rate_mid,
                "bucket": "candidate_comparable",
                "same_category_match": same_cat,
                "same_subtype_match": same_sub,
                "anchor_strength": None,
                "used_in_forecast": False,
                "display_warning": "Considered, but did not meet the forecast anchor threshold.",
            })

    return {
        "forecast_anchors": forecast_anchors,
        "candidate_comparables": candidate_comparables,
        "fallback_neighbors": [],
        "exploratory_comparables": [],
    }
