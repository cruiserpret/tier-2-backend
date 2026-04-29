# Retrieval Drift Cases — Captured Apr 29, 2026

Source: `backend/dtc_v3/eval_baseline_apr29.json` (25 cases, run on tier2-v3-ui-rebuild HEAD 8fb57b0).

These are cases where the v3-lite retrieval pipeline pulled anchors that look like cross-category drift, NOT legitimate semantic matches. Captured as input to future Phase 2 (DB expansion) and any later retrieval tuning. Phase 1 (Evidence Panel) does not fix these — it only surfaces them honestly in the report via anchor-strength labels and Confidence Ledger bullets.

Per friend's Apr 29 ruling: do not touch SIMILARITY_FLOOR, category bonus, source weights, weighted median formula, coverage gates, confidence caps, or retrieval ranking until a separate gated patch is approved.

---

## Drift Case 1 — cloudfoam_mattress

- **Product:** Cloudfoam Mattress (mattress)
- **Expected subtype:** mattress
- **Actual top anchors:** Casper Original Mattress (4.0%), Purple Original Mattress (3.0%), Brooklinen Luxe Sheets (8.0%)
- **Forecast:** 4.0%, confidence: high, coverage: strong, fallback: False
- **Drift:** Brooklinen Luxe Sheets is bedding, not a mattress. Pulled into forecast math at full weight.
- **Likely cause:** DB sparsity in mattress subtype (only Casper and Purple) — retrieval reaches into adjacent bedding category to fill the top-K slot.

## Drift Case 2 — luxebottle_drinkware

- **Product:** LuxeBottle Drinkware (premium drinkware)
- **Expected subtype:** premium_drinkware
- **Actual top anchors:** YETI Rambler 20oz (16.0%), Purple Original Mattress (3.0%), Casper Original Mattress (4.0%)
- **Forecast:** 16.0%, confidence: medium-high, coverage: strong, fallback: False
- **Drift:** Mattresses pulled in alongside YETI as drinkware anchors.
- **Likely cause:** Premium-drinkware subtype has only YETI as a record; retrieval cross-category leaks to high-AOV "premium home" products.

## Drift Case 3 — smoothshave_razor

- **Product:** SmoothShave Razor (razor subscription)
- **Expected subtype:** razor_subscription
- **Actual top anchors:** Dollar Shave Club Starter Kit (16.5%), Glossier Boy Brow (17.5%), Liquid IV Hydration Multiplier (20.5%)
- **Forecast:** 16.5%, confidence: medium-high, coverage: strong, fallback: False
- **Drift:** Glossier (beauty) and Liquid IV (hydration supplement) are not razor analogs.
- **Likely cause:** Sparse razor subtype (likely just Dollar Shave Club) — retrieval expands to "DTC subscription consumables" category-wide, mixing beauty + supplements with razors.

## Drift Case 4 — glowdrop_vitamin_c

- **Product:** GlowDrop Vitamin C (skincare active)
- **Expected subtype:** skincare_active (vitamin C serum)
- **Actual top anchors:** The Ordinary Niacinamide (14.0%), Glossier Boy Brow (17.5%), Dollar Shave Club Starter Kit (16.5%)
- **Forecast:** 14.0%, confidence: medium, coverage: medium, fallback: False
- **Drift:** Boy Brow (eyebrow gel) and Dollar Shave Club (razor) flagged as anchors for a vitamin C serum.
- **Likely cause:** skincare_active subtype is mixed — only The Ordinary is a true active. Glossier and Dollar Shave Club fill via "personal care" category.

## Drift Case 5 — clearskin_acne_patch

- **Product:** ClearSkin Acne Patch (skincare active)
- **Expected subtype:** skincare_active (acne patches)
- **Actual top anchors:** The Ordinary Niacinamide (14.0%), Glossier Boy Brow (17.5%), Dollar Shave Club Starter Kit (16.5%)
- **Forecast:** 14.0%, confidence: medium, coverage: medium, fallback: False
- **Drift:** Identical anchor set to glowdrop_vitamin_c despite product being a topical patch, not a serum.
- **Likely cause:** Same-as Case 4. Subtype routing collapses all "skincare_active" to the same neighbor set regardless of form factor (serum vs. patch).

---

## Aggregate observations

- All 5 drift cases sit in subtypes with ≤2 records in the current 37-brand DB.
- Drift behaves as: when subtype is sparse, retrieval leaks across category boundaries and pulls high-similarity neighbors from adjacent commerce categories (beauty ↔ skincare ↔ subscription ↔ supplements).
- Forecast math at full weight on these adjacent neighbors. Coverage gate did NOT downgrade confidence (strong/medium tiers, not weak) because the eligible-count gate still passed.
- These are exactly the cases the Phase 1 Evidence Panel must label honestly. Confidence Ledger should surface category/subtype mismatch as a minus bullet so users see "medium-high confidence" and "razor anchored on hydration supplement" simultaneously.

## Routing for later phases

- **Phase 2 (DB expansion):** prioritize packs that fill these sparse subtypes — mattress (more anchors beyond Casper/Purple), premium drinkware (more outdoor/lifestyle), razor subscription (more shave/grooming), skincare active (separate serums vs. patches vs. niacinamide-class).
- **Phase 7 (later retrieval-tuning patch):** consider category-bonus reweighting, hard subtype-mismatch penalty, or separate Exploratory retrieval pass distinct from forecast retrieval.
- **No changes to retrieval logic in Phase 1.**
