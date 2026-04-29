# Retrieval Drift Cases — Apr 30, 2026

**Status:** Phase 2 input
**Source:** Phase 1 (P1.6) evidence panel surfaced these cases when rendered
**Friend ruling Apr 30:** Do NOT fix in Phase 1. Phase 1 = transparency only.
Fixes belong in Phase 2 (DB expansion, retrieval purity) with before/after evals.

---

## Case 1 — Liquid IV (excluded) anchors on energy drinks

When `exclude_brand="Liquid IV"`, the forecast pulls 8 forecast anchors at
similarity floor 0.45+. The forecast is deterministic at 22.0% but the
underlying anchor set crosses category boundaries.

### Direct anchors (correct, hydration_supplement subtype)
- Pedialyte — 22.0% trial, sim 0.79
- DripDrop Hydration — 9.0% trial, sim 0.75
- LMNT Electrolytes — 6.5% trial, sim 0.78

### Adjacent / weak anchors entering forecast math
- Liquid Death Mountain Water — 20.0% trial, sim 0.59 (weak, branded water)
- Monster Energy — 20.0% trial, sim 0.46 (weak, energy drink)
- Red Bull — 22.0% trial, sim 0.46 (weak, energy drink)
- AG1 Athletic Greens — 11.0% trial, sim 0.55 (adjacent, greens powder)
- Celsius — 14.0% trial, sim 0.50 (weak, energy drink)

### Risk

The 22.0% forecast is deterministic and reproducible. But it is not purely
hydration-subtype anchored. Three of eight anchors are direct hydration
matches; five are cross-category beverages. The forecast lands at 22.0%
partly because A-grade direct anchors (Pedialyte) and weak anchors with
similar trial rates (Monster, Red Bull) happen to coincide numerically.

In Phase 1 this is shown transparently:
- Evidence panel labels each anchor (direct/adjacent/weak)
- Confidence ledger emits negative signal `category_subtype_mismatch`
  with text: "Some forecast anchors are adjacent or weak category matches,
  so this estimate should be treated as directional."

The drift is visible. The forecast math itself is unchanged.

### Phase 2 candidate fixes

Per friend Apr 30, the following are candidates for Phase 2 — to be
evaluated as a separate permission-gated block with before/after evals:

1. **Subtype purity weighting** — downweight non-direct anchors when
   computing weighted median; e.g., direct=1.0, adjacent=0.5, weak=0.25
   relative weight in the median calculation.

2. **Category/subtype gate before weighted median** — require N direct
   anchors before allowing adjacent/weak anchors to participate in the
   forecast prior. If insufficient direct, fall back rather than dilute.

3. **Source-grade × subtype-strength weight interaction** — current
   `source_weight` (A=1.0, B=0.6, C=0.25, D=0.1) is applied independent
   of subtype strength. Combined weighting could prevent A-grade weak
   anchors from dominating direct C-grade matches.

4. **Stronger hydration_supplement DB expansion** — current DB has only
   3 hydration-subtype brands (Pedialyte, DripDrop, LMNT). Adding more
   direct comparables (Nuun, Hydrant, Cure, Pickle Juice, Skratch,
   Mountain Drop, etc.) reduces the model's reliance on cross-category
   neighbors when retrieval falls back.

5. **Confidence cap when weak anchors contribute materially** — if >50%
   of weighted median mass comes from non-direct anchors, hard-cap
   confidence to medium-low regardless of other signals.

### Acceptance criteria for any Phase 2 fix

- Liquid IV (excluded) forecast anchors should be ≥80% direct
  hydration_supplement subtype matches OR confidence should drop to medium
  with explicit ledger signal.
- Liquid IV trial rate forecast may shift from 22.0% to something else;
  this is acceptable. Determinism (5/5 identical runs) MUST be preserved.
- No regression on test_invariants.py — all 12 tests must pass.
- Eval suite MAE comparison: before vs. after must be reported per case.

---

## Other cases

(Carried forward from RETRIEVAL_DRIFT_CASES_APR29.md — Phase 0 baseline.)

- cloudfoam_mattress → Brooklinen drift (luxury bedding wrongly anchoring DTC mattress)
- luxebottle_drinkware → Casper / Purple drift (mattress brands anchoring drinkware)
- smoothshave_razor → Glossier / LiquidIV drift (cross-category)
- glowdrop_vitamin_c → Glossier drift (single anchor only)
- clearskin_acne_patch → Glossier drift (single anchor only)

These remain Phase 2 priority. The Apr 30 Liquid IV case becomes the
first Phase 2 task because the evidence panel makes it visible to YC
viewers in real time.

---

## Phase 1 framing for demo

Per friend's Apr 30 ruling, Liquid IV is the "determinism + transparency"
demo case, not the "perfectly clean direct-anchor" demo case. Pitch:

> Here Assembly returns the same forecast every time, but it also shows
> the quality of the comparable evidence underneath. Three hydration
> anchors are direct matches, while several adjacent/weak beverage
> comparables are visibly labeled. We don't hide that — this is exactly
> how sellers know whether to trust, test, or collect more evidence.

Different demos for different points:
- **Liquid IV** = determinism + transparency
- **Triton Drinks** = category coverage after energy-drink expansion
- **YETI** = fallback honesty / low-confidence trust case
- **SoberCraft** = routing correction / NA beer specificity
