# Production Baseline — Assembly Tier 2

**Last Updated:** 2026-04-19
**Status:** LOCKED. Do not modify production without updating this document.

---

## Current Production State

**Commit hash:** `fd10ee6`
**Commit message:** `LOCK: Deploy 1 coefficients (best validation: 5/10 A-grade, MAE 5.78%)`
**Branch deployed:** `main` (pushed via `git push tier2 hamza/railway-deploy:main --force`)
**Railway URL:** `https://tier-2-backend-production.up.railway.app`

## Locked Coefficients

Located in `backend/dtc/market_report_agent.py`:

```python
BEHAVIORAL_COMPENSATION_COEF = 0.15
BEHAVIORAL_COMPENSATION_FLOOR = 0.40
COMPOUND_PENALTY_MULTIPLIER = 0.80
```

## Validation Performance

Last full validation: 10 products, 50-agent runs.

| Metric | Value |
|--------|-------|
| A-grade predictions | 5/10 (50%) |
| MAE | 5.78% |
| In-range hits | 2/10 |
| Correct verdict direction | 9/10 |

## A-Grade Products (Regression Invariants)

These products MUST remain A-grade in any future change. If a change drops any of these below A-grade, it is a **regression** and must not ship to production.

| Product | Ground Truth Range | Deploy 1 Prediction | Grade |
|---------|-------------------|---------------------|-------|
| Olipop Prebiotic Soda 12-pack | 24-30% | 26.4% | A+ |
| YETI Rambler 20oz Travel Mug | 14-18% | 14.8% | A+ |
| Oura Ring Gen 3 | 6-9% | 7.6% | A+ |
| Everlane Cotton T-Shirt | 8-12% | 10.9% | A+ |
| Warby Parker Prescription Glasses | 11-14% | 14.6% | A |
| AG1 Athletic Greens | 9-13% | 8.1% | A |

## Known Limitations (Out of Scope for v1.0)

These products consistently fail and are outside target customer range. Not blocking:

- **Casper Mattress ($1,095):** 13.4% predicted vs 3-5% ground truth. Category has low repurchase rate (every 8-10 years). Structural fix requires purchase-frequency modeling.
- **Hims GLP-1 ($199 + $249/mo):** 15.8% predicted vs 4-7% ground truth. Medical Rx has unique regulatory friction. Out of Pransh's target customer scope.
- **Dollar Shave Club ($10 + $20/mo):** 9.4% predicted vs 15-18% ground truth. Low-commitment subscription logic needs refinement.
- **Liquid Death ($20):** 15.1% predicted vs 18-22% ground truth. Near-miss, may self-correct with better validation coverage.

## Research Basis

Core math is research-backed:
- Juster (1966): intent-to-behavior scale
- Deffuant (2000): bounded confidence opinion dynamics
- Morwitz (1993): behavioral compensation in intent surveys
- Chandon (2005): purchase intent and actual purchase
- Monroe (2003): reference price and elasticity
- Van Westendorp (1976): price sensitivity meter
- Burnham (2003): switching costs

## Revert Instructions

If production is ever compromised, revert to this baseline:

```bash
cd ~/Desktop/Aseembly/assembly-backend
git checkout fd10ee6 -- backend/dtc/market_report_agent.py
git checkout fd10ee6 -- backend/dtc/dtc_ingestor.py

# Verify revert is clean:
grep "BEHAVIORAL_COMPENSATION_COEF = 0.15" backend/dtc/market_report_agent.py
grep "BEHAVIORAL_COMPENSATION_FLOOR = 0.40" backend/dtc/market_report_agent.py
grep "COMPOUND_PENALTY_MULTIPLIER = 0.80" backend/dtc/market_report_agent.py

# Commit and push:
git add backend/dtc/market_report_agent.py backend/dtc/dtc_ingestor.py
git commit -m "REVERT: restore Deploy 1 baseline"
git push tier2 hamza/railway-deploy:main --force
```

## Customer-Facing Disclosure

Current report label: **"Assembly Tier 2 — Beta v0.1"**

Customer messaging:
> "Validated against 10 DTC products (Nielsen, IQVIA, Mintel sources). Current accuracy: 50% A-grade predictions, 90% correct verdict direction. Use for directional intelligence. Not a substitute for traditional research."

---

*This document must be updated whenever production changes. Treat every production push as a version update requiring this file's refresh.*
