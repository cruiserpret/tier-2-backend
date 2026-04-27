"""
backend/dtc_v3/calibration/fit_beta_coefficients.py

Step 2 of friend's calibration sequence:
Fit 3 beta coefficients via ridge regression on RAG residuals.

Per friend's spec:
- Target: logit(gt_mid) - logit(rag_prior)
- Features: desirability_z, awareness_z, friction_z
- Sample weights: confidence_weight (A=1.0, B=0.6, C=0.25)
- Method: Ridge regression (alpha=1.0)
- Sign constraints applied post-hoc:
    beta_desirability = max(beta_desirability, 0)
    beta_awareness = max(beta_awareness, 0)
    beta_friction = min(beta_friction, 0)
- Shrinkage: beta *= 0.7 (anti-overfitting on small dataset)
- Clean 16 rows only

Output: beta_coefficients_v1.json
"""

from __future__ import annotations
import json
import math
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

CONTAMINATED_BRANDS = {
    "Athletic Brewing NA Beer",
    "MUD\\WTR Coffee Alternative",
    "Liquid IV Hydration Multiplier",
}

CALIBRATION_DIR = Path(__file__).parent
SIGNAL_TABLE = CALIBRATION_DIR / "signal_table_v0_from_v2.jsonl"
NORMALIZERS = CALIBRATION_DIR / "normalizer_constants_v1.json"
OUTPUT = CALIBRATION_DIR / "beta_coefficients_v1.json"

RIDGE_ALPHA = 1.0
SHRINKAGE = 0.7


def logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def main():
    # Load normalizers
    norm = json.loads(NORMALIZERS.read_text())["normalizers"]
    print(f"Loaded normalizers from {NORMALIZERS.name}")

    # Load signal table
    rows = []
    with open(SIGNAL_TABLE) as f:
        for line in f:
            rows.append(json.loads(line))

    # Filter clean rows
    clean = [r for r in rows if r["brand"] not in CONTAMINATED_BRANDS]
    print(f"Clean rows for beta fitting: {len(clean)}")

    # Build feature matrix and target
    X = []
    y = []
    weights = []
    metadata = []
    for r in clean:
        des_z = (r["desirability_raw"] - norm["desirability"]["center"]) / norm["desirability"]["scale"]
        awa_z = (r["awareness_raw"] - norm["awareness"]["center"]) / norm["awareness"]["scale"]
        fri_z = (r["friction_raw"] - norm["friction"]["center"]) / norm["friction"]["scale"]
        target = logit(r["gt_mid"]) - logit(r["rag_prior"])
        X.append([des_z, awa_z, fri_z])
        y.append(target)
        weights.append(r["confidence_weight"])
        metadata.append({
            "brand": r["brand"],
            "des_z": des_z, "awa_z": awa_z, "fri_z": fri_z,
            "target": target, "weight": r["confidence_weight"],
        })

    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)

    print()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Sample weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    print()

    # Show pre-fit feature stats
    print("Feature stats (z-scored, weighted):")
    for i, name in enumerate(["desirability_z", "awareness_z", "friction_z"]):
        col = X[:, i]
        print(f"  {name:<18} mean={col.mean():+.3f}  std={col.std():.3f}  range=[{col.min():+.2f}, {col.max():+.2f}]")
    print()

    # Fit ridge regression
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    ridge.fit(X, y, sample_weight=weights)
    raw_betas = ridge.coef_

    print(f"Raw ridge coefficients (alpha={RIDGE_ALPHA}):")
    print(f"  desirability:  {raw_betas[0]:+.4f}")
    print(f"  awareness:     {raw_betas[1]:+.4f}")
    print(f"  friction:      {raw_betas[2]:+.4f}")
    print()

    # Apply sign constraints
    constrained = [
        max(raw_betas[0], 0),  # desirability ≥ 0
        max(raw_betas[1], 0),  # awareness ≥ 0
        min(raw_betas[2], 0),  # friction ≤ 0
    ]

    sign_violations = []
    if raw_betas[0] < 0:
        sign_violations.append(f"desirability was {raw_betas[0]:+.4f} (clamped to 0)")
    if raw_betas[1] < 0:
        sign_violations.append(f"awareness was {raw_betas[1]:+.4f} (clamped to 0)")
    if raw_betas[2] > 0:
        sign_violations.append(f"friction was {raw_betas[2]:+.4f} (clamped to 0)")

    if sign_violations:
        print("⚠ Sign constraint violations:")
        for v in sign_violations:
            print(f"   {v}")
    else:
        print("✓ All ridge coefficients had expected signs")
    print()

    # Apply shrinkage
    final_betas = [b * SHRINKAGE for b in constrained]

    print(f"After shrinkage (×{SHRINKAGE}):")
    print(f"  beta_desirability:  {final_betas[0]:+.4f}")
    print(f"  beta_awareness:     {final_betas[1]:+.4f}")
    print(f"  beta_friction:      {final_betas[2]:+.4f}")
    print()

    # Validation: predict residuals + compute MAE
    predicted_targets = X @ np.array(final_betas)
    residual_errors = np.abs(predicted_targets - y)
    weighted_mae_logit = (residual_errors * weights).sum() / weights.sum()

    # Also compute predicted_trial_rate vs gt_mid
    print("Per-product prediction (post-fit):")
    print(f"{'Brand':<35}{'GT':>6}{'RAG':>6}{'Adj':>8}{'Pred':>7}{'Err':>7}")
    print("-" * 70)
    pred_trial_rates = []
    actual_trial_rates = []
    for r, target_pred in zip(clean, predicted_targets):
        # Apply clamping (per friend: clamp(adjustment, -0.75, 0.75))
        clamped_adj = max(-0.75, min(0.75, target_pred))
        final_pred_logit = logit(r["rag_prior"]) + clamped_adj
        final_pred = 1 / (1 + math.exp(-final_pred_logit))
        err = abs(final_pred - r["gt_mid"])
        pred_trial_rates.append(final_pred)
        actual_trial_rates.append(r["gt_mid"])
        print(f"{r['brand'][:33]:<35}"
              f"{r['gt_mid']*100:5.1f}%"
              f"{r['rag_prior']*100:5.1f}%"
              f"{clamped_adj:+7.3f} "
              f"{final_pred*100:5.1f}%"
              f"{err*100:6.1f}pp")

    pred_arr = np.array(pred_trial_rates)
    actual_arr = np.array(actual_trial_rates)
    mae_pp = np.mean(np.abs(pred_arr - actual_arr)) * 100
    in_range_count = sum(
        1 for r, p in zip(clean, pred_trial_rates)
        if r["gt_low"] <= p <= r["gt_high"]
    )

    print()
    print(f"MAE (residual logit, weighted): {weighted_mae_logit:.4f}")
    print(f"MAE (trial rate, unweighted):   {mae_pp:.2f}pp")
    print(f"In ground-truth range:          {in_range_count}/{len(clean)} ({in_range_count*100//len(clean)}%)")
    print()

    # Save
    output = {
        "version": "v1",
        "method": f"Ridge regression (alpha={RIDGE_ALPHA}) + sign constraints + shrinkage ({SHRINKAGE})",
        "computed_from": "signal_table_v0_from_v2.jsonl",
        "n_clean_rows_used": len(clean),
        "excluded_brands": list(CONTAMINATED_BRANDS),
        "ridge_alpha": RIDGE_ALPHA,
        "shrinkage_factor": SHRINKAGE,
        "raw_ridge_coefficients": {
            "desirability": float(raw_betas[0]),
            "awareness": float(raw_betas[1]),
            "friction": float(raw_betas[2]),
        },
        "sign_violations": sign_violations,
        "final_betas": {
            "desirability": float(final_betas[0]),
            "awareness": float(final_betas[1]),
            "friction": float(final_betas[2]),
        },
        "validation_metrics": {
            "weighted_mae_logit": float(weighted_mae_logit),
            "mae_trial_rate_pp": float(mae_pp),
            "in_range_count": in_range_count,
            "in_range_pct": float(in_range_count / len(clean) * 100),
        },
    }
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved to {OUTPUT.name}")


if __name__ == "__main__":
    main()
