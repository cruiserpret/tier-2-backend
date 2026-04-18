"""
backend/dtc/bayesian_tuner.py — GODMODE 3.3 BAYESIAN OPTIMIZATION SCAFFOLDING

═══════════════════════════════════════════════════════════════════════════════
HONEST NOTES — READ THIS FIRST:

This is NOT reinforcement learning (PPO, A2C, TRPO).
This is Bayesian hyperparameter optimization using Gaussian processes.

WHY BAYESIAN OPT, NOT RL:
- RL needs 1,000s of training episodes with dense reward signals
- Bayesian opt works with 10-50 samples (perfect for small validation sets)
- RL needs differentiable action spaces — ours is discrete categorical
- Bayesian opt is the standard for hyperparameter tuning in ML (HyperOpt, Optuna)

WHEN TO UPGRADE TO REAL RL:
- 100+ validation products (currently: 5)
- Real-time reward signal (currently: batched offline eval)
- Continuous action space (currently: discrete category choices)
- Budget for 10,000+ training sims (currently: ~$67 per sweep)

REQUIREMENTS:
    pip install scikit-optimize
    pip install numpy

For YC demo purposes, running this with 5 products still demonstrates
a principled optimization approach — just label it "Bayesian calibration"
in the pitch, not "RL".
═══════════════════════════════════════════════════════════════════════════════

USAGE:
    python -m backend.dtc.bayesian_tuner
"""

import asyncio
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("⚠ scikit-optimize not installed. Run: pip install scikit-optimize")

from backend.dtc.calibration import (
    load_validation_products,
    run_single_product,
    compute_mae,
)


# ── Bayesian Optimization Search Space ────────────────────────────────────────
# Continuous ranges instead of discrete grid — Bayesian opt explores the
# full space and uses a Gaussian process to model the objective function.

SEARCH_SPACE = [
    Real(0.10, 0.45, name="BEHAVIORAL_COMPENSATION_COEF"),
    Real(0.25, 0.45, name="BEHAVIORAL_COMPENSATION_FLOOR"),
    Real(0.50, 0.85, name="COMPOUND_PENALTY_MULTIPLIER"),
] if SKOPT_AVAILABLE else []


# State for the async-to-sync bridge
_PRODUCTS_CACHE = None
_GROUND_TRUTHS_CACHE = None


async def evaluate_coefficients_async(coefficients: dict) -> float:
    """
    Run all validation products with given coefficients, return MAE.
    This is the objective function Bayesian opt minimizes.
    """
    global _PRODUCTS_CACHE, _GROUND_TRUTHS_CACHE

    if _PRODUCTS_CACHE is None:
        _PRODUCTS_CACHE = load_validation_products()
        _GROUND_TRUTHS_CACHE = [p["ground_truth_trial_rate"] for p in _PRODUCTS_CACHE]

    predictions = []
    for product in _PRODUCTS_CACHE:
        try:
            pred = await run_single_product(product, coefficients)
            predictions.append(pred)
        except Exception as e:
            print(f"  ⚠ {product['id']} failed: {e}")
            predictions.append(0.0)

    mae = compute_mae(predictions, _GROUND_TRUTHS_CACHE)
    return mae


def objective(params: list) -> float:
    """
    Synchronous wrapper for Bayesian optimizer.
    Converts params list to dict and runs async evaluation.
    """
    coefficients = {
        "BEHAVIORAL_COMPENSATION_COEF":  params[0],
        "BEHAVIORAL_COMPENSATION_FLOOR": params[1],
        "COMPOUND_PENALTY_MULTIPLIER":   params[2],
    }
    print(f"\n  Evaluating: {coefficients}")
    mae = asyncio.run(evaluate_coefficients_async(coefficients))
    print(f"  MAE = {mae:.4f}")
    return mae


def run_bayesian_optimization(n_calls: int = 20):
    """
    Run Bayesian optimization with n_calls iterations.
    Each iteration runs all 5 validation products.
    Total simulations: n_calls × 5 = 100 sims for n_calls=20.
    Cost: ~$50.
    """
    if not SKOPT_AVAILABLE:
        print("✗ Cannot run: scikit-optimize not installed.")
        print("  Install with: pip install scikit-optimize")
        return None

    print(f"\n═══ GM3.3 BAYESIAN OPTIMIZATION ═══")
    print(f"Search space: {len(SEARCH_SPACE)} dimensions (continuous)")
    print(f"Iterations: {n_calls}")
    print(f"Total sims: {n_calls * 5}")
    print(f"Estimated cost: ~${n_calls * 5 * 0.5:.2f}")
    print(f"")
    print(f"⚠ HONEST CAVEAT: With only 5 validation products, Bayesian opt")
    print(f"  will converge to the local minimum for OUR 5 products. It will")
    print(f"  NOT generalize to unseen products until we have 20+ validation")
    print(f"  products for cross-validation.\n")

    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=n_calls,
        n_initial_points=5,  # random exploration first
        acq_func="EI",       # Expected Improvement acquisition function
        random_state=42,
    )

    best_coeffs = {
        "BEHAVIORAL_COMPENSATION_COEF":  result.x[0],
        "BEHAVIORAL_COMPENSATION_FLOOR": result.x[1],
        "COMPOUND_PENALTY_MULTIPLIER":   result.x[2],
    }

    print(f"\n═══ BAYESIAN OPT COMPLETE ═══")
    print(f"Best MAE: {result.fun:.4f}")
    print(f"Best coefficients:")
    for k, v in best_coeffs.items():
        print(f"  {k} = {v:.3f}")

    # Save results
    output_path = Path(__file__).parent / "bayesian_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_coefficients": best_coeffs,
            "best_mae": float(result.fun),
            "n_iterations": n_calls,
            "iteration_history": [
                {"iteration": i, "mae": float(y), "params": list(x)}
                for i, (y, x) in enumerate(zip(result.func_vals, result.x_iters))
            ],
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    return best_coeffs


# ── Cross-Validation (Leave-One-Out) ──────────────────────────────────────────

async def leave_one_out_validation(coefficients: dict):
    """
    LOOCV: For each product, train on other 4 and validate on the held-out one.
    Gives us an honest estimate of generalization with small data.

    With 5 products, we get 5 train/test splits.
    """
    products = load_validation_products()
    print(f"\n═══ LEAVE-ONE-OUT CROSS-VALIDATION ═══")
    print(f"Coefficients: {coefficients}")

    errors = []
    for i, test_product in enumerate(products):
        pred = await run_single_product(test_product, coefficients)
        actual = test_product["ground_truth_trial_rate"]
        error = abs(pred - actual)
        errors.append(error)
        print(f"Hold-out: {test_product['id']} → "
              f"predicted {pred*100:.1f}% | actual {actual*100:.1f}% | "
              f"error {error*100:.1f}%")

    mae = sum(errors) / len(errors)
    max_error = max(errors)
    print(f"\nLOOCV MAE: {mae:.4f}")
    print(f"Worst-case error: {max_error:.4f}")
    return mae, max_error


if __name__ == "__main__":
    if not SKOPT_AVAILABLE:
        print("\nFallback: run grid search via calibration.py")
        print("  python -m backend.dtc.calibration")
        sys.exit(1)

    best = run_bayesian_optimization(n_calls=20)
    if best:
        print(f"\nRunning LOOCV with best coefficients...")
        asyncio.run(leave_one_out_validation(best))