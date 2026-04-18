"""
backend/dtc/calibration.py — GODMODE 3.3 GRID-SEARCH CALIBRATION

═══════════════════════════════════════════════════════════════════════════════
HONEST: This is grid-search calibration, not reinforcement learning.
For 5 validation products, grid search is the right tool.
Upgrade to Bayesian optimization (bayesian_tuner.py) with 20+ products.
Upgrade to real RL (PPO/TRPO) with 100+ products.
═══════════════════════════════════════════════════════════════════════════════

USAGE:
    python -m backend.dtc.calibration

This runs the simulation for each validation product, with a grid of
coefficient combinations, and outputs the best-performing set.

OUTPUT: calibration_results.json with MSE ranking of coefficient sets.

COST: ~$0.50 per sim × 5 products × 27 combinations = ~$67.50 per full sweep
      Reduce grid size for cheaper runs.
"""

import asyncio
import json
import sys
import os
import itertools
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.dtc.dtc_ingestor import ProductBrief, run_market_ingestion
from backend.dtc.buyer_persona_generator import generate_buyer_personas
from backend.dtc.market_debate_engine import run_market_debate
from backend.dtc.market_report_agent import generate_market_report


# ── Grid-Search Coefficient Space ─────────────────────────────────────────────
# Each coefficient is swept over a small range, creating a grid.
# 5 products × 27 combinations = 135 simulations per sweep.
# Runtime: ~30 minutes on Railway. Cost: ~$67.

COEFFICIENT_GRID = {
    "BEHAVIORAL_COMPENSATION_COEF": [0.15, 0.25, 0.35],
    "BEHAVIORAL_COMPENSATION_FLOOR": [0.30, 0.35, 0.40],
    "COMPOUND_PENALTY_MULTIPLIER":   [0.55, 0.65, 0.75],
}


def load_validation_products():
    """Load ground truth data from validation_products.json."""
    json_path = Path(__file__).parent / "validation_products.json"
    with open(json_path) as f:
        data = json.load(f)
    return data["products"]


def compute_mse(predicted: list[float], ground_truth: list[float]) -> float:
    """Mean Squared Error between predictions and ground truth."""
    if len(predicted) != len(ground_truth):
        raise ValueError("Length mismatch")
    n = len(predicted)
    return sum((p - g) ** 2 for p, g in zip(predicted, ground_truth)) / n


def compute_mae(predicted: list[float], ground_truth: list[float]) -> float:
    """Mean Absolute Error — easier to interpret than MSE."""
    n = len(predicted)
    return sum(abs(p - g) for p, g in zip(predicted, ground_truth)) / n


def compute_in_range_score(predicted: list[float], ranges: list[tuple]) -> float:
    """Proportion of predictions within ground-truth range."""
    hits = sum(1 for p, (low, high) in zip(predicted, ranges) if low <= p <= high)
    return hits / len(predicted) if predicted else 0.0


async def run_single_product(product_data: dict, coefficients: dict) -> float:
    """
    Run Assembly sim for one product with given coefficients.
    Returns the predicted trial rate (0-1).
    """
    # Monkey-patch the coefficients onto market_report_agent
    from backend.dtc import market_report_agent as mra
    mra.BEHAVIORAL_COMPENSATION_COEF = coefficients["BEHAVIORAL_COMPENSATION_COEF"]
    mra.BEHAVIORAL_COMPENSATION_FLOOR = coefficients["BEHAVIORAL_COMPENSATION_FLOOR"]
    mra.COMPOUND_PENALTY_MULTIPLIER = coefficients["COMPOUND_PENALTY_MULTIPLIER"]

    # Build product brief
    product = ProductBrief(
        name=product_data["name"],
        description=f"{product_data['name']} — calibration run",
        price=product_data["price"],
        category=product_data["category"],
        competitors=[],  # calibration runs use minimal competitors
    )

    # Run pipeline
    intel = await run_market_ingestion(product, num_agents=50)
    agents = generate_buyer_personas(intel, num_agents=50)
    debate = await run_market_debate(agents, intel, simulation_id=f"cal_{product_data['id']}")
    report = await generate_market_report(intel, debate, simulation_id=f"cal_{product_data['id']}")

    return report["juster_trial_rate"]["trial_rate_pct"] / 100.0


async def run_calibration_sweep():
    """
    Main calibration entry point.
    Grid-searches coefficients, returns best performer by MAE.
    """
    products = load_validation_products()
    ground_truths = [p["ground_truth_trial_rate"] for p in products]
    ground_ranges = [tuple(p["ground_truth_range"]) for p in products]

    # Generate all coefficient combinations
    keys = list(COEFFICIENT_GRID.keys())
    values = list(COEFFICIENT_GRID.values())
    combinations = list(itertools.product(*values))

    print(f"\n═══ GM3.3 CALIBRATION SWEEP ═══")
    print(f"Products: {len(products)}")
    print(f"Coefficient combinations: {len(combinations)}")
    print(f"Total simulations: {len(products) * len(combinations)}")
    print(f"Estimated cost: ~${len(products) * len(combinations) * 0.5:.2f}\n")

    results = []

    for combo_idx, combo in enumerate(combinations):
        coeffs = dict(zip(keys, combo))
        print(f"\n[{combo_idx+1}/{len(combinations)}] Testing: {coeffs}")

        predictions = []
        for product in products:
            try:
                pred = await run_single_product(product, coeffs)
                predictions.append(pred)
                print(f"  {product['id']}: predicted {pred*100:.1f}% | "
                      f"ground truth {product['ground_truth_trial_rate']*100:.1f}%")
            except Exception as e:
                print(f"  {product['id']}: FAILED ({e})")
                predictions.append(0.0)

        mse = compute_mse(predictions, ground_truths)
        mae = compute_mae(predictions, ground_truths)
        in_range = compute_in_range_score(predictions, ground_ranges)

        results.append({
            "coefficients": coeffs,
            "predictions": predictions,
            "mse": mse,
            "mae": mae,
            "in_range_score": in_range,
        })
        print(f"  MSE={mse:.4f} | MAE={mae:.4f} | In-range={in_range*100:.0f}%")

    # Sort by MAE (lower is better) — more interpretable than MSE
    results.sort(key=lambda r: r["mae"])

    print(f"\n═══ TOP 5 COEFFICIENT SETS (by MAE) ═══")
    for i, r in enumerate(results[:5]):
        print(f"#{i+1}: MAE={r['mae']:.4f} | In-range={r['in_range_score']*100:.0f}%")
        print(f"     {r['coefficients']}")

    # Save full results
    output_path = Path(__file__).parent / "calibration_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "sweep_metadata": {
                "n_products": len(products),
                "n_combinations": len(combinations),
                "total_simulations": len(products) * len(combinations),
            },
            "best_coefficients": results[0]["coefficients"],
            "best_mae": results[0]["mae"],
            "all_results": results,
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ Best coefficients: {results[0]['coefficients']}")
    print(f"✓ Best MAE: {results[0]['mae']:.4f}")

    return results[0]


if __name__ == "__main__":
    asyncio.run(run_calibration_sweep())