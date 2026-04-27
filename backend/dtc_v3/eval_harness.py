"""
backend/dtc_v3/eval_harness.py — Multi-mode evaluation framework.

Modes:
    mode_0_v1 — existing v1 production
    mode_1_rag_only — RAG prior alone
    mode_2_rag_plus_persona — RAG + 3-feature persona delta (clamped)
    mode_3_rag_plus_calibrated — RAG + persona + ridge calibrator

Outputs metrics per mode for cross-comparison.
"""

from __future__ import annotations
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class EvalResult:
    product_id: str
    product_name: str
    ground_truth_low: float
    ground_truth_high: float
    ground_truth_mid: float
    predicted: float
    predicted_low: float = 0.0
    predicted_high: float = 0.0
    error_abs: float = 0.0
    in_range: bool = False
    in_3pp: bool = False
    direction_correct: bool = False
    confidence: str = "medium"
    mode: str = ""


@dataclass
class ModeReport:
    mode: str
    n_products: int
    n_in_range: int
    n_in_3pp: int
    pct_in_range: float
    pct_in_3pp: float
    mae: float
    median_abs_error: float
    over_predict_count: int
    under_predict_count: int
    over_predict_pct: float
    under_predict_pct: float
    by_category: dict = field(default_factory=dict)


def evaluate_predictions(
    mode: str,
    predictions: list[dict],
) -> ModeReport:
    """
    predictions: list of dicts with keys:
        product_id, product_name, gt_low, gt_high, predicted,
        predicted_low (optional), predicted_high (optional), category
    """
    results = []
    for p in predictions:
        gt_low = p["gt_low"]
        gt_high = p["gt_high"]
        gt_mid = (gt_low + gt_high) / 2
        predicted = p["predicted"]
        error = abs(predicted - gt_mid)

        result = EvalResult(
            product_id=p["product_id"],
            product_name=p.get("product_name", ""),
            ground_truth_low=gt_low,
            ground_truth_high=gt_high,
            ground_truth_mid=gt_mid,
            predicted=predicted,
            predicted_low=p.get("predicted_low", predicted * 0.8),
            predicted_high=p.get("predicted_high", predicted * 1.2),
            error_abs=error,
            in_range=gt_low <= predicted <= gt_high,
            in_3pp=error <= 0.03,
            direction_correct=(
                (predicted >= gt_mid and gt_mid >= 0.10)
                or (predicted < gt_mid and gt_mid < 0.10)
            ),
            confidence=p.get("confidence", "medium"),
            mode=mode,
        )
        results.append(result)

    n = len(results)
    if n == 0:
        return ModeReport(mode=mode, n_products=0, n_in_range=0, n_in_3pp=0,
                          pct_in_range=0, pct_in_3pp=0, mae=0,
                          median_abs_error=0, over_predict_count=0,
                          under_predict_count=0, over_predict_pct=0,
                          under_predict_pct=0)

    n_in_range = sum(1 for r in results if r.in_range)
    n_in_3pp = sum(1 for r in results if r.in_3pp)
    errors = [r.error_abs for r in results]
    over = sum(1 for r in results if r.predicted > r.ground_truth_high)
    under = sum(1 for r in results if r.predicted < r.ground_truth_low)

    by_category = {}
    for p, r in zip(predictions, results):
        cat = p.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "in_range": 0, "errors": []}
        by_category[cat]["total"] += 1
        if r.in_range:
            by_category[cat]["in_range"] += 1
        by_category[cat]["errors"].append(r.error_abs)

    for cat, stats in by_category.items():
        stats["pct_in_range"] = stats["in_range"] / stats["total"] * 100
        stats["mae"] = statistics.mean(stats["errors"]) if stats["errors"] else 0

    return ModeReport(
        mode=mode,
        n_products=n,
        n_in_range=n_in_range,
        n_in_3pp=n_in_3pp,
        pct_in_range=n_in_range / n * 100,
        pct_in_3pp=n_in_3pp / n * 100,
        mae=statistics.mean(errors),
        median_abs_error=statistics.median(errors),
        over_predict_count=over,
        under_predict_count=under,
        over_predict_pct=over / n * 100,
        under_predict_pct=under / n * 100,
        by_category=by_category,
    )


def print_report(report: ModeReport):
    print(f"\n{'='*80}")
    print(f"MODE: {report.mode}")
    print(f"{'='*80}")
    print(f"Products evaluated: {report.n_products}")
    print(f"In ground-truth range: {report.n_in_range}/{report.n_products} ({report.pct_in_range:.1f}%)")
    print(f"Within ±3pp:          {report.n_in_3pp}/{report.n_products} ({report.pct_in_3pp:.1f}%)")
    print(f"MAE:                  {report.mae*100:.2f}pp")
    print(f"Median abs error:     {report.median_abs_error*100:.2f}pp")
    print(f"Over-prediction:      {report.over_predict_count} ({report.over_predict_pct:.1f}%)")
    print(f"Under-prediction:     {report.under_predict_count} ({report.under_predict_pct:.1f}%)")
    if report.by_category:
        print(f"\nBy category:")
        for cat, stats in sorted(report.by_category.items()):
            print(f"  {cat:<22} {stats['in_range']}/{stats['total']} in-range ({stats['pct_in_range']:.0f}%) MAE={stats['mae']*100:.1f}pp")


def compare_modes(reports: list[ModeReport]):
    print(f"\n{'='*80}")
    print(f"CROSS-MODE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Mode':<35}{'In-Range':<12}{'±3pp':<10}{'MAE':<10}{'Bias'}")
    print("-" * 80)
    for r in reports:
        bias = "over" if r.over_predict_pct > r.under_predict_pct + 10 else \
               "under" if r.under_predict_pct > r.over_predict_pct + 10 else "balanced"
        print(f"{r.mode:<35}{r.pct_in_range:<11.1f}%{r.pct_in_3pp:<9.1f}%{r.mae*100:<9.1f}pp{bias}")


def save_results(reports: list[ModeReport], path: str = "v3_eval_results.json"):
    Path(path).write_text(json.dumps([asdict(r) for r in reports], indent=2))
    print(f"\nResults saved to {path}")
