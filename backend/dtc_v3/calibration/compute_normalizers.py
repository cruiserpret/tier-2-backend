"""
backend/dtc_v3/calibration/compute_normalizers.py

Step 1 of friend's calibration sequence:
Compute empirical z-score normalizers (median + IQR/1.349) from CLEAN rows only.

Per friend's spec:
- Use clean 16 rows (exclude Athletic Brewing, MUD-WTR, Liquid IV — retrieval contaminated)
- Use median (robust to outliers, not mean)
- Use IQR/1.349 as scale (robust SD equivalent)
- Floor at 0.05 to avoid divide-by-tiny

Output: normalizer_constants_v1.json
"""

from __future__ import annotations
import json
import statistics
from pathlib import Path

CONTAMINATED_BRANDS = {
    "Athletic Brewing NA Beer",
    "MUD\\WTR Coffee Alternative",
    "Liquid IV Hydration Multiplier",
}

CALIBRATION_DIR = Path(__file__).parent
SIGNAL_TABLE = CALIBRATION_DIR / "signal_table_v0_from_v2.jsonl"
OUTPUT = CALIBRATION_DIR / "normalizer_constants_v1.json"


def robust_normalizer(values: list[float]) -> dict:
    """Compute median + IQR/1.349 (robust SD equivalent)."""
    sorted_v = sorted(values)
    n = len(sorted_v)
    median = statistics.median(values)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    scale = max(0.05, iqr / 1.349)  # floor to prevent divide-by-tiny
    return {
        "center": round(median, 4),
        "scale": round(scale, 4),
        "iqr": round(iqr, 4),
        "n_samples": n,
    }


def main():
    # Load signal table
    rows = []
    with open(SIGNAL_TABLE) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} signal rows from {SIGNAL_TABLE.name}")

    # Filter to clean rows
    clean = [r for r in rows if r["brand"] not in CONTAMINATED_BRANDS]
    contaminated_in_data = [r["brand"] for r in rows if r["brand"] in CONTAMINATED_BRANDS]

    print(f"Clean rows: {len(clean)}")
    print(f"Excluded contaminated: {contaminated_in_data}")
    print()

    # Compute normalizers per signal
    normalizers = {}
    for field in ["desirability_raw", "awareness_raw", "friction_raw"]:
        signal_name = field.replace("_raw", "")
        values = [r[field] for r in clean]
        normalizers[signal_name] = robust_normalizer(values)
        n = normalizers[signal_name]
        print(f"  {signal_name:<15} center={n['center']:.4f}  scale={n['scale']:.4f}  IQR={n['iqr']:.4f}")

    # Add provenance
    output = {
        "version": "v1",
        "method": "median + IQR/1.349 (robust)",
        "computed_from": "signal_table_v0_from_v2.jsonl",
        "n_clean_rows_used": len(clean),
        "excluded_brands": list(CONTAMINATED_BRANDS),
        "exclusion_reason": "retrieval-contaminated (no in-subtype comparable in DB)",
        "scale_floor": 0.05,
        "normalizers": normalizers,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved to {OUTPUT.name}")


if __name__ == "__main__":
    main()
