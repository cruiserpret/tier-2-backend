"""
backend/dtc_v3/build_signal_table_from_v2.py

Per friend's Path C ruling:
  USE v2 artifacts: debate state, agent stances, market intel, objections
  DO NOT USE: v2 trial_rate, funnel coefficients, awareness/reach/conversion scores
  FRESH compute: RAG prior using v3 retrieval

Output: backend/dtc_v3/calibration/signal_table_v0_from_v2.jsonl
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Add repo root to path
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root))

from backend.dtc_v3.models import ProductBrief
from backend.dtc_v3.rag_retrieval import retrieve_neighbors, compute_rag_prior
from backend.dtc_v3.persona_signals import (
    extract_desirability, extract_awareness, extract_friction
)
from backend.dtc_v3.ground_truth_db import GROUND_TRUTH_DB


# ═══════════════════════════════════════════════════════════════════════
# PRODUCT BRIEFS (matching the 19 v1/v2 baseline products)
# ═══════════════════════════════════════════════════════════════════════

VALIDATION_PRODUCTS = {
    "olipop_2024": {
        "brand_canonical": "Olipop Prebiotic Soda",  # for GT lookup + exclude
        "name": "Olipop Prebiotic Soda 12-pack",
        "description": "Prebiotic soda, 9g fiber, 2-5g sugar, plant-based ingredients. 12-pack of 12oz cans.",
        "price": 35.99, "category": "food_beverage",
        "demographic": "Health-conscious adults 25-45",
        "competitors": [{"name": "Health-Ade Kombucha"}, {"name": "Culture Pop Soda"}, {"name": "Poppi Prebiotic Soda"}],
    },
    "everlane_2024": {
        "brand_canonical": "Everlane Cotton T-Shirt",
        "name": "Everlane Cotton T-Shirt",
        "description": "Premium organic cotton t-shirt, sustainably manufactured.",
        "price": 130, "category": "fashion_apparel",
        "demographic": "Eco-conscious professionals 28-45",
        "competitors": [{"name": "Uniqlo Supima Cotton T-Shirt"}, {"name": "Buck Mason Slub Cotton T-Shirt"}, {"name": "Outerknown Sojourn Tee"}],
    },
    "yeti_2024": {
        "brand_canonical": "YETI Rambler 20oz",
        "name": "YETI Rambler 20oz Travel Mug",
        "description": "Vacuum insulated stainless steel travel mug, MagSlider lid.",
        "price": 40, "category": "home_lifestyle",
        "demographic": "Outdoor enthusiasts and professionals 25-55",
        "competitors": [{"name": "Stanley Quencher H2.0"}, {"name": "Hydro Flask 21oz"}, {"name": "Contigo West Loop"}],
    },
    "oura_2024": {
        "brand_canonical": "Oura Ring Gen 3",
        "name": "Oura Ring Gen 3",
        "description": "Smart ring tracking sleep, heart rate, body temperature, activity. Subscription required at $5.99/month.",
        "price": 349, "category": "electronics_tech",
        "demographic": "Health-conscious professionals 28-50",
        "competitors": [{"name": "Whoop 4.0 Strap"}, {"name": "Apple Watch Series 9"}, {"name": "Fitbit Charge 6"}],
    },
    "ag1_subscription": {
        "brand_canonical": "AG1 Athletic Greens",
        "name": "AG1 Athletic Greens Subscription",
        "description": "Daily greens powder subscription. 75 vitamins, minerals, whole-food ingredients.",
        "price": 79, "category": "supplements_health",
        "demographic": "Wellness-focused adults 28-50",
        "competitors": [{"name": "Bloom Greens Powder"}, {"name": "Naked Nutrition Super Greens"}, {"name": "Garden of Life Perfect Food"}],
    },
    "liquid_death": {
        "brand_canonical": "Liquid Death Mountain Water",
        "name": "Liquid Death Mountain Water 12-pack",
        "description": "Canned mountain water with edgy branding. Sugar-free, calorie-free.",
        "price": 19.99, "category": "food_beverage",
        "demographic": "Adults 22-45 turned off by traditional water marketing",
        "competitors": [{"name": "Aquafina Purified Water"}, {"name": "LIFEWTR Premium Bottled Water"}, {"name": "Just Water Spring Water"}],
    },
    "warby_parker": {
        "brand_canonical": "Warby Parker Glasses",
        "name": "Warby Parker Prescription Glasses",
        "description": "Designer prescription glasses with home try-on program.",
        "price": 95, "category": "fashion_apparel",
        "demographic": "Vision-corrected adults 22-50",
        "competitors": [{"name": "Zenni Optical"}, {"name": "GlassesUSA"}, {"name": "EyeBuyDirect"}],
    },
    "casper_mattress": {
        "brand_canonical": "Casper Original Mattress",
        "name": "Casper Original Mattress Queen",
        "description": "Memory foam mattress in a box. 100-night trial, 10-year warranty.",
        "price": 1095, "category": "home_lifestyle",
        "demographic": "Adults 28-55 in market for new mattress",
        "competitors": [{"name": "Tempur-Pedic"}, {"name": "Purple Original Mattress"}, {"name": "Saatva Classic"}],
    },
    "dollar_shave_club": {
        "brand_canonical": "Dollar Shave Club",
        "name": "Dollar Shave Club Starter Kit",
        "description": "Razor and grooming starter kit. First box discounted to $10.",
        "price": 10, "category": "beauty_skincare",
        "demographic": "Men 22-45",
        "competitors": [{"name": "Gillette Fusion5"}, {"name": "Harry's Truman"}, {"name": "Bevel Safety Razor"}],
    },
    "bombas_socks": {
        "brand_canonical": "Bombas",
        "name": "Bombas Ankle Socks 4-pack",
        "description": "Performance ankle socks with honeycomb arch support, blister tab.",
        "price": 54, "category": "fashion_apparel",
        "demographic": "Active adults 25-55 valuing comfort and social impact",
        "competitors": [{"name": "Nike Everyday Cushion"}, {"name": "Adidas Athletic Cushioned"}, {"name": "Under Armour Essential"}],
    },
    "athletic_brewing": {
        "brand_canonical": "Athletic Brewing NA Beer",
        "name": "Athletic Brewing Upside Dawn Non-Alcoholic IPA 6-pack",
        "description": "Craft non-alcoholic IPA. Zero alcohol, 50 calories per can.",
        "price": 13, "category": "food_beverage",
        "demographic": "Health-conscious adults 25-55 reducing alcohol consumption",
        "competitors": [{"name": "Heineken 0.0"}, {"name": "Budweiser Zero"}, {"name": "Partake Brewing IPA"}],
    },
    "allbirds_wool_runners": {
        "brand_canonical": "Allbirds Wool Runners",
        "name": "Allbirds Wool Runners",
        "description": "Machine-washable merino wool running sneakers.",
        "price": 110, "category": "fashion_apparel",
        "demographic": "Eco-conscious professionals 25-50",
        "competitors": [{"name": "Nike Free Run"}, {"name": "Adidas Ultraboost"}, {"name": "Rothy's"}],
    },
    "glossier_boy_brow": {
        "brand_canonical": "Glossier Boy Brow",
        "name": "Glossier Boy Brow",
        "description": "Tinted eyebrow pomade that thickens, fills, and grooms.",
        "price": 18, "category": "beauty_skincare",
        "demographic": "Women 18-35 seeking minimalist beauty",
        "competitors": [{"name": "Benefit Gimme Brow"}, {"name": "Anastasia Beverly Hills"}, {"name": "Maybelline Brow Drama"}],
    },
    "mudwtr_coffee": {
        "brand_canonical": "MUD\\WTR Coffee Alternative",
        "name": "MUD WTR Coffee Alternative Starter Kit",
        "description": "Mushroom-based coffee alternative with cacao, masala chai, turmeric, adaptogens.",
        "price": 40, "category": "food_beverage",
        "demographic": "Wellness-focused adults 25-45 seeking coffee alternatives",
        "competitors": [{"name": "Four Sigmatic"}, {"name": "Ryze Mushroom Coffee"}, {"name": "Starbucks Pike Place"}],
    },
    "quince_cashmere": {
        "brand_canonical": "Quince Cashmere Sweater",
        "name": "Quince Mongolian Cashmere Crewneck Sweater",
        "description": "100% Grade-A Mongolian cashmere crewneck. Direct-to-consumer pricing.",
        "price": 75, "category": "fashion_apparel",
        "demographic": "Value-conscious professionals 28-55",
        "competitors": [{"name": "Everlane Cashmere"}, {"name": "Naadam Essential Cashmere"}, {"name": "J.Crew Cashmere"}],
    },
    "rothys_flats": {
        "brand_canonical": "Rothy's Classic Flats",
        "name": "Rothys Classic Point Flats",
        "description": "Machine-washable flats made from recycled plastic water bottles.",
        "price": 145, "category": "fashion_apparel",
        "demographic": "Professional women 28-50 valuing sustainability",
        "competitors": [{"name": "Sam Edelman Felicia"}, {"name": "Tory Burch Minnie"}, {"name": "Cole Haan Tali Bow"}],
    },
    "parachute_sheets": {
        "brand_canonical": "Parachute Home Percale Sheets",
        "name": "Parachute Home Percale Sheet Set Queen",
        "description": "Egyptian long-staple cotton percale sheet set. Crisp cool feel.",
        "price": 169, "category": "home_lifestyle",
        "demographic": "Home-focused adults 30-55",
        "competitors": [{"name": "Brooklinen Luxe"}, {"name": "Boll Branch Signature"}, {"name": "Pottery Barn Classic"}],
    },
    "liquid_iv": {
        "brand_canonical": "Liquid IV Hydration Multiplier",
        "name": "Liquid IV Hydration Multiplier Variety Pack",
        "description": "Electrolyte drink mix powder. 3x more electrolytes than sports drinks.",
        "price": 25, "category": "supplements_health",
        "demographic": "Active adults 22-50 focused on hydration",
        "competitors": [{"name": "Gatorade Gatorlyte"}, {"name": "LMNT Recharge"}, {"name": "Nuun Sport"}],
    },
    "gymshark_sports_bra": {
        "brand_canonical": "Gymshark",
        "name": "Gymshark Vital Seamless Sports Bra",
        "description": "Seamless medium-impact sports bra with four-way stretch.",
        "price": 38, "category": "fashion_apparel",
        "demographic": "Fitness-focused women 18-40",
        "competitors": [{"name": "Nike Swoosh"}, {"name": "Lululemon Energy Bra"}, {"name": "Alo Yoga Alosoft"}],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# V2 ARTIFACT EXTRACTORS
# Per friend: extract debate_state, agent stances, objections, market intel.
# DO NOT touch v2's trial_rate breakdown, awareness coefficient, etc.
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_debate_state(report: dict) -> dict:
    """
    Build a debate_state dict matching v3 persona_signals.extract_*() expectations.
    Uses ONLY round_summaries + agent_summaries (clean v2 artifacts).
    """
    rounds = []
    for r in report.get("round_summaries", []):
        rounds.append({
            "round_num": r.get("round_num", 0),
            "round_name": r.get("round_name", ""),
            "for_count": r.get("for_count", 0),
            "against_count": r.get("against_count", 0),
            "neutral_count": r.get("neutral_count", 0),
            "avg_score": r.get("avg_score", 5.0),
            "shifted_count": r.get("shifted_count", 0),
        })
    return {"rounds": rounds, "agent_summaries": report.get("agent_summaries", [])}


def reconstruct_market_intel(report: dict, validation_entry: dict) -> dict:
    """
    Build market_intel dict for v3 awareness extraction.
    Uses competitive_positioning + top_objections (clean v2 artifacts).
    DOES NOT use v2's awareness_coefficient (broken funnel value).
    """
    # Parse competitive_positioning string for review counts/ratings
    # Format: "Dominant competitor: X (12,345 reviews, 4.5★). Your price ratio: 1.5x market."
    cp = report.get("competitive_positioning", "")

    # Extract review count from positioning string
    competitors = []
    import re
    review_match = re.search(r'\(([\d,]+)\s+reviews,\s*([\d.]+)★\)', cp)
    if review_match:
        rev_count = int(review_match.group(1).replace(",", ""))
        rating = float(review_match.group(2))
        competitors.append({"total_reviews": rev_count, "rating": rating})

    # Extract price ratio
    pr_match = re.search(r'price ratio:\s*([\d.]+)x', cp)
    price_ratio = float(pr_match.group(1)) if pr_match else 1.0

    return {
        "competitors": competitors,
        "reddit_signal_count": 12,  # v2 always pulled ~12-16 signals; using midpoint
        "top_objections": report.get("top_objections", []),
        "price_ratio": price_ratio,
    }


def get_gt_record(brand_canonical: str):
    """Look up GT record by canonical brand name."""
    for record in GROUND_TRUTH_DB:
        if brand_canonical.lower() in record.brand.lower() or \
           record.brand.lower() in brand_canonical.lower():
            return record
    return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    # _repo_root resolves to assembly-backend/. validation/ is one level up at ~/Desktop/Aseembly/validation/
    aseembly_root = _repo_root.parent
    v2_path = aseembly_root / "validation" / "v2_validation_results.json"
    output_dir = _repo_root / "backend" / "dtc_v3" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "signal_table_v0_from_v2.jsonl"

    print(f"Reading v2 artifacts from: {v2_path}")
    with open(v2_path) as f:
        v2_data = json.load(f)

    rows = []
    skipped = []

    for product_id, brief_dict in VALIDATION_PRODUCTS.items():
        brand_canonical = brief_dict.pop("brand_canonical")
        v2_entry = v2_data.get(product_id)

        if not v2_entry or "error" in v2_entry.get("report", {}):
            skipped.append((product_id, "no v2 data"))
            continue

        report = v2_entry["report"]

        # Build ProductBrief
        brief = ProductBrief(**brief_dict)

        # ── 1. Reconstruct CLEAN artifacts from v2 ──
        debate_state = reconstruct_debate_state(report)
        market_intel = reconstruct_market_intel(report, v2_entry)

        # ── 2. Extract raw composites using v3 logic ──
        # NOTE: these are RAW values (not z-scored yet — z-scoring happens later
        # using empirical normalizers fit on this very table)
        desirability_raw = extract_desirability(debate_state)
        awareness_raw = extract_awareness(market_intel)
        friction_raw = extract_friction(debate_state, market_intel, brief)

        # ── 3. FRESH RAG prior using v3 retrieval (exclude self) ──
        neighbors = retrieve_neighbors(brief, k=6, exclude_brand=brand_canonical)
        rag_prior = compute_rag_prior(neighbors)

        # ── 4. GT lookup ──
        gt_record = get_gt_record(brand_canonical)
        if gt_record is None:
            skipped.append((product_id, f"no GT record for '{brand_canonical}'"))
            continue

        row = {
            # Provenance (per friend's spec)
            "brand": brand_canonical,
            "product_id": product_id,
            "signal_source": "v2_debate_artifacts",
            "rag_source": "v3_hybrid_retrieval",
            "used_v2_prediction": False,
            "used_v2_funnel_math": False,
            "calibration_version": "v0_from_v2_artifacts",

            # Ground truth
            "gt_low": gt_record.trial_rate_low,
            "gt_mid": gt_record.trial_rate_mid,
            "gt_high": gt_record.trial_rate_high,
            "confidence": gt_record.confidence,
            "confidence_weight": gt_record.source_weight,

            # FRESH v3 RAG prior
            "rag_prior": round(rag_prior, 4),
            "rag_neighbors": [
                {"brand": n.brand, "trial_rate": n.trial_rate_mid,
                 "weight": round(n.source_weight, 4), "confidence": n.confidence}
                for n in neighbors
            ],

            # RAW composite signals (will be normalized in next step)
            "desirability_raw": round(desirability_raw, 4),
            "awareness_raw": round(awareness_raw, 4),
            "friction_raw": round(friction_raw, 4),

            # Diagnostic
            "rag_residual_logit": None,  # filled in calibration step
            "category": brief.category,
            "price": brief.price,
        }
        rows.append(row)

    # Write JSONL
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n✓ Wrote {len(rows)} signal rows to {output_path}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for pid, reason in skipped:
            print(f"  - {pid}: {reason}")

    # Print summary
    print(f"\n{'─'*100}")
    print(f"{'Brand':<35}{'GT':<10}{'RAG':<10}{'Desire':<9}{'Aware':<9}{'Frict':<9}{'Conf'}")
    print("─" * 100)
    for r in rows:
        print(f"{r['brand'][:33]:<35}"
              f"{r['gt_mid']*100:.1f}%    "
              f"{r['rag_prior']*100:5.1f}%   "
              f"{r['desirability_raw']:.2f}     "
              f"{r['awareness_raw']:.2f}     "
              f"{r['friction_raw']:.2f}     "
              f"{r['confidence']}")


if __name__ == "__main__":
    main()
