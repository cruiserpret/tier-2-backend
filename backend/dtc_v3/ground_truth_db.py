"""
backend/dtc_v3/ground_truth_db.py

Curated benchmark of 30 DTC products with known/inferred trial rates.

Confidence labels:
    A = hard public number (IPO filings, earnings reports, official disclosures)
    B = credible analyst estimate (eMarketer, Statista, brand tracker reports)
    C = inferred from public revenue/adoption data
    D = weak estimate (founder interviews, podcast mentions)

Source weights:
    A → 1.00
    B → 0.60
    C → 0.25
    D → 0.10

Use:
    A/B for calibration training
    A/B/C for retrieval / RAG priors
    D for retrieval only, never calibration

Sources cited where applicable. Updated as new data becomes available.
"""

from __future__ import annotations
from .models import GroundTruthRecord


GROUND_TRUTH_DB: list[GroundTruthRecord] = [
    # ═══════════════════════════════════════════════════════════════════
    # ELECTRONICS / WEARABLES (5)
    # ═══════════════════════════════════════════════════════════════════
    GroundTruthRecord(
        brand="Oura Ring Gen 3",
        category="electronics_tech",
        trial_rate_low=0.06, trial_rate_high=0.09, trial_rate_mid=0.075,
        confidence="B",
        price_band="premium",
        purchase_frequency="durable",
        target_demo=["health optimizers", "sleep trackers", "high income"],
        frictions=["high price", "subscription fee", "sizing"],
        drivers=["sleep tracking accuracy", "recovery insights", "ring form factor"],
        source_notes="2.5M users (2023), addressable health-conscious segment ~20-30M US",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="wearable_health"
    ),
    GroundTruthRecord(
        brand="Whoop 4.0",
        category="electronics_tech",
        trial_rate_low=0.04, trial_rate_high=0.07, trial_rate_mid=0.055,
        confidence="C",
        price_band="premium",
        purchase_frequency="durable",
        target_demo=["athletes", "fitness optimizers", "biohackers"],
        frictions=["mandatory subscription", "screen-less complexity"],
        drivers=["athletic recovery", "strain analysis", "celebrity endorsements"],
        source_notes="~1M subscribers reported (2022)",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="subscription_led", category_role="specialist",
        category_subtype="wearable_health"
    ),
    GroundTruthRecord(
        brand="Eight Sleep Pod",
        category="electronics_tech",
        trial_rate_low=0.02, trial_rate_high=0.04, trial_rate_mid=0.03,
        confidence="C",
        price_band="luxury",
        purchase_frequency="durable",
        target_demo=["sleep optimizers", "high income", "tech enthusiasts"],
        frictions=["very high price", "installation complexity"],
        drivers=["temperature control", "sleep tracking", "premium positioning"],
        source_notes="$2K-$4K mattress cover; small luxury market",
        market_tier="luxury", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="smart_sleep"
    ),
    GroundTruthRecord(
        brand="Fitbit Charge 6",
        category="electronics_tech",
        trial_rate_low=0.12, trial_rate_high=0.16, trial_rate_mid=0.14,
        confidence="B",
        price_band="mid",
        purchase_frequency="durable",
        target_demo=["mainstream fitness", "casual trackers"],
        frictions=["Google ecosystem dependency", "battery life"],
        drivers=["affordability", "step counting", "Google integration"],
        source_notes="Owned by Google; mass-market fitness tracker",
        market_tier="mass_market", brand_scale="global_giant",
        distribution_model="mass_retail", category_role="generalist",
        category_subtype="fitness_tracker"
    ),
    GroundTruthRecord(
        brand="Apple Watch Series 9",
        category="electronics_tech",
        trial_rate_low=0.20, trial_rate_high=0.28, trial_rate_mid=0.24,
        confidence="A",
        price_band="premium",
        purchase_frequency="durable",
        target_demo=["iPhone owners", "mainstream"],
        frictions=["price", "iPhone-only"],
        drivers=["Apple ecosystem", "health features", "social proof"],
        source_notes="Apple disclosed 100M+ wearables users",
        market_tier="mass_platform", brand_scale="global_giant",
        distribution_model="mass_retail", category_role="platform",
        category_subtype="smartwatch_platform"
    ),

    # ═══════════════════════════════════════════════════════════════════
    # HOME / LIFESTYLE (5)
    # ═══════════════════════════════════════════════════════════════════
    GroundTruthRecord(
        brand="YETI Rambler 20oz",
        category="home_lifestyle",
        trial_rate_low=0.14, trial_rate_high=0.18, trial_rate_mid=0.16,
        confidence="A",
        price_band="premium",
        purchase_frequency="occasional",
        target_demo=["outdoor enthusiasts", "professionals", "gift buyers"],
        frictions=["price premium vs basic mugs"],
        drivers=["durability", "brand status", "lifetime warranty"],
        source_notes="YETI 2023 revenue $1.6B, public company filings",
        market_tier="premium_niche", brand_scale="large_public",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="premium_drinkware"
    ),
    GroundTruthRecord(
        brand="Casper Original Mattress",
        category="home_lifestyle",
        trial_rate_low=0.03, trial_rate_high=0.05, trial_rate_mid=0.04,
        confidence="A",
        price_band="luxury",
        purchase_frequency="durable",
        target_demo=["mattress shoppers", "DTC-curious"],
        frictions=["high commitment purchase", "trust gap", "size of decision"],
        drivers=["100-night trial", "mattress in a box", "marketing"],
        source_notes="Casper IPO 2020, ~80K mattresses/year run rate",
        market_tier="challenger", brand_scale="large_public",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="mattress"
    ),
    GroundTruthRecord(
        brand="Parachute Home Percale Sheets",
        category="home_lifestyle",
        trial_rate_low=0.04, trial_rate_high=0.07, trial_rate_mid=0.055,
        confidence="C",
        price_band="premium",
        purchase_frequency="occasional",
        target_demo=["home design enthusiasts", "premium bedding shoppers"],
        frictions=["price vs Target/Pottery Barn", "brand awareness"],
        drivers=["quality positioning", "aesthetic", "Oeko-Tex certification"],
        source_notes="~$50M revenue estimated; small premium bedding niche",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="bedding_premium"
    ),
    GroundTruthRecord(
        brand="Brooklinen Luxe Sheets",
        category="home_lifestyle",
        trial_rate_low=0.06, trial_rate_high=0.10, trial_rate_mid=0.08,
        confidence="B",
        price_band="premium",
        purchase_frequency="occasional",
        target_demo=["bedding upgraders", "millennials buying first home"],
        frictions=["competitor parity (Parachute, Boll & Branch)"],
        drivers=["sateen quality", "marketing reach", "first-mover DTC"],
        source_notes="$100M+ revenue, broader than Parachute",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="bedding_premium"
    ),
    GroundTruthRecord(
        brand="Purple Original Mattress",
        category="home_lifestyle",
        trial_rate_low=0.02, trial_rate_high=0.04, trial_rate_mid=0.03,
        confidence="A",
        price_band="luxury",
        purchase_frequency="durable",
        target_demo=["mattress shoppers seeking innovation"],
        frictions=["hyper-distinct feel polarizes", "high price"],
        drivers=["proprietary grid technology", "viral marketing"],
        source_notes="Purple Innovation public filings",
        market_tier="challenger", brand_scale="large_public",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="mattress"
    ),

    # ═══════════════════════════════════════════════════════════════════
    # FOOD / BEVERAGE (6)
    # ═══════════════════════════════════════════════════════════════════
    GroundTruthRecord(
        brand="Olipop Prebiotic Soda",
        category="food_beverage",
        trial_rate_low=0.24, trial_rate_high=0.30, trial_rate_mid=0.27,
        confidence="B",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["health-conscious soda drinkers", "millennials/Gen-Z"],
        frictions=["price 3x normal soda"],
        drivers=["functional health claims", "TikTok virality", "retail distribution"],
        source_notes="$200M+ revenue 2023, Series C funding deck",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="functional_soda"
    ),
    GroundTruthRecord(
        brand="Liquid Death Mountain Water",
        category="food_beverage",
        trial_rate_low=0.18, trial_rate_high=0.22, trial_rate_mid=0.20,
        confidence="B",
        price_band="mid",
        purchase_frequency="consumable",
        target_demo=["edgy branding lovers", "millennial water drinkers"],
        frictions=["price for water"],
        drivers=["heavy metal branding", "viral marketing", "aluminum sustainability"],
        source_notes="$250M+ revenue 2023, Series D",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="branded_water"
    ),
    GroundTruthRecord(
        brand="Athletic Brewing NA Beer",
        category="food_beverage",
        trial_rate_low=0.04, trial_rate_high=0.06, trial_rate_mid=0.05,
        confidence="B",
        price_band="mid",
        purchase_frequency="consumable",
        target_demo=["sober-curious", "health-conscious adults"],
        frictions=["NA beer category skepticism"],
        drivers=["craft NA leader", "low calorie", "category growth"],
        source_notes="$100M+ revenue, NA category leader",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="nonalcoholic_beer"
    ),
    GroundTruthRecord(
        brand="MUD\\WTR Coffee Alternative",
        category="food_beverage",
        trial_rate_low=0.03, trial_rate_high=0.05, trial_rate_mid=0.04,
        confidence="C",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["wellness adopters", "coffee replacers"],
        frictions=["habit replacement", "taste vs coffee", "price"],
        drivers=["adaptogens", "low caffeine", "ritual"],
        source_notes="Small but growing; ~$50M est",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="coffee_alternative"
    ),
    GroundTruthRecord(
        brand="Health-Ade Kombucha",
        category="food_beverage",
        trial_rate_low=0.08, trial_rate_high=0.12, trial_rate_mid=0.10,
        confidence="B",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["gut-health conscious", "yoga moms"],
        frictions=["price premium", "kombucha taste polarizing"],
        drivers=["whole-foods retail", "organic positioning"],
        source_notes="Major retail presence, ~$100M est revenue",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="mass_retail", category_role="specialist",
        category_subtype="functional_fermented"
    ),
    GroundTruthRecord(
        brand="Poppi Prebiotic Soda",
        category="food_beverage",
        trial_rate_low=0.20, trial_rate_high=0.26, trial_rate_mid=0.23,
        confidence="A",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["functional beverage seekers", "Gen-Z TikTok"],
        frictions=["niche soda category", "premium pricing"],
        drivers=["strong branding", "functional positioning", "celebrity endorsements"],
        source_notes="Acquired by PepsiCo 2025 for $1.95B; valued growth trajectory",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="functional_soda"
    ),

    # ═══════════════════════════════════════════════════════════════════
    # SUPPLEMENTS / HEALTH (4)
    # ═══════════════════════════════════════════════════════════════════
    GroundTruthRecord(
        brand="AG1 Athletic Greens",
        category="supplements_health",
        trial_rate_low=0.09, trial_rate_high=0.13, trial_rate_mid=0.11,
        confidence="B",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["wellness optimizers", "podcast listeners", "executives"],
        frictions=["$79/month subscription", "taste"],
        drivers=["podcast advertising", "celebrity endorsements", "convenience"],
        source_notes="Reported ~$600M revenue; influencer-driven growth",
        market_tier="premium_niche", brand_scale="growth_challenger",
        distribution_model="subscription_led", category_role="specialist",
        category_subtype="greens_powder"
    ),
    GroundTruthRecord(
        brand="Liquid IV Hydration Multiplier",
        category="supplements_health",
        trial_rate_low=0.18, trial_rate_high=0.23, trial_rate_mid=0.205,
        confidence="A",
        price_band="mid",
        purchase_frequency="consumable",
        target_demo=["active adults", "hangover-curious", "Costco shoppers"],
        frictions=["hydration category skepticism"],
        drivers=["Unilever acquisition", "Costco distribution", "brand recognition"],
        source_notes="Acquired by Unilever 2020; mass retail penetration",
        market_tier="mass_market", brand_scale="large_public",
        distribution_model="mass_retail", category_role="generalist",
        category_subtype="hydration_supplement"
    ),
    GroundTruthRecord(
        brand="LMNT Electrolytes",
        category="supplements_health",
        trial_rate_low=0.05, trial_rate_high=0.08, trial_rate_mid=0.065,
        confidence="C",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["keto", "athletes", "biohacker community"],
        frictions=["taste polarizing", "high sodium concerns"],
        drivers=["Rogan podcast endorsement", "keto positioning"],
        source_notes="DTC-only, podcast-driven growth",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="hydration_supplement"
    ),
    GroundTruthRecord(
        brand="Ritual Multivitamin",
        category="supplements_health",
        trial_rate_low=0.04, trial_rate_high=0.07, trial_rate_mid=0.055,
        confidence="C",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["women 25-45", "wellness-focused"],
        frictions=["price vs CVS multivitamins", "subscription fatigue"],
        drivers=["transparent ingredient sourcing", "design", "minimalist branding"],
        source_notes="Subscription DTC; ~$100M est",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="subscription_led", category_role="specialist",
        category_subtype="multivitamin_subscription"
    ),

    # ═══════════════════════════════════════════════════════════════════
    # FASHION / APPAREL (6)
    # ═══════════════════════════════════════════════════════════════════
    GroundTruthRecord(
        brand="Allbirds Wool Runners",
        category="fashion_apparel",
        trial_rate_low=0.08, trial_rate_high=0.12, trial_rate_mid=0.10,
        confidence="A",
        price_band="premium",
        purchase_frequency="occasional",
        target_demo=["eco-conscious professionals", "Silicon Valley uniform"],
        frictions=["narrow style range", "competitor parity"],
        drivers=["sustainability narrative", "comfort", "tech-worker adoption"],
        source_notes="Allbirds public filings; 2021 IPO",
        market_tier="challenger", brand_scale="large_public",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="premium_sneaker"
    ),
    GroundTruthRecord(
        brand="Bombas Ankle Socks",
        category="fashion_apparel",
        trial_rate_low=0.06, trial_rate_high=0.09, trial_rate_mid=0.075,
        confidence="B",
        price_band="premium",
        purchase_frequency="frequent",
        target_demo=["active adults", "social-impact buyers"],
        frictions=["3x price of basic socks"],
        drivers=["one-for-one donation", "comfort claims", "Shark Tank fame"],
        source_notes="$300M+ revenue, post-Shark Tank growth",
        market_tier="premium_niche", brand_scale="growth_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="premium_basics"
    ),
    GroundTruthRecord(
        brand="Rothy's Classic Flats",
        category="fashion_apparel",
        trial_rate_low=0.05, trial_rate_high=0.08, trial_rate_mid=0.065,
        confidence="C",
        price_band="luxury",
        purchase_frequency="occasional",
        target_demo=["professional women", "sustainability-conscious"],
        frictions=["high price", "narrow style"],
        drivers=["recycled plastic story", "machine washable", "comfort"],
        source_notes="$140M est revenue; niche premium positioning",
        market_tier="luxury", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="luxury_fashion"
    ),
    GroundTruthRecord(
        brand="Warby Parker Glasses",
        category="fashion_apparel",
        trial_rate_low=0.11, trial_rate_high=0.14, trial_rate_mid=0.125,
        confidence="A",
        price_band="mid",
        purchase_frequency="occasional",
        target_demo=["vision-corrected adults", "millennials"],
        frictions=["online glasses skepticism", "fit uncertainty"],
        drivers=["home try-on", "brick-and-mortar expansion", "$95 price point"],
        source_notes="WRBY public filings 2021 IPO",
        market_tier="challenger", brand_scale="large_public",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="eyewear"
    ),
    GroundTruthRecord(
        brand="Everlane Cotton T-Shirt",
        category="fashion_apparel",
        trial_rate_low=0.08, trial_rate_high=0.12, trial_rate_mid=0.10,
        confidence="C",
        price_band="premium",
        purchase_frequency="frequent",
        target_demo=["eco-conscious", "minimalists", "millennials"],
        frictions=["price vs Uniqlo/Target"],
        drivers=["radical transparency narrative", "minimalist aesthetic"],
        source_notes="Private; ~$200M est revenue",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="premium_basics"
    ),
    GroundTruthRecord(
        brand="Quince Cashmere Sweater",
        category="fashion_apparel",
        trial_rate_low=0.08, trial_rate_high=0.11, trial_rate_mid=0.095,
        confidence="C",
        price_band="mid",
        purchase_frequency="occasional",
        target_demo=["value-conscious professionals", "DTC enthusiasts"],
        frictions=["new brand awareness", "shipping wait"],
        drivers=["disruptive pricing ($75 cashmere)", "factory-direct narrative"],
        source_notes="Fast-growing DTC, ~$100M est",
        market_tier="challenger", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="premium_basics"
    ),

    # ═══════════════════════════════════════════════════════════════════
    # BEAUTY / SKINCARE (4)
    # ═══════════════════════════════════════════════════════════════════
    GroundTruthRecord(
        brand="Glossier Boy Brow",
        category="beauty_skincare",
        trial_rate_low=0.15, trial_rate_high=0.20, trial_rate_mid=0.175,
        confidence="B",
        price_band="mid",
        purchase_frequency="frequent",
        target_demo=["women 18-35", "minimalist beauty"],
        frictions=["beauty SKU saturation"],
        drivers=["cult product status", "Instagram aesthetic", "no-makeup look"],
        source_notes="Iconic Glossier hero product; ~25% of company revenue",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="brow_makeup"
    ),
    GroundTruthRecord(
        brand="Dollar Shave Club Starter Kit",
        category="beauty_skincare",
        trial_rate_low=0.15, trial_rate_high=0.18, trial_rate_mid=0.165,
        confidence="A",
        price_band="budget",
        purchase_frequency="frequent",
        target_demo=["men 22-45", "convenience seekers"],
        frictions=["razor incumbency (Gillette)"],
        drivers=["$1 first month", "viral 2012 video", "subscription model"],
        source_notes="Unilever acquired DSC for $1B (2016)",
        market_tier="mass_market", brand_scale="large_public",
        distribution_model="subscription_led", category_role="generalist",
        category_subtype="razor_subscription"
    ),
    GroundTruthRecord(
        brand="Gymshark Sports Bra",
        category="fashion_apparel",
        trial_rate_low=0.06, trial_rate_high=0.09, trial_rate_mid=0.075,
        confidence="C",
        price_band="mid",
        purchase_frequency="occasional",
        target_demo=["fitness-focused women", "Gen-Z athletes", "social media followers"],
        frictions=["sizing uncertainty online", "competitor saturation"],
        drivers=["influencer marketing", "TikTok virality", "athleisure crossover"],
        source_notes="Private; ~$500M revenue est; UK-origin DTC fitness apparel",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="athletic_apparel"
    ),
    GroundTruthRecord(
        brand="The Ordinary Niacinamide",
        category="beauty_skincare",
        trial_rate_low=0.12, trial_rate_high=0.16, trial_rate_mid=0.14,
        confidence="B",
        price_band="budget",
        purchase_frequency="consumable",
        target_demo=["skincare enthusiasts", "Gen-Z", "Reddit r/SkincareAddiction"],
        frictions=["clinical packaging vs sephora aesthetic"],
        drivers=["radical pricing transparency", "Reddit cult following", "$8 price"],
        source_notes="Acquired by Estee Lauder 2021; high-frequency repeat",
        market_tier="mass_market", brand_scale="large_public",
        distribution_model="retail_plus_dtc", category_role="substitute",
        category_subtype="skincare_active"
    ),
    GroundTruthRecord(
        brand="Drunk Elephant Protini",
        category="beauty_skincare",
        trial_rate_low=0.04, trial_rate_high=0.07, trial_rate_mid=0.055,
        confidence="B",
        price_band="luxury",
        purchase_frequency="frequent",
        target_demo=["affluent skincare buyers"],
        frictions=["$68 price point", "beauty SKU saturation"],
        drivers=["clean beauty positioning", "Sephora prominence"],
        source_notes="Acquired by Shiseido 2019 for $845M",
        market_tier="luxury", brand_scale="large_public",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="skincare_premium"
    ),
    # ═══════════════════════════════════════════════════════════════════
    # v3.1 SURGICAL ADDITIONS — retrieval-only, structural coverage gaps
    # Per friend: "Add comparables for structural coverage, not desired output"
    # All marked C/D confidence — used for retrieval only, NOT calibration
    # ═══════════════════════════════════════════════════════════════════
    GroundTruthRecord(
        brand="Heineken 0.0",
        category="food_beverage",
        trial_rate_low=0.06, trial_rate_high=0.10, trial_rate_mid=0.08,
        confidence="C",
        price_band="mid",
        purchase_frequency="consumable",
        target_demo=["sober-curious", "moderate drinkers"],
        frictions=["NA beer skepticism", "taste vs regular beer"],
        drivers=["mainstream brand recognition", "category leader"],
        source_notes="Mass-market NA beer; Heineken NV public estimates",
        market_tier="mass_market", brand_scale="global_giant",
        distribution_model="mass_retail", category_role="generalist",
        category_subtype="nonalcoholic_beer"
    ),
    GroundTruthRecord(
        brand="Partake Brewing IPA",
        category="food_beverage",
        trial_rate_low=0.02, trial_rate_high=0.05, trial_rate_mid=0.035,
        confidence="D",
        price_band="mid",
        purchase_frequency="consumable",
        target_demo=["health-conscious", "Whole30 / dry-January participants"],
        frictions=["niche awareness", "limited retail"],
        drivers=["craft NA positioning", "low calorie"],
        source_notes="Small NA beer brand; ~$10M est, weak signal",
        market_tier="premium_niche", brand_scale="early_stage",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="nonalcoholic_beer"
    ),
    GroundTruthRecord(
        brand="Four Sigmatic Mushroom Coffee",
        category="food_beverage",
        trial_rate_low=0.02, trial_rate_high=0.04, trial_rate_mid=0.03,
        confidence="C",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["wellness adopters", "biohackers"],
        frictions=["taste polarization", "habit replacement"],
        drivers=["functional mushroom positioning", "lifestyle branding"],
        source_notes="Niche specialty coffee category; ~$30M est",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="coffee_alternative"
    ),
    GroundTruthRecord(
        brand="Ryze Mushroom Coffee",
        category="food_beverage",
        trial_rate_low=0.04, trial_rate_high=0.07, trial_rate_mid=0.055,
        confidence="D",
        price_band="premium",
        purchase_frequency="consumable",
        target_demo=["coffee replacers", "Instagram wellness audience"],
        frictions=["taste vs coffee", "subscription resistance"],
        drivers=["aggressive influencer marketing", "morning routine"],
        source_notes="Recent fast-grower; weak public data",
        market_tier="premium_niche", brand_scale="venture_challenger",
        distribution_model="dtc_led", category_role="specialist",
        category_subtype="coffee_alternative"
    ),
    GroundTruthRecord(
        brand="DripDrop Hydration",
        category="supplements_health",
        trial_rate_low=0.07, trial_rate_high=0.11, trial_rate_mid=0.09,
        confidence="C",
        price_band="mid",
        purchase_frequency="consumable",
        target_demo=["medical/dehydration users", "athletes"],
        frictions=["clinical positioning niche"],
        drivers=["medical-grade claims", "WHO formula"],
        source_notes="Medical hydration brand; ~$50M est",
        market_tier="challenger", brand_scale="growth_challenger",
        distribution_model="retail_plus_dtc", category_role="specialist",
        category_subtype="hydration_supplement"
    ),
    GroundTruthRecord(
        brand="Pedialyte",
        category="supplements_health",
        trial_rate_low=0.18, trial_rate_high=0.26, trial_rate_mid=0.22,
        confidence="C",
        price_band="mid",
        purchase_frequency="consumable",
        target_demo=["parents", "hangover recovery", "medical use"],
        frictions=["pediatric branding (changing)"],
        drivers=["medical authority", "Walmart/CVS distribution"],
        source_notes="Abbott Labs; mass-market hydration with adult crossover",
        market_tier="mass_market", brand_scale="global_giant",
        distribution_model="mass_retail", category_role="generalist",
        category_subtype="hydration_supplement"
    ),

]


# ═══════════════════════════════════════════════════════════════════════
# QUERY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_db_stats() -> dict:
    """Return summary stats about the DB."""
    by_confidence = {}
    by_category = {}
    for record in GROUND_TRUTH_DB:
        by_confidence[record.confidence] = by_confidence.get(record.confidence, 0) + 1
        by_category[record.category] = by_category.get(record.category, 0) + 1
    return {
        "total": len(GROUND_TRUTH_DB),
        "by_confidence": by_confidence,
        "by_category": by_category,
    }


def get_calibration_subset() -> list[GroundTruthRecord]:
    """Return only A/B confidence records for calibration training."""
    return [r for r in GROUND_TRUTH_DB if r.confidence in ("A", "B")]


def get_retrieval_set() -> list[GroundTruthRecord]:
    """Return all records (A/B/C/D) for RAG retrieval."""
    return GROUND_TRUTH_DB


def get_by_brand(brand_query: str) -> GroundTruthRecord | None:
    """Find record by partial brand match."""
    q = brand_query.lower()
    for record in GROUND_TRUTH_DB:
        if q in record.brand.lower():
            return record
    return None


def get_by_category(category: str) -> list[GroundTruthRecord]:
    return [r for r in GROUND_TRUTH_DB if r.category == category]


# ═══════════════════════════════════════════════════════════════════════
# v3.1 — Aliases for canonical exclude_brand matching (P5.2)
# Per friend's spec: surgical aliases for validation/demo-critical brands.
# Populated post-construction to keep record syntax stable during YC sprint.
# TODO post-YC: move aliases inline into GroundTruthRecord definitions.
# ═══════════════════════════════════════════════════════════════════════

_ALIAS_MAP = {
    "YETI Rambler 20oz":              ["Yeti", "YETI"],
    "Liquid IV Hydration Multiplier": ["Liquid I.V.", "Liquid IV"],
    "MUD\\WTR Coffee Alternative":    ["MUD", "MUD WTR", "MUDWTR", "MUD\\WTR"],
    "Athletic Brewing NA Beer":       ["Athletic", "Athletic Brewing", "Athletic Brewing Co", "Athletic Brewing Company"],
    "Oura Ring Gen 3":                ["Oura Ring", "Oura"],
    "AG1 Athletic Greens":            ["Athletic Greens", "AG 1", "AG-1", "Athletic-Greens"],
}

for _record in GROUND_TRUTH_DB:
    if _record.brand in _ALIAS_MAP:
        _record.aliases = list(_ALIAS_MAP[_record.brand])
