# `/api/dtc_v3/discuss` — Frozen Schema Contract

**Branch:** `tier2-v3-ui-rebuild`
**Status:** Frozen for the UI rebuild. Pransh builds against this. Hamza implements to this.
**Last updated:** 2026-04-28

This file is the **single source of truth** for the new `/discuss` API and the data the report page consumes. Pransh: build frontend mocks against this exact JSON shape. Hamza: implement backend to match this exact shape. Any changes go through friend.

---

## 1. Endpoint

```
POST /api/dtc_v3/discuss
```

### Request body

```json
{
  "product": {
    "product_name": "Triton Drinks",
    "name": "Triton Drinks",
    "description": "Energy drink for US market",
    "price": 3.99,
    "category": "food_beverage",
    "demographic": "active adults 18-35",
    "competitors": [{ "name": "Red Bull" }, { "name": "Monster" }]
  },
  "forecast": {
    "trial_rate_median": 0.18,
    "confidence": "medium-high",
    "fallback_used": false,
    "coverage_tier": "strong"
  },
  "agent_count": 50,
  "mode": "template"
}
```

### Query string mode override (developer use only)

```
POST /api/dtc_v3/discuss?mode=llm
```

`mode` is `template` (default, free, fast, deterministic) or `llm` (single structured call, cached). Public frontend always uses `template`. LLM is debug/showcase only.

---

## 2. Response shape

```json
{
  "agent_panel": {
    "agent_count": 50,
    "mode": "template",
    "seed": "sha256-hexdigest-32chars",
    "coverage_warning": "",

    "intent_distribution": {
      "buy": 0.36,
      "considering": 0.34,
      "resistant": 0.30
    },

    "rounds": [
      { "round": 1, "title": "First Impression" },
      { "round": 2, "title": "Competitor Comparison" },
      { "round": 3, "title": "Consensus Building" }
    ],

    "agents": [
      {
        "id": "agent_01",
        "name": "Fatima Hassan",
        "segment": "Exhausted Medical Professional",
        "profile": "Nurse working long shifts who buys convenient energy products.",
        "age_band": "25-34",
        "income_band": "$75K-$100K",
        "verdict": "CONSIDERING",
        "score": 0.62,
        "stance": "neutral",
        "top_objection": "Needs more proof this works better than existing caffeine drinks.",
        "what_would_change_mind": "Clinical proof, trusted reviews, or a lower-risk trial offer.",
        "round_responses": [
          {
            "round": 1,
            "title": "First Impression",
            "response": "I understand the promise, but I need proof the benefits are real."
          },
          {
            "round": 2,
            "title": "Competitor Comparison",
            "response": "Compared with Red Bull or Celsius, this needs a clearer reason to switch."
          },
          {
            "round": 3,
            "title": "Consensus Building",
            "response": "I would consider trying it if the brand led with proof and a starter pack."
          }
        ],
        "journey": {
          "initial_verdict": "CONSIDERING",
          "final_verdict": "CONSIDERING",
          "shift_reason": "Interest stayed moderate because trust proof remained the main gap.",
          "key_quote": "I would consider trying it if the brand led with proof and a starter pack."
        }
      }
    ],

    "buyer_journeys": [
      {
        "agent_id": "agent_01",
        "name": "Fatima Hassan",
        "segment": "Exhausted Medical Professional",
        "initial_verdict": "CONSIDERING",
        "final_verdict": "CONSIDERING",
        "shift_reason": "Interest stayed moderate because trust proof remained the main gap.",
        "key_quote": "I would consider trying it if the brand led with proof and a starter pack."
      }
    ],

    "top_drivers": [
      {
        "label": "Sharp focus, sustained energy, zero compromise.",
        "agent_count": 18,
        "representative_quote": "I would try this if it gives me energy without needing a separate hydration drink.",
        "representative_agent_id": "agent_07"
      }
    ],

    "top_objections": [
      {
        "label": "9 of 50 agents resisted — primarily on price vs Red Bull.",
        "agent_count": 9,
        "representative_quote": "Red Bull is everywhere and I trust it. Why switch?",
        "representative_agent_id": "agent_22"
      }
    ],

    "most_receptive_segment": {
      "highest_intent": [
        { "agent_id": "agent_07", "name": "Chloe Bernard", "segment": "Caffeine-Driven Graduate Student", "score": 0.91 },
        { "agent_id": "agent_12", "name": "Mei Lin Zhao", "segment": "Overworked Professional", "score": 0.88 },
        { "agent_id": "agent_19", "name": "Brooke Stephens", "segment": "Fitness Optimizer", "score": 0.84 }
      ],
      "hardest_to_convert": [
        { "agent_id": "agent_22", "name": "Robert Chen", "segment": "Brand-Loyal Red Bull Buyer", "score": 0.18 },
        { "agent_id": "agent_31", "name": "Walter Kimura", "segment": "Ingredient Skeptic", "score": 0.21 },
        { "agent_id": "agent_44", "name": "Carlos Mendoza", "segment": "Price-Sensitive Parent", "score": 0.24 }
      ]
    },

    "winning_message": {
      "message": "Sharp focus, sustained energy, zero compromise.",
      "source": "Pulled from agent debate — not generated.",
      "use_in": ["landing_page_headline", "ad_copy", "product_page_hero"]
    },

    "risk_factors": {
      "summary": "9 of 50 agents resisted — primarily on price vs Red Bull.",
      "detail": "Positioned as a more affordable, smoother energy drink alternative with emphasis on cognitive function. Currently underpriced but with potential for strategic repositioning.",
      "holdout_agents": ["agent_22", "agent_31", "agent_44"]
    },

    "actionable_insight": "Lead with proof, reviews, and a low-risk starter offer before scaling paid ads.",

    "consensus": "Forecast of 18.0% trial is grounded in comparable energy drink anchors. Agents support directional verdict.",

    "comparable_price_range": {
      "min": 2.0,
      "max": 5.0,
      "user_price": 3.99,
      "anchor_brands": ["Red Bull", "Monster Energy", "Celsius", "C4 Energy"]
    },

    "counterfactuals": [
      {
        "label": "Lower price 15%",
        "description": "Scenario estimate: lowering price to ~$3.40 could move forecast toward ~21%, assuming execution matches comparable brands.",
        "delta_logit": 0.15,
        "direction": "up",
        "new_prediction_pct": 21.0
      },
      {
        "label": "Add money-back guarantee",
        "description": "Scenario estimate: adding a guarantee could move forecast toward ~20%, assuming execution matches comparable brands.",
        "delta_logit": 0.10,
        "direction": "up",
        "new_prediction_pct": 20.0
      }
    ]
  }
}
```

---

## 3. Field meanings (frontend reference)

### `agent_panel.intent_distribution`

```json
{ "buy": 0.36, "considering": 0.34, "resistant": 0.30 }
```

- `buy` = fraction of agents with `verdict: "BUY"` (capped: ≤ 2 × `forecast.trial_rate_median`, clamp 0.6 max)
- `considering` = fraction with `verdict: "CONSIDERING"`
- `resistant` = fraction with `verdict: "WON'T BUY"` (the JSON key is `resistant` for code stability; display label is `WON'T BUY`)
- All three sum to 1.0 ± float rounding.

**Display labels (visible to user):**
- `BUY` (green)
- `CONSIDERING` (yellow / amber)
- `WON'T BUY` (red / pink)

**Sidebar subtitle (verbatim):**

> AI buyer-panel signal, not market-size forecast.

### `agent_panel.agents[]`

One object per agent. `agent_count` of them. Order is deterministic given the same `(product, forecast, agent_count, mode)`.

| Field | Type | Notes |
|---|---|---|
| `id` | string | `agent_01`, `agent_02`, ... |
| `name` | string | From persona bank, deterministically sampled |
| `segment` | string | Persona archetype label |
| `profile` | string | One-line description |
| `age_band` | string | e.g. "25-34" |
| `income_band` | string | e.g. "$75K-$100K" |
| `verdict` | enum | `"BUY"` \| `"CONSIDERING"` \| `"WON'T BUY"` |
| `score` | float 0..1 | Intent score |
| `stance` | enum | `"for"` \| `"neutral"` \| `"against"` (legacy field, kept for backend internals) |
| `top_objection` | string | One-sentence objection |
| `what_would_change_mind` | string | One-sentence trigger that flips them |
| `round_responses` | array | 3 entries, one per round |
| `journey` | object | Initial → final verdict trajectory |

### `agent_panel.buyer_journeys[]`

A flattened slice of `agents[].journey` for easy rendering on Section 08. Subset of agents whose journey is most narrative (e.g. those who shifted, plus the strongest BUY and strongest RESISTANT). Backend selects ~6-12 journeys depending on `agent_count`.

### Verdict mapping rule (backend → frontend)

```
stance="for"     → verdict="BUY"
stance="neutral" → verdict="CONSIDERING"
stance="against" → verdict="WON'T BUY"
```

### Pre-generation allocation rule (forecast → bucket counts)

```python
trial_rate = forecast.trial_rate_median   # e.g. 0.18, not 18
confidence = forecast.confidence

max_buy_share = min(2.0 * trial_rate, 0.60)

if confidence in ("high", "medium-high"):
    buy_share = max_buy_share
    considering_share = min(0.35, 1.0 - buy_share)
elif confidence == "medium":
    buy_share = min(max_buy_share, 0.35)
    considering_share = 0.35
elif confidence == "medium-low":
    buy_share = min(max_buy_share, 0.25)
    considering_share = 0.35
else:  # low
    buy_share = min(max_buy_share, 0.15)
    considering_share = 0.30

resistant_share = 1.0 - buy_share - considering_share

if forecast.fallback_used or confidence == "low":
    n_buy = min(n_buy, round(n_agents * 0.15))   # hard cap

n_buy         = round(n_agents * buy_share)
n_considering = round(n_agents * considering_share)
n_resistant   = n_agents - n_buy - n_considering
```

### `agent_panel.comparable_price_range`

```json
{
  "min": 2.0,
  "max": 5.0,
  "user_price": 3.99,
  "anchor_brands": ["Red Bull", "Monster Energy", "Celsius", "C4 Energy"]
}
```

- Computed from `forecast.neighbors[]` (anchor brand price bands).
- May be `null` if forecast has fewer than 2 eligible neighbors (weak coverage).
- Frontend hides Section 04 if `null`.

### `agent_panel.counterfactuals[]`

Same shape as the existing `forecast.counterfactuals[]` field. Already implemented backend-side.

### `agent_panel.coverage_warning`

- Empty string `""` if forecast confidence is `medium` or higher.
- Populated with a sentence like `"Coverage is thin — treat this forecast as directional."` if `confidence in ["low", "medium-low"]` or `fallback_used: true`.
- Frontend renders as a banner above the agent grid when non-empty.

---

## 4. Forbidden language (DO NOT render)

Frontend code must NOT contain any of these strings, anywhere:

- `Juster`
- `Van Westendorp`
- `Optimal Price Point`
- `real buyers`
- `probability scale`
- `r=0.895`
- `survey respondents`
- `causal proof`
- `guaranteed demand`
- `would buy` *(when used as a forecast label — OK as agent-verdict label)*

Replace with v3-honest copy:

| ❌ Forbidden | ✅ Replacement |
|---|---|
| Juster probability composite | Comparable-anchored forecast |
| Van Westendorp PSM | Directional price scenario |
| Optimal Price Point | Potential lower-friction price |
| Real buyers | AI buyer personas |
| Would buy (forecast) | Estimated 12-month trial rate |
| Probability scale | Comparable evidence |

---

## 5. The 13 Report Sections

The report page (`/dtc-v3/report/:id`) renders the following sections in order. Frontend reads from forecast response + this discuss response.

| # | Section | Data source | Hide if missing? |
|---|---|---|---|
| 01 | Executive Summary | `forecast.verdict`, `forecast.confidence`, `agent_panel.consensus`, `agent_panel.actionable_insight` | No (always show) |
| 02 | Predicted Trial Rate | `forecast.trial_rate.percentage`, `forecast.trial_rate.range`, `forecast.confidence` | No |
| 03 | Coverage & Confidence | `forecast.diagnostics.coverage_tier`, `forecast.confidence_reasons`, `forecast.diagnostics.eligible_count`, `forecast.fallback_used` | No |
| 04 | Price / Offer Sensitivity | `agent_panel.comparable_price_range`, `agent_panel.counterfactuals` | Hide if `comparable_price_range == null` |
| 05 | Most Receptive Segment | `agent_panel.most_receptive_segment` | No |
| 06 | Drivers & Objections | `agent_panel.top_drivers`, `agent_panel.top_objections` | No |
| 07 | Winning Message | `agent_panel.winning_message` | Hide if empty |
| 08 | Buyer Journeys | `agent_panel.buyer_journeys` | No |
| 09 | Individual Buyer Opinions | `agent_panel.agents[]` (with filter chips: All / Buy / Considering / Won't Buy) | No |
| 10 | Risk Factors | `agent_panel.risk_factors` | Hide if no holdouts |
| 11 | Counterfactual Scenarios | `agent_panel.counterfactuals` | Hide if empty |
| 12 | Why This Might Be Wrong | `forecast.why_might_be_wrong` | No |
| 13 | Recommended Next Action | Derived from `forecast.confidence` (template logic in `customer_report.py`) | No |

---

## 6. Mock JSON for frontend dev

While backend is being implemented, frontend devs use this exact mock to render against:

`frontend/src/mocks/discuss_mock_triton.json` — 50-agent example
`frontend/src/mocks/discuss_mock_yeti.json` — low-confidence example
`frontend/src/mocks/discuss_mock_liquidiv.json` — high-confidence example

Hamza will publish these mocks as soon as Phase 1 backend code is structured (~next few hours). Until then, you can hand-author against the schema above.

---

## 7. Acceptance criteria (before merge to `tier2-v3-lite-yc`)

Backend tests:

```
✓ /discuss returns agents with names
✓ /discuss returns verdict per agent
✓ /discuss returns round_responses per agent (3 entries)
✓ /discuss returns buyer_journeys
✓ /discuss does not change trial_rate
✓ low-confidence forecast creates coverage_warning
✓ agent_count=20 returns 20 agents
✓ agent_count=50 returns 50 agents
✓ same input returns same agents (determinism)
✓ BUY count <= ceil(2 * trial_rate * agent_count)
✓ low-confidence/fallback caps BUY count at 15%
✓ verdict distribution sums to agent_count
✓ templates match assigned verdict bucket
```

Frontend manual acceptance:

```
✓ Custom Triton Drinks → simulation page → 50 individual agent cards
✓ Each agent has name, persona, verdict, reasoning, objection
✓ Report page shows all 13 numbered sections (or honestly hides them)
✓ No forbidden v2 strings present anywhere
✓ Predicted trial rate matches /forecast output exactly
✓ Liquid IV deterministic still passes (22.0% × 5)
✓ YETI report shows weak coverage banner honestly
✓ Verdict color tags: BUY=green, CONSIDERING=yellow, WON'T BUY=red
```

---

## 8. Branch discipline

- All UI rebuild work happens on `tier2-v3-ui-rebuild`
- `tier2-v3-lite-yc` (current production branch) stays untouched until merge
- Vercel preview deploys from `tier2-v3-ui-rebuild` for testing
- Production demo URL stays on stable d7dbb07 throughout the rebuild
- Merge to `tier2-v3-lite-yc` only when all backend + frontend acceptance criteria pass

---

**Schema is frozen. Any changes require explicit approval.**
