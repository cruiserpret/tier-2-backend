"""
correction_store.py — Reflexion-based correction memory for Assembly

Grounded in Shinn et al. 2023 (Reflexion, Princeton/MIT):
Instead of updating model weights, we store verbal memories of past
prediction failures and inject them as context into future prompts.
The model can't learn permanently, but it can learn across sessions
through retrieved correction context.

Workflow:
  1. After each simulation, call store_correction() with real polling data
  2. At simulation start, call get_correction_context() to retrieve
     similar past failures
  3. Inject returned context into calibration prompt
"""

import json
import os
import re
from datetime import datetime

# ── Storage path ──────────────────────────────────────────────────
# Stored as a flat JSON file alongside the backend
STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "correction_memory.json"
)

# ── Topic type taxonomy ───────────────────────────────────────────
# Used for grouping similar topics — corrections on UBI apply to
# other economic welfare topics, healthcare corrections apply to
# other access/coverage topics, etc.
TOPIC_TYPES = {
    "economic_welfare":   ["ubi", "universal basic income", "welfare", "food stamps",
                           "housing assistance", "poverty", "cash transfer", "guaranteed income"],
    "healthcare_access":  ["healthcare", "health insurance", "single payer", "medicare",
                           "medicaid", "hospital", "medical", "coverage", "uninsured"],
    "labor_policy":       ["work week", "minimum wage", "union", "labor", "workers",
                           "employment", "salary", "overtime", "gig economy"],
    "climate_policy":     ["climate", "paris", "emissions", "carbon", "green",
                           "fossil fuel", "renewable", "environment"],
    "drug_policy":        ["marijuana", "cannabis", "drug", "legalize", "decriminalize",
                           "opioid", "fentanyl", "addiction"],
    "rights_social":      ["abortion", "gun", "immigration", "voting", "lgbtq",
                           "affirmative action", "death penalty", "criminal justice"],
    "tech_policy":        ["tiktok", "social media", "ai regulation", "data privacy",
                           "antitrust", "big tech", "surveillance"],
    "trade_fiscal":       ["tariff", "trade", "tax", "deficit", "spending",
                           "budget", "inflation", "interest rate"],
}


def classify_topic_type(topic: str) -> str:
    """Classify a topic string into one of the taxonomy types."""
    topic_lower = topic.lower()
    for topic_type, keywords in TOPIC_TYPES.items():
        if any(kw in topic_lower for kw in keywords):
            return topic_type
    return "general"


def tokenize(text: str) -> set:
    """Simple word tokenization for similarity matching."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stopwords = {"the", "and", "for", "that", "this", "with", "should",
                 "would", "could", "have", "from", "they", "their",
                 "will", "what", "when", "how", "are", "was", "were",
                 "been", "has", "had", "its", "not", "but", "more"}
    return set(w for w in words if w not in stopwords)


def topic_similarity(topic_a: str, topic_b: str) -> float:
    """
    Compute similarity between two topics using Jaccard coefficient
    on tokenized word sets.

    Returns 0.0 (no overlap) to 1.0 (identical).
    """
    tokens_a = tokenize(topic_a)
    tokens_b = tokenize(topic_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def load_store() -> list:
    """Load correction memory from disk."""
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_store(records: list) -> None:
    """Save correction memory to disk."""
    try:
        with open(STORE_PATH, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"[CorrectionStore] Save error: {e}")


def store_correction(
    topic: str,
    predicted: dict,
    actual: dict,
    root_cause: str = "",
) -> None:
    """
    Store a prediction error after comparing simulation output to real polling.

    Call this after each test where you have real polling benchmarks.

    Args:
        topic: The exact simulation topic string
        predicted: {"for": 0.70, "against": 0.15, "neutral": 0.15}
        actual:    {"for": 0.45, "against": 0.54, "neutral": 0.08}
        root_cause: Human-written explanation of why the error occurred
    """
    topic_type = classify_topic_type(topic)

    error = {
        stance: round(predicted.get(stance, 0) - actual.get(stance, 0), 2)
        for stance in ["for", "against", "neutral"]
    }

    # Auto-generate correction rule from error direction
    correction_parts = []
    if error["for"] > 0.10:
        correction_parts.append(
            f"reduce FOR signal by ~{int(error['for']*100)}% — "
            f"pro voices overrepresented in online discourse"
        )
    if error["against"] < -0.10:
        correction_parts.append(
            f"increase AGAINST signal by ~{int(abs(error['against'])*100)}% — "
            f"opposition voices underrepresented online"
        )
    if error["for"] < -0.10:
        correction_parts.append(
            f"increase FOR signal by ~{int(abs(error['for'])*100)}% — "
            f"support voices underrepresented online"
        )
    correction_rule = "; ".join(correction_parts) if correction_parts else "no systematic correction identified"

    record = {
        "topic":           topic,
        "topic_type":      topic_type,
        "predicted":       predicted,
        "actual":          actual,
        "error":           error,
        "root_cause":      root_cause or "not specified",
        "correction_rule": correction_rule,
        "stored_at":       datetime.now().isoformat(),
    }

    records = load_store()
    # Deduplicate — replace if same topic exists
    records = [r for r in records if r["topic"].lower() != topic.lower()]
    records.append(record)
    save_store(records)

    print(f"[CorrectionStore] Stored correction for: {topic}")
    print(f"[CorrectionStore] Error — for: {error['for']:+.2f} / "
          f"against: {error['against']:+.2f} / neutral: {error['neutral']:+.2f}")
    print(f"[CorrectionStore] Correction rule: {correction_rule}")


def retrieve_similar_topics(topic: str, top_k: int = 3) -> list:
    """
    Retrieve the top_k most similar past correction records.

    Similarity computed by:
    1. Topic type match (same category gets bonus)
    2. Jaccard word overlap on topic strings
    """
    records = load_store()
    if not records:
        return []

    topic_type = classify_topic_type(topic)
    scored = []

    for record in records:
        # Word overlap similarity
        word_sim = topic_similarity(topic, record["topic"])

        # Topic type bonus — same category is more relevant
        type_bonus = 0.20 if record.get("topic_type") == topic_type else 0.0

        # Only include records with meaningful errors
        max_error = max(
            abs(record["error"].get("for", 0)),
            abs(record["error"].get("against", 0))
        )
        if max_error < 0.10:
            continue  # Skip records with small errors — not useful

        scored.append((word_sim + type_bonus, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


def get_correction_context(topic: str) -> str:
    """
    Build a correction context string to inject into calibration prompts.

    Returns empty string if no relevant corrections found.
    Called before calibration in identify_stakeholders().

    Example output:
    ---
    PAST PREDICTION ERRORS ON SIMILAR TOPICS — learn from these:

    Topic: "Should the US implement UBI of $1000/month?" (economic_welfare)
    Predicted FOR: 70% | Real FOR: 45% | Error: +25%
    Root cause: online discourse overrepresents economic grievance voices
    Correction: reduce FOR by ~25%, increase AGAINST — silent majority doesn't post online

    INSTRUCTION: If this new topic has similar characteristics, apply the same
    correction direction when estimating population stance distribution.
    ---
    """
    similar = retrieve_similar_topics(topic, top_k=3)
    if not similar:
        return ""

    lines = ["PAST PREDICTION ERRORS ON SIMILAR TOPICS — learn from these:\n"]

    for record in similar:
        pred = record["predicted"]
        actual = record["actual"]
        error = record["error"]

        lines.append(f"Topic: \"{record['topic']}\" ({record['topic_type']})")
        lines.append(
            f"Predicted FOR: {pred.get('for', 0)*100:.0f}% | "
            f"Real FOR: {actual.get('for', 0)*100:.0f}% | "
            f"Error: {error.get('for', 0):+.0%}"
        )
        lines.append(
            f"Predicted AGAINST: {pred.get('against', 0)*100:.0f}% | "
            f"Real AGAINST: {actual.get('against', 0)*100:.0f}% | "
            f"Error: {error.get('against', 0):+.0%}"
        )
        lines.append(f"Root cause: {record['root_cause']}")
        lines.append(f"Correction: {record['correction_rule']}")
        lines.append("")

    lines.append(
        "INSTRUCTION: If this new topic has similar systematic biases, "
        "apply the same correction direction when estimating the population "
        "stance distribution. Especially watch for silent majorities that "
        "don't dominate online discourse."
    )

    context = "\n".join(lines)
    print(f"[CorrectionStore] Injecting {len(similar)} correction(s) for: {topic}")
    return context


# ── Pre-seeded corrections from sprint testing ────────────────────
# These are the errors we observed across 5 tests this sprint.
# Pre-seeded so the system has memory from day one.

SPRINT_CORRECTIONS = [
    {
        "topic": "Should the US federal government mandate a 4-day work week (32 hours) for all full-time employees?",
        "predicted": {"for": 0.55, "against": 0.25, "neutral": 0.20},
        "actual":    {"for": 0.38, "against": 0.42, "neutral": 0.20},
        "root_cause": "concept-vs-mandate gap — people support 4-day weeks (71%) but oppose federal mandate (38%). Keyword system read concept support as mandate support."
    },
    {
        "topic": "Should the United States rejoin the Paris Climate Agreement?",
        "predicted": {"for": 0.55, "against": 0.30, "neutral": 0.15},
        "actual":    {"for": 0.69, "against": 0.20, "neutral": 0.12},
        "root_cause": "fossil fuel industry content in graph scores AGAINST via keyword matching even when the article argues FOR climate action. False positive AGAINST hits on pro-climate content."
    },
    {
        "topic": "Should the US federal government implement a Universal Basic Income of $1,000 per month for every adult citizen?",
        "predicted": {"for": 0.70, "against": 0.15, "neutral": 0.15},
        "actual":    {"for": 0.45, "against": 0.54, "neutral": 0.08},
        "root_cause": "economic grievance voices dominate online UBI discourse. Workers facing hardship post loudly FOR. Upper-income, middle-class, and small business AGAINST voices are silent online."
    },
    {
        "topic": "Should the US replace its current healthcare system with a single government-run insurance program covering all Americans?",
        "predicted": {"for": 0.60, "against": 0.25, "neutral": 0.15},
        "actual":    {"for": 0.37, "against": 0.50, "neutral": 0.13},
        "root_cause": "healthcare horror stories dominate online discourse. 91% of insured Americans are satisfied with their coverage but don't post about it. AGAINST camp (satisfied insured, small business, conservatives) is systematically quiet online."
    },
]


def seed_sprint_corrections():
    """
    Pre-seed correction memory with observed errors from the sprint.
    Call once at startup or manually.
    """
    for correction in SPRINT_CORRECTIONS:
        store_correction(
            topic=correction["topic"],
            predicted=correction["predicted"],
            actual=correction["actual"],
            root_cause=correction["root_cause"],
        )
    print(f"[CorrectionStore] Seeded {len(SPRINT_CORRECTIONS)} sprint corrections")


if __name__ == "__main__":
    # Run directly to seed the correction database
    seed_sprint_corrections()
    print(f"\nStored at: {STORE_PATH}")

    # Test retrieval
    test_topic = "Should the US provide free college tuition for all Americans?"
    print(f"\nTest retrieval for: {test_topic}")
    context = get_correction_context(test_topic)
    print(context)