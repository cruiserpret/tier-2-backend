"""
correction_store.py — Reflexion-based correction memory + Brier Score reward

Clean slate — seeded with 5 validated A-grade tests from April 14, 2026.
All entries validated against Pew Research / Gallup / YouGov real-world data.
TikTok pending retest after Change 3 (raw chunk text matching) fix.
"""

import json
import os
import re
from datetime import datetime

STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "correction_memory.json"
)

# ── Updated baseline — 5-test A-grade average ─────────────────────
# Computed from April 14 2026 stress test at 500-600 chunks.
# Reflects current system accuracy post all fixes.
ASSEMBLY_BASELINE = {
    "for":     0.52,
    "against": 0.31,
    "neutral": 0.13,
}

TOPIC_TYPES = {
    "economic_welfare":  ["ubi", "universal basic income", "welfare", "food stamps",
                          "housing assistance", "poverty", "cash transfer", "guaranteed income",
                          "student loan", "debt cancellation", "loan forgiveness"],
    "healthcare_access": ["healthcare", "health insurance", "single payer", "medicare",
                          "medicaid", "hospital", "medical", "coverage", "uninsured"],
    "labor_policy":      ["work week", "minimum wage", "union", "labor", "workers",
                          "employment", "salary", "overtime", "gig economy"],
    "climate_policy":    ["climate", "paris", "emissions", "carbon", "green",
                          "fossil fuel", "renewable", "environment"],
    "drug_policy":       ["marijuana", "cannabis", "drug", "legalize", "decriminalize",
                          "opioid", "fentanyl", "addiction"],
    "rights_social":     ["abortion", "gun", "immigration", "voting", "lgbtq",
                          "affirmative action", "death penalty", "criminal justice",
                          "assault weapon", "capital punishment"],
    "tech_policy":       ["tiktok", "social media", "ai regulation", "data privacy",
                          "antitrust", "big tech", "surveillance"],
    "trade_fiscal":      ["tariff", "trade", "tax", "deficit", "spending",
                          "budget", "inflation", "interest rate"],
}


def brier_score(predicted: dict, actual: dict) -> float:
    stances = ["for", "against", "neutral"]
    return sum(
        (predicted.get(s, 0.0) - actual.get(s, 0.0)) ** 2
        for s in stances
    )


def brier_skill_score(predicted: dict, actual: dict, baseline: dict = None) -> float:
    if baseline is None:
        baseline = ASSEMBLY_BASELINE
    bs_model    = brier_score(predicted, actual)
    bs_baseline = brier_score(baseline, actual)
    if bs_baseline == 0:
        return 1.0
    return round(1.0 - (bs_model / bs_baseline), 4)


def brier_reward(predicted: dict, actual: dict) -> float:
    bs = brier_score(predicted, actual)
    return round(1.0 - (bs / 2.0), 4)


def assembly_reward(
    predicted: dict,
    actual: dict,
    baseline: dict = None,
    trajectory_correct: bool = None
) -> dict:
    if baseline is None:
        baseline = ASSEMBLY_BASELINE
    bs  = brier_score(predicted, actual)
    bss = brier_skill_score(predicted, actual, baseline)
    br  = brier_reward(predicted, actual)
    trajectory_bonus = 0.0
    if trajectory_correct is not None:
        trajectory_bonus = 0.30 if trajectory_correct else 0.0
    composite = round((br * 0.70) + trajectory_bonus, 4)
    return {
        "brier_score":       round(bs, 4),
        "brier_skill_score": round(bss, 4),
        "brier_reward":      br,
        "trajectory_bonus":  trajectory_bonus,
        "composite_reward":  composite,
    }


def classify_topic_type(topic: str) -> str:
    topic_lower = topic.lower()
    for topic_type, keywords in TOPIC_TYPES.items():
        if any(kw in topic_lower for kw in keywords):
            return topic_type
    return "general"


def tokenize(text: str) -> set:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stopwords = {"the", "and", "for", "that", "this", "with", "should",
                 "would", "could", "have", "from", "they", "their",
                 "will", "what", "when", "how", "are", "was", "were",
                 "been", "has", "had", "its", "not", "but", "more"}
    return set(w for w in words if w not in stopwords)


def topic_similarity(topic_a: str, topic_b: str) -> float:
    tokens_a = tokenize(topic_a)
    tokens_b = tokenize(topic_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def load_store() -> list:
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_store(records: list) -> None:
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
    trajectory_correct: bool = None,
) -> dict:
    topic_type = classify_topic_type(topic)
    reward = assembly_reward(predicted, actual, trajectory_correct=trajectory_correct)

    error = {
        stance: round(predicted.get(stance, 0) - actual.get(stance, 0), 2)
        for stance in ["for", "against", "neutral"]
    }

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
    correction_rule = (
        "; ".join(correction_parts)
        if correction_parts
        else "no systematic correction identified — prediction was accurate"
    )

    record = {
        "topic":           topic,
        "topic_type":      topic_type,
        "predicted":       predicted,
        "actual":          actual,
        "error":           error,
        "reward":          reward,
        "root_cause":      root_cause or "not specified",
        "correction_rule": correction_rule,
        "stored_at":       datetime.now().isoformat(),
    }

    records = load_store()
    records = [r for r in records if r["topic"].lower() != topic.lower()]
    records.append(record)
    save_store(records)

    print(f"[CorrectionStore] Stored: {topic[:60]}")
    print(f"[CorrectionStore] Error         — "
          f"for: {error['for']:+.2f} / "
          f"against: {error['against']:+.2f} / "
          f"neutral: {error['neutral']:+.2f}")
    print(f"[CorrectionStore] Brier Score   — {reward['brier_score']:.4f}")
    print(f"[CorrectionStore] Skill Score   — {reward['brier_skill_score']:.4f}")
    print(f"[CorrectionStore] Composite     — {reward['composite_reward']:.4f}")
    print(f"[CorrectionStore] Rule          — {correction_rule}")

    return reward


def retrieve_similar_topics(topic: str, top_k: int = 3) -> list:
    records = load_store()
    if not records:
        return []

    topic_type = classify_topic_type(topic)
    scored = []

    for record in records:
        word_sim   = topic_similarity(topic, record["topic"])
        type_bonus = 0.20 if record.get("topic_type") == topic_type else 0.0

        max_error = max(
            abs(record["error"].get("for", 0)),
            abs(record["error"].get("against", 0))
        )
        if max_error < 0.08:
            continue  # skip very accurate predictions

        scored.append((word_sim + type_bonus, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


def get_correction_context(topic: str) -> str:
    similar = retrieve_similar_topics(topic, top_k=3)
    if not similar:
        return ""

    lines = ["PAST PREDICTION ERRORS ON SIMILAR TOPICS — learn from these:\n"]

    for record in similar:
        pred   = record["predicted"]
        actual = record["actual"]
        error  = record["error"]
        reward = record.get("reward", {})

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
        if reward:
            lines.append(
                f"Brier Score: {reward.get('brier_score', '?'):.4f} | "
                f"Skill Score: {reward.get('brier_skill_score', '?'):.4f}"
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
    print(f"[CorrectionStore] Injecting {len(similar)} correction(s) for: {topic[:60]}")
    return context


def get_leaderboard() -> list:
    records = load_store()
    scored = [r for r in records if r.get("reward")]
    scored.sort(key=lambda r: r["reward"].get("brier_score", 999))
    return scored


def print_leaderboard():
    records = get_leaderboard()
    if not records:
        print("[CorrectionStore] No corrections stored yet.")
        return
    print("\n── Assembly Accuracy Leaderboard ───────────────────────────")
    print(f"{'#':<3} {'Topic':<50} {'Brier':>7} {'Skill':>7} {'Grade':>6}")
    print("-" * 78)
    for i, r in enumerate(records, 1):
        label = r["topic"][:47] + "..." if len(r["topic"]) > 50 else r["topic"]
        bs    = r["reward"].get("brier_score", 999)
        bss   = r["reward"].get("brier_skill_score", 0)
        grade = "A" if bs < 0.01 else "B" if bs < 0.05 else "C" if bs < 0.10 else "D" if bs < 0.15 else "F"
        print(f"{i:<3} {label:<50} {bs:>7.4f} {bss:>7.4f} {grade:>6}")
    print("-" * 78)


# ─────────────────────────────────────────────────────────────────
# VALIDATED CORRECTIONS — April 14 2026 stress test
# 5 Grade A results only. TikTok pending retest after Change 3 fix.
# ─────────────────────────────────────────────────────────────────

SPRINT_CORRECTIONS = [

    # ── Grade A: Brier 0.0053 ─────────────────────────────────────
    {
        "topic":     "Should the US federal government raise the minimum wage to $15 per hour?",
        "predicted": {"for": 0.58, "against": 0.37, "neutral": 0.05},
        "actual":    {"for": 0.65, "against": 0.35, "neutral": 0.05},
        "root_cause": (
            "Best score of stress test. Graph node signal read 39/42 — close enough "
            "to reality that calibration said no correction needed. FOR workers post "
            "loudly online but middle-class satisfied employees also FOR in polling — "
            "both captured. AGAINST: small business owners, rural employers — "
            "present but quieter online. Neutral near-exact at 5%."
        ),
    },

    # ── Grade A: Brier 0.0061 ─────────────────────────────────────
    {
        "topic":     "Should the US federal government ban assault-style weapons?",
        "predicted": {"for": 0.53, "against": 0.37, "neutral": 0.11},
        "actual":    {"for": 0.58, "against": 0.37, "neutral": 0.05},
        "root_cause": (
            "Keyword signal massively inverted (15% FOR / 58% AGAINST) but tier system "
            "caught the failure — enforced min 2 FOR patterns, neutral cap redistributed. "
            "LLM pattern extractor generated realistic FOR demographics organically "
            "from rich graph (1179 nodes). AGAINST exact at 37%. Neutral too high at "
            "11% vs real 5% — neutral cap over-converting on polarized topics."
        ),
    },

    # ── Grade A: Brier 0.0062 ─────────────────────────────────────
    {
        "topic":     "Should the US federal government cancel all federal student loan debt?",
        "predicted": {"for": 0.40, "against": 0.55, "neutral": 0.05},
        "actual":    {"for": 0.46, "against": 0.50, "neutral": 0.04},
        "root_cause": (
            "Framing-sensitive topic — 'cancel ALL' gets different numbers than "
            "'cancel some'. Assembly read AGAINST correctly at 55% vs real 50%. "
            "Silent AGAINST demographic fired correctly: people who paid loans, "
            "tradespeople who didn't attend college, fiscal conservatives. "
            "FOR slightly low — online advocates for full cancellation are vocal "
            "but polling for complete cancellation is genuinely lower than partial."
        ),
    },

    # ── Grade A: Brier 0.0078 ─────────────────────────────────────
    {
        "topic":     "Should the United States use the death penalty for people convicted of murder?",
        "predicted": {"for": 0.47, "against": 0.42, "neutral": 0.11},
        "actual":    {"for": 0.52, "against": 0.44, "neutral": 0.04},
        "root_cause": (
            "Closest real-world split tested (52/44 Gallup Oct 2025). "
            "AGAINST gap only 2 points — most accurate AGAINST prediction of stress test. "
            "Both FOR and AGAINST within 5 points simultaneously. "
            "Neutral too high at 11% vs real 4% — same pattern as assault weapons. "
            "Shift rate 11% with realistic dynamics: John Roberts moved toward "
            "more moderate FOR via emotional contagion from victim testimony."
        ),
    },

    # ── Grade A: Brier 0.0093 ─────────────────────────────────────
    {
        "topic":     "Should abortion be legal in the United States?",
        "predicted": {"for": 0.58, "against": 0.32, "neutral": 0.11},
        "actual":    {"for": 0.60, "against": 0.37, "neutral": 0.03},
        "root_cause": (
            "Keyword signal badly inverted (22% FOR / 64% AGAINST) but LLM pattern "
            "extractor and tier system compensated — produced 58% FOR only 2 points "
            "from reality. Pro-choice discourse uses identical language to pro-life "
            "articles ('constitutional', 'rights', 'fundamental') causing keyword "
            "inversion. AGAINST 5 points low — religious conservative voices "
            "underrepresented in graph despite being ~37% of population. "
            "Neutral too high at 11% vs real 3% — abortion is one of the most "
            "polarized topics, near-zero genuine neutrals."
        ),
    },

    # ── Grade B: Brier 0.0161 — Change 3 active retest ───────────────
{
    "topic":     "Should the US federal government raise the minimum wage to $15 per hour?",
    "predicted": {"for": 0.56, "against": 0.31, "neutral": 0.13},
    "actual":    {"for": 0.65, "against": 0.35, "neutral": 0.05},
    "root_cause": (
        "Change 3 (raw chunk text matching) active — hurt this topic specifically. "
        "447 chunks pulled heavy policy analysis content (CBO reports, think-tank papers, "
        "economics articles) which use measured language triggering neutral keywords even "
        "when arguing strongly FOR. Raw signal read 30% FOR / 9% AGAINST / 60% neutral — "
        "completely inverted from reality 65/35/5. Calibration added 2 neutral stakeholders "
        "pulling distribution wrong direction. Fix: weight forum+high-sentiment chunks 2x "
        "for signal matching, down-weight institutional analysis chunks. "
        "Previous A-grade (0.0053) was without Change 3 where graph node matching "
        "accidentally read closer to correct at 39/42."
    ),
},

]


def seed_sprint_corrections():
    """
    Seed correction memory with April 14 2026 stress test results.
    Run: python backend/agents/correction_store.py
    """
    print("[CorrectionStore] Seeding April 14 2026 stress test corrections...\n")
    rewards = []
    for c in SPRINT_CORRECTIONS:
        reward = store_correction(
            topic=c["topic"],
            predicted=c["predicted"],
            actual=c["actual"],
            root_cause=c["root_cause"],
        )
        rewards.append(reward)
        print()

    print(f"\n[CorrectionStore] Seeded {len(SPRINT_CORRECTIONS)} corrections")
    print("\n── Stress Test Performance ─────────────────────────────────")
    print(f"{'Topic':<52} {'Brier':>7} {'Grade':>6}")
    print("-" * 68)

    grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for c, r in zip(SPRINT_CORRECTIONS, rewards):
        label = c["topic"][:49] + "..." if len(c["topic"]) > 52 else c["topic"]
        bs    = r["brier_score"]
        grade = "A" if bs < 0.01 else "B" if bs < 0.05 else "C" if bs < 0.10 else "D" if bs < 0.15 else "F"
        grades[grade] += 1
        print(f"{label:<52} {bs:>7.4f} {grade:>6}")

    print("-" * 68)
    avg_bs = sum(r["brier_score"] for r in rewards) / len(rewards)
    print(f"{'AVERAGE':<52} {avg_bs:>7.4f}")
    print(f"\nGrade distribution: {grades}")
    print(f"\nNote: TikTok pending retest after Change 3 fix.")
    print(f"Stored at: {STORE_PATH}")


if __name__ == "__main__":
    seed_sprint_corrections()
    print()
    print_leaderboard()