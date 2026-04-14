import json
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json

# ── Continuous spectrum formula ───────────────────────────────────
# inst_ratio = 0.80 - (public_score * 0.60)
# pub_ratio  = 0.20 + (public_score * 0.60)
#
# public_score = 0.0  → 80/20 (pure institutional)
# public_score = 0.5  → 50/50 (balanced)
# public_score = 1.0  → 20/80 (pure public)

def calculate_split(public_score: float) -> tuple[float, float]:
    """Convert public_score (0-1) to institutional/public ratio."""
    public_score = max(0.0, min(1.0, public_score))
    inst_ratio = round(0.80 - (public_score * 0.60), 2)
    pub_ratio  = round(0.20 + (public_score * 0.60), 2)
    return inst_ratio, pub_ratio

def safe_parse(raw: str) -> dict:
    if not raw:
        return {}
    for attempt in [
        lambda r: json.loads(re.sub(r'[\x00-\x1f\x7f]', ' ', r)),
        lambda r: json.loads(re.sub(r'[\x00-\x1f\x7f]', ' ', r)[r.find('{'):r.rfind('}')+1]),
        lambda r: json.loads(re.sub(r'```json|```', '', r).strip()),
    ]:
        try:
            return attempt(raw)
        except Exception:
            pass
    return {}

async def classify_topic_primary(topic: str) -> dict:
    """
    Call 1 — Primary classifier.
    Rates topic on 5 dimensions, returns public_score and reasoning.
    """
    system = """You are an expert at understanding who drives public debates.
Rate this topic on 5 dimensions to determine the institutional/public split.
Respond in valid JSON only."""

    prompt = f"""Rate this debate topic on 5 dimensions: "{topic}"

For each dimension, give a score from 0.0 to 1.0:

1. personal_impact (0-1)
   How directly does this affect ordinary people's daily lives?
   0 = affects only institutions (central bank policy, merger approvals)
   1 = affects every person's daily life (abortion, workplace rights, healthcare)

2. emotional_salience (0-1)
   How strongly do ordinary people feel about this personally?
   0 = technical/abstract (antitrust law, trade agreements)
   1 = deeply personal and emotional (abortion, gun violence, immigration)

3. institutional_decision (0-1)
   How much is the final outcome determined by institutions vs public behavior?
   0 = entirely public behavior (consumer choices, voting)
   1 = entirely institutional decision (regulatory approval, court ruling)

4. public_discourse_volume (0-1)
   How much do ordinary people discuss this in forums, social media, daily life?
   0 = mostly expert/institutional discussion
   1 = massive public discourse — Reddit, Twitter, dinner table conversations

5. expert_dominance (0-1)
   How much do experts/technocrats dominate the real debate?
   0 = ordinary people's lived experience is the primary evidence
   1 = requires specialized knowledge — economists, lawyers, scientists only

Examples to calibrate:
- "Should the Fed raise interest rates" → personal_impact: 0.2, emotional_salience: 0.1, institutional_decision: 0.95, public_discourse: 0.1, expert_dominance: 0.95
- "Should abortion be legal" → personal_impact: 0.95, emotional_salience: 0.98, institutional_decision: 0.4, public_discourse: 0.95, expert_dominance: 0.1
- "Should AI be regulated" → personal_impact: 0.5, emotional_salience: 0.4, institutional_decision: 0.8, public_discourse: 0.5, expert_dominance: 0.7

Respond in this exact JSON format:
{{
    "personal_impact": 0.7,
    "emotional_salience": 0.6,
    "institutional_decision": 0.5,
    "public_discourse_volume": 0.6,
    "expert_dominance": 0.4,
    "reasoning": "one sentence explanation of the dominant characteristic",
    "key_actors": "who primarily drives this debate"
}}"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = safe_parse(result)

        personal_impact        = float(parsed.get("personal_impact", 0.5))
        emotional_salience     = float(parsed.get("emotional_salience", 0.5))
        institutional_decision = float(parsed.get("institutional_decision", 0.5))
        public_discourse       = float(parsed.get("public_discourse_volume", 0.5))
        expert_dominance       = float(parsed.get("expert_dominance", 0.5))

        # Weighted formula — personal impact and emotional salience
        # are the strongest predictors of public-dominant debate
        public_score = (
            personal_impact        * 0.30 +
            emotional_salience     * 0.25 +
            (1 - institutional_decision) * 0.20 +
            public_discourse       * 0.15 +
            (1 - expert_dominance) * 0.10
        )
        public_score = round(public_score, 3)

        return {
            "public_score": public_score,
            "dimensions": {
                "personal_impact": personal_impact,
                "emotional_salience": emotional_salience,
                "institutional_decision": institutional_decision,
                "public_discourse_volume": public_discourse,
                "expert_dominance": expert_dominance,
            },
            "reasoning": parsed.get("reasoning", ""),
            "key_actors": parsed.get("key_actors", "")
        }

    except Exception as e:
        print(f"[TopicClassifier] Primary classification error: {e}")
        return {"public_score": 0.4, "dimensions": {}, "reasoning": "Error", "key_actors": "Unknown"}

async def critique_classification(topic: str, primary: dict) -> float:
    """
    Call 2 — Judge reviewer.
    Only intervenes when the primary classification is clearly wrong.
    High bar for correction — confirms when reasonable, corrects only when off by >0.20.

    Unlike a critic (who finds flaws), a judge evaluates fairly:
    "Is this classification accurate enough?" not "What's wrong with it?"
    """
    system = """You are an impartial judge reviewing a topic classification.
Your default position is to CONFIRM the classification unless it is clearly wrong.
You are NOT looking for flaws — you are asking if the score is reasonable.
Respond in valid JSON only."""

    dims = primary.get("dimensions", {})
    prompt = f"""Review this topic classification as an impartial judge.

Topic: "{topic}"

Classification to review:
- public_score: {primary['public_score']} (0=purely institutional, 1=purely public)
- personal_impact: {dims.get('personal_impact', '?')}
- emotional_salience: {dims.get('emotional_salience', '?')}
- institutional_decision: {dims.get('institutional_decision', '?')}
- public_discourse_volume: {dims.get('public_discourse_volume', '?')}
- expert_dominance: {dims.get('expert_dominance', '?')}
- reasoning: {primary.get('reasoning', '')}

Your job as judge:
1. Ask yourself: "Is this public_score reasonable for this topic?"
2. If yes — confirm it. Do not change it just because you might score it slightly differently.
3. Only intervene if the score is CLEARLY wrong by more than 0.20 points.

HIGH BAR FOR INTERVENTION — only correct if one of these is true:
- The topic is clearly institutional but scored above 0.7 (e.g. Fed interest rates at 0.8)
- The topic is clearly public but scored below 0.3 (e.g. abortion rights at 0.2)
- A critical dimension is obviously wrong (e.g. expert_dominance 0.9 for a topic that ordinary people clearly dominate)

DO NOT correct for:
- Minor disagreements (you'd score 0.65, it scored 0.68 — that's fine)
- Stylistic differences in reasoning
- Wanting to make it "more precise"
- Emotional salience on economic topics — high salience does NOT always mean high public score

Respond in this exact JSON format:
{{
    "classification_is_reasonable": true/false,
    "corrected_score": {primary['public_score']},
    "judge_reasoning": "one sentence — why you confirmed or why you intervened"
}}"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = safe_parse(result)

        is_reasonable = parsed.get("classification_is_reasonable", True)
        corrected = float(parsed.get("corrected_score", primary["public_score"]))
        original = primary["public_score"]
        diff = abs(corrected - original)

        print(f"[TopicClassifier] Judge: {'confirmed' if is_reasonable else 'intervened'} — {parsed.get('judge_reasoning', '')}")

        if is_reasonable or diff < 0.20:
            return original

        print(f"[TopicClassifier] Judge corrected: {original:.2f} → {corrected:.2f}")
        return corrected

    except Exception as e:
        print(f"[TopicClassifier] Judge error: {e} — keeping original score")
        return primary["public_score"]
    
async def classify_topic(topic: str) -> dict:
    """
    Main classifier with continuous spectrum + reflection loop.

    Call 1: Rate 5 dimensions → compute public_score
    Call 2: Critic reviews → corrects if major flaw found
    Final: Calculate inst/pub ratio from corrected score
    """
    print(f"[TopicClassifier] Classifying: {topic}")

    # Call 1 — primary classification
    primary = await classify_topic_primary(topic)
    original_score = primary["public_score"]

    # Call 2 — reflection critic
    final_score = await critique_classification(topic, primary)

    # Calculate final split
    inst_ratio, pub_ratio = calculate_split(final_score)

    # Determine label for logging
    if final_score < 0.35:
        label = "institutional"
    elif final_score < 0.65:
        label = "mixed"
    else:
        label = "public"

    print(f"[TopicClassifier] Score: {original_score:.2f} → {final_score:.2f} (after reflection)")
    print(f"[TopicClassifier] Split: {inst_ratio*100:.0f}% institutional / {pub_ratio*100:.0f}% public ({label})")
    print(f"[TopicClassifier] Reasoning: {primary.get('reasoning', '')}")

    return {
        "public_score": final_score,
        "institutional_ratio": inst_ratio,
        "public_ratio": pub_ratio,
        "label": label,
        "reasoning": primary.get("reasoning", ""),
        "key_actors": primary.get("key_actors", ""),
        "dimensions": primary.get("dimensions", {}),
        "original_score": original_score,
        "reflection_applied": final_score != original_score,
    }