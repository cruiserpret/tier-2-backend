import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json

# Binary split ratios
# institutional_ratio = fraction of agents that are institutional actors
# public_ratio = fraction of agents that are public/demographic actors

INSTITUTIONAL_TOPIC = {
    "institutional_ratio": 0.80,
    "public_ratio": 0.20,
    "label": "institutional"
}

PUBLIC_TOPIC = {
    "institutional_ratio": 0.40,
    "public_ratio": 0.60,
    "label": "public"
}

async def classify_topic(topic: str) -> dict:
    """
    Classify topic as institutional or public to determine agent split ratio.

    Institutional topics — primarily debated by organizations, governments,
    corporations. Public sentiment exists but institutional actors drive outcomes.
    Examples: AI regulation, corporate restructuring, trade policy, mergers

    Public topics — primarily debated by everyday people. Institutional actors
    exist but public sentiment drives outcomes.
    Examples: RTO policy, social media bans, consumer products, workplace rights
    """

    system = """You are an expert at understanding who drives debates on different topics.
Classify whether a topic is primarily driven by institutional actors or public sentiment.
Respond in valid JSON only."""

    prompt = f"""Classify this debate topic: {topic}

Determine whether this is primarily an INSTITUTIONAL or PUBLIC topic.

INSTITUTIONAL topic — the outcome is primarily determined by:
- Governments, regulators, corporations, international bodies
- Experts, academics, policy makers
- The general public has opinions but institutions decide
- Examples: AI regulation, merger approvals, trade agreements, corporate restructuring

PUBLIC topic — the outcome is primarily determined by:
- Employees, consumers, communities, general population
- Public pressure drives institutional response
- Lived experience matters more than policy expertise
- Examples: workplace policies, consumer product decisions, social media behavior, lifestyle choices

Respond in this exact JSON format:
{{
    "topic_type": "institutional/public",
    "reasoning": "one sentence explanation of why",
    "key_actors": "who primarily drives this debate",
    "confidence": 0.85
}}"""

    try:
        result = await call_llm_json(prompt, system)
        parsed = json.loads(result)
        topic_type = parsed.get("topic_type", "institutional").lower().strip()

        if topic_type == "public":
            classification = PUBLIC_TOPIC.copy()
        else:
            classification = INSTITUTIONAL_TOPIC.copy()

        classification["reasoning"] = parsed.get("reasoning", "")
        classification["key_actors"] = parsed.get("key_actors", "")
        classification["confidence"] = parsed.get("confidence", 0.8)

        print(f"[TopicClassifier] Topic type: {classification['label']} "
              f"(institutional: {classification['institutional_ratio']*100:.0f}% / "
              f"public: {classification['public_ratio']*100:.0f}%)")
        print(f"[TopicClassifier] Reasoning: {classification['reasoning']}")

        return classification

    except Exception as e:
        print(f"[TopicClassifier] Error: {e} — defaulting to institutional")
        result = INSTITUTIONAL_TOPIC.copy()
        result["reasoning"] = "Defaulted due to classification error"
        result["key_actors"] = "Unknown"
        result["confidence"] = 0.5
        return result