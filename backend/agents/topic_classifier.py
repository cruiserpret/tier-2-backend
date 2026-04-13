import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.utils.llm_client import call_llm_json

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
    system = """You are an expert at understanding who drives debates on different topics.
Classify whether a topic is primarily driven by institutional actors or public sentiment.
Respond in valid JSON only."""

    prompt = f"""Classify this debate topic: {topic}

Determine whether this is primarily an INSTITUTIONAL or PUBLIC topic.

INSTITUTIONAL topic — outcome primarily determined by:
- Governments, regulators, corporations, international bodies
- Experts, academics, policy makers in closed rooms
- General public has opinions but institutions make the final call
- Examples: AI regulation frameworks, merger approvals, trade agreements,
  corporate restructuring, OpenAI nonprofit status, antitrust law,
  central bank interest rate policy, military defense contracts

PUBLIC topic — outcome primarily determined by:
- Everyday people — their votes, their behavior, their lived experience
- Public pressure forces institutional response
- The topic directly affects how millions of ordinary people live daily
- Examples: workplace policies (RTO, remote work), consumer product bans,
  social media behavior, lifestyle choices, free college tuition,
  student loan debt, healthcare access, housing affordability,
  minimum wage, artist compensation, UBI, immigration lived impact,
  social media bans for children, voting rights and participation,
  mandatory voting, compulsory civic duties, election reform,
  democratic participation, gun control, abortion rights,
  drug legalization, criminal justice reform, police reform,
  climate change personal impact, food labeling, public health mandates,
  caste reservations, affirmative action, racial equity policies

CRITICAL RULE — THE MOST IMPORTANT TEST:
Ask yourself: "Does this topic directly affect how everyday people vote,
work, raise children, pay bills, access healthcare, or exercise their
democratic rights?"

If YES -> classify as PUBLIC, even if governments make the final decision.
The question is WHO IS MOST AFFECTED, not who has the final vote.

MANDATORY VOTING, CIVIC PARTICIPATION, AFFIRMATIVE ACTION, CASTE POLICY
are always PUBLIC topics. They affect every citizen's daily life directly.

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