import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "anthropic/claude-3.5-haiku")
ZEP_API_KEY = os.getenv("ZEP_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "assembly-dev")
# ── Tier 2 DTC ────────────────────────────────────────────────
RAINFOREST_API_KEY = os.getenv("RAINFOREST_API_KEY", "")
EASYPARSER_API_KEY = os.getenv("EASYPARSER_API_KEY", "")
APIFY_API_KEY      = os.getenv("APIFY_API_KEY", "")
