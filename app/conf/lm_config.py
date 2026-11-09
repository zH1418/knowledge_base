from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    lv_model: str
    llm_model: str
    extract_model: str
    llm_temperature: float
    llm_max_tokens: int      # 默认 max_tokens，0 表示不限

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    lv_model=os.getenv("VL_MODEL", ""),
    llm_model=os.getenv("LLM_DEFAULT_MODEL", ""),
    extract_model=os.getenv("LLM_EXTRACT_MODEL", ""),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")),
    llm_max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "0")),
)
