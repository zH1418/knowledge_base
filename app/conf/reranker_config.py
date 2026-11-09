from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RerankerConfig:
    bge_reranker_large: str       # 本地模型路径（见 .env BGE_RERANKER_LARGE）
    bge_reranker_device: str      # 模型运行设备 cpu / cuda
    bge_reranker_fp16: bool       # 半精度（CPU 下无意义，GPU 下省显存）

    dashscope_api_key: str        # DashScope API 密钥（复用百炼 MCP_WEBSEARCH_API_KEY）
    text_rerank_model: str        # 云端重排序模型名
    text_rerank_instruct: str     # 重排序任务指令

    rerank_enabled: bool          # ★ 是否启用重排（False ⇒ 直接用 RRF 序取 Top-K，跳过断崖）
    rerank_provider: str          # dashscope | voyage | local_kalm
    voyage_api_key: str           # Voyage AI 密钥（.env VOYAGE_API_KEY）
    voyage_rerank_model: str      # rerank-3 | rerank-3-lite

    local_rerank_exe: str         # llama-kalm-reranker.exe 路径
    local_rerank_model: str       # *.gguf 路径
    local_rerank_threads: int
    local_rerank_max_chars: int   # ★ 送入 rerank 的正文截断字数（0=不截断）

    siliconflow_api_key: str      # SiliconFlow 密钥（.env SILICONFLOW_API_KEY）
    siliconflow_rerank_model: str # BAAI/bge-reranker-v2-m3 | Pro/BAAI/bge-reranker-v2-m3


reranker_config = RerankerConfig(
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE", ""),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE", "cpu"),
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16", "0") in ("1", "True", "true", 1),

    dashscope_api_key=os.getenv("MCP_WEBSEARCH_API_KEY", ""),
    text_rerank_model=os.getenv("TEXT_RERANK_MODEL", "qwen3-rerank"),
    text_rerank_instruct=os.getenv(
        "TEXT_RERANK_INSTRUCT", "针对给定的查询，检索能够解答该查询的相关段落"
    ),

    rerank_enabled=os.getenv("RERANK_ENABLED", "1") not in ("0", "false", "False"),
    rerank_provider=os.getenv("RERANK_PROVIDER", "dashscope").strip().lower(),
    voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
    voyage_rerank_model=os.getenv("VOYAGE_RERANK_MODEL", "rerank-3"),

    local_rerank_exe=os.getenv("LOCAL_RERANK_EXE", ""),
    local_rerank_model=os.getenv("LOCAL_RERANK_MODEL", ""),
    local_rerank_threads=int(os.getenv("LOCAL_RERANK_THREADS", "4")),
    local_rerank_max_chars=int(os.getenv("LOCAL_RERANK_MAX_CHARS", "512")),

    siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
    siliconflow_rerank_model=os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
)
