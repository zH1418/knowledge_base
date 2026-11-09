from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # 本地模型路径
    bge_m3: str       # 模型仓库标识
    bge_device: str   # 运行设备(cuda:0/cpu)
    bge_fp16: bool    # 是否开启半精度（1=True/0=False）
    dense_vector_dim: int  # BGE-M3 稠密向量维度（模型固定 1024）

embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH", ""),
    bge_m3=os.getenv("BGE_M3", "BAAI/bge-m3"),
    bge_device=os.getenv("BGE_DEVICE", "cpu"),
    bge_fp16=os.getenv("BGE_FP16", "0") in ("1", "True", "true", 1),
    dense_vector_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
)