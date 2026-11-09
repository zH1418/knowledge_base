from __future__ import annotations

import concurrent.futures as cf
import os
import random
import threading
import time
from typing import Any

import httpx

from app.core.logger import logger
from app.core.decorators import log_execution, safe_execute
from app.core.exceptions import EmbeddingException, ParameterError
from app.conf.embedding_config import embedding_config
from app.utils.normalize_sparse_vector import normalize_sparse_vector

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "doubao-embedding-vision")
EMBED_ENDPOINT = os.getenv("EMBED_ENDPOINT", "/embeddings/multimodal")
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "4"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))          # 单请求文本条数
EMBED_MAX_RETRY = int(os.getenv("EMBED_MAX_RETRY", "6"))
EMBED_BACKOFF = float(os.getenv("EMBED_BACKOFF", "2.0"))

_url = EMBED_BASE_URL.rstrip("/") + EMBED_ENDPOINT


def _api_key() -> str:
    for k in ("EMBED_API_KEY", "ARK_API_KEY", "OPENAI_API_KEY"):
        v = os.getenv(k, "")
        if v:
            return v
    raise EmbeddingException(message="缺少嵌入 API key（EMBED_API_KEY / OPENAI_API_KEY）")


@safe_execute(component="bge-m3", reraise=True)
def get_bge_m3_ef():  # noqa: D401 —— 兼容旧引用（仅测试脚本使用）
    """已废弃：本地 BGE-M3 实现于 2026-10-08 移除（备份见 .backup_20261008）。"""
    raise EmbeddingException(
        message="本地 BGE-M3 已下线：嵌入已切换为独立嵌入服务接口（generate_embeddings 接口不变）",
    )


@log_execution(component="bge-m3")
@safe_execute(component="bge-m3", reraise=True)
def generate_embeddings(texts):
    if not isinstance(texts, list) or len(texts) == 0:
        raise ParameterError(
            message="参数 texts 必须是包含文本的非空列表",
            detail={"received_type": type(texts).__name__},
        )

    logger.info(f"[EmbedAPI] 开始为 {len(texts)} 条文本生成混合向量嵌入 | model={EMBED_MODEL}")

    dense_out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    sparse_out: list[dict] = [None] * len(texts)  # type: ignore[list-item]

    def _run_one(idx: int) -> None:
        dv, sv = _embed_one(texts[idx])
        dense_out[idx] = dv
        sparse_out[idx] = sv

    lock = threading.Lock()
    done = [0]

    with cf.ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
        futs = [ex.submit(_run_one, i) for i in range(len(texts))]
        for f in cf.as_completed(futs):
            f.result()
            with lock:
                done[0] += 1
                if done[0] % max(1, len(texts) // 5) == 0 or done[0] == len(texts):
                    logger.debug(f"[EmbedAPI] 进度 {done[0]}/{len(texts)}")

    if any(v is None for v in dense_out):
        raise EmbeddingException(message="嵌入结果存在缺失（并发批次部分失败）")

    result = {"dense": dense_out, "sparse": sparse_out}
    logger.success(
        f"[EmbedAPI] {len(texts)} 条文本向量生成完成 | dense({len(dense_out[0])}维) + sparse(已归一)"
    )
    return result


def _embed_one(text: str) -> tuple[list[float], dict]:
    """单条：一次请求取回稠密 + 稀疏（含 429/5xx 指数退避）。"""
    payload = {
        "model": EMBED_MODEL,
        "input": [{"type": "text", "text": text}],
        "sparse_embedding": {"type": "enabled"},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_api_key()}"}

    last: Exception | None = None
    for attempt in range(1, EMBED_MAX_RETRY + 1):
        t0 = time.time()
        try:
            r = httpx.post(_url, headers=headers, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json().get("data")
                item = data if isinstance(data, dict) else (data or [{}])[0]
                dv = item.get("embedding")
                if dv is None:
                    raise EmbeddingException(f"响应缺 embedding: {str(item)[:120]}")
                raw = {}
                for pair in (item.get("sparse_embedding") or []):
                    raw[int(pair["index"])] = float(pair["value"])
                logger.debug(f"[EmbedAPI] 完成 | {time.time() - t0:.2f}s")
                return [float(x) for x in dv], normalize_sparse_vector(raw)
            body = r.text[:300]
            if r.status_code in (429, 500, 502, 503, 504):
                delay = EMBED_BACKOFF * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                logger.warning(f"[EmbedAPI] {r.status_code} 退避 {delay:.1f}s (attempt {attempt})")
                time.sleep(delay)
                last = EmbeddingException(f"HTTP {r.status_code}: {body}")
                continue
            raise EmbeddingException(f"嵌入调用失败 HTTP {r.status_code}: {body}")
        except httpx.HTTPError as e:
            delay = EMBED_BACKOFF * (2 ** (attempt - 1))
            logger.warning(f"[EmbedAPI] 网络异常退避 {delay:.1f}s | {type(e).__name__}")
            time.sleep(delay)
            last = e
    raise EmbeddingException(message=f"嵌入多次重试仍失败: {last}")


if __name__ == "__main__":
    demo = generate_embeddings(["万用表怎么测电压", "苏伯尔电磁炉故障维修"])
    print("dense dim =", len(demo["dense"][0]))
    print("sparse nnz =", len(demo["sparse"][0]))
