import os
import threading
import time
from typing import List

from app.conf.reranker_config import reranker_config
from app.core.logger import logger
from app.core.decorators import log_execution, safe_execute


@log_execution(component="reranker_http")
@safe_execute(component="reranker_http", default_return=[], reraise=False)
def rerank_documents(
    query: str,
    documents: List[str],
    top_n: int = 10,
) -> List[float]:
    if not documents:
        return []

    provider = (reranker_config.rerank_provider or "dashscope").lower()
    if provider == "voyage":
        return _rerank_voyage(query, documents, top_n)
    if provider == "siliconflow":
        return _rerank_siliconflow(query, documents, top_n)
    if provider == "local_kalm":
        return _rerank_local_kalm(query, documents, top_n)
    return _rerank_dashscope(query, documents, top_n)



def _rerank_dashscope(query: str, documents: List[str], top_n: int) -> List[float]:
    api_key = reranker_config.dashscope_api_key
    if not api_key:
        logger.warning("[RerankerHTTP] dashscope_api_key 未配置")
        return [0.0] * len(documents)

    import dashscope
    dashscope.api_key = api_key
    from dashscope import TextReRank

    model = reranker_config.text_rerank_model
    logger.info(
        f"[RerankerHTTP:dashscope] 开始重排序 | model={model}"
        f" | docs={len(documents)} | top_n={top_n}"
    )

    result = TextReRank.call(
        model=model,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        return_documents=False,
        instruction=reranker_config.text_rerank_instruct,
    )

    if result.status_code != 200:
        logger.error(
            f"[RerankerHTTP:dashscope] API 调用失败 | status={result.status_code}"
            f" | message={result.message}"
        )
        return [0.0] * len(documents)

    scores = [0.0] * len(documents)
    for item in result.output.get("results", []):
        scores[item["index"]] = float(item["relevance_score"])

    logger.info(
        f"[RerankerHTTP:dashscope] 完成 | 分数范围=[{min(scores):.4f}, {max(scores):.4f}]"
    )
    return scores



_VOYAGE_URL = "https://api.voyageai.com/v1/rerank"

_MIN_INTERVAL: float = float(os.getenv("VOYAGE_MIN_INTERVAL_SEC", "21"))
_MAX_RETRY: int = int(os.getenv("VOYAGE_MAX_RETRY", "4"))
_rate_lock = threading.Lock()
_last_call_ts: float = 0.0


def _pace() -> None:
    """全局限速：保证两次调用间隔 >= _MIN_INTERVAL（防 429）。"""
    global _last_call_ts
    with _rate_lock:
        wait = _MIN_INTERVAL - (time.time() - _last_call_ts)
        if wait > 0:
            logger.info(f"[RerankerHTTP:voyage] 限速等待 {wait:.1f}s（3 RPM 免费档）")
            time.sleep(wait)
        _last_call_ts = time.time()


def _rerank_voyage(query: str, documents: List[str], top_n: int) -> List[float]:
    api_key = reranker_config.voyage_api_key
    if not api_key:
        logger.warning("[RerankerHTTP] voyage_api_key 未配置")
        return [0.0] * len(documents)

    import httpx

    model = reranker_config.voyage_rerank_model
    logger.info(
        f"[RerankerHTTP:voyage] 开始重排序 | model={model}"
        f" | docs={len(documents)} | top_n={top_n}"
    )

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_k": min(top_n, len(documents)),
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err = ""
    for attempt in range(1, _MAX_RETRY + 1):
        _pace()
        try:
            r = httpx.post(_VOYAGE_URL, headers=headers, json=payload, timeout=90.0)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            logger.warning(f"[RerankerHTTP:voyage] 请求异常（{attempt}/{_MAX_RETRY}）| {last_err}")
            time.sleep(5.0 * attempt)
            continue

        if r.status_code == 200:
            data = r.json().get("data") or []
            scores = [0.0] * len(documents)
            for item in data:
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(documents):
                    scores[idx] = float(item.get("relevance_score", 0.0))
            logger.info(
                f"[RerankerHTTP:voyage] 完成 | tokens={r.json().get('usage', {}).get('total_tokens', '?')}"
                f" | 分数范围=[{min(scores):.4f}, {max(scores):.4f}]"
            )
            return scores

        last_err = f"status={r.status_code} body={r.text[:150]}"
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("Retry-After") or 0) or (25.0 * attempt)
            logger.warning(
                f"[RerankerHTTP:voyage] {r.status_code} 退避 {wait:.0f}s"
                f"（{attempt}/{_MAX_RETRY}）"
            )
            time.sleep(wait)
            continue
        logger.error(f"[RerankerHTTP:voyage] API 调用失败（不可重试）| {last_err}")
        return [0.0] * len(documents)

    logger.error(f"[RerankerHTTP:voyage] 重试耗尽，降级全零 | {last_err}")
    return [0.0] * len(documents)



_SILICONFLOW_URL = "https://api.siliconflow.cn/v1/rerank"


def _rerank_siliconflow(query: str, documents: List[str], top_n: int) -> List[float]:
    api_key = reranker_config.siliconflow_api_key
    if not api_key:
        logger.warning("[RerankerHTTP] siliconflow_api_key 未配置")
        return [0.0] * len(documents)

    import httpx

    model = reranker_config.siliconflow_rerank_model
    logger.info(
        f"[RerankerHTTP:siliconflow] 开始重排序 | model={model}"
        f" | docs={len(documents)} | top_n={top_n}"
    )

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = httpx.post(_SILICONFLOW_URL, headers=headers, json=payload, timeout=60.0)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[RerankerHTTP:siliconflow] 请求异常 | {type(e).__name__}: {e}")
        return [0.0] * len(documents)

    if r.status_code != 200:
        logger.error(
            f"[RerankerHTTP:siliconflow] API 调用失败 | status={r.status_code}"
            f" | body={r.text[:200]}"
        )
        return [0.0] * len(documents)

    scores = [0.0] * len(documents)
    for item in (r.json().get("results") or []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(documents):
            scores[idx] = float(item.get("relevance_score", 0.0))

    logger.info(
        f"[RerankerHTTP:siliconflow] 完成"
        f" | 分数范围=[{min(scores):.4f}, {max(scores):.4f}]"
    )
    return scores



def _rerank_local_kalm(query: str, documents: List[str], top_n: int) -> List[float]:
    import json
    import subprocess
    import tempfile

    exe = reranker_config.local_rerank_exe
    model = reranker_config.local_rerank_model
    if not exe or not model:
        logger.warning("[RerankerHTTP] local_kalm 未配置 exe/model 路径")
        return [0.0] * len(documents)

    logger.info(
        f"[RerankerHTTP:local_kalm] 开始重排序 | model={model.split(chr(92))[-1]}"
        f" | docs={len(documents)} | top_n={top_n}"
    )

    tmpdir = tempfile.mkdtemp(prefix="kalm_rerank_")
    in_path = os.path.join(tmpdir, "in.jsonl")
    out_path = os.path.join(tmpdir, "out.jsonl")
    try:
        max_chars = reranker_config.local_rerank_max_chars
        with open(in_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(documents):
                passage = doc[:max_chars] if max_chars > 0 else doc
                f.write(json.dumps(
                    {"qid": "q", "pid": str(i), "query": query, "passage": passage},
                    ensure_ascii=False) + "\n")

        r = subprocess.run(
            [exe, "-m", model, "-ngl", "0", "-t", str(reranker_config.local_rerank_threads),
             "--jsonl", in_path, "--out", out_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=900,
        )
        if r.returncode != 0:
            logger.error(
                f"[RerankerHTTP:local_kalm] 子进程失败 | rc={hex(r.returncode & 0xFFFFFFFF)}"
                f" | stderr={(r.stderr or '')[-200:]}"
            )
            return [0.0] * len(documents)

        scores = [0.0] * len(documents)
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                idx = int(item.get("pid", -1))
                if 0 <= idx < len(documents):
                    scores[idx] = float(item.get("score", 0.0))

        logger.info(
            f"[RerankerHTTP:local_kalm] 完成 | 分数范围=[{min(scores):.4f}, {max(scores):.4f}]"
        )
        return scores
    except Exception as e:  # noqa: BLE001
        logger.error(f"[RerankerHTTP:local_kalm] 异常 | {type(e).__name__}: {e}")
        return [0.0] * len(documents)
    finally:
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
