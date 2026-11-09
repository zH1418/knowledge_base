from __future__ import annotations

import json
from typing import Any

from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings


def milvus_search(query: str, scope: str = "all", top_k: int = 5) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []
    try:
        from app.clients.milvus_utils import (
            get_milvus_client, create_hybrid_search_requests, hybrid_search,
        )
        from app.conf.milvus_config import milvus_config

        embeddings = generate_embeddings([query])
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        expr = "part == 0" if scope == "toc" else None

        reqs = create_hybrid_search_requests(
            dense_vector=dense, sparse_vector=sparse, expr=expr, limit=top_k,
        )
        client = get_milvus_client()
        res = hybrid_search(
            client=client, collection_name=milvus_config.chunks_collection,
            reqs=reqs, norm_score=True, limit=top_k,
            output_fields=["title", "file_title", "item_name", "content"],
        )

        hits: list[dict[str, Any]] = []
        if res and res[0]:
            for hit in res[0]:
                ent = hit.entity
                hits.append({
                    "title": ent.get("title", ""),
                    "item_name": ent.get("item_name", ""),
                    "file_title": ent.get("file_title", ""),
                    "score": round(float(hit.score), 4),
                    "snippet": (ent.get("content", "") or "")[:120],
                })
        return hits
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[MilvusSearchTool] 检索失败 | {type(e).__name__}: {e}")
        return []


def format_hits(hits: list[dict[str, Any]], label: str = "") -> str:
    """把命中列表格式化为 LLM 可读文本；空结果有明确标识。"""
    if not hits:
        return f"（{label}无命中——库中没有与该文本相关的内容）" if label else "（无命中）"
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(
            f"[{i}] score={h['score']:.3f} | 主体={h['item_name'] or '（空）'}"
            f" | 文档={h['file_title'][:40]} | 章节={h['title'][:30]}"
            f" | 摘要: {h['snippet'][:80]}"
        )
    return "\n".join(lines)



TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "milvus_search",
        "description": (
            "检索文档知识库（混合检索：语义+关键词）。"
            "用于：① 验证你提取的商品/主题名在库中是否存在（scope='toc' 查文档目录）；"
            "② 查看库内与某话题相关的实际内容，据此决定改写方向。"
            "返回命中列表（score 越高越相关，>=0.6 视为强相关）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索文本（候选主体名或检索句）"},
                "scope": {
                    "type": "string",
                    "enum": ["toc", "all"],
                    "description": "toc=仅文档目录（验证主体名用）；all=全量内容（默认）",
                },
                "top_k": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
            },
            "required": ["query"],
        },
    },
}


def run_tool(args: dict[str, Any]) -> str:
    """执行工具调用并序列化为 ToolMessage 内容。"""
    hits = milvus_search(
        query=str(args.get("query", "")),
        scope=str(args.get("scope", "all")),
        top_k=int(args.get("top_k", 5)),
    )
    return json.dumps({"hits": hits, "formatted": format_hits(hits)},
                      ensure_ascii=False)
