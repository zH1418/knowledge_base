from __future__ import annotations

from typing import Any

SOURCE_CONTENT_MAX: int = 0


def build_sources(
    docs: list[dict[str, Any]] | None, *, with_content: bool = True,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs or [], start=1):
        if not isinstance(doc, dict):
            continue
        item: dict[str, Any] = {
            "idx": idx,
            "title": doc.get("title") or "",
            "file_title": doc.get("file_title") or "",
            "url": doc.get("url") or "",
            "source": doc.get("source") or "",
            "chunk_id": doc.get("chunk_id"),
            "score": doc.get("score"),
        }
        if with_content:
            content = doc.get("content") or ""
            if SOURCE_CONTENT_MAX > 0:
                content = content[:SOURCE_CONTENT_MAX]
            item["content"] = content
        out.append(item)
    return out
