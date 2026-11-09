from app.core.logger import logger
from app.lm.reranker_http_utils import rerank_documents as compute_scores
from app.conf.reranker_config import reranker_config
from app.query.agent.node_base import NodeBase
from app.query.agent.state import QueryGraphState

CLIFF_ABS_GAP: float = 0.5        # 绝对分差阈值
CLIFF_REL_GAP: float = 0.25       # 相对下降比例
CLIFF_DENOM_FLOOR: float = 0.05   # ★ 相对判据分母下限（防近零爆炸：abs/cur 在 cur→0 时失真）
RELEVANCE_FLOOR: float = 0.05     # ★ 相关性地板：低于此分判为「不相干」→ 直接截断（可 0 条）
CLIFF_MAX_K: int = 10             # 成本上限（保留：相关条目多时按分数择优挤掉较不相关的）
_RERANK_MAX_DOCS: int = int(__import__('os').getenv('RERANK_MAX_DOCS', '0'))


class NodeRerank(NodeBase):
    name = "重排序"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query", "")
        rrf_docs = state.get("rrf_chunks") or []
        web_docs = state.get("web_search_docs") or []

        if not query:
            logger.warning(f"【{self.name}】rewritten_query 为空，跳过")
            state["reranked_docs"] = []
            return state

        merged = self._merge_docs(rrf_docs, web_docs)
        if not merged:
            logger.info(f"【{self.name}】无合并文档，跳过")
            state["reranked_docs"] = []
            return state

        if _RERANK_MAX_DOCS > 0 and len(merged) > _RERANK_MAX_DOCS:
            logger.info(
                f"【{self.name}】rerank 输入截断 {len(merged)} → {_RERANK_MAX_DOCS}"
                f"（RERANK_MAX_DOCS 限制）"
            )
            merged = merged[:_RERANK_MAX_DOCS]

        if not reranker_config.rerank_enabled:
            limit = min(CLIFF_MAX_K, len(merged))
            state["reranked_docs"] = merged[:limit]
            logger.info(
                f"【{self.name}】重排已禁用（RERANK_ENABLED=0）| 合并={len(merged)}"
                f" → 直接取 RRF 序 Top-{limit}（跳过断崖截断）"
            )
            return state

        scored = self._rerank(query, merged)

        cutoff = self._cliff_cutoff(scored)

        if not cutoff:
            state["reranked_docs"] = []
            state["answer"] = (
                "知识库中未找到与您问题相关的内容。"
                "建议更换关键词，或提供更具体的产品名称后重试。"
            )
            logger.info(f"【{self.name}】相关性地板过滤后 0 条 → 未找到兜底（跳过生成）")
            return state

        logger.info(
            f"【{self.name}】重排序完成"
            f" | 合并={len(merged)} | 含分={len(scored)} | 截断={len(cutoff)}"
        )
        state["reranked_docs"] = cutoff
        return state


    @staticmethod
    def _merge_docs(
        rrf_docs: list[dict], web_docs: list[dict],
    ) -> list[dict]:
        seen_ids: set[str] = set()
        seen_fingerprints: set[str] = set()
        merged: list[dict] = []

        for doc in rrf_docs:
            if not isinstance(doc, dict):
                continue
            cid = doc.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)

            content = doc.get("content", "")
            fp = _fingerprint(content)
            if fp:
                seen_fingerprints.add(fp)

            merged.append({
                "content": content,
                "title": doc.get("title", ""),
                "file_title": doc.get("file_title", ""),   # ★ 保留所属文档名（引用回查/评测用）
                "chunk_id": cid,
                "url": "",
                "source": "local",
            })

        for doc in web_docs:
            if not isinstance(doc, dict):
                continue
            snip = doc.get("snippet", "")
            if not snip:
                continue

            fp = _fingerprint(snip)
            if fp and fp in seen_fingerprints:
                continue

            merged.append({
                "content": snip,
                "title": doc.get("title", ""),
                "file_title": "",                          # 网络结果无所属文档
                "chunk_id": None,
                "url": doc.get("url", ""),
                "source": "web",
            })

        return merged


    @staticmethod
    def _rerank(query: str, docs: list[dict]) -> list[dict]:
        """用 qwen3-rerank 云端 API 对每条文档计算相关性分数。"""
        contents = [doc["content"] for doc in docs]
        try:
            scores = compute_scores(query, contents)
        except Exception as e:
            logger.warning(f"【NodeRerank】打分失败 | {type(e).__name__}: {e}")
            return [{**doc, "score": 0.0} for doc in docs]

        scored = [
            {**doc, "score": round(float(s), 4)}
            for doc, s in zip(docs, scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored


    @staticmethod
    def _cliff_cutoff(ranked: list[dict]) -> list[dict]:
        if not ranked:
            return []

        limit = min(CLIFF_MAX_K, len(ranked))
        for i in range(limit):
            score = ranked[i].get("score")
            if score is None or float(score) < RELEVANCE_FLOOR:
                limit = i
                logger.info(
                    f"【NodeRerank】相关性地板 | i={i} score={float(score or 0):.4f}"
                    f" < {RELEVANCE_FLOOR} → 截断至 {i} 条"
                )
                break

        cutoff = limit
        for i in range(limit - 1):
            cur = ranked[i].get("score")
            nxt = ranked[i + 1].get("score")
            if cur is None or nxt is None:
                continue
            cur_f, nxt_f = float(cur), float(nxt)
            abs_gap = cur_f - nxt_f
            rel_gap = abs_gap / max(abs(cur_f), CLIFF_DENOM_FLOOR)  # ★ 分母带地板
            if abs_gap >= CLIFF_ABS_GAP or rel_gap >= CLIFF_REL_GAP:
                cutoff = i + 1
                logger.info(
                    f"【NodeRerank】断崖检测 i={i + 1}"
                    f" | abs={abs_gap:.4f} rel={rel_gap:.4f}"
                )
                break

        return ranked[:cutoff]



def _fingerprint(text: str) -> str:
    """取文本前 80 字标准化后作为轻量指纹，用于跨源去重。"""
    t = " ".join(text.split())[:80].lower()
    return t if t else ""

