import concurrent.futures as cf

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.query.agent.node_base import NodeBase
from app.query.agent.state import QueryGraphState


class NodeSearchEmbedding(NodeBase):
    name = "向量混合检索"

    _EMB_LIMIT = 50           # 每路候选池条数
    _DENSE_WEIGHT = 0.8       # 稠密向量权重（语义主导）
    _SPARSE_WEIGHT = 0.2      # 稀疏向量权重（关键词纠偏）
    _FILTER_MIN_HITS = 10
    _TITLE_MIN_CHARS = 50
    _TITLE_EXPAND_K = 2      # 每次标题命中，回查该文档的正文块条数

    def process(self, state: QueryGraphState) -> QueryGraphState:
        if state.get("query_type") == "unknown":
            logger.info(f"【{self.name}】unknown 模式 → 静默跳过")
            return {}

        query = state.get("rewritten_query", "")
        original = state.get("original_query", "")
        items = state.get("item_names", [])
        hyde_doc = state.get("hyde_doc", "")

        if not query:
            logger.warning(f"【{self.name}】rewritten_query 为空")
            return {}

        jobs: list[tuple[str, str, str]] = [("base", query, "改写句")]
        if hyde_doc:
            jobs.append(("hyde", query + " " + hyde_doc, "HyDE"))
        if original and original.strip() and original.strip() != query.strip():
            jobs.append(("original", original, "原query保底"))

        results = self._search_concurrent(jobs, items)
        base_chunks = results.get("base", [])
        hyde_chunks = results.get("hyde", [])
        original_chunks = results.get("original", [])

        for key, tag in (("base", "基础检索"), ("hyde", "HyDE 检索"), ("original", "原 query 保底路")):
            if key in results:
                logger.info(f"【{self.name}】{tag} | {len(results[key])} 条")

        return {
            "embedding_chunks": base_chunks,
            "hyde_embedding_chunks": hyde_chunks,
            "original_embedding_chunks": original_chunks,
        }

    def _search_concurrent(
        self, jobs: list[tuple[str, str, str]], items: list[str],
    ) -> dict[str, list[dict]]:
        """并发执行多路检索；单路失败只影响该路（返回空列表）。"""
        out: dict[str, list[dict]] = {}
        if len(jobs) == 1:
            key, text, tag = jobs[0]
            out[key] = self._do_search(text, items, tag=tag)
            return out
        with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = {
                ex.submit(self._do_search, text, items, tag=tag): key
                for key, text, tag in jobs
            }
            for fut in cf.as_completed(futures):
                key = futures[fut]
                try:
                    out[key] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"【{self.name}】{key} 并发检索失败 | {type(e).__name__}: {e}")
                    out[key] = []
        return out


    def _do_search(self, text: str, item_names: list[str], *, tag: str = "") -> list[dict]:
        """单次混合检索：BGE 嵌入 → Milvus hybrid_search（含 Q3 软过滤回退）。"""
        if not text or not text.strip():
            return []

        try:
            embeddings = generate_embeddings([text])
            dense = embeddings["dense"][0]
            sparse = embeddings["sparse"][0]

            expr = self._build_expr(item_names)

            chunks = self._hybrid(dense, sparse, expr)

            chunks = self._expand_title_hits(chunks, dense, sparse, tag=tag)

            if expr and len(chunks) < self._FILTER_MIN_HITS:
                logger.info(
                    f"【{self.name}】{tag} 过滤命中 {len(chunks)} < {self._FILTER_MIN_HITS}"
                    f" → 去 filter 全局召回（软过滤回退）"
                )
                seen = {c.get("chunk_id") for c in chunks}
                for c in self._hybrid(dense, sparse, None):
                    cid = c.get("chunk_id")
                    if cid in seen:
                        continue
                    seen.add(cid)
                    chunks.append(c)

            return chunks
        except Exception as e:
            logger.warning(f"【{self.name}】{tag} 检索失败 | {type(e).__name__}: {e}")
            return []

    def _expand_title_hits(self, chunks: list[dict], dense, sparse, *, tag: str = "") -> list[dict]:
        if not chunks:
            return chunks

        out: list[dict] = []
        seen: set = set()
        expanded_docs: set[str] = set()
        n_title_hits = 0

        for c in chunks:
            content = c.get("content") or ""
            if len(content) >= self._TITLE_MIN_CHARS:
                cid = c.get("chunk_id")
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(c)
                continue

            n_title_hits += 1
            ft = (c.get("file_title") or "").strip()
            if not ft or ft in expanded_docs:
                continue
            expanded_docs.add(ft)
            expr = f'file_title == "{ft}" and part != 0'
            for h in self._hybrid(dense, sparse, expr)[: self._TITLE_EXPAND_K]:
                cid = h.get("chunk_id")
                if cid in seen or len(h.get("content") or "") < self._TITLE_MIN_CHARS:
                    continue
                seen.add(cid)
                out.append(h)

        if n_title_hits:
            logger.info(
                f"【{self.name}】{tag} 标题即入口 | 标题命中 {n_title_hits} 条"
                f" → 展开文档 {len(expanded_docs)} 篇 | 正文块 {len(out)} 条"
            )
        return out

    def _hybrid(self, dense, sparse, expr) -> list[dict]:
        """单次 Milvus 混合检索（候选池 = _EMB_LIMIT），返回标准化 chunk 列表。"""
        reqs = create_hybrid_search_requests(
            dense_vector=dense,
            sparse_vector=sparse,
            expr=expr,
            limit=self._EMB_LIMIT,
        )
        client = get_milvus_client()
        res = hybrid_search(
            client=client,
            collection_name=milvus_config.chunks_collection,
            reqs=reqs,
            ranker_weights=(self._DENSE_WEIGHT, self._SPARSE_WEIGHT),
            norm_score=True,
            limit=self._EMB_LIMIT,
            output_fields=["chunk_id", "content", "title", "file_title", "item_name"],
        )
        return self._parse_results(res)


    @staticmethod
    def _build_expr(item_names: list[str]) -> str | None:
        """构建 item_name in [...] 过滤表达式。"""
        if not item_names:
            return None
        quoted = ", ".join(f'"{name}"' for name in item_names)
        return f"item_name in [{quoted}]"


    @staticmethod
    def _parse_results(res) -> list[dict]:
        """将 Milvus 返回值标准化为下游可消费的 chunk 列表。"""
        if not res or not res[0]:
            return []

        return [
            {
                "chunk_id": str(hit.id),
                "content": hit.entity.get("content", ""),
                "title": hit.entity.get("title", ""),
                "file_title": hit.entity.get("file_title", ""),
                "item_name": hit.entity.get("item_name", ""),
                "score": round(float(hit.score), 4) if hasattr(hit, "score") else 0.0,
                "source": "local",
            }
            for hit in res[0]
        ]
