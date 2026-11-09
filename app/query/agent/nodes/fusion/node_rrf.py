from app.core.logger import logger
from app.query.agent.node_base import NodeBase
from app.query.agent.state import QueryGraphState


class NodeRRF(NodeBase):
    name = "倒数排序融合"

    _K: int = 60            # RRF 平滑常数
    _MAX_RESULTS: int = 50  # 最大返回条数
    _WEIGHT: float = 1.0    # 各路权重（等权）

    def process(self, state: QueryGraphState) -> QueryGraphState:
        embedding = state.get("embedding_chunks") or []
        hyde = state.get("hyde_embedding_chunks") or []
        original = state.get("original_embedding_chunks") or []

        if not embedding and not hyde and not original:
            logger.info(f"【{self.name}】无检索结果，跳过")
            state["rrf_chunks"] = []
            return state

        rrf_inputs = [
            (embedding, self._WEIGHT),
            (hyde, self._WEIGHT),
            (original, self._WEIGHT),
        ]

        scored = self._rrf_merge(rrf_inputs)

        top = scored[:self._MAX_RESULTS]
        rrf_chunks = [doc for doc, _ in top]

        logger.info(
            f"【{self.name}】融合完成 | 输入={len(embedding)}+{len(hyde)}+{len(original)}"
            f" | 去重={len(scored)} | 输出={len(rrf_chunks)}"
        )
        state["rrf_chunks"] = rrf_chunks
        return state


    def _rrf_merge(
        self, rrf_inputs: list[tuple[list[dict], float]]
    ) -> list[tuple[dict, float]]:
        chunk_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for docs, weight in rrf_inputs:
            if not docs:
                continue
            for rank, doc in enumerate(docs, start=1):
                cid = doc.get("chunk_id", "")
                if not cid:
                    continue
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + weight / (self._K + rank)
                chunk_data.setdefault(cid, doc)

        items = [(chunk_data[cid], chunk_scores[cid]) for cid in chunk_scores]
        items.sort(key=lambda x: x[1], reverse=True)
        return items
