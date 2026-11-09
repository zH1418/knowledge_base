from app.core.exceptions import InvalidStateError, ParameterError
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings


class NodeBgeEmbedding(NodeBase):

    name = "BGE向量生成"

    _BATCH_SIZE: int = 5                # 批量大小（CPU 安全值）
    _ITEM_CONTENT_SEP: str = "\n"        # item_name 与 content 拼接分隔符


    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks, item_name = self._validate_inputs(state)

        embeddings = self._encode_chunks(chunks, item_name)

        state["embeddings_content"] = embeddings
        logger.success(f"BGE-M3 向量化完成，共处理 {len(embeddings)} 个切片")
        return state


    @staticmethod
    def _validate_inputs(state: ImportGraphState) -> tuple[list[dict], str]:
        chunks = state.get("chunks", [])
        if not chunks or not isinstance(chunks, list):
            raise InvalidStateError("chunks 为空或非列表类型，无法执行向量化")

        item_name = state.get("item_name", "").strip()
        if not item_name:
            raise ParameterError("item_name 缺失，无法拼接 chunk 编码文本")

        for i, chunk in enumerate(chunks):
            if not chunk.get("content", "").strip():
                raise InvalidStateError(f"chunk[{i}] 的 content 字段为空")

        logger.info(f"向量化输入校验通过，待处理 {len(chunks)} 个切片，item_name={item_name}")
        return chunks, item_name


    def _encode_chunks(
        self, chunks: list[dict], item_name: str
    ) -> list[dict]:
        output: list[dict] = []
        total = len(chunks)

        for batch_start in range(0, total, self._BATCH_SIZE):
            batch = chunks[batch_start:batch_start + self._BATCH_SIZE]
            start_num, end_num = batch_start + 1, min(batch_start + len(batch), total)

            input_texts = [f"{item_name}{self._ITEM_CONTENT_SEP}{c['content']}" for c in batch]

            try:
                result = generate_embeddings(input_texts)
            except Exception as e:
                logger.exception(f"第 {start_num}-{end_num} 条切片：向量生成失败")
                raise InvalidStateError(
                    f"BGE-M3 向量化失败（切片 {start_num}-{end_num}）: {e}"
                ) from e

            if not result or "dense" not in result or "sparse" not in result:
                raise InvalidStateError(
                    f"第 {start_num}-{end_num} 条切片：BGE-M3 返回空结果"
                )

            for j, chunk in enumerate(batch):
                item = dict(chunk)  # 浅拷贝，不污染上游原始 chunk
                item["dense_vector"] = result["dense"][j]
                item["sparse_vector"] = result["sparse"][j]
                output.append(item)

            logger.info(f"切片 {start_num}-{end_num}/{total}：双向量生成成功")

        return output
