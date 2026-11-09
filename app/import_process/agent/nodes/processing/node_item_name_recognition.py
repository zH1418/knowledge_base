from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.embedding_config import embedding_config
from app.conf.milvus_config import milvus_config
from app.core.exceptions import InvalidStateError, ParameterError
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client


class NodeItemNameRecognition(NodeBase):

    name = "商品名称识别"

    _CONTEXT_CHUNK_K: int = 5       # LLM 上下文切片数
    _CONTEXT_MAX_CHARS: int = 2500  # 上下文字符上限


    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title, chunks = self._load_inputs(state)

        context = self._build_context(chunks)

        item_name = self._call_llm(file_title, context)

        self._update_chunks(state, chunks, item_name)

        dense_vector, sparse_vector = self._generate_vectors(item_name)

        self._save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        logger.success(f"主体识别完成: {item_name}")
        return state


    @staticmethod
    def _load_inputs(
        state: ImportGraphState
    ) -> tuple[str, list[dict]]:
        file_title = state.get("file_title", "").strip()
        if not file_title:
            raise ParameterError("核心参数 file_title 缺失")

        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise InvalidStateError("核心参数 chunks 为空或非列表类型")

        logger.info(f"输入校验完成，获取到 {len(chunks)} 个有效切片")
        return file_title, chunks


    def _build_context(self, chunks: list[dict]) -> str:
        """取前 K 个切片拼接为结构化上下文，双重限流（数量+字符）。"""
        parts: list[str] = []
        total_chars = 0

        for idx, chunk in enumerate(chunks[:self._CONTEXT_CHUNK_K], start=1):
            title = chunk.get("title", "").strip()
            content = chunk.get("content", "").strip()
            piece = f"【切片{idx}】\n标题：{title}\n内容：{content}"
            parts.append(piece)

            total_chars += len(piece)
            if total_chars > self._CONTEXT_MAX_CHARS:
                logger.info(f"上下文已超过 {self._CONTEXT_MAX_CHARS} 字符，停止拼接")
                break

        context = "\n\n".join(parts).strip()
        final_context = context[:self._CONTEXT_MAX_CHARS]
        logger.info(f"上下文构建完成，最终长度 {len(final_context)} 字符")
        return final_context


    @staticmethod
    def _call_llm(file_title: str, context: str) -> str:
        """调用 LLM 识别商品名称，异常时返回 file_title 兜底。"""
        if not context:
            logger.warning("上下文为空，使用 file_title 作为兜底")
            return file_title

        try:
            system_prompt = load_prompt("product_recognition_system")
            human_prompt = load_prompt(
                "item_name_recognition",
                file_title=file_title,
                context=context,
            )
            llm = get_llm_client(json_mode=False)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]
            resp = llm.invoke(messages)

            item_name = ""
            for attr in ("content", "text", "message", "response"):
                val = getattr(resp, attr, None)
                if val:
                    item_name = str(val).strip()
                    break

            item_name = item_name.replace(" ", "").replace("\n", "").replace("\r", "")
            for prefix in ("商品名称：", "产品名称：", "商品名：", "商品:", "产品:"):
                if item_name.startswith(prefix):
                    item_name = item_name[len(prefix):]

            if not item_name:
                logger.warning("LLM 返回空结果，使用 file_title 兜底")
                return file_title

            logger.info(f"LLM 识别商品名称: {item_name}")
            return item_name

        except Exception as e:
            logger.exception(f"LLM 调用失败: {e}")
            return file_title


    @staticmethod
    def _update_chunks(
        state: ImportGraphState,
        chunks: list[dict],
        item_name: str,
    ) -> None:
        """将 item_name 写入 state 和所有 chunk。"""
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["chunks"] = chunks
        logger.info(f"item_name 已回填到 state 和 {len(chunks)} 个切片")


    @staticmethod
    def _generate_vectors(item_name: str) -> tuple[list, dict]:
        """为商品名称生成 BGE-M3 稠密+稀疏向量。"""
        if not item_name:
            raise InvalidStateError("商品名称为空，无法生成向量")

        result = generate_embeddings([item_name])
        if not result or "dense" not in result or "sparse" not in result:
            raise InvalidStateError(f"向量生成返回空结果: {item_name}")

        dense_vector = result["dense"][0]
        sparse_vector = result["sparse"][0]
        logger.info(f"向量生成完成，稠密维度={len(dense_vector)}")
        return dense_vector, sparse_vector


    def _save_to_milvus(
        self,
        state: ImportGraphState,
        file_title: str,
        item_name: str,
        dense_vector: list,
        sparse_vector: dict,
    ) -> None:
        """将商品名称和向量持久化到 Milvus kb_item_names 集合。"""
        collection_name = milvus_config.item_name_collection
        if not collection_name:
            logger.warning("Milvus 配置缺失 ITEM_NAME_COLLECTION，跳过入库")
            return

        client = get_milvus_client()
        if not client:
            logger.error("无法获取 Milvus 客户端，跳过入库")
            return

        logger.info(f"准备将 [{item_name}] 存入 Milvus 集合 [{collection_name}]")

        try:
            is_new = self._ensure_collection(client, collection_name)

            pk = f"{file_title}::{item_name}"
            data = {
                "pk": pk,
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
            client.upsert(collection_name=collection_name, data=[data])
            logger.debug(f"upsert 完成, pk={pk}")

            if is_new:
                client.load_collection(collection_name=collection_name)
                logger.info(f"新集合 [{collection_name}] 已加载到内存")

            state["item_name"] = item_name
            logger.success(f"[{item_name}] 已存入 Milvus 集合 [{collection_name}]")

        except Exception as e:
            logger.warning(f"Milvus 入库失败: {e}", exc_info=True)

    @staticmethod
    def _ensure_collection(client, collection_name: str) -> bool:
        """确保集合存在，不存在则创建并返回 True（表示新创建）。"""
        if client.has_collection(collection_name=collection_name):
            return False

        logger.info(f"创建 Milvus 集合 [{collection_name}]")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

        schema.add_field(
            field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=512,
        )
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_config.dense_vector_dim,
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "normalize": True,
                "quantization": "none",
            },
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Milvus 集合 [{collection_name}] 创建完成")
        return True
