from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.embedding_config import embedding_config
from app.conf.milvus_config import milvus_config
from app.core.exceptions import InvalidStateError, ParameterError
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState


class NodeImportMilvus(NodeBase):

    name = "Milvus向量入库"

    _DEFAULT_PART: int = 0


    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks, dimension = self._validate_inputs(state)

        client, is_new = self._ensure_collection(dimension)

        written = self._upsert_chunks(client, chunks)

        state["embeddings_content"] = written
        logger.success(
            f"Milvus 入库完成，{len(written)} 个切片 → 集合 [{milvus_config.chunks_collection}]"
        )
        return state


    @staticmethod
    def _validate_inputs(state: ImportGraphState) -> tuple[list[dict], int]:
        chunks = state.get("embeddings_content", [])
        if not chunks or not isinstance(chunks, list):
            raise InvalidStateError("embeddings_content 为空或非列表类型")

        first = chunks[0]
        if "dense_vector" not in first:
            raise InvalidStateError("chunk 缺失 dense_vector 字段")
        if "sparse_vector" not in first:
            raise InvalidStateError("chunk 缺失 sparse_vector 字段")

        dimension = embedding_config.dense_vector_dim
        logger.info(f"输入校验通过，{len(chunks)} 个切片，向量维度 {dimension}")
        return chunks, dimension


    @staticmethod
    def _ensure_collection(dimension: int):
        """确保 kb_chunks 集合存在，返回 (client, is_new)。"""
        collection_name = milvus_config.chunks_collection
        if not collection_name:
            raise ParameterError("Milvus 配置缺失 chunks_collection")

        client = get_milvus_client()
        if not client:
            raise InvalidStateError("Milvus 客户端连接失败")

        if client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus 集合 [{collection_name}] 已存在，直接复用")
            return client, False

        logger.info(f"创建 Milvus 集合 [{collection_name}]，维度 {dimension}")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

        schema.add_field(
            field_name="pk", datatype=DataType.VARCHAR,
            is_primary=True, max_length=512,
        )
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="part", datatype=DataType.INT8,
        )
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR, max_length=65535,
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dimension,
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
        client.load_collection(collection_name=collection_name)
        logger.info(f"Milvus 集合 [{collection_name}] 创建并加载完成")
        return client, True


    def _upsert_chunks(self, client, chunks: list[dict]) -> list[dict]:
        """构建确定性 pk → 批量 upsert → 回填 pk 后返回副本列表。"""
        data: list[dict] = []

        for chunk in chunks:
            item = dict(chunk)  # 不污染上游
            item["part"] = item.get("part", self._DEFAULT_PART)
            title = item.get("title", "")
            item["pk"] = f"{item.get('file_title', '')}::{title}::{item['part']}"
            data.append(item)

        logger.info(f"准备 upsert {len(data)} 个切片 → 集合 [{milvus_config.chunks_collection}]")
        client.upsert(collection_name=milvus_config.chunks_collection, data=data)
        logger.info(f"upsert 完成，{len(data)} 个切片已写入")
        return data
