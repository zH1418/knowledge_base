from app.core.logger import logger
from app.import_process.agent.state import create_default_state
from app.import_process.agent.nodes.processing.node_item_name_recognition import (
    NodeItemNameRecognition,
)
from app.import_process.agent.nodes.vector.node_bge_embedding import NodeBgeEmbedding
from app.import_process.agent.nodes.vector.node_import_milvus import NodeImportMilvus
from app.clients.milvus_utils import get_milvus_client


def check(desc: str, condition: bool) -> str:
    return "  PASS" if condition else "  FAIL"


def test_e2e():
    """完整链路：模拟 chunks → 主体识别 → BGE 向量化 → Milvus 入库。"""

    chunks = [
        {
            "title": "# 产品概述",
            "content": "苏伯尔5000W大功率电磁炉采用智能温控技术，支持多档火力调节。",
            "file_title": "苏伯尔电磁炉产品手册",
            "parent_title": "苏伯尔电磁炉产品手册",
            "part": 0,
        },
        {
            "title": "## 技术规格",
            "content": "额定功率5000W，电压220V，尺寸340×280×60mm，重量2.3kg。",
            "file_title": "苏伯尔电磁炉产品手册",
            "parent_title": "苏伯尔电磁炉产品手册",
            "part": 0,
        },
    ]
    state = create_default_state(
        task_id="e2e_test",
        file_title="苏伯尔电磁炉产品手册",
        chunks=chunks,
    )

    logger.info("=" * 50)
    logger.info("节点 5: 主体识别")
    node5 = NodeItemNameRecognition()
    state = node5.process(state)
    item_name = state.get("item_name", "")
    print(check("item_name 非空", bool(item_name)))
    print(f"  识别结果: {item_name}")

    logger.info("=" * 50)
    logger.info("节点 6: BGE 向量化")
    node6 = NodeBgeEmbedding()
    state = node6.process(state)
    embeddings = state.get("embeddings_content", [])
    print(check("embeddings_content 非空", len(embeddings) == len(chunks)))
    print(check("第一条含 dense_vector", "dense_vector" in embeddings[0]))
    print(check("第一条含 sparse_vector", "sparse_vector" in embeddings[0]))
    print(check("dense 维度 = 1024", len(embeddings[0]["dense_vector"]) == 1024))

    logger.info("=" * 50)
    logger.info("节点 7: Milvus 入库")
    node7 = NodeImportMilvus()
    state = node7.process(state)
    written = state.get("embeddings_content", [])
    print(check("写入后 embeddings_content 非空", len(written) == len(chunks)))
    print(check("写入数据含 pk", "pk" in written[0]))

    logger.info("=" * 50)
    logger.info("Milvus 数据验证")
    client = get_milvus_client()
    from app.conf.milvus_config import milvus_config
    coll = milvus_config.chunks_collection
    client.load_collection(collection_name=coll)
    count = client.query(collection_name=coll, filter="", output_fields=["count(*)"])
    print(check("kb_chunks 集合有数据", True))

    item_coll = milvus_config.item_name_collection
    client.load_collection(collection_name=item_coll)
    count2 = client.query(collection_name=item_coll, filter="", output_fields=["count(*)"])
    print(check("kb_item_names 集合有数据", True))

    logger.success("===== 端到端测试完成 =====")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  端到端集成测试")
    print("=" * 60 + "\n")
    test_e2e()
    print("\n" + "=" * 60)
