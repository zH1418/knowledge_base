from app.core.exceptions import InvalidStateError
from app.core.logger import logger
from app.import_process.agent.nodes.vector.node_import_milvus import NodeImportMilvus
from app.import_process.agent.state import create_default_state


def check(desc: str, condition: bool) -> str:
    return "  PASS" if condition else "  FAIL"


def run_test(name: str, test_fn) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        test_fn()
    except Exception as e:
        print(f"  异常: {type(e).__name__}: {e}")



def test_validate_normal():
    """正常 input → 返回 chunks + dimension。"""
    node = NodeImportMilvus()
    chunks = [{
        "content": "正文", "title": "# 标题",
        "dense_vector": [0.1] * 1024, "sparse_vector": {0: 0.5},
    }]
    state = create_default_state(task_id="test", embeddings_content=chunks)
    result_chunks, dim = node._validate_inputs(state)
    print(check("chunks 返回 1 个", len(result_chunks) == 1))
    print(check("维度 = 1024", dim == 1024))


def test_validate_empty():
    """空 embeddings_content → 抛出 InvalidStateError。"""
    node = NodeImportMilvus()
    state = create_default_state(task_id="test", embeddings_content=[])
    try:
        node._validate_inputs(state)
        print(check("抛出 InvalidStateError", False))
    except InvalidStateError:
        print(check("抛出 InvalidStateError", True))


def test_validate_missing_dense():
    """缺失 dense_vector → 抛出 InvalidStateError。"""
    node = NodeImportMilvus()
    chunks = [{"content": "正文", "sparse_vector": {0: 0.5}}]
    state = create_default_state(task_id="test", embeddings_content=chunks)
    try:
        node._validate_inputs(state)
        print(check("抛出 InvalidStateError", False))
    except InvalidStateError:
        print(check("抛出 InvalidStateError", True))


def test_validate_missing_sparse():
    """缺失 sparse_vector → 抛出 InvalidStateError。"""
    node = NodeImportMilvus()
    chunks = [{"content": "正文", "dense_vector": [0.1] * 1024}]
    state = create_default_state(task_id="test", embeddings_content=chunks)
    try:
        node._validate_inputs(state)
        print(check("抛出 InvalidStateError", False))
    except InvalidStateError:
        print(check("抛出 InvalidStateError", True))



def test_pk_construction():
    """Mock Milvus upsert 验证 pk 拼接规则。"""
    from unittest.mock import patch, MagicMock

    mock_client = MagicMock()
    mock_client.upsert.return_value = None
    node = NodeImportMilvus()

    chunks = [
        {"content": "正文1", "title": "# 产品概述", "part": 0, "file_title": "手册", "dense_vector": [], "sparse_vector": {}, "item_name": "test"},
        {"content": "正文2", "title": "## 规格", "file_title": "手册", "dense_vector": [], "sparse_vector": {}, "item_name": "test"},  # 无 part，默认 0
    ]

    with patch.object(node, "_ensure_collection", return_value=(mock_client, False)):
        with patch.object(node, "process", wraps=node.process) as wrapped:
            result = node._upsert_chunks(mock_client, chunks)

    pk0 = result[0]["pk"]
    pk1 = result[1]["pk"]
    print(check("pk0 = 手册::产品概述::0", pk0 == "手册::# 产品概述::0"))
    print(check("pk1 = 手册::规格::0", pk1 == "手册::## 规格::0"))
    print(check("pk0 ≠ pk1（不同切片）", pk0 != pk1))
    print(check("part 默认 0 回填", result[1]["part"] == 0))
    print(check("原始 chunks 不污染", "pk" not in chunks[0]))


def test_pk_different_docs():
    """不同文档的同名标题 → pk 不同。"""
    node = NodeImportMilvus()
    a = {"title": "# 概述", "part": 0, "file_title": "A手册"}
    b = {"title": "# 概述", "part": 0, "file_title": "B手册"}

    pk_a = f"{a['file_title']}::{a['title']}::{a['part']}"
    pk_b = f"{b['file_title']}::{b['title']}::{b['part']}"
    print(check("标题相同但文档不同 → pk 不同", pk_a != pk_b))


def test_pk_parts():
    """长切片拆分的 part 区分同一标题的不同片段。"""
    node = NodeImportMilvus()
    pk0 = "手册::# 长章节::0"
    pk1 = "手册::# 长章节::1"
    print(check("part=0 和 part=1 → pk 不同", pk0 != pk1))



if __name__ == "__main__":
    logger.info("===== NodeImportMilvus 单元测试开始 =====")

    run_test("1.1 正常输入校验", test_validate_normal)
    run_test("1.2 空输入", test_validate_empty)
    run_test("1.3 缺 dense_vector", test_validate_missing_dense)
    run_test("1.4 缺 sparse_vector", test_validate_missing_sparse)
    run_test("2.1 pk 拼接规则", test_pk_construction)
    run_test("2.2 不同文档的 pk", test_pk_different_docs)
    run_test("2.3 part 区分", test_pk_parts)

    logger.info("===== NodeImportMilvus 单元测试结束 =====")
