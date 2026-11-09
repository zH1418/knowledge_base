from app.core.exceptions import InvalidStateError, ParameterError, KBException
from app.core.logger import logger
from app.import_process.agent.nodes.vector.node_bge_embedding import NodeBgeEmbedding
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
        print(f"  异常: {e}")



def test_validate_normal():
    """正常输入 → 返回 chunks + item_name。"""
    node = NodeBgeEmbedding()
    state = create_default_state(
        task_id="test",
        item_name="测试商品",
        chunks=[{"title": "# 标题", "content": "正文内容"}],
    )
    chunks, item_name = node._validate_inputs(state)
    print(check("chunks 非空", len(chunks) == 1))
    print(check("item_name 正确", item_name == "测试商品"))


def test_validate_empty_chunks():
    """空 chunks → 抛出 InvalidStateError。"""
    node = NodeBgeEmbedding()
    state = create_default_state(task_id="test", item_name="test", chunks=[])
    try:
        node._validate_inputs(state)
        print(check("抛出 InvalidStateError", False))
    except InvalidStateError:
        print(check("抛出 InvalidStateError", True))


def test_validate_missing_item_name():
    """缺少 item_name → 抛出 ParameterError。"""
    node = NodeBgeEmbedding()
    state = create_default_state(
        task_id="test",
        chunks=[{"title": "# 标题", "content": "正文"}],
    )
    try:
        node._validate_inputs(state)
        print(check("抛出 ParameterError", False))
    except ParameterError:
        print(check("抛出 ParameterError", True))


def test_validate_empty_content():
    """chunk 中 content 为空 → 抛出 InvalidStateError。"""
    node = NodeBgeEmbedding()
    state = create_default_state(
        task_id="test",
        item_name="test",
        chunks=[{"title": "# 标题", "content": ""}],
    )
    try:
        node._validate_inputs(state)
        print(check("抛出 InvalidStateError", False))
    except InvalidStateError:
        print(check("抛出 InvalidStateError", True))



def test_encode_chunks_structure():
    """Mock generate_embeddings 验证输出结构。"""
    from unittest.mock import patch

    mock_return = lambda texts: {
        "dense": [[0.1 * (i + 1)] * 1024 for i in range(len(texts))],
        "sparse": [{i: 0.5} for i in range(len(texts))],
    }

    with patch(
        "app.import_process.agent.nodes.vector.node_bge_embedding.generate_embeddings",
        side_effect=mock_return,
    ):
        node = NodeBgeEmbedding()
        chunks = [
            {"title": "# A", "content": "内容A", "file_title": "测试"},
            {"title": "# B", "content": "内容B", "file_title": "测试"},
        ]
        result = node._encode_chunks(chunks, "苏伯尔5000W")

        print(check("输出 2 个元素", len(result) == 2))
        print(check("元素含 dense_vector", "dense_vector" in result[0]))
        print(check("元素含 sparse_vector", "sparse_vector" in result[0]))
        print(check("dense 维度 = 1024", len(result[0]["dense_vector"]) == 1024))
        print(check("sparse 为 dict", isinstance(result[0]["sparse_vector"], dict)))
        print(check("原始字段保留", result[0]["title"] == "# A"))
        print(check("原始 chunks 未被修改", "dense_vector" not in chunks[0]))


def test_encode_chunks_batch():
    """6 个 chunks → 分 2 批（5+1）处理。"""
    from unittest.mock import patch

    call_counts = []

    def mock_generate(texts):
        call_counts.append(len(texts))
        return {
            "dense": [[0.0] * 1024 for _ in texts],
            "sparse": [{0: 0.1} for _ in texts],
        }

    with patch(
        "app.import_process.agent.nodes.vector.node_bge_embedding.generate_embeddings",
        side_effect=mock_generate,
    ):
        node = NodeBgeEmbedding()
        chunks = [{"title": f"# {i}", "content": f"内容{i}"} for i in range(6)]
        result = node._encode_chunks(chunks, "测试")

        print(check("共 2 批调用", len(call_counts) == 2))
        print(check("第 1 批 5 条", call_counts[0] == 5))
        print(check("第 2 批 1 条", call_counts[1] == 1))
        print(check("输出 6 个元素", len(result) == 6))



def test_process_e2e():
    """完整 process 流程 → state["embeddings_content"] 正确填充。"""
    from unittest.mock import patch

    mock_return = lambda texts: {
        "dense": [[0.0] * 1024 for _ in range(len(texts))],
        "sparse": [{i: 0.5} for i in range(len(texts))],
    }

    with patch(
        "app.import_process.agent.nodes.vector.node_bge_embedding.generate_embeddings",
        side_effect=mock_return,
    ):
        node = NodeBgeEmbedding()
        chunks = [
            {"title": "# 产品", "content": "产品说明", "file_title": "手册"},
            {"title": "## 规格", "content": "规格参数", "file_title": "手册"},
        ]
        state = create_default_state(
            task_id="test", item_name="测试商品", chunks=chunks,
        )
        result = node.process(state)

        emb = result.get("embeddings_content", [])
        print(check("embeddings_content 有 2 个", len(emb) == 2))
        print(check("第一个含 dense_vector", "dense_vector" in emb[0]))
        print(check("第一个含 sparse_vector", "sparse_vector" in emb[0]))
        print(check("原始 chunks 不变", "dense_vector" not in chunks[0]))



if __name__ == "__main__":
    logger.info("===== NodeBgeEmbedding 单元测试开始 =====")

    run_test("1.1 正常输入校验", test_validate_normal)
    run_test("1.2 空 chunks", test_validate_empty_chunks)
    run_test("1.3 缺 item_name", test_validate_missing_item_name)
    run_test("1.4 content 为空", test_validate_empty_content)
    run_test("2.1 输出结构验证", test_encode_chunks_structure)
    run_test("2.2 批量分片验证", test_encode_chunks_batch)
    run_test("3.1 process 端到端", test_process_e2e)

    logger.info("===== NodeBgeEmbedding 单元测试结束 =====")
