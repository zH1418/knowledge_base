from app.core.exceptions import InvalidStateError, ParameterError, KBException
from app.core.logger import logger
from app.import_process.agent.nodes.processing.node_item_name_recognition import (
    NodeItemNameRecognition,
)
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



def test_load_inputs_normal():
    """正常输入 → 返回 file_title + chunks。"""
    node = NodeItemNameRecognition()
    state = create_default_state(
        task_id="test",
        file_title="测试产品手册",
        chunks=[{"title": "H1", "content": "正文"}],
    )
    title, chunks = node._load_inputs(state)
    print(check("file_title 正确", title == "测试产品手册"))
    print(check("chunks 非空", len(chunks) == 1))


def test_load_inputs_missing_title():
    """缺少 file_title → 抛出 ParameterError。"""
    node = NodeItemNameRecognition()
    state = create_default_state(task_id="test", chunks=[{"title": "H"}])
    try:
        node._load_inputs(state)
        print(check("抛出 ParameterError", False))
    except ParameterError:
        print(check("抛出 ParameterError", True))


def test_load_inputs_empty_chunks():
    """空 chunks → 抛出 InvalidStateError。"""
    node = NodeItemNameRecognition()
    state = create_default_state(task_id="test", file_title="test", chunks=[])
    try:
        node._load_inputs(state)
        print(check("抛出 InvalidStateError", False))
    except InvalidStateError:
        print(check("抛出 InvalidStateError", True))


def test_load_inputs_missing_chunks_key():
    """state 中无 chunks 键 → 抛出 InvalidStateError。"""
    node = NodeItemNameRecognition()
    state = create_default_state(task_id="test", file_title="test")
    try:
        node._load_inputs(state)
        print(check("抛出 InvalidStateError", False))
    except InvalidStateError:
        print(check("抛出 InvalidStateError", True))



def test_build_context_normal():
    """正常 3 个切片 → 拼接为结构化上下文。"""
    node = NodeItemNameRecognition()
    chunks = [
        {"title": "# 第一章", "content": "第一章正文"},
        {"title": "## 1.1", "content": "第一节正文"},
        {"title": "## 1.2", "content": "第二节正文"},
    ]
    context = node._build_context(chunks)
    print(check("上下文非空", len(context) > 0))
    print(check("包含切片1", "【切片1】" in context))
    print(check("包含切片3", "【切片3】" in context))
    print(check("长度 ≤ 2500", len(context) <= 2500))


def test_build_context_exceeds_k():
    """超量切片 → 只取前 K=5 个。"""
    node = NodeItemNameRecognition()
    chunks = [{"title": f"# Ch{i}", "content": f"内容{i}"} for i in range(10)]
    context = node._build_context(chunks)
    print(check("不包含切片6", "【切片6】" not in context))
    print(check("包含切片5", "【切片5】" in context))


def test_build_context_exceeds_chars():
    """超字符 → 在 2500 字符处截断。"""
    node = NodeItemNameRecognition()
    long_content = "测试数据。" * 600  # ~3000 字符
    chunks = [{"title": "# 超长", "content": long_content}]
    context = node._build_context(chunks)
    print(check("长度 ≤ 2500", len(context) <= 2500))


def test_build_context_empty_chunks():
    """空切片列表 → 返回空字符串。"""
    node = NodeItemNameRecognition()
    context = node._build_context([])
    print(check("返回空字符串", context == ""))



def test_update_chunks():
    """回填 item_name → state 和所有 chunk 同步更新。"""
    node = NodeItemNameRecognition()
    state = create_default_state(task_id="test", file_title="test")
    chunks = [
        {"title": "# A", "content": "aa"},
        {"title": "# B", "content": "bb"},
    ]
    node._update_chunks(state, chunks, "苏伯尔5000W电磁炉")

    print(check("state.item_name 已设置", state["item_name"] == "苏伯尔5000W电磁炉"))
    print(check("chunk[0].item_name 已设置", chunks[0]["item_name"] == "苏伯尔5000W电磁炉"))
    print(check("chunk[1].item_name 已设置", chunks[1]["item_name"] == "苏伯尔5000W电磁炉"))
    print(check("state.chunks 已同步", state["chunks"] is chunks))



def test_process_validation():
    """process 入口校验 → 参数非法时抛出 KBException（NodeBase 包装）。"""
    node = NodeItemNameRecognition()
    state = create_default_state(task_id="test")
    try:
        node(state)
        print(check("抛出 KBException", False))
    except KBException:
        print(check("抛出 KBException（NodeBase 包装）", True))



if __name__ == "__main__":
    logger.info("===== NodeItemNameRecognition 单元测试开始 =====")

    run_test("1.1 正常输入", test_load_inputs_normal)
    run_test("1.2 缺少 file_title", test_load_inputs_missing_title)
    run_test("1.3 空 chunks", test_load_inputs_empty_chunks)
    run_test("1.4 无 chunks 键", test_load_inputs_missing_chunks_key)

    run_test("2.1 正常上下文构建", test_build_context_normal)
    run_test("2.2 超量切片截断", test_build_context_exceeds_k)
    run_test("2.3 超字符截断", test_build_context_exceeds_chars)
    run_test("2.4 空切片列表", test_build_context_empty_chunks)

    run_test("3.1 回填 item_name", test_update_chunks)
    run_test("4.1 process 参数校验", test_process_validation)

    logger.info("===== NodeItemNameRecognition 单元测试结束 =====")
