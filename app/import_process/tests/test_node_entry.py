import traceback
from app.core.exceptions import KBException
from app.core.logger import logger
from app.import_process.agent.nodes.entry.node_entry import NodeEntry
from app.import_process.agent.state import create_default_state



def check(desc: str, condition: bool) -> str:
    """单条断言，返回结果标记。"""
    return "  PASS" if condition else "  FAIL"


def run_test(name: str, test_fn) -> None:
    """包装测试执行，统一日志格式。"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        test_fn()
    except Exception:
        print(f"  异常: {traceback.format_exc()}")



def test_pdf_recognition():
    """PDF 后缀 → is_pdf_read_enabled=True, pdf_path 设置。"""
    node = NodeEntry()
    state = create_default_state(
        task_id="test_pdf",
        local_file_path="d:/docs/产品手册.pdf",
    )
    result = node(state)

    print(check("is_pdf_read_enabled=True", result["is_pdf_read_enabled"] is True))
    print(check("pdf_path 正确", result["pdf_path"] == "d:/docs/产品手册.pdf"))
    print(check("is_md_read_enabled=False", result["is_md_read_enabled"] is False))
    print(check("file_title='产品手册'", result["file_title"] == "产品手册"))


def test_md_recognition():
    """MD 后缀 → is_md_read_enabled=True, md_path 设置。"""
    node = NodeEntry()
    state = create_default_state(
        task_id="test_md",
        local_file_path="d:/docs/说明书.md",
    )
    result = node(state)

    print(check("is_md_read_enabled=True", result["is_md_read_enabled"] is True))
    print(check("md_path 正确", result["md_path"] == "d:/docs/说明书.md"))
    print(check("is_pdf_read_enabled=False", result["is_pdf_read_enabled"] is False))
    print(check("file_title='说明书'", result["file_title"] == "说明书"))


def test_unknown_extension():
    """未知后缀 → 两个标记均为 False，file_title 仍正常提取。"""
    node = NodeEntry()
    state = create_default_state(
        task_id="test_txt",
        local_file_path="d:/docs/notes.txt",
    )
    result = node(state)

    print(check("is_pdf_read_enabled=False", result["is_pdf_read_enabled"] is False))
    print(check("is_md_read_enabled=False", result["is_md_read_enabled"] is False))
    print(check("file_title='notes'", result["file_title"] == "notes"))


def test_empty_path_raises():
    """空路径 → 抛出异常（NodeBase.__call__ 会将 ValueError 包装为 KBException）。"""
    node = NodeEntry()
    state = create_default_state(task_id="test_empty", local_file_path="")
    try:
        node(state)
        print(check("抛出 KBException", False))
    except KBException:
        print(check("抛出 KBException（NodeBase 包装）", True))


def test_whitespace_path():
    """只有空格的路径 → 抛出异常。"""
    node = NodeEntry()
    state = create_default_state(task_id="test_space", local_file_path="   ")
    try:
        node(state)
        print(check("抛出 KBException", False))
    except KBException:
        print(check("抛出 KBException（NodeBase 包装）", True))


def test_file_title_with_nested_path():
    """深层路径 → file_title 只取文件名部分。"""
    node = NodeEntry()
    state = create_default_state(
        task_id="test_deep",
        local_file_path="/home/user/projects/docs/manual.pdf",
    )
    result = node(state)
    print(check("file_title='manual'", result["file_title"] == "manual"))


def test_uppercase_extension():
    """大写后缀 .PDF → 按课程实现 .endswith('.pdf') 区分大小写，.PDF 不匹配。"""
    node = NodeEntry()
    state = create_default_state(
        task_id="test_upper",
        local_file_path="d:/docs/MANUAL.PDF",
    )
    result = node(state)
    is_unknown = (not result["is_pdf_read_enabled"] and not result["is_md_read_enabled"])
    print(check(".PDF 被视为未知后缀（大小写敏感）", is_unknown))



if __name__ == "__main__":
    logger.info("===== NodeEntry 单元测试开始 =====")

    run_test("1. PDF 文件识别", test_pdf_recognition)
    run_test("2. Markdown 文件识别", test_md_recognition)
    run_test("3. 未知后缀处理", test_unknown_extension)
    run_test("4. 空路径应抛异常", test_empty_path_raises)
    run_test("5. 纯空格路径应抛异常", test_whitespace_path)
    run_test("6. 深层路径 file_title 提取", test_file_title_with_nested_path)
    run_test("7. 大写后缀行为", test_uppercase_extension)

    logger.info("===== NodeEntry 单元测试结束 =====")
