from app.core.logger import logger
from app.import_process.agent.nodes.processing.node_document_split import NodeDocumentSplit
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



def test_basic_split():
    node = NodeDocumentSplit()
    state = create_default_state(
        task_id="test", file_title="测试", md_content="# 第一章\n\n正文内容\n\n## 细节\n\n更多内容",
    )
    result = node.process(state)
    chunks = result["chunks"]
    print(check("产出 ≥ 1 chunk", len(chunks) >= 1))
    print(check("第一节 chunk 含正文", "正文内容" in chunks[0]["content"]))



def test_code_block_protection():
    node = NodeDocumentSplit()
    md = "# 标题\n\n正文\n\n```\n# 这不是标题\ncode\n```\n\n# 真实标题\n\n后续"
    state = create_default_state(task_id="t", file_title="t", md_content=md)
    result = node.process(state)
    chunks = result["chunks"]
    titles = [c["title"] for c in chunks]
    print(check("代码块内 # 不被识别为标题", "# 这不是标题" not in titles))
    print(check("真实标题 被正确识别", "真实标题" in titles))



def test_indented_code():
    node = NodeDocumentSplit()
    md = "# 章节\n\n正文\n\n    # 这是缩进（不出现在标题中）\n\n# 标题\n\n内容"
    state = create_default_state(task_id="t", file_title="t", md_content=md)
    chunks = node.process(state)["chunks"]
    titles = [c["title"] for c in chunks]
    print(check("缩进 # 不成为 heading title", all("缩进" not in t for t in titles)))



def test_toc_output():
    node = NodeDocumentSplit()
    md = "# 第一章\n\n正文\n\n# 第二章\n\n内容"
    state = create_default_state(task_id="t", file_title="t", md_content=md)
    toc = node.process(state)["toc_entries"]
    print(check("两个一级标题 → 2 个 TOC 条目", len(toc) == 2))
    print(check("TOC 含 chapter_start", all("chunk_start" in t for t in toc)))
    print(check("TOC 含 chapter_count", all("chunk_count" in t for t in toc)))



def test_no_title_doc():
    node = NodeDocumentSplit()
    md = "这是没有标题的文档。\n\n只有普通段落。"
    state = create_default_state(task_id="t", file_title="无标题文档", md_content=md)
    chunks = node.process(state)["chunks"]
    print(check("无标题文档仍有 chunk", len(chunks) > 0))
    print(check("chunk 含正文", "只有普通段落" in chunks[0]["content"]))



def test_chunk_fields():
    node = NodeDocumentSplit()
    md = "# 标题\n\n内容"
    state = create_default_state(task_id="t", file_title="t", md_content=md)
    chunks = node.process(state)["chunks"]
    required = {"title", "content", "parent_title", "section_root", "section_title", "file_title", "part"}
    for c in chunks:
        missing = required - set(c.keys())
        if missing:
            print(check(f"字段缺失: {missing}", False))
            return
    print(check("所有 chunk 字段完整", True))



def test_small_html_table_protected():
    from app.import_process.agent.nodes.processing.node_document_split import (
        extract_beautiful_tables, split_table_rows,
    )
    node = NodeDocumentSplit()
    html = (
        "<table><tr><td>长度:</td><td>90 mm 到 500 度</td></tr>"
        "<tr><td>宽度:</td><td>55 mm 到 225 mm</td></tr></table>"
    )
    md = f"# 规格\n\n正文说明\n\n{html}\n\n后续文字"
    state = create_default_state(task_id="t", file_title="HAK180", md_content=md)
    chunks = node.process(state)["chunks"]
    table_chunks = [c for c in chunks if c.get("type") == "table"]
    print(check("小表格独立成 1 个 table chunk", len(table_chunks) == 1))
    print(check("表格 HTML 完整未切破", "<table>" in table_chunks[0]["content"] and "</table>" in table_chunks[0]["content"]))
    print(check("表格含全部行", table_chunks[0]["content"].count("<tr>") == 2))



def test_large_html_table_row_split():
    from app.import_process.agent.nodes.processing.node_document_split import split_table_rows, _TR_PATTERN
    rows = ["<tr><td>参数{:03d}</td><td>测量值 {:d} mV</td><td>状态 正常</td></tr>".format(i, i * 10) for i in range(80)]
    html = "<table>" + "".join(rows) + "</table>"
    blocks = split_table_rows(html, max_chars=2000)
    print(check("大表按行族切成多块", len(blocks) > 1))
    for blk in blocks:
        print(check("每块整表结构完整", blk.startswith("<table>") and blk.endswith("</table>")))
        trs = _TR_PATTERN.findall(blk)
        print(check("每块行完整不半截", all(t.startswith("<tr>") and t.endswith("</tr>") for t in trs)))
    total_rows = sum(_TR_PATTERN.findall(blk).__len__() for blk in blocks)
    print(check("行总数守恒", total_rows == 80))



def test_large_table_via_node():
    node = NodeDocumentSplit()
    rows = ["<tr><td>参数{:03d}</td><td>测量值 {:d} mV</td><td>状态 正常</td></tr>".format(i, i) for i in range(80)]
    html = "<table>" + "".join(rows) + "</table>"
    md = f"# 技术参数\n\n设备规格说明如下。\n\n{html}\n\n以上为全部参数。"
    state = create_default_state(task_id="t", file_title="HAK180", md_content=md)
    chunks = node.process(state)["chunks"]
    table_chunks = [c for c in chunks if c.get("type") == "table"]
    print(check("大表切出多个 table chunk", len(table_chunks) > 1))
    for tc in table_chunks:
        print(check("每个表格 chunk 结构完整", "<table>" in tc["content"] and "</table>" in tc["content"]))
    print(check("表格 chunk 关联到章节", all(tc["section_root"] == "技术参数" for tc in table_chunks)))



def test_split_table_rows_unit():
    from app.import_process.agent.nodes.processing.node_document_split import split_table_rows
    small = "<table><tr><td>a</td></tr></table>"
    print(check("小表不切（单块）", len(split_table_rows(small, 2000)) == 1))
    empty = "<table></table>"
    print(check("空表兜底返回原表", split_table_rows(empty, 2000) == [empty]))
    big = "<table><thead><tr><th>参数</th><th>测量值</th><th>状态</th></tr></thead>" + "".join(
        "<tr><td>参数{:03d}</td><td>测量值 {:d} mV</td><td>状态 正常</td></tr>".format(i, i) for i in range(80)
    ) + "</table>"
    blocks = split_table_rows(big, 2000)
    print(check("大表切多块", len(blocks) > 1))
    print(check("表头复制到每块", all("<th>" in b for b in blocks)))


def test_image_split():
    from app.import_process.agent.nodes.processing.node_image_split import NodeImageSplit
    node = NodeImageSplit()
    state = create_default_state(
        task_id="t", file_title="图集", md_content="![产品正面](url1)\n![背部](url2)",
    )
    chunks = node.process(state)["chunks"]
    print(check("2 张图片 → 2 chunk", len(chunks) == 2))
    print(check("content 为 alt 文本", chunks[0]["content"] == "产品正面"))
    print(check("section_root = __image__", chunks[0]["section_root"] == "__image__"))
    required = {"title", "content", "parent_title", "section_root", "section_title", "file_title", "part"}
    print(check("字段完整", required <= set(chunks[0].keys())))



if __name__ == "__main__":
    logger.info("===== NodeDocumentSplit 重写测试 =====")

    run_test("1. 基本切分", test_basic_split)
    run_test("2. 代码块保护", test_code_block_protection)
    run_test("3. 缩进代码块", test_indented_code)
    run_test("4. TOC 产出", test_toc_output)
    run_test("5. 无标题文档", test_no_title_doc)
    run_test("6. chunk 字段统一", test_chunk_fields)
    run_test("7. 图片文档切分", test_image_split)
    run_test("8. HTML 小表格整块保留", test_small_html_table_protected)
    run_test("9. HTML 大表格行族细切", test_large_html_table_row_split)
    run_test("10. 大表经节点产出多 table chunk", test_large_table_via_node)
    run_test("11. 表格行细切单元", test_split_table_rows_unit)

    logger.info("===== NodeDocumentSplit 重写测试结束 =====")
