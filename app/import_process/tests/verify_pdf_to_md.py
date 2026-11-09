import os
from app.core.logger import logger
from app.import_process.agent.nodes.parsing.node_pdf_to_md import NodePdfToMd
from app.import_process.agent.state import create_default_state
from app.utils.path_util import PROJECT_ROOT


def verify():
    """查找项目中的 PDF 并测试转换。"""
    pdf_candidates = list(PROJECT_ROOT.rglob("*.pdf"))

    if not pdf_candidates:
        print("  未找到任何 PDF 文件，跳过验证（需要将 PDF 放入项目目录）")
        return

    pdf_path = str(pdf_candidates[0])
    local_dir = str(PROJECT_ROOT / "output" / "verify_pdf_test")

    print(f"  PDF: {pdf_path}")
    print(f"  输出目录: {local_dir}")

    state = create_default_state(
        task_id="verify_pdf",
        pdf_path=pdf_path,
        local_dir=local_dir,
    )

    node = NodePdfToMd()
    try:
        result = node(state)
        md_path = result.get("md_path", "")
        md_len = len(result.get("md_content", ""))

        print(f"  PASS - md_path: {md_path}")
        print(f"  PASS - md_content 长度: {md_len} 字符")
        print(f"  PASS - MinerU API 连通正常")
    except Exception as e:
        print(f"  FAIL - {e}")


if __name__ == "__main__":
    logger.info("===== NodePdfToMd 验证 =====")
    print("\n============================================================")
    print("  NodePdfToMd 连通性验证")
    print("============================================================")
    verify()
    logger.info("===== NodePdfToMd 验证结束 =====")
