from app.core.logger import logger
from app.import_process.agent.nodes.parsing.node_md_img import NodeMdImg
from app.import_process.agent.state import create_default_state
from app.utils.path_util import PROJECT_ROOT


def verify():
    """查找项目中的 Markdown 文件并测试图片处理。"""
    md_candidates = list(PROJECT_ROOT.rglob("*.md"))

    if not md_candidates:
        print("  未找到任何 MD 文件，跳过验证")
        return

    md_path = str(md_candidates[0])
    print(f"  MD 文件: {md_path}")

    images_dir = PROJECT_ROOT / md_candidates[0].parent / "images"
    has_images = images_dir.exists() and any(images_dir.iterdir())
    print(f"  Images 目录存在且有图片: {has_images}")

    state = create_default_state(
        task_id="verify_md_img",
        md_path=md_path,
    )

    node = NodeMdImg()
    try:
        result = node(state)

        new_md_path = result.get("md_path", "")
        new_content = result.get("md_content", "")

        print(f"  PASS - 处理完成")
        print(f"  PASS - 输出文件: {new_md_path}")
        print(f"  PASS - 新内容长度: {len(new_content)} 字符")

        if has_images and "minio" not in new_content.lower() and "http" not in new_content.lower():
            print(f"  WARN - 内容中未检测到 MinIO URL（可能无图片或缓存命中）")
    except Exception as e:
        print(f"  FAIL - {e}")


if __name__ == "__main__":
    logger.info("===== NodeMdImg 验证 =====")
    print("\n============================================================")
    print("  NodeMdImg 连通性验证")
    print("============================================================")
    verify()
    logger.info("===== NodeMdImg 验证结束 =====")
