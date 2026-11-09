import re

from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState


_IMAGE_LINE_PATTERN: re.Pattern = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


class NodeImageSplit(NodeBase):
    """纯图片文档切分节点。"""

    name = "图片文档切分"

    _DEFAULT_ALT: str = "图片内容"
    _DEFAULT_PART: int = 1


    def process(self, state: ImportGraphState) -> ImportGraphState:
        content = state.get("md_content", "")
        file_title = state.get("file_title", "")

        chunks = self._extract_image_chunks(content, file_title)

        state["chunks"] = chunks
        state["toc_entries"] = [{
            "level": 1,
            "title": file_title,
            "sub_headings": [],
            "chunk_start": 0,
            "chunk_count": len(chunks),
        }] if chunks else []

        logger.success(f"图片切分完成，{len(chunks)} 张图片 → {len(chunks)} 个切片")
        return state


    def _extract_image_chunks(
        self, content: str, file_title: str
    ) -> list[dict]:
        """从 MD 中提取每张图片行，生成统一字段的 chunk 列表。"""
        chunks: list[dict] = []
        idx = 0

        for line in content.split("\n"):
            matches = _IMAGE_LINE_PATTERN.findall(line)
            for alt_text in matches:
                alt = alt_text.strip() or self._DEFAULT_ALT
                idx += 1
                chunks.append({
                    "title": f"图片-{idx}",
                    "content": alt,
                    "parent_title": file_title,
                    "section_root": "__image__",
                    "section_title": "",
                    "file_title": file_title,
                    "part": self._DEFAULT_PART,
                })

        return chunks
