import re

import mistune
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exceptions import ParameterError
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState



def _extract_text(node: dict) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if "type" in node and node["type"] == "image":
            url = (node.get("attrs") or {}).get("url", "")
            alt = "".join(_extract_text(c) for c in node.get("children", []))
            return f"![{alt}]({url})" if url else alt
        if "children" in node:
            return "".join(_extract_text(c) for c in node["children"])
        if "raw" in node:
            return node["raw"]
    if isinstance(node, list):
        return "".join(_extract_text(c) for c in node)
    return ""



_TABLE_PATTERN = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_TR_PATTERN = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_THEAD_PATTERN = re.compile(r"<thead\b.*?</thead>", re.IGNORECASE | re.DOTALL)
_TBODY_PATTERN = re.compile(r"<tbody\b.*?</tbody>", re.IGNORECASE | re.DOTALL)
_CAPTION_PATTERN = re.compile(r"<caption\b.*?>(.*?)</caption>", re.IGNORECASE | re.DOTALL)


def extract_beautiful_tables(raw: str) -> list[str]:
    """通用：从 HTML 文本中提取所有 <table>...</table> 的完整 HTML 片段。"""
    return _TABLE_PATTERN.findall(raw)


def split_table_rows(table_html: str, max_chars: int) -> list[str]:
    if len(table_html) <= max_chars:
        return [table_html]

    thead_match = _THEAD_PATTERN.search(table_html)
    tbody_match = _TBODY_PATTERN.search(table_html)
    header = thead_match.group(0) if thead_match else ""

    if tbody_match:
        body = tbody_match.group(0)
        rows = _TR_PATTERN.findall(body)
    else:
        rows = _TR_PATTERN.findall(table_html)

    if not rows:
        return [table_html]

    blocks: list[str] = []
    current_rows: list[str] = []
    current_len = 0

    for row in rows:
        row_len = len(row)
        if current_rows and current_len + row_len > max_chars:
            blocks.append(_wrap_table(header, current_rows))
            current_rows = []
            current_len = 0
        current_rows.append(row)
        current_len += row_len

    if current_rows:
        blocks.append(_wrap_table(header, current_rows))

    return blocks if blocks else [table_html]


def _wrap_table(header: str, rows: list[str]) -> str:
    """将表头+行族包回 <table>...</table> 外壳。"""
    inner = "\n".join([header, *rows]) if header else "\n".join(rows)
    return f"<table>\n{inner}\n</table>"


def _table_caption(table_html: str) -> str:
    """提取表格 <caption> 内容作为标题提示（无则空串）。"""
    m = _CAPTION_PATTERN.search(table_html)
    return m.group(1).strip() if m else ""



class NodeDocumentSplit(NodeBase):
    """MD 文档切分节点（mistune 驱动）。"""

    name = "文档切分"

    _MAX_CHUNK_SIZE: int = 2000
    _CHUNK_OVERLAP: int = 200
    _SEPARATORS: list[str] = [
        "\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ",
    ]


    def process(self, state: ImportGraphState) -> ImportGraphState:
        content, file_title = self._load_inputs(state)

        sections = self._parse_sections(content, file_title)

        chunks, toc = self._build_chunks(sections, file_title)

        total_sections = len(toc)
        for t in toc:
            chunks.append({
                "title": t["title"],
                "content": f"{file_title} {t['title']}",  # 向量化文本 = 文件名+标题，确保商品名可被检索
                "parent_title": file_title,
                "section_root": t["title"],
                "section_title": "",
                "file_title": file_title,
                "part": 0,                        # 0 = TOC 条目
                "type": "toc",                    # 标记
                "total_sections": total_sections,
                "sub_headings": t.get("sub_headings", []),
                "chunk_start": t["chunk_start"],
                "chunk_count": t["chunk_count"],
            })

        state["chunks"] = chunks
        state["toc_entries"] = toc
        logger.success(f"切分完成：{len(chunks)} 个 chunk（含 {total_sections} 个 TOC），{total_sections} 个一级章节")
        return state


    @staticmethod
    def _load_inputs(state: ImportGraphState) -> tuple[str, str]:
        content = state.get("md_content", "")
        if not content or not isinstance(content, str):
            raise ParameterError("核心参数 md_content 缺失")
        file_title = state.get("file_title", "")
        if not file_title:
            raise ParameterError("核心参数 file_title 缺失")
        return content, file_title


    @staticmethod
    def _parse_sections(content: str, file_title: str) -> list[dict]:
        md = mistune.create_markdown(renderer="ast")
        ast = md(content)

        sections: list[dict] = []
        current_texts: list[str] = []
        stack: list[tuple[int, str]] = []

        def flush_section():
            """将当前收集的文本合并为一个段落。"""
            nonlocal current_texts
            text = "\n".join(current_texts).strip()
            if text:
                sections.append({
                    "level": 99,  # 非标题段落
                    "title": "",
                    "text": text,
                    "kind": "text",
                    "parent_title": stack[-1][1] if stack else file_title,
                })
            current_texts = []

        def append_table(tables: list[str]):
            """将一个或多个 HTML 表格作为独立段落追加（不进入 current_texts）。"""
            for table_html in tables:
                sections.append({
                    "level": 99,
                    "title": "",
                    "text": table_html,
                    "kind": "table",            # 标记为表格段
                    "parent_title": stack[-1][1] if stack else file_title,
                })

        for node in ast:
            ntype = node.get("type", "")

            if ntype == "heading":
                flush_section()
                level = node.get("attrs", {}).get("level", 1)
                title = _extract_text(node).strip()

                while stack and stack[-1][0] >= level:
                    stack.pop()

                parent = stack[-1][1] if stack else file_title
                stack.append((level, title))

                sections.append({
                    "level": level,
                    "title": title,
                    "text": "",
                    "kind": "heading",
                    "parent_title": parent,
                })

            elif ntype in ("paragraph", "block_text", "list", "list_item"):
                text = _extract_text(node).strip()
                if text:
                    current_texts.append(text)

            elif ntype == "block_code":
                raw = node.get("raw", "")
                if raw:
                    current_texts.append(raw)

            elif ntype == "block_quote":
                text = _extract_text(node).strip()
                if text:
                    current_texts.append(text)

            elif ntype in ("table", "table_head", "table_body", "table_row", "table_cell"):
                text = _extract_text(node).strip()
                if text:
                    current_texts.append(text)

            elif ntype == "block_html":
                raw = node.get("raw", "")
                if raw:
                    tables = extract_beautiful_tables(raw)
                    if tables:
                        append_table(tables)
                        rest = _TABLE_PATTERN.sub("", raw).strip()
                        if rest:
                            current_texts.append(rest)
                    else:
                        current_texts.append(raw)


        flush_section()
        logger.info(f"mistune 解析完成：{len(sections)} 个段落")
        return sections


    def _build_chunks(
        self, sections: list[dict], file_title: str
    ) -> tuple[list[dict], list[dict]]:
        if not sections:
            return [], []

        groups: list[dict] = []  # [{root_title, parent_title, items: [section]}]
        current_group = None

        for sec in sections:
            if sec["level"] == 1:
                if current_group:
                    groups.append(current_group)
                current_group = {
                    "root_title": sec["title"],
                    "parent_title": sec["parent_title"],
                    "items": [sec],
                }
            elif current_group is not None:
                current_group["items"].append(sec)
            else:
                if not groups:
                    groups.append({
                        "root_title": f"__root__::{file_title}",
                        "parent_title": file_title,
                        "items": [],
                    })
                groups[0]["items"].append(sec)

        if current_group:
            groups.append(current_group)

        all_chunks: list[dict] = []
        toc: list[dict] = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._MAX_CHUNK_SIZE,
            chunk_overlap=self._CHUNK_OVERLAP,
            separators=self._SEPARATORS,
        )

        for group in groups:
            root = group["root_title"]
            start_idx = len(all_chunks)

            sub_groups = self._split_by_level2(group["items"], group["parent_title"])

            prev_tail = ""  # 前一子组尾部，用于 H2 边界重叠
            for sg in sub_groups:
                table_sections: list[dict] = []
                text_sections: list[dict] = []
                for sec in sg:
                    if sec.get("kind") == "table":
                        table_sections.append(sec)
                    else:
                        text_sections.append(sec)

                lines: list[str] = []
                for sec in text_sections:
                    if sec["title"]:
                        lines.append(sec["title"])
                    if sec["text"]:
                        lines.append(sec["text"])
                combined = "\n\n".join(lines).strip()
                if combined:
                    if prev_tail:
                        tail_overlap = prev_tail[-80:] if len(prev_tail) > 80 else prev_tail
                        combined = tail_overlap + "\n\n" + combined
                prev_tail = combined

                first_title = ""
                parent_title = file_title
                for sec in text_sections:
                    if sec.get("kind") == "heading":
                        if not first_title:
                            first_title = sec["title"]
                        if parent_title == file_title or sec["level"] == 2:
                            parent_title = sec["title"]
                        break
                if not first_title:
                    first_title = root

                if combined:
                    if len(combined) <= self._MAX_CHUNK_SIZE:
                        all_chunks.append({
                            "title": first_title or root,
                            "content": combined,
                            "parent_title": parent_title,
                            "section_root": root,
                            "section_title": first_title,
                            "file_title": file_title,
                            "part": 1,
                            "chunk_index": len(all_chunks),
                        })
                    else:
                        texts = splitter.split_text(combined)
                        for i, text in enumerate(texts, start=1):
                            text = text.strip()
                            if not text:
                                continue
                            all_chunks.append({
                                "title": f"{first_title or root}-{i}",
                                "content": text,
                                "parent_title": parent_title,
                                "section_root": root,
                                "section_title": first_title,
                                "file_title": file_title,
                                "part": i,
                                "chunk_index": len(all_chunks),
                            })

                table_parent_title = parent_title
                for sec in text_sections:
                    if sec.get("kind") == "heading":
                        table_parent_title = sec["title"]
                        break
                for tsec in table_sections:
                    table_html = tsec["text"]
                    table_sec_title = table_parent_title
                    caption = _table_caption(table_html)
                    blocks = split_table_rows(table_html, self._MAX_CHUNK_SIZE)
                    for bi, blk in enumerate(blocks, start=1):
                        title = caption or table_sec_title or root
                        if len(blocks) > 1:
                            title = f"{title}-表{bi}"
                        all_chunks.append({
                            "title": title,
                            "content": blk,
                            "parent_title": table_sec_title or root,
                            "section_root": root,
                            "section_title": table_sec_title,
                            "file_title": file_title,
                            "part": bi,
                            "type": "table",       # 标记表格 chunk
                            "chunk_index": len(all_chunks),
                        })

            chunk_count = len(all_chunks) - start_idx
            if chunk_count > 0:
                display_title = root.replace(f"__root__::{file_title}", file_title) if root.startswith("__root__::") else root
                sub_headings = [
                    sec["title"] for sec in group["items"]
                    if sec["level"] == 2
                ]
                toc.append({
                    "level": 1,
                    "title": display_title,
                    "sub_headings": sub_headings,
                    "chunk_start": start_idx,
                    "chunk_count": chunk_count,
                })

        logger.info(f"_build_chunks：{len(groups)} 组 → {len(all_chunks)} chunk，TOC {len(toc)} 条")
        return all_chunks, toc


    @staticmethod
    def _split_by_level2(
        items: list[dict], default_parent: str
    ) -> list[list[dict]]:
        """将 items 按 level-2 标题拆分为子组，不跨二级边界。"""
        if not items:
            return []

        result: list[list[dict]] = []
        current: list[dict] = []
        current_parent = default_parent

        for sec in items:
            if sec["level"] in (2, 3):
                if current:
                    result.append(current)
                current = [sec]
                current_parent = sec["title"]
            elif sec["level"] <= 1:
                if current:
                    result.append(current)
                current = [sec]
                current_parent = sec.get("parent_title", default_parent)
            else:
                if sec.get("parent_title"):
                    sec = {**sec, "parent_title": current_parent}
                current.append(sec)

        if current:
            result.append(current)

        return result