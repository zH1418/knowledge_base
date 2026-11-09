import os

from app.core.exceptions import ParameterError
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState


class NodeEntry(NodeBase):
    name = "入口-文件类型判断"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        local_file_path = state.get("local_file_path", "").strip()
        if not local_file_path:
            raise ParameterError("缺失参数：local_file_path")
        if local_file_path.endswith(".pdf"):
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = local_file_path
        elif local_file_path.endswith(".md"):
            state["is_md_read_enabled"] = True
            state["md_path"] = local_file_path

        file_name = os.path.basename(local_file_path)
        state["file_title"] = os.path.splitext(file_name)[0]
        logger.debug(f"[入口] file_title={state['file_title']}, pdf={state['is_pdf_read_enabled']}, md={state['is_md_read_enabled']}")
        return state
