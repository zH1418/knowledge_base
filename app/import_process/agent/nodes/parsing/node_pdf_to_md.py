from pathlib import Path

from app.clients.mineru_client import MineruClient
from app.conf.mineru_config import mineru_config
from app.core.exceptions import InvalidStateError, ParameterError
from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState


class NodePdfToMd(NodeBase):
    name = "PDF转Markdown"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        pdf_path_obj, output_dir_obj = self._validate_paths(state)

        client = MineruClient(mineru_config)
        md_path = client.process_pdf(pdf_path_obj, output_dir_obj)
        logger.info(f"[PDF→MD] MinerU 解析完成: {md_path}")

        md_content = Path(md_path).read_text(encoding="utf-8")
        logger.info(f"[PDF→MD] 读取 MD 内容: {len(md_content)} 字符")

        state["md_path"] = md_path
        state["md_content"] = md_content
        return state

    @staticmethod
    def _validate_paths(state: ImportGraphState) -> tuple[Path, Path]:
        pdf_path_str = state.get("pdf_path", "").strip()
        local_dir_str = state.get("local_dir", "").strip()

        if not pdf_path_str or not local_dir_str:
            raise ParameterError("state 中缺少 pdf_path 或 local_dir")

        pdf_path = Path(pdf_path_str)
        output_dir = Path(local_dir_str)

        if not pdf_path.exists():
            raise InvalidStateError(f"PDF 文件不存在：{pdf_path.absolute()}")

        output_dir.mkdir(parents=True, exist_ok=True)
        return pdf_path, output_dir
