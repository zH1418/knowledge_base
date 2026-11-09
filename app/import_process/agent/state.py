import copy
from typing import TypedDict

from app.core.state_base import create_base_state


class ImportGraphState(TypedDict):

    task_id: str  # 任务唯一ID，用于追踪日志

    is_md_read_enabled: bool  # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取路径

    is_normal_split_enabled: bool  # 是否启用本地常规切分（基于 RecursiveCharacterTextSplitter，默认 True）
    is_silicon_flow_api_enabled: bool  # 预留：是否启用硅基流动 API 语义切分（云端中文智能分段，默认 True）
    is_advanced_split_enabled: bool  # 预留：是否启用高级切分策略（如知识图谱辅助切分，当前未实现）
    is_vllm_enabled: bool  # 预留：是否启用本地 vLLM 推理（替代云端 LLM，需 GPU 资源，默认 False）

    local_dir: str  # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径
    file_title: str  # 文件标题（文件名去后缀）
    pdf_path: str  # PDF 文件路径 (如果输入是PDF)
    md_path: str  # Markdown 文件路径 (转换后或直接输入的)
    split_path: str  # 分块后的文件路径
    embeddings_path: str  # 向量数据库文件路径

    md_content: str  # Markdown 的全文内容
    chunks: list  # 切片后的文本列表，包含 metadata
    toc_entries: list  # 目录条目列表（纯标题索引），与 chunks 分离存储
    item_name: str  # 识别出的主体名称 (如: "万用表")，用于增强检索

    embeddings_content: list  # 包含向量数据的列表，准备写入 Milvus

graph_default_state: ImportGraphState = {
    **create_base_state(),  # 继承公共字段（当前: task_id）
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "toc_entries": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides)->ImportGraphState:
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state

def get_default_state() -> ImportGraphState:
    return copy.deepcopy(graph_default_state)

if __name__ == '__main__':
    from app.utils.format_utils import format_state
    init_sate= create_default_state(
        task_id="task_001",
        local_file_path="d:/abc.md"
    )

    print(format_state(init_sate))