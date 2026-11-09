from typing import Optional

from app.core.logger import logger

_tasks_status: dict[str, str] = {}          # task_id → uploading|processing|completed|failed
_tasks_done_list: dict[str, list[str]] = {}  # task_id → ["文件预处理已完成", ...]
_tasks_running: dict[str, str] = {}          # task_id → "向量化入库"（兼容导入管线单节点）
_tasks_running_list: dict[str, list[str]] = {}  # task_id → ["检索中", "重排中"]（多节点并行）
_tasks_errors: dict[str, str] = {}           # task_id → "错误信息"
_tasks_result: dict[str, dict[str, str]] = {}  # task_id → {"answer": "...", ...}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_NODE_NAME_TO_CN: dict[str, str] = {
    "preprocess": "文件预处理",
    "entry": "文件类型判断",
    "pdf_to_md": "PDF转MD",
    "md_img": "图片处理",
    "image_split": "图片切分",
    "processing": "文档处理",
    "document_split": "文档切分",
    "item_name_recognition": "主体名称识别",
    "bge_embedding": "BGE向量生成",
    "import_milvus": "Milvus入库",
    "vector": "向量化入库",
    "item_name_confirm": "实体确认与问题改写",
    "search_embedding": "向量混合检索",
    "search_hyde": "HyDE假设性检索",
    "web_search_mcp": "联网搜索",
    "rrf": "倒数排序融合",
    "rerank": "重排序",
    "answer_output": "答案生成",
    "pre_search": "预检索",
    "multi_search": "多路并行检索",
    "fusion": "结果融合重排",
    "generation": "答案生成",
}


def _ensure_task(task_id: str) -> None:
    """惰性初始化 task_id 的数据结构（查询管线用）。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _task_push_queue(task_id: str) -> None:
    """SSE 推送当前进度到前端（仅查询管线调用）。"""
    try:
        from app.query.utils.sse_utils import push_to_session
        push_to_session(task_id, "progress", {
            "status": get_task_status(task_id) or "",
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })
    except ImportError:
        pass


def update_task_status(task_id: str, status: str) -> None:
    """更新任务整体状态。"""
    _tasks_status[task_id] = status
    logger.info(f"[Task] {task_id} 状态 → {status}")


def set_task_error(task_id: str, error: str) -> None:
    """记录任务错误信息。"""
    _tasks_errors[task_id] = error
    logger.error(f"[Task] {task_id} 错误：{error}")


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段（如 answer）。"""
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}
    _tasks_result[task_id][key] = value


def get_task_status(task_id: str) -> Optional[str]:
    """获取任务状态。"""
    return _tasks_status.get(task_id)


def get_task_error(task_id: str) -> str:
    """获取任务错误信息。"""
    return _tasks_errors.get(task_id, "")


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """获取任务结果字段。"""
    return _tasks_result.get(task_id, {}).get(key, default)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    cn_name = _NODE_NAME_TO_CN.get(node_name, node_name)
    if is_stream:
        _ensure_task(task_id)
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if cn_name not in _tasks_done_list[task_id]:
        _tasks_done_list[task_id].append(cn_name)
        logger.info(f"[Task] {task_id} | 节点完成: {cn_name}")
    if task_id in _tasks_running_list:
        _tasks_running_list[task_id] = [
            n for n in _tasks_running_list[task_id] if n != node_name
        ]
    if is_stream:
        _task_push_queue(task_id)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    cn_name = _NODE_NAME_TO_CN.get(node_name, node_name)
    if is_stream:
        _ensure_task(task_id)
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if node_name not in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].append(node_name)
    _tasks_running[task_id] = cn_name
    if is_stream:
        _task_push_queue(task_id)


def set_running_task(task_id: str, node_name: str) -> None:
    cn_name = _NODE_NAME_TO_CN.get(node_name, node_name)
    _tasks_running[task_id] = cn_name


def get_done_list(task_id: str) -> list[str]:
    """获取已完成节点列表（导入管线兼容接口）。"""
    return _tasks_done_list.get(task_id, [])


def get_done_task_list(task_id: str) -> list[str]:
    """获取已完成节点列表（查询管线兼容接口）。"""
    return _tasks_done_list.get(task_id, [])


def get_running_task(task_id: str) -> str:
    """获取当前运行节点名（导入管线兼容）。"""
    return _tasks_running.get(task_id, "")


def get_running_task_list(task_id: str) -> list[str]:
    """获取正在运行的节点列表（中文名，查询管线 SSE 推送用）。"""
    _ensure_task(task_id)
    return [_NODE_NAME_TO_CN.get(n, n) for n in _tasks_running_list.get(task_id, [])]


def clear_task(task_id: str) -> None:
    """清理任务数据。"""
    _tasks_status.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_running.pop(task_id, None)
    _tasks_running_list.pop(task_id, None)
    _tasks_errors.pop(task_id, None)
    _tasks_result.pop(task_id, None)
