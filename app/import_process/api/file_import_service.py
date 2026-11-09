import uuid
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.logger import logger
from app.import_process.agent.import_pipeline_v2 import KBImportPipelineV2
from app.import_process.agent.kb_import_workflow import KBImportWorkflow
from app.import_process.agent.state import create_default_state
from app.utils.task_utils import (
    get_done_list, get_running_task,
    get_task_error, get_task_status, set_running_task, set_task_error,
    update_task_status,
)


app = FastAPI(title="文档知识库 - 导入服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PAGE_DIR = Path(__file__).resolve().parent.parent / "page"
_UPLOAD_DIR = Path("output")
_UPLOAD_DIR.mkdir(exist_ok=True)



@app.get("/import.html")
async def import_page():
    """返回文件上传前端页面。"""
    return FileResponse(_PAGE_DIR / "import.html")



@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    task_id: str = Form(""),
):
    """接收文件 → 本地保存 → 启动后台 LangGraph 导入任务。"""
    task_id = task_id or str(uuid.uuid4())
    update_task_status(task_id, "uploading")

    task_dir = _UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for file in files:
        file_path = task_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        saved_files.append(str(file_path))
        logger.info(f"[Task] {task_id} 文件已保存: {file_path}")

    update_task_status(task_id, "processing")

    for file_path in saved_files:
        background_tasks.add_task(
            _run_import, task_id, file_path, str(task_dir)
        )

    return JSONResponse({
        "task_id": task_id,
        "status": "processing",
        "message": f"已接收 {len(saved_files)} 个文件",
    })



@app.get("/status/{task_id}")
async def task_status(task_id: str):
    """返回任务进度：status + done_list + running + error。"""
    return JSONResponse({
        "task_id": task_id,
        "status": get_task_status(task_id) or "unknown",
        "done_list": get_done_list(task_id),
        "running": get_running_task(task_id),
        "error": get_task_error(task_id),
    })



def _run_import(task_id: str, local_file_path: str, local_dir: str):
    try:
        is_md = Path(local_file_path).suffix.lower() == ".md"
        state = create_default_state(
            task_id=task_id,
            local_file_path=local_file_path,
            local_dir=local_dir,
            is_pdf_read_enabled=not is_md,
            is_md_read_enabled=is_md,
        )

        if is_md:
            wf = KBImportPipelineV2()
            for _chunk in wf.run(state, stream=True):
                pass  # v2 事件即阶段标记，直接喂给进度展示
            update_task_status(task_id, "completed")
            set_running_task(task_id, "")
            logger.success(f"[Task] {task_id} 导入完成（v2 管线）")
            return

        wf = KBImportWorkflow()
        for chunk in wf.run(state, stream=True):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                ns, _ = chunk
                if isinstance(ns, tuple) and len(ns) > 0:
                    node_name = ns[-1] if ns[-1] else ""
                    if node_name and ":" not in str(node_name):
                        set_running_task(task_id, node_name)

        update_task_status(task_id, "completed")
        set_running_task(task_id, "")  # 清除 running 状态，避免前端残留"正在进行..."
        logger.success(f"[Task] {task_id} 导入完成")

    except Exception as e:
        update_task_status(task_id, "failed")
        set_task_error(task_id, str(e))
        logger.error(f"[Task] {task_id} 导入失败: {e}")

    finally:
        pass
