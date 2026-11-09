import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
from starlette import status

from app.query.agent.kb_query_workflow import KBQueryWorkflow
from app.query.agent.state import create_default_state
from app.query.api.schema.query_request import QueryRequest
from app.query.utils.source_utils import build_sources
from app.query.utils.sse_utils import create_sse_queue, push_to_session, SSEEvent, sse_generator
from app.utils.task_utils import (
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
    update_task_status, get_task_result, get_done_task_list, set_task_result, get_task_status,
)
from app.core.logger import logger
from app.core.exceptions import QueryProcessingError


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/chat.html")
async def chat_html():
    """返回前端聊天界面。"""
    path = Path(__file__).resolve().parents[1] / "page" / "chat.html"
    if not path.exists():
        return PlainTextResponse(
            content=f"chat.html 未找到: {path}",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return FileResponse(path=str(path), media_type="text/html")


@app.post("/query")
async def query(req: QueryRequest, background_tasks: BackgroundTasks):
    session_id = req.session_id or str(uuid.uuid4())
    is_stream = req.is_stream

    from app.query.utils.sse_utils import remove_sse_queue
    from app.utils.task_utils import clear_task
    remove_sse_queue(session_id)
    clear_task(session_id)

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id, TASK_STATUS_PROCESSING)

    if is_stream:
        background_tasks.add_task(_run_graph, session_id, req.query, is_stream)
        return {"message": "任务已提交", "session_id": session_id}

    _run_graph(session_id, req.query, is_stream)
    return {
        "message": "处理完成",
        "session_id": session_id,
        "answer": get_task_result(session_id, "answer") or "",
        "error": "",
        "image_urls": json.loads(get_task_result(session_id, "image_urls") or "[]"),
        "query_type": get_task_result(session_id, "query_type"),
        "item_names": json.loads(get_task_result(session_id, "item_names") or "[]"),
        "done_list": get_done_task_list(session_id),
        "sources": json.loads(get_task_result(session_id, "sources") or "[]"),
    }


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """轮询查询任务进度（兼容前端 textbook chat.html）。"""
    from app.utils.task_utils import get_done_task_list, get_running_task_list
    done = get_done_task_list(session_id) or []
    running_raw = get_running_task_list(session_id) or []
    running = [r for r in running_raw if r not in done]
    return {
        "status": _resolve_status(get_task_status(session_id)),
        "done_list": done,
        "running_list": running,
        "answer": get_task_result(session_id, "answer") or "",
        "error": get_task_result(session_id, "error") or "",
        "image_urls": json.loads(get_task_result(session_id, "image_urls") or "[]"),
        "sources": json.loads(get_task_result(session_id, "sources") or "[]"),
    }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """查询指定会话的最近对话记录。"""
    try:
        from app.clients.mongo_history_utils import get_recent_messages
        messages = get_recent_messages(session_id, limit=20)
        for m in messages:
            if "_id" in m:
                m["_id"] = str(m["_id"])
            ts = m.get("ts")
            if hasattr(ts, "timestamp"):
                m["ts"] = int(ts.timestamp())
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        logger.warning(f"[QueryService] 获取历史失败 | {session_id} | {e}")
        return {"session_id": session_id, "messages": []}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """清除指定会话的全部历史记录。"""
    try:
        from app.clients.mongo_history_utils import clear_history
        count = clear_history(session_id)
        logger.info(f"[QueryService] 已清除历史 | {session_id} | {count} 条")
        return {"session_id": session_id, "deleted": count}
    except Exception as e:
        logger.warning(f"[QueryService] 清除历史失败 | {session_id} | {e}")
        return {"session_id": session_id, "deleted": 0}





def _resolve_status(raw: Optional[str]) -> str:
    """将 task_utils 状态映射为前端期望的状态字符串。"""
    if not raw:
        return "pending"
    return raw  # processing/completed/failed 直接透传


def _extract_image_urls(text: str) -> list[str]:
    """从答案中提取图片 URL。"""
    import re
    urls: list[str] = []
    for alt, url in re.findall(r"!\[.*?]\((.*?)\)", text):
        url = url.strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _run_graph(session_id: str, query: str, is_stream: bool) -> None:
    state = create_default_state(
        session_id=session_id,
        task_id=session_id,  # 复用 session_id 作为 task_id（task_utils 按此追踪节点进度）
        original_query=query,
        is_stream=is_stream,
    )

    try:
        workflow = KBQueryWorkflow()
        final_state = workflow.run(state, stream=is_stream)

        answer = final_state.get("answer", "")
        set_task_result(session_id, "answer", answer)
        set_task_result(session_id, "query_type", final_state.get("query_type", ""))
        set_task_result(session_id, "item_names", json.dumps(final_state.get("item_names") or [], ensure_ascii=False))
        img_urls = _extract_image_urls(answer)
        set_task_result(session_id, "image_urls", json.dumps(img_urls, ensure_ascii=False))
        sources = build_sources(final_state.get("reranked_docs"))
        set_task_result(session_id, "sources", json.dumps(sources, ensure_ascii=False))
        update_task_status(session_id, TASK_STATUS_COMPLETED)
        if is_stream:
            from app.utils.task_utils import get_done_task_list, get_running_task_list
            push_to_session(session_id, SSEEvent.PROGRESS, {
                "status": TASK_STATUS_COMPLETED,
                "done_list": get_done_task_list(session_id) or [],
                "running_list": get_running_task_list(session_id) or [],
            })

    except QueryProcessingError:
        update_task_status(session_id, TASK_STATUS_FAILED)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": "查询处理失败"})
        raise

    except Exception as e:
        logger.opt(exception=True).error(
            f"[QueryService] 未预期异常 | session={session_id} | {type(e).__name__}: {e}"
        )
        update_task_status(session_id, TASK_STATUS_FAILED)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": "服务器内部错误"})
        raise QueryProcessingError(
            message=f"查询工作流执行异常: {e}",
        ) from e
