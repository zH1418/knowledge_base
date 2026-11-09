import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request

from app.core.logger import logger


class SSEEvent:
    """SSE 事件类型常量。"""
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号


_session_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列。"""
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列。"""
    logger.debug(f"[SSE] 创建队列 | session={session_id}")
    q = queue.Queue()
    _session_stream[session_id] = q
    return q


def remove_sse_queue(session_id: str):
    """移除指定 session 的队列。"""
    logger.debug(f"[SSE] 移除队列 | session={session_id}")
    _session_stream.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})
    else:
        logger.warning(
            f"[SSE] 队列不存在，无法推送 | session={session_id} | event={event}"
        )


async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    logger.debug(f"[SSE] 生成器启动 | session={session_id}")
    stream_queue = get_sse_queue(session_id)

    if stream_queue is None:
        logger.error(
            f"[SSE] 队列未找到，无法启动生成器 | session={session_id}"
            f" | 可用队列={list(_session_stream.keys())}"
        )
        return

    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack("ready", {})
        logger.debug(f"[SSE] 已发送 ready 信号 | session={session_id}")

        while True:
            if await request.is_disconnected():
                logger.info(f"[SSE] 客户端断开连接 | session={session_id}")
                break

            try:
                msg = await loop.run_in_executor(
                    None, stream_queue.get, True, 1.0
                )
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            if event == "__close__":
                logger.debug(f"[SSE] 收到关闭信号 | session={session_id}")
                break

            yield _sse_pack(event, data)

    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        logger.info(f"[SSE] 连接异常终止 | session={session_id}")
        remove_sse_queue(session_id)
        return
    except Exception as e:
        logger.error(f"[SSE] 生成器异常 | session={session_id}: {type(e).__name__}: {e}")
        remove_sse_queue(session_id)
    finally:
        logger.debug(f"[SSE] 生成器结束 | session={session_id}")
        remove_sse_queue(session_id)
