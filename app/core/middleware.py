import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger  # type: ignore[import-untyped]


_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id_var.get()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的入口/出口和耗时。"""

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        _request_id_var.set(request_id)

        path_with_query = request.url.path
        if request.url.query:
            path_with_query += f"?{request.url.query}"

        logger.info(
            f"[{request_id}] --> {request.method} {path_with_query}"
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"[{request_id}] <-- {response.status_code} "
                f"| 耗时={elapsed_ms:.0f}ms"
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"[{request_id}] <!! 请求处理异常 | 耗时={elapsed_ms:.0f}ms"
            )
            raise


def setup_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestLoggingMiddleware)
    logger.info("[Middleware] 请求日志中间件已注册")
