from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger  # type: ignore[import-untyped]

from app.core.exceptions import (
    KBException,
    ValidationException,
    BusinessException,
    ServiceException,
    InternalError,
)
from app.core.middleware import get_request_id


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(KBException)
    async def kb_exception_handler(request: Request, exc: KBException):
        """处理所有知识库自定义异常。"""
        request_id = get_request_id()

        logger.opt(exception=True).warning(
            f"[API] [{request_id}] {request.method} {request.url.path} "
            f"-> {exc.error_code}: {exc.message}"
        )

        return JSONResponse(
            status_code=_http_status(exc),
            content={
                "success": False,
                "error": exc.to_dict(),
                "request_id": request_id or None,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """处理所有未预期的异常（兜底）。"""
        request_id = get_request_id()

        logger.opt(exception=True).error(
            f"[API] [{request_id}] 未处理异常 | "
            f"{request.method} {request.url.path} | "
            f"{type(exc).__name__}: {exc}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "error_code": "KB9999",
                    "message": "服务器内部错误",
                    "detail": {},
                },
                "request_id": request_id or None,
            },
        )

    logger.info("[ErrorHandlers] 全局异常处理器已注册")


def _http_status(exc: KBException) -> int:
    if isinstance(exc, ValidationException):
        return 400
    if isinstance(exc, BusinessException):
        if "NotFound" in type(exc).__name__:
            return 404
        return 422
    if isinstance(exc, ServiceException):
        return 502  # Bad Gateway: 上游服务不可用
    if isinstance(exc, InternalError):
        return 500
    return 500
