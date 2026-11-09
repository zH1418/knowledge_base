import asyncio
import functools
import inspect
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger  # type: ignore[import-untyped]

from app.core.exceptions import (
    KBException,
    InternalError,
    get_builtin_exception_map,
)

F = TypeVar("F", bound=Callable[..., Any])



def _find_exception_mapping(
    exc: Exception,
    mapping: dict[type[Exception], type[KBException]],
) -> type[KBException] | None:
    for exc_type in type(exc).__mro__:
        if exc_type in mapping:
            return mapping[exc_type]
    return None


def _truncate_repr(obj: Any, max_len: int) -> str:
    """将对象转为 repr 字符串并截断。"""
    s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + "...<truncated>"
    return s



def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    backoff_max: float = 30.0,
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    ),
    jitter: bool = True,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            total_attempts = max_retries + 1
            for attempt in range(1, total_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt >= total_attempts:
                        logger.error(
                            f"[重试耗尽] {func.__qualname__} "
                            f"重试 {max_retries} 次后仍然失败"
                        )
                        raise
                    delay = min(backoff_factor * (2 ** (attempt - 1)), backoff_max)
                    if jitter:
                        delay *= 0.5 + random.random()  # 0.5x ~ 1.5x
                    logger.warning(
                        f"[重试 {attempt}/{max_retries}] {func.__qualname__} "
                        f"因 {type(e).__name__}: {e} 失败, "
                        f"{delay:.1f}s 后重试"
                    )
                    time.sleep(delay)
                except Exception:
                    raise
            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            total_attempts = max_retries + 1
            for attempt in range(1, total_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt >= total_attempts:
                        logger.error(
                            f"[重试耗尽] {func.__qualname__} "
                            f"重试 {max_retries} 次后仍然失败"
                        )
                        raise
                    delay = min(backoff_factor * (2 ** (attempt - 1)), backoff_max)
                    if jitter:
                        delay *= 0.5 + random.random()
                    logger.warning(
                        f"[重试 {attempt}/{max_retries}] {func.__qualname__} "
                        f"因 {type(e).__name__}: {e} 失败, "
                        f"{delay:.1f}s 后重试"
                    )
                    await asyncio.sleep(delay)
                except Exception:
                    raise
            raise last_exception  # type: ignore[misc]

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator



def safe_execute(
    component: str = "unknown",
    default_return: Any = None,
    reraise: bool = True,
    log_level: str = "ERROR",
    exception_map: dict[type[Exception], type[KBException]] | None = None,
) -> Callable[[F], F]:
    _map = exception_map or get_builtin_exception_map()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except KBException:
                logger.opt(exception=True).log(
                    log_level,
                    f"[{component}] {func.__qualname__} 执行失败（已知异常）"
                )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                mapped_type = _find_exception_mapping(e, _map)
                if mapped_type is not None:
                    wrapped = mapped_type(
                        message=str(e),
                        original=e,
                    )
                    logger.opt(exception=True).log(
                        log_level,
                        f"[{component}] {func.__qualname__} "
                        f"-> {type(wrapped).__name__}({wrapped.error_code})"
                    )
                    if reraise:
                        raise wrapped from e
                    return default_return

                logger.opt(exception=True).log(
                    log_level,
                    f"[{component}] {func.__qualname__} "
                    f"发生未预期异常: {type(e).__name__}"
                )
                if reraise:
                    raise InternalError(
                        message=f"未预期错误: {e}",
                        detail={
                            "component": component,
                            "func": func.__qualname__,
                            "exception_type": type(e).__name__,
                        },
                    ) from e
                return default_return

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except KBException:
                logger.opt(exception=True).log(
                    log_level,
                    f"[{component}] {func.__qualname__} 执行失败（已知异常）"
                )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                mapped_type = _find_exception_mapping(e, _map)
                if mapped_type is not None:
                    wrapped = mapped_type(message=str(e), original=e)
                    logger.opt(exception=True).log(
                        log_level,
                        f"[{component}] {func.__qualname__} "
                        f"-> {type(wrapped).__name__}({wrapped.error_code})"
                    )
                    if reraise:
                        raise wrapped from e
                    return default_return

                logger.opt(exception=True).log(
                    log_level,
                    f"[{component}] {func.__qualname__} "
                    f"发生未预期异常: {type(e).__name__}"
                )
                if reraise:
                    raise InternalError(
                        message=f"未预期错误: {e}",
                        detail={
                            "component": component,
                            "func": func.__qualname__,
                            "exception_type": type(e).__name__,
                        },
                    ) from e
                return default_return

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator



def log_execution(
    component: str = "",
    log_args: bool = True,
    log_result: bool = False,
    log_level: str = "DEBUG",
    max_arg_length: int = 200,
    max_result_length: int = 500,
) -> Callable[[F], F]:
    prefix = f"[{component}] " if component else ""

    def decorator(func: F) -> F:
        if inspect.isgeneratorfunction(func):
            @functools.wraps(func)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.log(log_level,
                           f"{prefix}{func.__qualname__} 生成器启动")
                start = time.perf_counter()
                try:
                    gen = func(*args, **kwargs)
                    yield from gen
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.log(
                        log_level,
                        f"{prefix}{func.__qualname__} 生成器完成"
                        f" | 耗时={elapsed:.2f}ms"
                    )
                except Exception:
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.log(
                        log_level,
                        f"{prefix}{func.__qualname__} 生成器异常"
                        f" | 已耗时={elapsed:.2f}ms（将交由上层处理）"
                    )
                    raise
            return gen_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            args_repr = ""
            if log_args:
                parts = []
                if args:
                    parts.append(f"args={_truncate_repr(args, max_arg_length)}")
                if kwargs:
                    parts.append(f"kwargs={_truncate_repr(kwargs, max_arg_length)}")
                args_repr = ", ".join(parts)

            msg = f"{prefix}{func.__qualname__} 调用开始"
            if args_repr:
                msg += f" ({args_repr})"
            logger.log(log_level, msg)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000

                result_repr = ""
                if log_result:
                    result_repr = f" -> {_truncate_repr(result, max_result_length)}"

                logger.log(
                    log_level,
                    f"{prefix}{func.__qualname__} 调用完成"
                    f" | 耗时={elapsed:.2f}ms{result_repr}"
                )
                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(
                    log_level,
                    f"{prefix}{func.__qualname__} 调用异常"
                    f" | 已耗时={elapsed:.2f}ms（将交由上层处理）"
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            args_repr = ""
            if log_args:
                parts = []
                if args:
                    parts.append(f"args={_truncate_repr(args, max_arg_length)}")
                if kwargs:
                    parts.append(f"kwargs={_truncate_repr(kwargs, max_arg_length)}")
                args_repr = ", ".join(parts)

            msg = f"{prefix}{func.__qualname__} 调用开始"
            if args_repr:
                msg += f" ({args_repr})"
            logger.log(log_level, msg)

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000

                result_repr = ""
                if log_result:
                    result_repr = f" -> {_truncate_repr(result, max_result_length)}"

                logger.log(
                    log_level,
                    f"{prefix}{func.__qualname__} 调用完成"
                    f" | 耗时={elapsed:.2f}ms{result_repr}"
                )
                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(
                    log_level,
                    f"{prefix}{func.__qualname__} 调用异常"
                    f" | 已耗时={elapsed:.2f}ms（将交由上层处理）"
                )
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
