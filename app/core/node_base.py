import time
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger  # type: ignore[import-untyped]

from app.core.exceptions import KBException


class NodeBase(ABC):

    name: str = "base_node"

    def __init__(self):
        if self.name == "base_node":
            raise ValueError(
                f"子类 {type(self).__name__} 必须覆盖 name 类属性，"
                f"请设置为有意义的节点名称（如 'PDF转Markdown'）"
            )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        logger.info(f"{'=' * 20}【{self.name}】节点启动 {'=' * 20}")
        self._log_state_summary(state)

        task_id = state.get("task_id")
        is_stream = state.get("is_stream", False)
        if task_id:
            try:
                from app.utils.task_utils import add_running_task
                add_running_task(task_id, self.name, is_stream)
            except ImportError:
                pass  # task_utils 不可用时静默跳过

        start_time = time.perf_counter()

        try:
            state = self.process(state)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"【{self.name}】节点完成 | 耗时={elapsed_ms:.0f}ms"
            )

            if task_id:
                try:
                    from app.utils.task_utils import add_done_task
                    add_done_task(task_id, self.name, is_stream)
                except ImportError:
                    pass

            return state

        except KBException as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            e.detail.setdefault("node", self.name)
            e.detail.setdefault("elapsed_ms", round(elapsed_ms, 1))
            logger.opt(exception=True).error(
                f"【{self.name}】节点失败（已知异常） | "
                f"耗时={elapsed_ms:.0f}ms | "
                f"{e.error_code}: {e.message}"
            )
            raise

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.opt(exception=True).error(
                f"【{self.name}】节点失败（未知异常） | "
                f"耗时={elapsed_ms:.0f}ms | "
                f"类型={type(e).__name__}"
            )
            raise KBException(
                message=f"【{self.name}】节点执行异常: {e}",
                detail={
                    "node": self.name,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "exception_type": type(e).__name__,
                },
            ) from e

    @abstractmethod
    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        ...

    def _log_state_summary(self, state: dict[str, Any]) -> None:
        keys = list(state.keys())
        summary_items: list[str] = []
        show_limit = 10

        for k in keys[:show_limit]:
            v = state.get(k)
            if isinstance(v, list):
                summary_items.append(f"{k}=list[{len(v)}]")
            elif isinstance(v, dict):
                summary_items.append(f"{k}=dict[{len(v)} keys]")
            elif isinstance(v, str) and len(v) > 50:
                summary_items.append(f"{k}=str[{len(v)} chars]")
            else:
                summary_items.append(f"{k}={v!r}")

        if len(keys) > show_limit:
            summary_items.append(f"...(+{len(keys) - show_limit} more keys)")

        logger.debug(f"  State: {{{', '.join(summary_items)}}}")
