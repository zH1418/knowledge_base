import time
from typing import Deque
from app.core.logger import logger  # 复用项目全局logger


def apply_api_rate_limit(
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60
) -> None:
    current_time = time.time()

    while request_times and current_time - request_times[0] >= window_seconds:
        request_times.popleft()

    if len(request_times) >= max_requests:
        sleep_duration = window_seconds - (current_time - request_times[0])
        if sleep_duration > 0:
            logger.debug(f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
            time.sleep(sleep_duration)
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

    request_times.append(current_time)
    logger.debug(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")
