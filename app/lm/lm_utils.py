from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.core.decorators import log_execution, safe_execute, retry_on_failure
from app.core.exceptions import LLMException, ConfigurationError

_llm_client_cache = {}


@log_execution(component="dashscope", log_args=False)
@safe_execute(component="dashscope")
def get_llm_client(
    model: Optional[str] = None,
    json_mode: bool = False,
    enable_web_search: bool = False,
) -> ChatOpenAI:
    target_model = model or lm_config.llm_model or "qwen3-32b"
    cache_key = (target_model, json_mode, enable_web_search)

    if cache_key in _llm_client_cache:
        logger.debug(
            f"[LLM] 缓存命中 | model={target_model} | json_mode={json_mode} | "
        f"web_search={enable_web_search}"
        )
        return _llm_client_cache[cache_key]

    if not lm_config.api_key:
        raise ConfigurationError(
            message="LLM API 密钥缺失",
            detail={"missing": ["OPENAI_API_KEY"]},
        )
    if not lm_config.base_url:
        raise ConfigurationError(
            message="LLM API 基础地址缺失",
            detail={"missing": ["OPENAI_BASE_URL"]},
        )

    logger.info(
        f"[LLM] 开始初始化新实例 | model={target_model} | json_mode={json_mode}"
    )

    extra_body = {}
    max_tokens = lm_config.llm_max_tokens if lm_config.llm_max_tokens > 0 else None

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("[LLM] 已开启 JSON 输出模式")

    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or 0.1,
            max_tokens=max_tokens,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    except LangChainException as e:
        raise LLMException(
            message=f"模型 [{target_model}] 初始化失败（LangChain 层）: {e}",
            original=e,
        ) from e

    _llm_client_cache[cache_key] = llm_client
    logger.info(
        f"[LLM] 实例初始化成功并缓存 | model={target_model} | json_mode={json_mode}"
    )
    return llm_client


if __name__ == "__main__":
    logger.info("===== LLM 客户端工具测试 =====")
    try:
        client1 = get_llm_client()
        logger.success("测试1: 默认配置客户端创建成功")
        client2 = get_llm_client(model="qwen-vl-plus")
        logger.success("测试2: 指定模型客户端创建成功")
        client3 = get_llm_client(model="qwen-vl-plus")
        logger.success(f"测试3: 缓存验证 - 同一实例: {client2 is client3}")
        client4 = get_llm_client(model="qwen3-32b", json_mode=True)
        logger.success("测试4: JSON 模式客户端创建成功")
    except Exception as e:
        logger.error(f"测试失败: {e}")
    finally:
        logger.info("===== LLM 客户端工具测试结束 =====")
