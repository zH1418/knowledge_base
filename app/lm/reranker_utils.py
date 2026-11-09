from FlagEmbedding import FlagReranker
from app.core.logger import logger
from app.core.decorators import log_execution, safe_execute
from app.core.exceptions import RerankerException
from app.conf.reranker_config import reranker_config

_reranker_model = None
_init_lock = __import__("threading").Lock()


@log_execution(component="reranker")
@safe_execute(component="reranker")
def get_reranker_model():
    global _reranker_model
    if _reranker_model is not None:
        logger.debug("[Reranker] 单例已存在，直接返回")
        return _reranker_model

    with _init_lock:
        if _reranker_model is not None:
            return _reranker_model

        model_path = reranker_config.bge_reranker_large
        device = reranker_config.bge_reranker_device
        use_fp16 = reranker_config.bge_reranker_fp16

        logger.info(
            f"[Reranker] 开始初始化 | model={model_path}"
            f" | device={device} | fp16={use_fp16}"
        )

        _reranker_model = FlagReranker(
            model_name_or_path=model_path,
            device=device,
            use_fp16=use_fp16,
        )
        logger.success("[Reranker] 模型初始化成功")
        return _reranker_model


@log_execution(component="reranker")
@safe_execute(component="reranker", default_return=[], reraise=False)
def compute_scores(pairs: list) -> list:
    model = get_reranker_model()
    logger.info(f"[Reranker] 开始计算 {len(pairs)} 对的相似度")
    scores = model.compute_score(pairs)
    logger.success(f"[Reranker] 重排序完成 | 分数范围=[{min(scores):.4f}, {max(scores):.4f}]")
    return scores
