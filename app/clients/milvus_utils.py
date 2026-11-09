from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.core.decorators import log_execution, safe_execute, retry_on_failure
from app.core.exceptions import VectorDBException, ConfigurationError

_milvus_client = None


@log_execution(component="milvus")
@safe_execute(component="milvus", default_return=None, reraise=False)
def get_milvus_client():
    global _milvus_client
    if _milvus_client is None:
        milvus_uri = milvus_config.milvus_url
        if not milvus_uri:
            raise ConfigurationError(
                message="Milvus 配置缺失",
                detail={"missing": ["MILVUS_URL"]},
            )
        _milvus_client = MilvusClient(uri=milvus_uri)
        logger.info("[Milvus] 客户端连接成功")
    return _milvus_client


def _coerce_int64_ids(ids):
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


@log_execution(component="milvus")
@safe_execute(component="milvus", default_return=[], reraise=False)
def fetch_chunks_by_chunk_ids(
    client,
    collection_name: str,
    chunk_ids,
    *,
    output_fields=None,
    batch_size: int = 100,
):
    if client is None:
        return []
    if not collection_name:
        return []
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"[Milvus] 无法转换为 INT64 的 chunk_id，将跳过: {bad_ids}")
    if not ok_ids:
        return []

    results = []
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        if hasattr(client, "get"):
            try:
                got = client.get(
                    collection_name=collection_name,
                    ids=batch,
                    output_fields=output_fields,
                )
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"[Milvus] get 方法失败，回退至 query: {e}")

        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(
                collection_name=collection_name,
                filter=expr,
                output_fields=output_fields,
            )
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"[Milvus] query 方法批量查询失败: {e}")

    return results


def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5,
):
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,
    )
    return [dense_req, sparse_req]


@log_execution(component="milvus")
@safe_execute(component="milvus", default_return=None, reraise=False)
@retry_on_failure(max_retries=3, retryable_exceptions=(ConnectionError,))
def hybrid_search(
    client,
    collection_name,
    reqs,
    ranker_weights=(0.5, 0.5),
    norm_score=False,
    limit=5,
    output_fields=None,
    search_params=None,
):
    rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

    if output_fields is None:
        output_fields = ["item_name"]

    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=rerank,
        limit=limit,
        output_fields=output_fields,
        search_params=search_params,
    )

    logger.info(
        f"[Milvus] 混合搜索完成 | collection={collection_name}"
        f" | 结果数={len(res[0])}"
    )
    return res
