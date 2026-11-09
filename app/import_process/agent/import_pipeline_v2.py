from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient

from app.conf.embedding_config import embedding_config
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.nodes.processing.node_document_split import NodeDocumentSplit
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.normalize_sparse_vector import normalize_sparse_vector

CHUNK_MAX = 2000                       # 唯一可调上限
CHUNK_MIN = int(CHUNK_MAX * 0.2)       # 下限 = 20%·上限（不足则短合）
CHUNK_OVERLAP = int(CHUNK_MAX * 0.1)   # 重合 = 10%·上限

NodeDocumentSplit._MAX_CHUNK_SIZE = CHUNK_MAX
NodeDocumentSplit._CHUNK_OVERLAP = CHUNK_OVERLAP

_DIM = embedding_config.dense_vector_dim  # 2048（.env EMBEDDING_DIM）

_CTX_MAX_CHARS = 2500  # 主体识别单轮上下文上限
_ABSTAIN = "弃权"


HEAD_MIN_CHARS = 50  # ★ 标题块阈值：低于此长度且非 TOC 的块 = 重复标题，不成块


def merge_small_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    merged: list[dict] = []
    n_merged = 0
    n_dropped_head = 0
    for c in chunks:
        content = c.get("content", "") or ""
        is_toc = c.get("type") == "toc" or c.get("part") == 0
        if is_toc:
            merged.append(c)
            continue
        if len(content) < HEAD_MIN_CHARS:
            n_dropped_head += 1
            continue
        if merged and len(content) < CHUNK_MIN:
            prev = merged[-1]
            if prev.get("type") != "toc" and prev.get("part") != 0:
                prev["content"] = (prev.get("content", "") + "\n\n" + content).strip()
                n_merged += 1
                continue
        merged.append(c)
    if n_dropped_head:
        logger.info(f"[PipelineV2] 标题块丢弃 {n_dropped_head} 条（标题信息已在 title 字段）")
    return merged, n_merged


def _build_context(chunks: list[dict], window: int) -> str:
    parts: list[str] = []
    total = 0
    for idx, c in enumerate(chunks[:window], start=1):
        if c.get("type") == "toc" or c.get("part") == 0:
            continue
        piece = f"【切片{idx}】\n标题：{c.get('title', '')}\n内容：{c.get('content', '')}"
        parts.append(piece)
        total += len(piece)
        if total > _CTX_MAX_CHARS:
            break
    return "\n\n".join(parts)[:_CTX_MAX_CHARS]


def _llm_confirm(file_title: str, context: str) -> str:
    """单轮 LLM 确认。返回主体名；'弃权'/空 = 未确认。"""
    if not context:
        return _ABSTAIN
    sys_p = load_prompt("product_recognition_system")
    hu_p = load_prompt("item_name_recognition", file_title=file_title, context=context)
    hu_p += "\n\n补充规则：如果无法从以上内容确认商品/主题名称，不要猜测，只输出三个字：弃权"
    try:
        resp = get_llm_client(json_mode=False).invoke(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": hu_p}]
        )
        name = str(resp.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[PipelineV2] 主体识别 LLM 异常（视为弃权）| {type(e).__name__}: {e}")
        return _ABSTAIN
    name = name.replace(" ", "").replace("\n", "").replace("\r", "")
    for prefix in ("商品名称：", "产品名称：", "商品名：", "商品:", "产品:"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name or _ABSTAIN


def progressive_item_name(file_title: str, chunks: list[dict]) -> tuple[str, int]:
    """渐进窗口：3 → 5 → 7 → …；返回 (主体, 轮数)。"""
    n_text = sum(1 for c in chunks if c.get("type") != "toc" and c.get("part") != 0)
    window = 3
    rounds = 0
    while True:
        ctx = _build_context(chunks, window)
        name = _llm_confirm(file_title, ctx)
        rounds += 1
        if name != _ABSTAIN:
            logger.info(f"[PipelineV2] 渐进确认（第{rounds}轮, window={min(window, n_text)}）: {name}")
            return name, rounds
        if window >= n_text:
            break
        window += 2
    logger.info(f"[PipelineV2] 渐进未确认（{rounds} 轮）→ 兜底 file_title: {file_title[:30]}")
    return file_title, rounds


def _indexes(client: MilvusClient):
    ip = client.prepare_index_params()
    ip.add_index(field_name="dense_vector", index_name="dense_vector_index",
                 index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
    ip.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                 index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                 params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True,
                         "quantization": "none"})
    return ip


def ensure_collections(client: MilvusClient) -> None:
    """集合不存在则创建（2048 维双索引）。不主动 drop —— 维度变更需显式重建。"""
    for coll in (milvus_config.chunks_collection, milvus_config.item_name_collection):
        if client.has_collection(coll):
            continue
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=512)
        if coll == milvus_config.chunks_collection:
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
            schema.add_field("title", DataType.VARCHAR, max_length=65535)
            schema.add_field("parent_title", DataType.VARCHAR, max_length=65535)
            schema.add_field("part", DataType.INT8)
        schema.add_field("file_title", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=_DIM)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        client.create_collection(coll, schema=schema, index_params=_indexes(client))
        client.load_collection(coll)
        logger.info(f"[PipelineV2] created {coll} (dim={_DIM})")


def upsert_chunks(client: MilvusClient, chunks: list[dict]) -> None:
    data = []
    for c in chunks:
        item = dict(c)
        item["part"] = item.get("part", 0)
        item["pk"] = f"{item.get('file_title', '')}::{item.get('title', '')}::{item['part']}"
        data.append(item)
    for i in range(0, len(data), 64):
        client.upsert(collection_name=milvus_config.chunks_collection, data=data[i:i + 64])
    client.flush(milvus_config.chunks_collection)


def upsert_item_name(client: MilvusClient, file_title: str, item_name: str) -> None:
    emb = generate_embeddings([item_name])
    client.upsert(collection_name=milvus_config.item_name_collection, data=[{
        "pk": f"{file_title}::{item_name}",
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": emb["dense"][0],
        "sparse_vector": emb["sparse"][0],
    }])
    client.flush(milvus_config.item_name_collection)


def import_one_file(local_file_path: str, client: MilvusClient) -> dict[str, Any]:
    """读 md → 切块/短合 → 渐进主体 → API 嵌入 → 入库。返回统计 dict。"""
    p = Path(local_file_path)
    t0 = time.time()

    md = p.read_text(encoding="utf-8")

    splitter = NodeDocumentSplit()
    st: dict[str, Any] = {
        "file_title": p.stem, "md_content": md, "chunks": [],
    }
    splitter.process(st)                                   # mistune AST 切块 + TOC
    chunks, n_merged = merge_small_chunks(st["chunks"])    # ★ 短合（下限 20%）
    t1 = time.time()

    item_name, rounds = progressive_item_name(p.stem, chunks)   # ★ 渐进主体
    for c in chunks:
        c["item_name"] = item_name
    t2 = time.time()

    texts = [f"{item_name}\n{c['content']}" for c in chunks]    # ★ API 双向量
    embs = generate_embeddings(texts)
    for c, dv, sv in zip(chunks, embs["dense"], embs["sparse"]):
        c["dense_vector"] = dv
        c["sparse_vector"] = sv
    t3 = time.time()

    upsert_chunks(client, chunks)
    upsert_item_name(client, p.stem, item_name)
    t4 = time.time()

    return {
        "file_title": p.stem, "chunks": len(chunks), "merged": n_merged,
        "item_name": item_name, "rounds": rounds,
        "t_split": t1 - t0, "t_item": t2 - t1, "t_embed": t3 - t2, "t_upsert": t4 - t3,
        "t_total": t4 - t0, "chunks_data": chunks,
    }


class KBImportPipelineV2:
    """v2 导入管线（md 路径）。run(state, stream=True) 产出节点事件。"""

    _STAGES = ("md_read", "document_split", "item_name_recognition",
               "bge_embedding", "import_milvus")

    def __init__(self) -> None:
        self._client = MilvusClient(uri=milvus_config.milvus_url)
        ensure_collections(self._client)

    def run(self, state: dict, stream: bool = False) -> Any:
        """执行导入。stream=True 时 yield ((None, node_name), state) 事件。"""
        local_path = state.get("local_file_path", "")
        result = import_one_file(local_path, self._client)

        if stream:
            for name in self._STAGES:
                yield ((None, name), state)

        state.update({
            "file_title": result["file_title"],
            "item_name": result["item_name"],
            "chunks": result["chunks_data"],
        })
        logger.success(
            f"[PipelineV2] 导入完成 | {result['file_title'][:40]} | chunks={result['chunks']}"
            f"(短合{result['merged']}) | 主体={result['item_name'][:20]}"
            f" | 切分 {result['t_split']:.1f}s 主体 {result['t_item']:.1f}s"
            f" 嵌入 {result['t_embed']:.1f}s 入库 {result['t_upsert']:.1f}s"
            f" | 合计 {result['t_total']:.1f}s"
        )
        if not stream:
            return state
