import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.decorators import retry_on_failure
from app.core.exceptions import InvalidStateError, ParameterError
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.query.agent.node_base import NodeBase
from app.query.agent.state import QueryGraphState
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.conf.lm_config import lm_config
from app.utils.history_utils import format_history_text




@dataclass
class NamedEntityGroup:
    standard: str
    originals: list[str]
    best_score: float
    status: str


class NodeItemNameConfirm(NodeBase):

    name = "实体确认"

    _HISTORY_LIMIT: int = 10

    _OVERFLOW_MAX_SELECT: int = 5  # 超出此值触发多选引导，也是选项框硬上限

    _STATUS_PRIORITY: dict[str, int] = {"confirmed": 3, "candidate": 2, "missing": 1}

    _catalog_cache: set[tuple[str, str, str]] = set()  # (item_name, file_title, title)
    _cache_timestamp: float = 0
    _CACHE_TTL_SECONDS: int = 300
    _cache_lock = threading.Lock()


    def process(self, state: QueryGraphState) -> QueryGraphState:
        self._validate_input(state)

        history = self._try_load_history(state.get("session_id", ""))
        state["history"] = history

        self._save_to_mongo(state)

        result = self._extract_and_evaluate(
            query=state["original_query"],
            history=history,
        )
        state["item_names"] = result.get("item_names", [])
        query_type = result.get("query_type", "search")
        state["query_type"] = query_type  # 写 state 供下游节点读取
        logger.info(
            f"【{self.name}】提取完成 | item_names={state['item_names']}"
            f" | query_type={query_type}"
        )

        if query_type == "llm_error":
            state["answer"] = "抱歉，实体识别服务暂时不可用，请稍后重试或换一种方式描述产品名称。"
            state["item_names"] = []
        elif query_type == "chitchat":
            state["item_names"] = []
            state["answer"] = ""  # 不硬编码，让 answer_output 用 LLM 自由发挥
            logger.info(f"【{self.name}】chitchat → LLM 自由应答")
        elif query_type == "enumeration":
            self._handle_enumeration(state)
        else:
            groups = self._verify_product_names(
                item_names=state["item_names"],
            )
            self._align_and_decide(state, groups)

        self._update_current_message(state)

        return state


    @staticmethod
    def _validate_input(state: QueryGraphState) -> None:
        query = state.get("original_query", "")
        if not query or not query.strip():
            raise ParameterError(
                message="缺少查询内容",
                detail={"field": "original_query"},
            )
        session_id = state.get("session_id", "")
        if not session_id or not session_id.strip():
            raise InvalidStateError(
                message="缺少会话标识",
                detail={"field": "session_id"},
            )


    def _try_load_history(self, session_id: str) -> list[dict[str, Any]]:
        if not session_id:
            return []
        try:
            from app.clients.mongo_history_utils import get_recent_messages
            return get_recent_messages(session_id=session_id, limit=self._HISTORY_LIMIT)
        except ImportError:
            logger.warning(f"【{self.name}】MongoDB 模块不可用，跳过历史")
            return []
        except Exception as e:
            logger.warning(f"【{self.name}】加载历史失败 | {type(e).__name__}: {e}")
            return []


    @retry_on_failure(
        retryable_exceptions=(json.JSONDecodeError, ConnectionError, TimeoutError),
    )
    def _call_llm_extract(self, prompt: str) -> dict[str, Any]:
        """纯 LLM 调用 + JSON 解析（json_mode=False，模型分类更准）。"""
        model = lm_config.extract_model or lm_config.llm_model
        response = get_llm_client(model=model, json_mode=False).invoke(prompt)
        content = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        m = re.search(r'\{[^{}]*"item_names"[^{}]*\}', content, re.DOTALL)
        if m:
            content = m.group(0)
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
        return json.loads(content)

    def _extract_and_evaluate(
        self, query: str, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        prompt = self._build_extract_prompt(query, history)
        try:
            return self._call_llm_extract(prompt)
        except Exception as e:
            logger.warning(f"【{self.name}】LLM 重试耗尽，降级 | {type(e).__name__}: {e}")
            return {"item_names": [], "query_type": "llm_error"}

    @staticmethod
    def _build_extract_prompt(
        query: str, history: list[dict[str, Any]]
    ) -> str:
        history_text = format_history_text(history, max_entries=6)
        return load_prompt(
            "item_name_confirm",
            history_text=history_text,
            query=query,
            current_date=datetime.now().strftime("%Y年%m月%d日"),
        )


    _TOC_SCORE_THRESHOLD: float = 0.70
    _FULL_SCORE_THRESHOLD: float = 0.60

    def _verify_product_names(
        self, item_names: list[str],
    ) -> list[NamedEntityGroup]:
        if not item_names:
            return []

        item_names = list(dict.fromkeys(item_names))

        try:
            from app.clients.milvus_utils import (
                get_milvus_client, create_hybrid_search_requests, hybrid_search,
            )
            from app.conf.milvus_config import milvus_config

            embeddings = generate_embeddings(item_names)
            dense_list = embeddings["dense"]
            sparse_list = embeddings["sparse"]
            client = get_milvus_client()
            collection = milvus_config.chunks_collection
        except Exception as e:
            logger.warning(
                f"【{self.name}】BGE/Milvus 初始化失败，降级空结果"
                f" | {type(e).__name__}: {e}"
            )
            return []

        raw_results = [
            self._search_single_item(
                dense_list[i], sparse_list[i], name,
                client, collection, create_hybrid_search_requests, hybrid_search,
            )
            for i, name in enumerate(item_names)
        ]

        return self._merge_into_groups(raw_results)


    @staticmethod
    def _search_with_filter(
        dense, sparse, expr, client, collection,
        create_hybrid_search_requests, hybrid_search,
    ):
        """单次 hybrid_search 调用，统一参数封装。"""
        reqs = create_hybrid_search_requests(
            dense_vector=dense, sparse_vector=sparse,
            expr=expr, limit=5,
        )
        return hybrid_search(
            client=client, collection_name=collection, reqs=reqs,
            output_fields=["title", "file_title", "item_name"], limit=5,
        )


    def _search_single_item(
        self,
        dense, sparse,
        name: str,
        client, collection,
        create_hybrid_search_requests, hybrid_search,
    ) -> dict[str, Any]:
        """对单个商品名执行 TOC → 全量两层检索，返回原始命中记录。"""
        try:
            toc_result = self._search_with_filter(
                dense, sparse, "part == 0",
                client, collection, create_hybrid_search_requests, hybrid_search,
            )

            best_toc = None
            if toc_result and toc_result[0]:
                for hit in toc_result[0]:
                    if hit.score >= self._TOC_SCORE_THRESHOLD:
                        best_toc = hit
                        break

            if best_toc:
                assert best_toc is not None
                standard = best_toc.entity.get("item_name") or best_toc.entity.get("title", name) or name  # type: ignore[union-attr]
                logger.info(
                    f"【{self.name}】TOC 确认 | {name} → "
                    f"{standard} score={best_toc.score:.4f}"  # type: ignore[union-attr]
                )
                return {
                    "original": name,
                    "standard": standard,
                    "score": best_toc.score,  # type: ignore[union-attr]
                    "status": "confirmed",
                }

            full_result = self._search_with_filter(
                dense, sparse, None,
                client, collection, create_hybrid_search_requests, hybrid_search,
            )

            full_hits = []
            if full_result and full_result[0]:
                for hit in full_result[0]:
                    if hit.score >= self._FULL_SCORE_THRESHOLD:
                        full_hits.append(hit)

            if full_hits:
                best = max(full_hits, key=lambda h: h.score)
                standard = best.entity.get("item_name") or best.entity.get("title", name) or name
                logger.info(
                    f"【{self.name}】全量候选 | {name} → "
                    f"{standard} score={best.score:.4f}"
                )
                return {"original": name, "standard": standard,
                        "score": best.score, "status": "candidate"}

            return {"original": name, "standard": "", "score": 0, "status": "missing"}

        except Exception as e:
            logger.warning(
                f"【{self.name}】Milvus 检索异常，跳过 {name}"
                f" | {type(e).__name__}: {e}"
            )
            return {"original": name, "standard": "", "score": 0, "status": "missing"}


    def _merge_into_groups(
        self, raw_results: list[dict[str, Any]],
    ) -> list[NamedEntityGroup]:
        """将 raw_results 按 KB 标准标题合并分组，含状态升级。"""
        groups: dict[str, NamedEntityGroup] = {}
        for r in raw_results:
            key = r["standard"] if r["status"] != "missing" else r["original"]
            if key in groups:
                groups[key].originals.append(r["original"])
                if r["score"] > groups[key].best_score:
                    groups[key].best_score = r["score"]
                if self._STATUS_PRIORITY.get(r["status"], 0) > self._STATUS_PRIORITY.get(groups[key].status, 0):
                    groups[key].status = r["status"]
            else:
                groups[key] = NamedEntityGroup(
                    standard=key,
                    originals=[r["original"]],
                    best_score=r["score"],
                    status=r["status"],
                )

        result = list(groups.values())
        confirmed_n = sum(1 for g in result if g.status == "confirmed")
        candidate_n = sum(1 for g in result if g.status == "candidate")
        missing_n = sum(1 for g in result if g.status == "missing")
        logger.info(
            f"【{self.name}】验证完成 | 确认={confirmed_n}"
            f" | 候选={candidate_n} | 未匹配={missing_n}"
            f" | 分组后共 {len(result)} 个标准实体"
        )
        return result


    def _align_and_decide(
        self,
        state: QueryGraphState,
        groups: list[NamedEntityGroup],
    ) -> None:
        confirmed = [g for g in groups if g.status == "confirmed"]
        candidates_all = [g for g in groups if g.status == "candidate"]
        candidates_sorted = sorted(
            candidates_all, key=lambda g: g.best_score, reverse=True
        )
        missing = [g for g in groups if g.status == "missing"]

        all_total = len(confirmed) + len(candidates_all) + len(missing)

        if all_total == 0:
            if state.get("query_type") == "unknown":
                extracted = state.get("item_names")
                state["item_names"] = extracted if extracted else [state.get("original_query", "")]
                logger.info(f"【{self.name}】unknown 模式 | 联网兜底 items={state['item_names']}")
                return
            state["answer"] = "您好，请描述您想了解的产品信息。"
            state["item_names"] = []
            state["query_type"] = ""
            logger.info(f"【{self.name}】无商品名 → 友好提示")
            return

        if all_total > self._OVERFLOW_MAX_SELECT:
            all_entities = confirmed + candidates_all + missing
            all_standards = [g.standard for g in all_entities]
            state["overflow_entities"] = all_standards
            state["overflow_max_select"] = self._OVERFLOW_MAX_SELECT
            state["item_names"] = all_standards
            state["answer"] = (
                f"您提到了 {all_total} 个产品，为避免信息过载，"
                f"请选择想优先查询的产品（最多 {self._OVERFLOW_MAX_SELECT} 个）：\n"
                + "\n".join(f"  {i}. {n}" for i, n in enumerate(all_standards, 1))
            )
            state["query_type"] = "too_many_entities"
            logger.info(
                f"【{self.name}】统一弹窗 | total={all_total}"
                f" | 确认={len(confirmed)} 候选={len(candidates_all)} 未匹配={len(missing)}"
            )
            return

        if confirmed or candidates_sorted or missing:
            retrieval_top = confirmed + missing + candidates_sorted
            state["item_names"] = [g.standard for g in retrieval_top]
            state["answer"] = ""
            state["query_type"] = (
                "unknown" if (not confirmed and not candidates_sorted) else "search"
            )
            if missing:
                state["missing_names"] = [g.standard for g in missing]
            supplementary = [g.standard for g in retrieval_top[1:]]
            if supplementary:
                state["supplementary_names"] = supplementary
            logger.info(
                f"【{self.name}】直接检索 | total={len(retrieval_top)}"
                f" | 确认={len(confirmed)} 未匹配={len(missing)} 候选={len(candidates_sorted)}"
            )


    def _save_to_mongo(self, state: QueryGraphState) -> None:
        session_id = state.get("session_id", "")
        if not session_id:
            return
        try:
            from app.clients.mongo_history_utils import save_chat_message
            msg_id = save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
            )
            state["user_message_id"] = msg_id if msg_id else ""
        except ImportError:
            logger.warning(f"【{self.name}】MongoDB 模块不可用，跳过写入")
        except Exception as e:
            logger.warning(f"【{self.name}】MongoDB 写入失败 | {type(e).__name__}: {e}")


    def _get_catalog(self) -> set[tuple[str, str, str]]:
        now = time.time()
        if (self._catalog_cache
                and (now - self._cache_timestamp < self._CACHE_TTL_SECONDS)):
            return self._catalog_cache

        with self._cache_lock:
            if (self._catalog_cache
                    and (time.time() - self._cache_timestamp < self._CACHE_TTL_SECONDS)):
                return self._catalog_cache

            try:
                from app.clients.milvus_utils import get_milvus_client
                from app.conf.milvus_config import milvus_config

                client = get_milvus_client()
                collection = milvus_config.chunks_collection

                results = client.query(
                    collection_name=collection,
                    filter="part == 0",
                    output_fields=["item_name", "file_title", "title"],
                    limit=16384,
                )

                catalog: set[tuple[str, str, str]] = set()
                for r in results or []:
                    item = (r.get("item_name") or "").strip()
                    ft = (r.get("file_title") or "").strip()
                    t = (r.get("title") or "").strip()
                    if item or ft:
                        catalog.add((item, ft, t))

                NodeItemNameConfirm._catalog_cache = catalog
                NodeItemNameConfirm._cache_timestamp = now
                logger.info(
                    f"【{self.name}】产品名录缓存刷新 | 共 {len(catalog)} 条唯一记录"
                )
                return catalog

            except Exception as e:
                logger.warning(
                    f"【{self.name}】产品名录加载失败，使用过期缓存"
                    f" | {type(e).__name__}: {e}"
                )
                return self._catalog_cache


    def _handle_enumeration(self, state: QueryGraphState) -> None:
        item_names = state.get("item_names", [])
        keyword = item_names[0] if item_names else None

        try:
            catalog = self._get_catalog()

            if not catalog:
                state["answer"] = "知识库产品名录暂无数据，请联系管理员。"
                state["item_names"] = []
                logger.warning(f"【{self.name}】枚举查询失败 | 名录缓存为空")
                return

            if keyword:
                kw_lower = keyword.lower()
                matched_names: set[tuple[str, str, str]] = set()
                for item_name, ft, t in catalog:
                    searchable = f"{item_name} {ft} {t}".lower()
                    if kw_lower in searchable:
                        matched_names.add((item_name, ft, t))
            else:
                matched_names = set(sorted(catalog, key=lambda x: x[1] or x[0])[:100])

            if matched_names:
                seen: set[str] = set()
                lines: list[str] = []
                for item_name, ft, t in sorted(matched_names, key=lambda x: x[0]):
                    label = _clean_enum_label(item_name, ft, t)
                    if label not in seen:
                        seen.add(label)
                        lines.append(f"- {label}")

                suffix = ""
                if len(matched_names) > 100:
                    suffix = f"\n（仅展示前 100 条，共 {len(matched_names)} 条匹配）"
                elif not keyword and len(matched_names) >= 100:
                    suffix = f"\n（仅展示前 100 条产品）"

                intro = f"知识库中有以下{'与「' + str(keyword) + '」相关的' if keyword else ''}产品："
                state["answer"] = intro + "\n" + "\n".join(lines) + suffix
                state["item_names"] = []
            else:
                state["answer"] = (
                    f"未找到{'与「' + str(keyword) + '」相关的' if keyword else ''}产品。"
                )
                state["item_names"] = []

            logger.info(
                f"【{self.name}】枚举完成 | keyword={keyword or '全部'}"
                f" | 命中 {len(matched_names)} 条"
            )

        except Exception as e:
            logger.warning(f"【{self.name}】枚举查询失败 | {type(e).__name__}: {e}")
            state["answer"] = ""
            state["query_type"] = "unknown"
            state["item_names"] = [state.get("original_query", "")]


    def _update_current_message(self, state: QueryGraphState) -> None:
        msg_id = state.get("user_message_id", "")
        if not msg_id:
            return
        try:
            from app.clients.mongo_history_utils import update_message_item_names
            update_message_item_names(
                ids=[msg_id],
                item_names=state.get("item_names", []),
            )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"【{self.name}】消息回填失败 | {type(e).__name__}: {e}")



import re as _re

def _clean_enum_label(item_name: str, file_title: str, title: str) -> str:
    """将 (item_name, file_title, title) 清洗为可读的产品名。"""
    ft = file_title or ''
    ft = _re.sub(r'\.(pdf|md|txt|docx?)$', '', ft, flags=_re.IGNORECASE)
    ft = _re.sub(r'用户手册.*|使用说明.*|说明书.*|Manuals.*|配置指南.*', '', ft, flags=_re.IGNORECASE)
    ft = ft.strip('-_ ')
    t = (title or '').strip()
    extra = ft if ft and ft != item_name else ''
    if t and t != item_name and t != ft:
        extra = f'{extra} — {t}' if extra else t
    return f'{item_name}' + (f'（{extra}）' if extra else '')
