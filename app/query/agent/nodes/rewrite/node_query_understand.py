from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.core.exceptions import InvalidStateError, ParameterError
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query.agent.node_base import NodeBase
from app.query.agent.routing.pre_intent_rules import classify_early
from app.query.agent.state import QueryGraphState
from app.query.agent.tools.milvus_search_tool import (
    TOOL_SCHEMA, format_hits, milvus_search, run_tool,
)
from app.utils.history_utils import format_history_text


class NodeQueryUnderstand(NodeBase):
    """查询理解节点：探针检索 + 单 LLM 工具闭环，产出下游所需全部字段。"""

    name = "查询理解"

    _HISTORY_LIMIT: int = 10
    _MAX_TOOL_ROUNDS: int = 3      # 工具调用轮数上限（防失控）
    _PROBE_TOP_K: int = 5          # 探针返回条数
    _ENABLE_TOOL_LOOP: bool = False

    def process(self, state: QueryGraphState) -> QueryGraphState:
        self._validate_input(state)

        history = self._try_load_history(state.get("session_id", ""))
        state["history"] = history

        self._save_to_mongo(state)

        early = classify_early(state.get("original_query", ""))
        if early is not None:
            q_type, answer = early
            state["query_type"] = q_type
            state["rewritten_query"] = state["original_query"]
            state["hyde_doc"] = ""
            state["item_names"] = []
            if q_type == "enumeration":
                self._handle_enumeration(state)      # 名录缓存 + 内存匹配（毫秒级）
            elif q_type == "chitchat":
                state["answer"] = answer             # "" → answer_output 自由应答
            else:
                state["answer"] = answer             # 空输入 → 友好提示
            logger.info(f"【{self.name}】L0 规则短路 | type={q_type or 'empty'} | 免探针/免LLM")
            self._update_current_message(state)
            return state

        original_query = state["original_query"]
        probe_hits = milvus_search(original_query, scope="all", top_k=self._PROBE_TOP_K)
        probe_text = format_hits(probe_hits, label="探针检索")
        logger.info(f"【{self.name}】探针检索 | {len(probe_hits)} 条命中")

        result = self._understand(original_query, history, probe_text)
        query_type = result.get("query_type", "search")
        item_names = result.get("item_names", []) or []
        rewritten = (result.get("rewritten_query", "") or original_query).strip() or original_query
        hyde_doc = result.get("hyde_doc", "") or ""

        if query_type == "search" and item_names:
            verdict = self._verify_items(item_names)
            if verdict == "all_missing":
                query_type = "unknown"
                logger.info(f"【{self.name}】确定性验证全未命中 → unknown（联网兜底）| items={item_names}")
            elif verdict == "has_hit":
                logger.info(f"【{self.name}】确定性验证命中 | items={item_names}")

        if query_type == "unknown":
            hyde_doc = ""

        if query_type == "llm_error":
            state["answer"] = "抱歉，实体识别服务暂时不可用，请稍后重试或换一种方式描述产品名称。"
            item_names = []
        elif query_type == "chitchat":
            item_names = []
            state["answer"] = ""  # 不硬编码，让 answer_output 用 LLM 自由发挥
            logger.info(f"【{self.name}】chitchat → LLM 自由应答")
        elif query_type == "enumeration":
            state["query_type"] = "enumeration"
            state["item_names"] = item_names
            self._handle_enumeration(state)   # 复用旧枚举实现（名录缓存+内存匹配）
            self._update_current_message(state)
            return state

        state["item_names"] = item_names
        state["query_type"] = query_type
        state["rewritten_query"] = rewritten
        state["hyde_doc"] = hyde_doc

        logger.info(
            f"【{self.name}】完成 | type={query_type} | items={item_names}"
            f" | rewritten={len(rewritten)} chars | hyde={'有' if hyde_doc else '无'}"
        )

        self._update_current_message(state)
        return state


    def _understand(
        self, query: str, history: list[dict[str, Any]], probe_text: str,
    ) -> dict[str, Any]:
        sys_p = load_prompt(
            "query_understand",
            current_date=datetime.now().strftime("%Y年%m月%d日"),
        )
        history_text = format_history_text(history, max_entries=6)
        user_p = load_prompt(
            "query_understand_user",
            history_text=history_text,
            query=query,
            probe_result=probe_text,
        )

        messages: list = [SystemMessage(content=sys_p), HumanMessage(content=user_p)]
        try:
            llm = get_llm_client()
            if self._ENABLE_TOOL_LOOP:
                llm = llm.bind_tools([TOOL_SCHEMA])
                logger.info(f"【{self.name}】工具闭环模式（可自查）")
            for round_i in range(self._MAX_TOOL_ROUNDS):
                resp = llm.invoke(messages)
                tool_calls = getattr(resp, "tool_calls", None)
                if not tool_calls:
                    try:
                        return self._parse_final(resp.content)
                    except (ValueError, json.JSONDecodeError):
                        logger.warning(f"【{self.name}】JSON 解析失败，追加纠正指令重试")
                        messages.append(resp)
                        messages.append(HumanMessage(
                            content="只输出一个 JSON 对象（含 query_type/item_names/"
                                    "rewritten_query/hyde_doc 四个字段），不要任何其他文字。"
                        ))
                        continue
                messages.append(resp)
                for tc in tool_calls:
                    out = run_tool(tc.get("args") or {})
                    logger.info(f"【{self.name}】工具自查（第{round_i + 1}轮）| args={tc.get('args')}")
                    messages.append(ToolMessage(content=out, tool_call_id=tc["id"]))
            return self._parse_final(resp.content)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"【{self.name}】LLM 调用失败，降级 | {type(e).__name__}: {e}")
            return {"query_type": "llm_error", "item_names": [],
                    "rewritten_query": "", "hyde_doc": ""}

    @staticmethod
    def _parse_final(content: Any) -> dict[str, Any]:
        """解析最终 JSON 输出。容错阶梯：严格 loads → 去尾逗号 → 字段级正则兜底。"""
        text = content if isinstance(content, str) else str(content)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                      flags=re.MULTILINE)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"无 JSON 对象: {text[:120]}")
        raw = text[start:end + 1]

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
        except json.JSONDecodeError:
            pass

        result: dict[str, Any] = {}
        m = re.search(r'"query_type"\s*:\s*"(\w+)"', raw)
        if m:
            result["query_type"] = m.group(1)
        m = re.search(r'"item_names"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if m:
            try:
                import ast
                result["item_names"] = ast.literal_eval(f"[{m.group(1)}]")
            except (ValueError, SyntaxError):
                result["item_names"] = re.findall(r'"([^"]+)"', m.group(1))
        for key in ("rewritten_query", "hyde_doc"):
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
            if m:
                try:
                    result[key] = json.loads(f'"{m.group(1)}"')
                except json.JSONDecodeError:
                    result[key] = m.group(1)
        if not result:
            raise ValueError(f"JSON 解析全失败: {raw[:160]}")
        logger.warning(f"[NodeQueryUnderstand] JSON 严格解析失败，字段级兜底成功")
        return result


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

    def _handle_enumeration(self, state: QueryGraphState) -> None:
        """枚举查询：复用旧节点实现（名录缓存 + 内存子串匹配）。"""
        from app.query.agent.nodes.rewrite.node_item_name_confirm import NodeItemNameConfirm
        NodeItemNameConfirm()._handle_enumeration(state)

    def _verify_items(self, item_names: list[str]) -> str:
        try:
            from app.query.agent.nodes.rewrite.node_item_name_confirm import NodeItemNameConfirm
            groups = NodeItemNameConfirm()._verify_product_names(item_names)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"【{self.name}】确定性验证异常，保持 LLM 判定 | {type(e).__name__}: {e}")
            return "unknown_infra"
        if not groups:
            return "unknown_infra"
        if any(g.status in ("confirmed", "candidate") for g in groups):
            return "has_hit"
        return "all_missing"
