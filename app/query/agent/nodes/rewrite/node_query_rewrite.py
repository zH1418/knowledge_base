import json
import re
from datetime import datetime
from typing import Any

from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.query.agent.node_base import NodeBase
from app.query.agent.state import QueryGraphState
from app.utils.history_utils import format_history_text


class NodeQueryRewrite(NodeBase):
    name = "问题改写与HyDE生成"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        item_names = state.get("item_names", [])
        original_query = state.get("original_query", "")
        query_type = state.get("query_type", "search")

        if not original_query and not item_names:
            logger.info(f"【{self.name}】无输入数据，跳过")
            return state

        history = state.get("history", [])
        history_text = format_history_text(history, max_entries=6)
        query_mode = (
            "【模式说明】本次为 unknown（联网兜底）模式：产品未在知识库中找到，"
            "请仅产出 rewritten_query 用于联网搜索，hyde_doc 必须设置为空字符串。"
            if query_type == "unknown" else ""
        )
        prompt = load_prompt(
            "query_rewrite",
            item_names=", ".join(item_names) if item_names else "（无）",
            original_query=original_query,
            history_text=history_text,
            current_date=datetime.now().strftime("%Y年%m月%d日"),
            query_mode=query_mode,
        )

        if query_type == "unknown":
            logger.info(f"【{self.name}】unknown 模式 | 仅产出 rewritten_query")

        try:
            result = self._call_llm(prompt)
            rewritten = result.get("rewritten_query", original_query) or original_query
            hyde_doc = result.get("hyde_doc", "")

            if query_type == "unknown":
                hyde_doc = ""  # 强制清空，不依赖 LLM 遵从指令

            state["rewritten_query"] = rewritten
            state["hyde_doc"] = hyde_doc

            logger.info(
                f"【{self.name}】完成 | rewritten={len(rewritten)} chars"
                f" | hyde_doc={'有' if hyde_doc else '无'} ({len(hyde_doc)} chars)"
            )
        except Exception as e:
            logger.warning(
                f"【{self.name}】LLM 失败，降级 | {type(e).__name__}: {e}"
            )
            state["rewritten_query"] = original_query
            state["hyde_doc"] = ""

        return state


    @staticmethod
    def _call_llm(prompt: str) -> dict[str, Any]:
        """调用 LLM 并解析 JSON 输出。"""
        response = get_llm_client(json_mode=True).invoke(prompt)
        content = response.content.strip() if isinstance(response.content, str) else ""
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
        return json.loads(content)
