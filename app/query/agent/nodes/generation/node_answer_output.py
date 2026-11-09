import re
from datetime import datetime

from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.query.agent.node_base import NodeBase
from app.query.agent.state import QueryGraphState
from app.query.utils.source_utils import build_sources
from app.query.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import set_task_result

HISTORY_MAX_CHARS: int = 2000   # 历史消息字符预算
BASE_TOKENS: int = 4096         # max_tokens 基础值（给推理留余量）
PER_DOC_TOKENS: int = 600       # 每条文档增加的 token 预算
MAX_TOKENS_CAP: int = 8192      # max_tokens 上限
CHAR_RATIO: float = 0.5         # ★ 正文目标占预算比例（原 0.8 把预算吃满、不给推理留空间）


class NodeAnswerOutput(NodeBase):
    name = "答案生成"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        if state.get("answer"):
            result = self._pass_through(state)
            result.setdefault("query_type", state.get("query_type", ""))
            return result

        if state.get("query_type") == "chitchat":
            return self._handle_chitchat(state)

        context_str = self._format_context(state.get("reranked_docs") or [])

        history_str = self._format_history(state.get("history") or [])

        doc_count = len(state.get("reranked_docs") or [])
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names_str = ", ".join(state.get("item_names") or []) or "无指定产品"

        max_tokens = max(BASE_TOKENS, min(BASE_TOKENS + doc_count * PER_DOC_TOKENS, MAX_TOKENS_CAP))
        limit_hint = ("硬性字数上限：正文（含标点，不含角标与图片区块）不超过 600 字。"
                      "只围绕问题的核心主题作答，不展开延伸内容；"
                      "若要点过多，宁可舍弃次要要点，也不得超长。")

        prompt = load_prompt(
            "answer_out",
            context=context_str or "无参考内容",
            history=history_str or "暂无历史对话",
            item_names=item_names_str,
            question=question,
            limit=limit_hint,
            current_date=datetime.now().strftime("%Y年%m月%d日"),
        )

        answer = self._generate(state, prompt, max_tokens)

        image_urls = self._extract_images(state.get("reranked_docs") or [])

        disclaimer = self._build_disclaimer(state)
        if disclaimer:
            answer = disclaimer + answer

        self._write_history(state, answer, image_urls)

        self._final_push(state, answer, image_urls)

        state["answer"] = answer
        return state


    def _pass_through(self, state: QueryGraphState) -> QueryGraphState:
        """上游短路的 answer 透传 + 免责前缀 + MongoDB 落盘。"""
        disclaimer = self._build_disclaimer(state)
        answer = state["answer"]
        if disclaimer:
            answer = disclaimer + answer
            state["answer"] = answer

        is_stream = state.get("is_stream", False)
        sid = state.get("session_id") or ""

        self._write_history(state, answer, [])

        if is_stream:
            push_to_session(sid, SSEEvent.DELTA, {"delta": answer})
            push_to_session(sid, SSEEvent.FINAL, {
                "answer": answer,
                "status": "completed",
                "image_urls": [],
            })
        else:
            set_task_result(sid, "answer", answer)
        return state


    def _handle_chitchat(self, state: QueryGraphState) -> QueryGraphState:
        """闲聊：无检索上下文，仅带历史对话让 LLM 自由回应。"""
        question = state.get("original_query", "")
        history = state.get("history") or []
        history_str = self._format_history(history)

        prompt = load_prompt(
            "answer_out",
            context="（这是一句闲聊，无需参考任何文档）",
            history=history_str or "暂无历史对话",
            item_names="",
            question=question,
            limit="简洁回复，50 字以内。",
            current_date=datetime.now().strftime("%Y年%m月%d日"),
        )

        answer = self._generate(state, prompt, BASE_TOKENS)
        image_urls: list[str] = []

        state["answer"] = answer
        if state.get("is_stream", False):
            push_to_session(state.get("session_id", ""), SSEEvent.DELTA, {"delta": answer})
            push_to_session(state.get("session_id", ""), SSEEvent.FINAL, {
                "answer": answer,
                "status": "completed",
                "image_urls": image_urls,
            })
        else:
            set_task_result(state.get("session_id", ""), "answer", answer)

        self._write_history(state, answer, image_urls)
        return state


    @staticmethod
    def _format_context(docs: list[dict]) -> str:
        """带元标签的上下文字符串，网络摘要截断到 500 字。"""
        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            if not isinstance(doc, dict):
                continue

            source = doc.get("source", "")
            content = doc.get("content", "")

            tags = [f"[{idx}]"]
            if source:
                tags.append(f"[source={source}]")
            cid = doc.get("chunk_id")
            if cid:
                tags.append(f"[chunk_id={cid}]")
            url = doc.get("url", "")
            if url:
                tags.append(f"[url={url}]")
            title = doc.get("title", "")
            if title:
                tags.append(f"[title={title}]")
            score = doc.get("score")
            if score is not None:
                tags.append(f"[score={float(score):.4f}]")  # type: ignore[arg-type]

            parts.append(" ".join(tags) + "\n" + content)

        return "\n\n".join(parts)


    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """对话历史，角色标记 + 预算控制。"""
        role_map = {"user": "用户", "assistant": "助手"}
        lines: list[str] = []
        used = 0

        for msg in reversed(history):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            text = msg.get("text", "")
            if role not in role_map or not text:
                continue
            line = f"{role_map[role]}: {text}"
            if used + len(line) > HISTORY_MAX_CHARS:
                break
            lines.append(line)
            used += len(line)

        lines.reverse()
        return "\n".join(lines)


    def _generate(self, state: QueryGraphState, prompt: str, max_tokens: int) -> str:
        """调用 LLM 生成答案，流式时 SSE DElta 推送。"""
        is_stream = state.get("is_stream", False)
        sid = state.get("session_id") or ""

        logger.info(
            f"【{self.name}】LLM 调用 | stream={is_stream}"
            f" | max_tokens={max_tokens}"
            f" | prompt_chars={len(prompt)}"
        )

        llm = get_llm_client().bind(max_tokens=max_tokens)

        if is_stream:
            final = ""
            try:
                for chunk in llm.stream(prompt):
                    delta = getattr(chunk, "content", "") or ""
                    if delta:
                        final += delta
                        push_to_session(sid, SSEEvent.DELTA, {"delta": delta})
            except Exception as e:
                logger.error(f"【{self.name}】流式失败 | {type(e).__name__}: {e}")
                push_to_session(sid, SSEEvent.ERROR, {"error": str(e)})
            if not final.strip():
                logger.warning(
                    f"【{self.name}】流式输出为空 | max_tokens={max_tokens}"
                    f"（推理 token 可能吃光预算；流式已推送增量，无法重试）"
                )
            logger.info(f"【{self.name}】LLM 输出 | answer_chars={len(final)} | ratio={len(final)/max(1,len(prompt)):.2%}")
            return final
        else:
            last_err = ""
            for attempt, mt in enumerate((max_tokens, MAX_TOKENS_CAP), 1):
                try:
                    resp = llm.bind(max_tokens=mt).invoke(prompt)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    logger.error(f"【{self.name}】非流式失败（{attempt}/2）| {last_err}")
                    continue
                answer = str(resp.content or "")
                if answer.strip():
                    set_task_result(sid, "answer", answer)
                    logger.info(
                        f"【{self.name}】LLM 输出 | attempt={attempt} | max_tokens={mt}"
                        f" | answer_chars={len(answer)} | ratio={len(answer)/max(1,len(prompt)):.2%}"
                    )
                    return answer
                logger.warning(
                    f"【{self.name}】空响应（{attempt}/2）| max_tokens={mt}"
                    f" → 提高预算重试（推理 token 可能吃光了预算）"
                )
            logger.error(f"【{self.name}】重试后仍为空响应 | 降级为显式失败文案 | {last_err}")
            fallback = "抱歉，本次未能生成有效回答，请重试或换一种问法。"
            set_task_result(sid, "answer", fallback)
            return fallback


    @staticmethod
    def _extract_images(docs: list[dict]) -> list[str]:
        """从 reranked_docs 提取 Markdown 图片 URL。"""
        pattern = re.compile(r"!\[.*?]\((.*?)\)")
        images: list[str] = []
        seen: set[str] = set()

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            u = (doc.get("url") or "").strip()
            if u and u.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")):
                if u not in seen:
                    seen.add(u)
                    images.append(u)
            text = (doc.get("content") or "").strip()
            for img_url in pattern.findall(text):
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    seen.add(img_url)
                    images.append(img_url)

        if images:
            logger.info(f"【{NodeAnswerOutput.name}】提取 {len(images)} 张图片")
        return images


    @staticmethod
    def _build_disclaimer(state: QueryGraphState) -> str:
        """按 query_type 分级免责声明。"""
        qt = state.get("query_type", "search")
        note = state.get("expansion_note", "")

        if qt == "enumeration":
            return "以下为知识库现有产品名录，如需特定型号请进一步说明。\n\n"
        if qt == "unknown":
            return "知识库中暂无该产品的详细信息，以下结果来自网络检索，仅供参考。\n\n"
        if note:
            return f"问题比较宽泛，已选择常见方向：{note}。如有特定需求请明确方向。\n\n"
        return ""


    @staticmethod
    def _write_history(
        state: QueryGraphState, answer: str, image_urls: list[str],
    ) -> None:
        """写入 MongoDB，失败不影响主链路。"""
        sid = state.get("session_id") or ""
        if not sid or not answer:
            return
        sources_for_db = build_sources(state.get("reranked_docs"))
        try:
            save_chat_message(
                session_id=sid,
                role="assistant",
                text=answer,
                rewritten_query=state.get("rewritten_query", ""),
                item_names=state.get("item_names"),
                image_urls=image_urls,
                sources=sources_for_db,
            )
            logger.info(f"【{NodeAnswerOutput.name}】MongoDB 写入完成 | session={sid}")
        except Exception as e:
            logger.warning(f"【{NodeAnswerOutput.name}】MongoDB 写入失败 | {type(e).__name__}: {e}")


    @staticmethod
    def _final_push(
        state: QueryGraphState, answer: str, image_urls: list[str],
    ) -> None:
        """最后一次 SSE 推送，触发前端图片渲染 + 来源参考区。"""
        sid = state.get("session_id") or ""
        if not state.get("is_stream", False) or not sid:
            return
        sources = build_sources(state.get("reranked_docs"))
        push_to_session(sid, SSEEvent.FINAL, {
            "answer": answer,
            "status": "completed",
            "image_urls": image_urls,
            "sources": sources,
        })
