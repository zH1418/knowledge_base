import asyncio
import atexit
import concurrent.futures
import os
import signal
import subprocess

from app.conf.mcp_config import mcp_config
from app.core.logger import logger
from app.query.agent.node_base import NodeBase
from app.query.agent.state import QueryGraphState

WEB_SNIPPET_MAX: int = 500  # 网络摘要长度上限，在产出源头截断


def _kill_orphan_npx():
    """退出时终止残留的 npx 进程，防止 Ctrl+C 卡死。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/IM", "npx.exe"],
                capture_output=True, timeout=5,
            )
        else:
            subprocess.run(
                ["pkill", "-f", "npx"],
                capture_output=True, timeout=5,
            )
    except Exception:
        pass


atexit.register(_kill_orphan_npx)


class NodeWebSearchMCP(NodeBase):
    name = "联网搜索"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query", "")
        query_type = state.get("query_type", "search")

        if not query:
            logger.warning(f"【{self.name}】rewritten_query 为空，跳过")
            return self._fallback_empty(state)

        logger.info(f"【{self.name}】{query_type} 模式 | 搜索词: {query[:80]}")

        docs = self._search_tavily(query)

        if query_type == "unknown":
            bailian_docs = self._search_bailian(query)
            docs = self._merge_dedup(docs, bailian_docs)

        state["web_search_docs"] = docs
        logger.info(f"【{self.name}】完成 | {len(docs)} 条结果")
        return state


    @staticmethod
    def _run_async(coro):
        """在同步方法中安全执行异步协程。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=30)


    def _search_tavily(self, query: str) -> list[dict]:
        cfg = mcp_config.servers.get("omnisearch")
        if not cfg:
            logger.warning(f"【{self.name}】omnisearch 未启用")
            return []

        return self._call_stdio_mcp(cfg, "web_search", {
            "provider": "tavily",
            "query": query,
            "limit": 5,
        })


    def _search_bailian(self, query: str) -> list[dict]:
        """百炼 MCP 搜索（原生 HTTP，绕过 streamablehttp_client 兼容问题）。"""
        import json

        cfg = mcp_config.servers.get("websearch")
        if not cfg:
            return []

        try:
            import httpx
            headers = {
                "Authorization": cfg.headers.get("Authorization", ""),
                "Content-Type": "application/json",
            }
            body = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "bailian_web_search",
                    "arguments": {"query": query, "count": 5},
                },
            }
            r = httpx.post(cfg.url, json=body, headers=headers, timeout=15)
            if r.status_code != 200:
                logger.warning(f"【{self.name}】百炼 HTTP {r.status_code}")
                return []
            raw_body = r.json()
            result = raw_body.get("result", {}) if isinstance(raw_body, dict) else {}
            content_list = result.get("content", [])
            if not content_list:
                return []
            raw_text = content_list[0].get("text", "") if content_list else ""
            if not raw_text:
                return []
            data = json.loads(raw_text)
            return [
                {
                    "title": (p.get("title") or "").strip() or "无标题",
                    "url": (p.get("url") or "").strip(),
                    "snippet": _trim(p.get("snippet", "")),
                    "source": "bailian",
                }
                for p in data.get("pages", [])
                if isinstance(p, dict) and (p.get("snippet") or "").strip()
            ]
        except Exception as e:
            logger.warning(f"【{self.name}】百炼失败 | {type(e).__name__}: {e}")
            return []


    def _call_stdio_mcp(
        self, cfg, tool_name: str, arguments: dict
    ) -> list[dict]:
        """通过 stdio MCP 客户端调用工具。"""
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters
        import os

        env = {**os.environ, **(cfg.env or {})}
        params = StdioServerParameters(
            command=cfg.command, args=cfg.args or [], env=env,
        )

        async def _run():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return self._parse_tool_result(result, source="tavily")

        try:
            return self._run_async(_run())
        except Exception as e:
            logger.warning(f"【{self.name}】stdio MCP 失败 | {type(e).__name__}: {e}")
            return []


    def _parse_tool_result(self, result, source: str) -> list[dict]:
        """将 MCP 工具返回值解析为标准文档列表。"""
        import json

        if not result or not result.content:
            return []

        text = result.content[0].text if result.content else ""
        if not text:
            return []

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"【{self.name}】无法解析 JSON: {text[:200]}")
            return []

        items = data.get("pages", []) if source != "tavily" else data
        if not isinstance(items, list):
            return []
        return [
            {
                "title": (item.get("title") or "").strip() or "无标题",
                "url": (item.get("url") or "").strip(),
                "snippet": _trim(item.get("snippet", "")),
                "source": source,
            }
            for item in items
            if isinstance(item, dict) and (item.get("snippet") or "").strip()
        ]


    @staticmethod
    def _merge_dedup(
        primary: list[dict], secondary: list[dict],
    ) -> list[dict]:
        """按 URL 去重合并，主结果优先，空 URL 视为独立条目。"""
        seen: set[str] = set()
        merged: list[dict] = []

        for doc in primary:
            url = doc.get("url", "")
            if not url:          # 空 URL 直接保留（独立条目）
                merged.append(doc)
                continue
            if url in seen:
                continue
            seen.add(url)
            merged.append(doc)

        for doc in secondary:
            url = doc.get("url", "")
            if not url:
                merged.append(doc)
                continue
            if url in seen:
                continue
            seen.add(url)
            merged.append(doc)

        return merged


    @staticmethod
    def _fallback_empty(state: QueryGraphState) -> QueryGraphState:
        """无可用搜索词时的空结果兜底。"""
        state["web_search_docs"] = []
        return state


def _trim(text: str) -> str:
    """网络摘要截断，在产出源头执行，不污染下游。"""
    s = (text or "").strip()
    return s if len(s) <= WEB_SNIPPET_MAX else s[:WEB_SNIPPET_MAX] + "..."
