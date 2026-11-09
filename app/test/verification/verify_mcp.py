import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from app.conf.mcp_config import mcp_config


async def _test_stdio():
    """测试本地 stdio MCP 搜索（mcp-omnisearch）。"""
    srv = mcp_config.servers.get("omnisearch")
    if not srv:
        print("  omnisearch 未启用，跳过")
        return

    params = StdioServerParameters(
        command=srv.command, args=srv.args, env=srv.env,
    )
    print(f"  连接: omnisearch (stdio: {srv.command} {' '.join(srv.args)})")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"  PASS - {len(tools.tools)} 个工具: {[t.name for t in tools.tools]}")

            for provider in ["tavily", "firecrawl"]:  # jina 是处理工具非搜索
                r = await session.call_tool("web_search", {
                    "query": "Python RAG", "max_results": 1, "provider": provider,
                })
                ok = len(str(r.content[0].text)) > 10
                print(f"    {provider}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    print("\n============================================================")
    print("  MCP 搜索验证")
    print("============================================================")
    asyncio.run(_test_stdio())
    print("============================================================\n")
