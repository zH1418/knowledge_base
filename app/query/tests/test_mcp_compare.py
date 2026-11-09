import asyncio, os
from dotenv import load_dotenv
load_dotenv()


async def test_local_omnisearch():
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp import ClientSession

    print("--- 本地 mcp-omnisearch (Tavily) ---")
    env = {
        **os.environ,
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY", ""),
    }
    params = StdioServerParameters(
        command=os.getenv("NPX_CMD", "npx"),
        args=["mcp-omnisearch"],
        env=env,
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                print(f"  工具: {names}")

                if "web_search" in names:
                    result = await session.call_tool(
                        "web_search", {"query": "万用表使用方法"}
                    )
                else:
                    result = await session.call_tool(
                        names[0], {"query": "万用表使用方法"}
                    )
                content = result.content[0].text if result.content else ""
                print(f"  搜索'万用表使用方法': {len(content)} chars")
                print(f"  {content[:600]}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")


async def test_remote_bailian():
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    url = os.getenv("MCP_WEBSEARCH_URL", "")
    api_key = os.getenv("MCP_WEBSEARCH_API_KEY", "")

    print(f"\n--- 远程百炼 MCP (streamable HTTP) ---")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    print(f"  URL: {url[:60]}...")
    print(f"  Key: {api_key[:12]}...")

    try:
        import httpx
        r = httpx.get(url, headers=headers, timeout=10)
        print(f"  HTTP直接访问: {r.status_code}")
        print(f"  响应头: {dict(r.headers)[:5]}")
        print(f"  响应体前200: {r.text[:200]}")
    except Exception as e:
        print(f"  HTTP直接访问失败: {type(e).__name__}: {str(e)[:200]}")

    try:
        async with streamablehttp_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"  连接成功! 工具: {[t.name for t in tools.tools]}")
    except Exception as e:
        print(f"  streamablehttp_client 失败: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(test_local_omnisearch())
    asyncio.run(test_remote_bailian())
