from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MCPServerConfig:
    """单个 MCP 服务器的连接配置。"""
    name: str               # 服务器逻辑名称
    transport: str          # streamable_http | sse | stdio | websocket
    url: str = ""           # 服务器端点（stdio 模式可为空）
    command: str = ""       # stdio: 启动命令
    args: list = None       # stdio: 命令参数
    env: dict = None        # stdio: 进程环境变量
    headers: dict = None    # http: 鉴权头
    timeout: float = 60.0       # 请求超时
    sse_read_timeout: float = 300.0  # SSE 长连接读取超时

    def __post_init__(self):
        self.args = self.args or []
        self.env = self.env or {}
        self.headers = self.headers or {}


@dataclass
class MCPGlobalConfig:
    """MCP 全局配置 + 已注册的服务器列表。"""
    servers: dict[str, MCPServerConfig]
    default_timeout: float = 60.0
    sse_read_timeout: float = 300.0



def _build_servers() -> dict[str, MCPServerConfig]:
    """从环境变量构建 MCP 服务器配置列表。"""
    servers: dict[str, MCPServerConfig] = {}

    if os.getenv("MCP_OMNISEARCH_ENABLED", "").lower() in ("1", "true", "yes"):
        servers["omnisearch"] = MCPServerConfig(
            name="omnisearch",
            transport="stdio",
            command="npx",
            args=["mcp-omnisearch"],
            env={
                k: v for k, v in {
                    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
                    "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY", ""),
                    "JINA_AI_API_KEY": os.getenv("JINA_AI_API_KEY", ""),
                }.items() if v
            },
        )

    websearch_url = os.getenv("MCP_WEBSEARCH_URL", "")
    if websearch_url:
        servers["websearch"] = MCPServerConfig(
            name="websearch",
            transport="streamable_http",
            url=websearch_url,
            headers={"Authorization": f"Bearer {os.getenv('MCP_WEBSEARCH_API_KEY', '')}"},
        )

    return servers



mcp_config = MCPGlobalConfig(servers=_build_servers())
