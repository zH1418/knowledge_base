from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.query.agent.state import QueryGraphState
from app.query.agent.nodes.search.node_search_embedding import NodeSearchEmbedding
from app.query.agent.nodes.search.node_web_search_mcp import NodeWebSearchMCP


def build_multi_search_subgraph() -> StateGraph:
    builder = StateGraph(QueryGraphState)

    builder.add_node("search_embedding", NodeSearchEmbedding())
    builder.add_node("search_web", NodeWebSearchMCP())

    builder.add_edge(START, "search_embedding")
    builder.add_edge(START, "search_web")

    builder.add_edge("search_embedding", END)
    builder.add_edge("search_web", END)

    return builder.compile()
