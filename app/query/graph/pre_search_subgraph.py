from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.query.agent.state import QueryGraphState
from app.query.agent.nodes.rewrite.node_query_understand import NodeQueryUnderstand


def build_pre_search_subgraph() -> StateGraph:
    builder = StateGraph(QueryGraphState)
    builder.add_node("query_understand", NodeQueryUnderstand())
    builder.add_edge(START, "query_understand")
    builder.add_edge("query_understand", END)
    return builder.compile()
