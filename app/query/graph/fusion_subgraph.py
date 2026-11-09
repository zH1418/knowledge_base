from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.query.agent.state import QueryGraphState
from app.query.agent.nodes.fusion.node_rrf import NodeRRF
from app.query.agent.nodes.fusion.node_rerank import NodeRerank


def build_fusion_subgraph() -> StateGraph:
    builder = StateGraph(QueryGraphState)
    builder.add_node("rrf", NodeRRF())
    builder.add_node("rerank", NodeRerank())
    builder.add_edge(START, "rrf")
    builder.add_edge("rrf", "rerank")
    builder.add_edge("rerank", END)
    return builder.compile()
