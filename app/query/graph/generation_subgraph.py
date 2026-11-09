from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.query.agent.state import QueryGraphState
from app.query.agent.nodes.generation.node_answer_output import NodeAnswerOutput


def build_generation_subgraph() -> StateGraph:
    builder = StateGraph(QueryGraphState)
    builder.add_node("answer_output", NodeAnswerOutput())
    builder.add_edge(START, "answer_output")
    builder.add_edge("answer_output", END)
    return builder.compile()
