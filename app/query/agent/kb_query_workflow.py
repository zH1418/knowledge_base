from typing import Optional, Any, Iterator

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.query.agent.state import QueryGraphState
from app.query.graph.pre_search_subgraph import build_pre_search_subgraph
from app.query.graph.multi_search_subgraph import build_multi_search_subgraph
from app.query.graph.fusion_subgraph import build_fusion_subgraph
from app.query.graph.generation_subgraph import build_generation_subgraph
from app.core.decorators import log_execution, safe_execute
from app.core.logger import logger


def _route_after_pre_search(state: QueryGraphState) -> str:
    if state.get("answer") or state.get("query_type") == "chitchat":
        return "generation_subgraph"
    return "multi_search_subgraph"


class KBQueryWorkflow:

    def __init__(self):
        self._pre_search = build_pre_search_subgraph()
        self._multi_search = build_multi_search_subgraph()
        self._fusion = build_fusion_subgraph()
        self._generation = build_generation_subgraph()
        self._build_main_graph()
        self._compiled_app: Optional[Any] = None

    def _build_main_graph(self):
        self.workflow = StateGraph(QueryGraphState)

        self.workflow.add_node("pre_search", self._pre_search)
        self.workflow.add_node("multi_search", self._multi_search)
        self.workflow.add_node("fusion", self._fusion)
        self.workflow.add_node("generation", self._generation)

        self.workflow.add_edge(START, "pre_search")
        self.workflow.add_conditional_edges(
            "pre_search", _route_after_pre_search,
            {
                "generation_subgraph": "generation",
                "multi_search_subgraph": "multi_search",
            },
        )
        self.workflow.add_edge("multi_search", "fusion")
        self.workflow.add_edge("fusion", "generation")
        self.workflow.add_edge("generation", END)

    def compile(self):
        if self._compiled_app is None:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    @log_execution(component="query")
    @safe_execute(component="query", reraise=True)
    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        app = self.compile()
        if stream:
            logger.info("[QueryWorkflow] 流式模式启动")
            final_state: QueryGraphState = {}  # type: ignore[assignment]
            for chunk in app.stream(initial_state, subgraphs=True):
                _namespace, node_updates = chunk
                for _node_name, node_state in node_updates.items():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)
            return final_state
        return app.invoke(initial_state)
