from langgraph.graph import StateGraph
from langgraph.constants import START, END

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.import_process.agent.graph.pre_processing_subgraph import build_pre_processing_subgraph
from app.import_process.agent.graph.chunking_subgraph import build_chunking_subgraph
from app.import_process.agent.graph.indexing_subgraph import build_indexing_subgraph


class KBImportWorkflow:
    def __init__(self):
        self._pre = build_pre_processing_subgraph()
        self._chunk = build_chunking_subgraph()
        self._index = build_indexing_subgraph()

    def compile(self) -> StateGraph:
        builder = StateGraph(ImportGraphState)
        builder.add_node("pre_processing", self._pre)
        builder.add_node("chunking", self._chunk)
        builder.add_node("indexing", self._index)
        builder.add_edge(START, "pre_processing")
        builder.add_edge("pre_processing", "chunking")
        builder.add_edge("chunking", "indexing")
        builder.add_edge("indexing", END)
        return builder.compile()

    def run(self, initial_state: ImportGraphState, stream: bool = False) -> ImportGraphState:
        """执行导入工作流。"""
        app = self.compile()
        if stream:
            logger.info("[ImportWorkflow] 流式模式启动")
            final_state: ImportGraphState = {}
            for chunk in app.stream(initial_state, subgraphs=True):
                _namespace, node_updates = chunk
                for _node_name, node_state in node_updates.items():
                    if isinstance(node_state, dict):
                        final_state.update(node_state)
            return final_state
        return app.invoke(initial_state)
