from langgraph.graph import StateGraph
from langgraph.constants import START, END

from app.import_process.agent.state import ImportGraphState
from app.import_process.agent.nodes.vector.node_bge_embedding import NodeBgeEmbedding
from app.import_process.agent.nodes.vector.node_import_milvus import NodeImportMilvus


def build_indexing_subgraph() -> StateGraph:
    builder = StateGraph(ImportGraphState)
    builder.add_node("bge", NodeBgeEmbedding())
    builder.add_node("milvus", NodeImportMilvus())
    builder.add_edge(START, "bge")
    builder.add_edge("bge", "milvus")
    builder.add_edge("milvus", END)
    return builder.compile()
