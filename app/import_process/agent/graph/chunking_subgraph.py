from langgraph.graph import StateGraph
from langgraph.constants import START, END

from app.import_process.agent.state import ImportGraphState
from app.import_process.agent.nodes.processing.node_document_split import NodeDocumentSplit
from app.import_process.agent.nodes.processing.node_image_split import NodeImageSplit
from app.import_process.agent.nodes.processing.node_item_name_recognition import NodeItemNameRecognition


def _should_image_split(state: ImportGraphState) -> str:
    """纯图片文档（chunks 为空且 image_chunks 非空）→ 进入图片切分。"""
    chunks = state.get("chunks") or []
    image_chunks = state.get("image_chunks") or []
    if not chunks and image_chunks:
        return "image_split"
    return "item_name"


def build_chunking_subgraph() -> StateGraph:
    builder = StateGraph(ImportGraphState)
    builder.add_node("doc_split", NodeDocumentSplit())
    builder.add_node("image_split", NodeImageSplit())
    builder.add_node("item_name", NodeItemNameRecognition())
    builder.add_edge(START, "doc_split")
    builder.add_conditional_edges("doc_split", _should_image_split, {
        "image_split": "image_split",
        "item_name": "item_name",
    })
    builder.add_edge("image_split", "item_name")
    builder.add_edge("item_name", END)
    return builder.compile()
