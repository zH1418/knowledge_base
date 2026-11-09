from langgraph.graph import StateGraph
from langgraph.constants import START, END

from app.import_process.agent.state import ImportGraphState
from app.import_process.agent.nodes.entry.node_entry import NodeEntry
from app.import_process.agent.nodes.parsing.node_pdf_to_md import NodePdfToMd
from app.import_process.agent.nodes.parsing.node_md_img import NodeMdImg


def build_pre_processing_subgraph() -> StateGraph:
    builder = StateGraph(ImportGraphState)
    builder.add_node("entry", NodeEntry())
    builder.add_node("pdf_to_md", NodePdfToMd())
    builder.add_node("md_img", NodeMdImg())
    builder.add_edge(START, "entry")
    builder.add_edge("entry", "pdf_to_md")
    builder.add_edge("pdf_to_md", "md_img")
    builder.add_edge("md_img", END)
    return builder.compile()
