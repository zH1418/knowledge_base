from typing import TypedDict, Annotated, Any
from copy import deepcopy

from app.core.state_base import create_base_state


def _keep_first(a: Any, b: Any) -> Any:
    """Reducer: 并行写入同键时取第一个非默认值。"""
    if a:
        return a
    return b


class QueryGraphState(TypedDict, total=False):
    task_id: str
    session_id: str
    original_query: str
    history: list
    rewritten_query: Annotated[str, _keep_first]
    item_names: Annotated[list, _keep_first]
    candidate_scores: Annotated[list, _keep_first]
    user_message_id: str
    query_type: str
    overflow_entities: Annotated[list, _keep_first]
    overflow_max_select: int
    missing_names: Annotated[list, _keep_first]
    supplementary_names: Annotated[list, _keep_first]
    hyde_doc: Annotated[str, _keep_first]
    embedding_chunks: Annotated[list, _keep_first]
    hyde_embedding_chunks: Annotated[list, _keep_first]
    original_embedding_chunks: Annotated[list, _keep_first]
    web_search_docs: Annotated[list, _keep_first]
    rrf_chunks: Annotated[list, _keep_first]
    reranked_docs: Annotated[list, _keep_first]
    prompt: Annotated[str, _keep_first]
    answer: Annotated[str, _keep_first]
    expansion_note: Annotated[str, _keep_first]
    is_stream: bool


_DEFAULT_STATE: dict = {
    **create_base_state(),
    "session_id": "",
    "original_query": "",
    "history": [],
    "rewritten_query": "",
    "item_names": [],
    "candidate_scores": [],
    "user_message_id": "",
    "query_type": "search",
    "overflow_entities": [],
    "overflow_max_select": 0,
    "missing_names": [],
    "supplementary_names": [],
    "hyde_doc": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "original_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "expansion_note": "",
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    state = deepcopy(_DEFAULT_STATE)
    state.update(overrides)
    return state
