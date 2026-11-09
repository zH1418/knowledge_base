from app.core.node_base import NodeBase as CoreNodeBase
from app.query.agent.state import QueryGraphState


class NodeBase(CoreNodeBase):

    def __call__(self, state: QueryGraphState) -> QueryGraphState:
        result = super().__call__(state)
        sid = state.get("session_id")
        if state.get("is_stream") and sid:
            try:
                from app.query.utils.sse_utils import push_to_session, SSEEvent
                from app.utils.task_utils import get_done_task_list
                done = get_done_task_list(sid) or []
                push_to_session(sid, SSEEvent.PROGRESS, {
                    "done_list": done,
                    "running_list": [],
                    "status": "processing",
                })
            except Exception:
                pass
        return result
