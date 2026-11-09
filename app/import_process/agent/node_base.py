from app.core.node_base import NodeBase as CoreNodeBase
from app.import_process.agent.state import ImportGraphState


class NodeBase(CoreNodeBase):

    def __call__(self, state: ImportGraphState) -> ImportGraphState:
        return super().__call__(state)
