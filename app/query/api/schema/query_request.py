from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    is_stream: bool = Field(default=False, description="是否流式返回")