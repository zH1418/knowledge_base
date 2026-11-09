import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv

from app.core.logger import logger
from app.core.decorators import log_execution, safe_execute, retry_on_failure
from app.core.exceptions import DocumentDBException, ConfigurationError

load_dotenv()


class HistoryMongoTool:
    """MongoDB 历史对话记录读写工具类。"""

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            if not self.mongo_url:
                raise ConfigurationError(
                    message="MongoDB 配置缺失",
                    detail={"missing": ["MONGO_URL"]},
                )
            if not self.db_name:
                self.db_name = "kb"

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logger.info(f"[MongoDB] 连接成功 | db={self.db_name}")
        except (ConfigurationError, DocumentDBException):
            raise
        except Exception as e:
            raise DocumentDBException(
                message=f"MongoDB 初始化失败: {e}",
                original=e,
            ) from e



_history_mongo_tool: Optional[HistoryMongoTool] = None

try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    logger.warning(f"[MongoDB] 模块预加载初始化失败（将在首次调用时重试）: {e}")


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取 HistoryMongoTool 单例（懒加载）。"""
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool



@log_execution(component="mongodb")
@safe_execute(component="mongodb", default_return=0, reraise=False)
def clear_history(session_id: str) -> int:
    mongo_tool = get_history_mongo_tool()
    result = mongo_tool.chat_message.delete_many({"session_id": session_id})
    logger.info(f"[MongoDB] 已删除 {result.deleted_count} 条记录 | session={session_id}")
    return result.deleted_count


@log_execution(component="mongodb")
@safe_execute(component="mongodb", default_return="", reraise=False)
def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: List[str] = None,
    image_urls: List[str] = None,
    sources: List[Dict] = None,
    message_id: str = None,
) -> str:
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "image_urls": image_urls,
        "sources": sources,
        "ts": ts,
    }

    mongo_tool = get_history_mongo_tool()

    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)
        return str(result.inserted_id)


@log_execution(component="mongodb")
@safe_execute(component="mongodb", default_return=0, reraise=False)
def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    mongo_tool = get_history_mongo_tool()
    object_ids = [ObjectId(i) for i in ids]
    result = mongo_tool.chat_message.update_many(
        {"_id": {"$in": object_ids}},
        {"$set": {"item_names": item_names}},
    )
    logger.info(
        f"[MongoDB] 已更新 {result.modified_count} 条记录的 item_names"
        f" -> {item_names}"
    )
    return result.modified_count


@log_execution(component="mongodb", log_result=True)
@safe_execute(component="mongodb", default_return=[], reraise=False)
def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    mongo_tool = get_history_mongo_tool()
    query = {"session_id": session_id}
    cursor = mongo_tool.chat_message.find(query).sort("ts", DESCENDING).limit(limit)
    messages = list(cursor)
    messages.reverse()
    return messages



if __name__ == "__main__":
    sid = "000015_hybrid"
    save_chat_message(sid, "user", "你好 (Hybrid)")
    save_chat_message(sid, "assistant", "你好！我是基于原生 Mongo + LangChain 对象的助手。")
    save_chat_message(sid, "user", "这个万用表怎么换电池？", item_names=["混合万用表"])

    print("--- 查询记录 ---")
    messages = get_recent_messages(sid, limit=5)
    print(f"查询到的记录数: {len(messages)}")
    for m in messages:
        print(f" {m}  ")
