from typing import Any



class KBException(Exception):
    error_code: str = "KB0000"

    def __init__(self, message: str = "", detail: dict[str, Any] | None = None):
        self.message = message
        self.detail = detail or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，适配 API 响应和结构化日志。"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.error_code}): {self.message}>"



class ServiceException(KBException):
    error_code = "KB1000"

    def __init__(
        self,
        service_name: str,
        message: str = "",
        detail: dict[str, Any] | None = None,
        original: Exception | None = None,
    ):
        self.service_name = service_name
        super().__init__(
            message=message or f"[{service_name}] 服务调用失败",
            detail=detail,
        )
        if original:
            self.__cause__ = original


class VectorDBException(ServiceException):
    """Milvus 向量数据库异常。"""
    error_code = "KB1001"

    def __init__(self, message: str = "", detail=None, original=None):
        super().__init__("Milvus", message, detail, original)


class DocumentDBException(ServiceException):
    """MongoDB 文档数据库异常。"""
    error_code = "KB1002"

    def __init__(self, message: str = "", detail=None, original=None):
        super().__init__("MongoDB", message, detail, original)


class ObjectStorageException(ServiceException):
    """MinIO 对象存储异常。"""
    error_code = "KB1003"

    def __init__(self, message: str = "", detail=None, original=None):
        super().__init__("MinIO", message, detail, original)


class LLMException(ServiceException):
    """大语言模型（DashScope）服务异常。"""
    error_code = "KB1004"

    def __init__(self, message: str = "", detail=None, original=None):
        super().__init__("DashScope", message, detail, original)


class EmbeddingException(ServiceException):
    """BGE-M3 嵌入模型异常。"""
    error_code = "KB1005"

    def __init__(self, message: str = "", detail=None, original=None):
        super().__init__("BGE-M3", message, detail, original)


class RerankerException(ServiceException):
    """BGE-Reranker 重排序模型异常。"""
    error_code = "KB1006"

    def __init__(self, message: str = "", detail=None, original=None):
        super().__init__("BGE-Reranker", message, detail, original)


class PDFParsingException(ServiceException):
    """MinerU PDF 解析服务异常。"""
    error_code = "KB1007"

    def __init__(self, message: str = "", detail=None, original=None):
        super().__init__("MinerU", message, detail, original)



class TransientServiceError(ServiceException):
    error_code = "KB1100"

    def __init__(self, service_name: str = "", message: str = "",
                 detail=None, original=None):
        super().__init__(service_name or "unknown",
                         message or "服务暂时不可用，可重试",
                         detail, original)


class PermanentServiceError(ServiceException):
    error_code = "KB1200"

    def __init__(self, service_name: str = "", message: str = "",
                 detail=None, original=None):
        super().__init__(service_name or "unknown",
                         message or "服务调用永久失败",
                         detail, original)



class BusinessException(KBException):
    """业务逻辑异常基类。"""
    error_code = "KB2000"


class DocumentNotFoundError(BusinessException):
    """文档未找到。"""
    error_code = "KB2001"

    def __init__(self, message: str = "文档未找到", detail=None):
        super().__init__(message, detail)


class KnowledgeBaseNotFoundError(BusinessException):
    """知识库未找到。"""
    error_code = "KB2002"

    def __init__(self, message: str = "知识库未找到", detail=None):
        super().__init__(message, detail)


class DocumentProcessingError(BusinessException):
    """文档处理流程错误。"""
    error_code = "KB2003"

    def __init__(self, message: str = "文档处理失败", detail=None):
        super().__init__(message, detail)


class QueryProcessingError(BusinessException):
    """查询检索流程错误。"""
    error_code = "KB2004"

    def __init__(self, message: str = "查询处理失败", detail=None):
        super().__init__(message, detail)


class TaskStateError(BusinessException):
    """任务状态异常（如非法状态转换）。"""
    error_code = "KB2005"

    def __init__(self, message: str = "任务状态异常", detail=None):
        super().__init__(message, detail)



class ValidationException(KBException):
    """输入校验异常基类。"""
    error_code = "KB3000"


class ConfigurationError(ValidationException):
    """配置错误（缺少必要配置项或配置值非法）。"""
    error_code = "KB3001"

    def __init__(self, message: str = "配置错误", detail=None):
        super().__init__(message, detail)


class InvalidStateError(ValidationException):
    """LangGraph 状态非法（缺少必要字段或字段类型错误）。"""
    error_code = "KB3002"

    def __init__(self, message: str = "状态数据非法", detail=None):
        super().__init__(message, detail)


class ParameterError(ValidationException):
    """函数参数非法。"""
    error_code = "KB3003"

    def __init__(self, message: str = "参数非法", detail=None):
        super().__init__(message, detail)



class InternalError(KBException):
    """内部未预期错误（通常表示 bug，需要开发人员介入）。"""
    error_code = "KB9999"

    def __init__(self, message: str = "服务器内部错误", detail=None):
        super().__init__(message, detail)




_BUILTIN_EXCEPTION_MAP: dict[type[Exception], type[KBException]] = {
    ConnectionError: TransientServiceError,
    TimeoutError: TransientServiceError,
    ValueError: ParameterError,
    KeyError: InvalidStateError,
    TypeError: ParameterError,
}


def get_builtin_exception_map() -> dict[type[Exception], type[KBException]]:
    """获取内置异常映射表（只读副本）。"""
    return dict(_BUILTIN_EXCEPTION_MAP)
