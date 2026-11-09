from typing import Union


def escape_milvus_string(value: Union[str, None]) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
