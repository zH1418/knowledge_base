import json
from typing import Any, Dict


def format_state(state: Dict[str, Any], indent: int = 4) -> str:

    return json.dumps(state, indent=indent, ensure_ascii=False)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

