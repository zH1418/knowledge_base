from typing import Any


def format_history_text(
    history: list[dict[str, Any]], max_entries: int = 6
) -> str:
    if not history:
        return "（无历史记录）"
    lines: list[str] = []
    for msg in history[-max_entries:]:
        role = msg.get("role", "")
        text = msg.get("text", "")
        if not text:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {text}")
    return "\n".join(lines) if lines else "（无历史记录）"
