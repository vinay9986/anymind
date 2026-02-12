from __future__ import annotations

from langchain_core.messages import BaseMessage


def message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)
