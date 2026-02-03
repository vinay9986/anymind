from __future__ import annotations

from typing import Any, Iterable, List

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import BaseMessage, ToolMessage


def _strip_ids_from_blocks(blocks: Iterable[Any]) -> List[Any]:
    cleaned: List[Any] = []
    for item in blocks:
        if isinstance(item, dict):
            new_item = dict(item)
            if new_item.get("type") == "tool_result":
                content = new_item.get("content")
                if isinstance(content, list):
                    new_item["content"] = _strip_ids_from_blocks(content)
            if new_item.get("type") == "text":
                new_item.pop("id", None)
                text_val = new_item.get("text")
                if isinstance(text_val, dict):
                    text_val = dict(text_val)
                    text_val.pop("id", None)
                    new_item["text"] = text_val
            cleaned.append(new_item)
        else:
            cleaned.append(item)
    return cleaned


class BedrockToolResultSanitizer(AgentMiddleware[dict, dict]):
    """Remove text.id fields from tool results before Bedrock invocation."""

    def before_model(self, state: dict, runtime: Any) -> dict[str, Any]:
        messages: List[BaseMessage] = list(state.get("messages", []))
        updated: List[BaseMessage] = []
        for message in messages:
            if isinstance(message, ToolMessage) and isinstance(message.content, list):
                cleaned = _strip_ids_from_blocks(message.content)
                updated.append(message.model_copy(update={"content": cleaned}))
            else:
                updated.append(message)
        return {"messages": updated}
