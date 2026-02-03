from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from anymind.agents.base import AgentContext


class ChatAgent:
    name = "chat_agent"
    _SYSTEM_PROMPT = "You are a helpful assistant. Use tools when helpful."

    def build(self, context: AgentContext) -> Any:
        return create_agent(
            context.model_client,
            context.tools,
            system_prompt=self._SYSTEM_PROMPT,
            checkpointer=context.checkpointer,
        )
