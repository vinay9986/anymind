from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from anymind.agents.base import AgentContext
from anymind.agents.bedrock_middleware import BedrockToolResultSanitizer


class ChatAgent:
    name = "chat_agent"
    _SYSTEM_PROMPT = "You are a helpful assistant. Use tools when helpful."

    def build(self, context: AgentContext) -> Any:
        middleware = []
        if context.model_config.model_provider == "bedrock":
            middleware.append(BedrockToolResultSanitizer())
        return create_agent(
            context.model_client,
            context.tools,
            system_prompt=self._SYSTEM_PROMPT,
            middleware=middleware,
            checkpointer=context.checkpointer,
        )
