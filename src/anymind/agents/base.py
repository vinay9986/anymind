from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver

from anymind.config.schemas import ModelConfig, PricingConfig
from anymind.policies.tool_policy import ToolPolicy


@dataclass
class AgentContext:
    model_config: ModelConfig
    pricing: PricingConfig
    tools: list[Any]
    tool_policy: ToolPolicy
    model_client: Any
    checkpointer: BaseCheckpointSaver


class BaseAgent(Protocol):
    name: str

    def build(self, context: AgentContext) -> Any: ...
