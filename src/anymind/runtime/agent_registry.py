from __future__ import annotations

from typing import Dict, Type

from anymind.agents.base import BaseAgent
from anymind.agents.chat_agent import ChatAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self.register(ChatAgent())

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not registered")
        return self._agents[name]

    def list(self) -> list[str]:
        return sorted(self._agents.keys())
