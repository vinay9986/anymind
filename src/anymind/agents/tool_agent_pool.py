from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator


class ToolAgentPool:
    def __init__(self, agents: list[Any]) -> None:
        if not agents:
            raise ValueError("ToolAgentPool requires at least one agent")
        self._agents = agents
        self._queue: asyncio.Queue[int] = asyncio.Queue(maxsize=len(agents))
        for idx in range(len(agents)):
            self._queue.put_nowait(idx)

    def agent_at(self, idx: int) -> Any:
        return self._agents[idx]

    @property
    def size(self) -> int:
        return len(self._agents)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        idx = await self._queue.get()
        try:
            yield self._agents[idx]
        finally:
            self._queue.put_nowait(idx)
