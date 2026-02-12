from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Tuple

from anymind.runtime.session import Session
from anymind.runtime.session_factory import SessionFactory


@dataclass
class SessionEntry:
    session: Session
    last_used: float


class SessionStore:
    def __init__(
        self, *, session_factory: SessionFactory, max_sessions: int = 128
    ) -> None:
        self._session_factory = session_factory
        self._max_sessions = max(1, int(max_sessions))
        self._sessions: Dict[Tuple[str, str], SessionEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, *, agent_name: str, thread_id: str) -> Session:
        key = (agent_name, thread_id)
        async with self._lock:
            entry = self._sessions.get(key)
            if entry is not None:
                entry.last_used = time.time()
                return entry.session

            session = await self._session_factory.create_session(agent_name=agent_name)
            self._sessions[key] = SessionEntry(session=session, last_used=time.time())
            await self._evict_if_needed()
            return session

    async def _evict_if_needed(self) -> None:
        if len(self._sessions) <= self._max_sessions:
            return
        overflow = len(self._sessions) - self._max_sessions
        if overflow <= 0:
            return
        oldest = sorted(self._sessions.items(), key=lambda item: item[1].last_used)
        for key, entry in oldest[:overflow]:
            await entry.session.close()
            self._sessions.pop(key, None)

    async def close_all(self) -> None:
        async with self._lock:
            entries = list(self._sessions.values())
            self._sessions.clear()
        for entry in entries:
            await entry.session.close()
