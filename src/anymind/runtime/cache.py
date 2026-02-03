from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import redis.asyncio as redis

from anymind.config.schemas import CacheConfig


class CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[dict[str, Any]]: ...

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None: ...


@dataclass
class InMemoryCache:
    _store: dict[str, tuple[float, dict[str, Any]]]

    def __init__(self) -> None:
        self._store = {}

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        now = time.time()
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at < now:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, value)


@dataclass
class RedisCache:
    client: redis.Redis

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        raw = await self.client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        payload = json.dumps(value)
        await self.client.set(key, payload, ex=ttl_seconds)


async def create_cache(config: Optional[CacheConfig]) -> Optional[CacheBackend]:
    if config is None or not config.redis_url:
        return None
    client = redis.Redis.from_url(config.redis_url)
    try:
        await client.ping()
    except Exception:
        return None
    return RedisCache(client=client)
