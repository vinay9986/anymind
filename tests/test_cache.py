from types import SimpleNamespace

import pytest

from anymind.config.schemas import CacheConfig
from anymind.runtime import cache as cache_module
from anymind.runtime.cache import InMemoryCache, RedisCache, create_cache


@pytest.mark.asyncio
async def test_inmemory_cache_expiration(monkeypatch) -> None:
    cache = InMemoryCache()
    monkeypatch.setattr(cache_module.time, "time", lambda: 1000.0)
    await cache.set("k", {"v": 1}, ttl_seconds=5)
    assert await cache.get("k") == {"v": 1}
    monkeypatch.setattr(cache_module.time, "time", lambda: 1006.0)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_redis_cache_serializes() -> None:
    store = {}

    class DummyRedis:
        async def get(self, key):
            return store.get(key)

        async def set(self, key, value, ex=None):
            store[key] = value

    cache = RedisCache(client=DummyRedis())
    await cache.set("k", {"v": 2}, ttl_seconds=10)
    assert await cache.get("k") == {"v": 2}


@pytest.mark.asyncio
async def test_create_cache_returns_none_without_redis() -> None:
    assert await create_cache(None) is None
    config = CacheConfig(redis_url=None)
    assert await create_cache(config) is None


@pytest.mark.asyncio
async def test_create_cache_handles_failed_ping(monkeypatch) -> None:
    class DummyRedis:
        async def ping(self):
            raise RuntimeError("fail")

    monkeypatch.setattr(cache_module.redis.Redis, "from_url", lambda url: DummyRedis())
    config = CacheConfig(redis_url="redis://localhost:6379/0")
    assert await create_cache(config) is None
