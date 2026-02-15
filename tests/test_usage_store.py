import sys
from types import SimpleNamespace

from anymind.runtime import usage_store
from anymind.runtime.usage import UsageTotals


def test_inmemory_usage_store_add_and_get() -> None:
    store = usage_store.InMemoryUsageStore()
    store.add(session_id="s1", model="m1", input_tokens=5, output_tokens=7)
    store.add(session_id="s1", model="m1", input_tokens=3, output_tokens=1)
    store.add(session_id="s1", model="m2", input_tokens=2, output_tokens=4)

    snapshot = store.get("s1")
    assert isinstance(snapshot.totals, UsageTotals)
    assert snapshot.totals.input_tokens == 10
    assert snapshot.totals.output_tokens == 12
    assert snapshot.per_model["m1"].input_tokens == 8
    assert snapshot.per_model["m1"].output_tokens == 8
    assert snapshot.per_model["m2"].input_tokens == 2
    assert snapshot.per_model["m2"].output_tokens == 4


def test_inmemory_usage_store_empty_session() -> None:
    store = usage_store.InMemoryUsageStore()
    snapshot = store.get("")
    assert snapshot.totals.input_tokens == 0
    assert snapshot.totals.output_tokens == 0
    assert snapshot.per_model == {}


def test_inmemory_usage_store_ignores_empty_session_on_add() -> None:
    store = usage_store.InMemoryUsageStore()
    store.add(session_id="", model="m1", input_tokens=5, output_tokens=7)
    snapshot = store.get("")
    assert snapshot.totals.input_tokens == 0
    assert snapshot.per_model == {}


def test_build_store_falls_back_to_memory(monkeypatch) -> None:
    monkeypatch.setenv("ANYMIND_USAGE_REDIS_URL", "redis://example")

    class FakeRedisStore:
        def __init__(self, url: str) -> None:
            self._client = SimpleNamespace(
                ping=lambda: (_ for _ in ()).throw(RuntimeError("no redis"))
            )

    monkeypatch.setattr(usage_store, "RedisUsageStore", FakeRedisStore)
    store = usage_store._build_store()
    assert isinstance(store, usage_store.InMemoryUsageStore)


def test_get_usage_store_caches(monkeypatch) -> None:
    monkeypatch.delenv("ANYMIND_USAGE_REDIS_URL", raising=False)
    usage_store._STORE = None
    store1 = usage_store.get_usage_store()
    store2 = usage_store.get_usage_store()
    assert store1 is store2


def test_redis_usage_store_add_and_get(monkeypatch) -> None:
    class FakePipeline:
        def __init__(self, client) -> None:
            self._client = client
            self._ops = []

        def incrby(self, key, amount):
            self._ops.append(("incrby", key, int(amount)))
            return self

        def hincrby(self, key, field, amount):
            self._ops.append(("hincrby", key, field, int(amount)))
            return self

        def get(self, key):
            self._ops.append(("get", key))
            return self

        def hgetall(self, key):
            self._ops.append(("hgetall", key))
            return self

        def execute(self):
            results = []
            for op in self._ops:
                if op[0] == "incrby":
                    _, key, amount = op
                    self._client.kv[key] = self._client.kv.get(key, 0) + amount
                    results.append(self._client.kv[key])
                elif op[0] == "hincrby":
                    _, key, field, amount = op
                    bucket = self._client.hashes.setdefault(key, {})
                    bucket[field] = int(bucket.get(field, 0)) + amount
                    results.append(bucket[field])
                elif op[0] == "get":
                    _, key = op
                    results.append(self._client.kv.get(key))
                elif op[0] == "hgetall":
                    _, key = op
                    results.append(dict(self._client.hashes.get(key, {})))
            self._ops = []
            return results

    class FakeClient:
        def __init__(self) -> None:
            self.kv = {}
            self.hashes = {}

        def pipeline(self):
            return FakePipeline(self)

        def ping(self):
            return True

    class FakeRedis:
        @classmethod
        def from_url(cls, url, decode_responses=True):
            return FakeClient()

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedis))

    store = usage_store.RedisUsageStore("redis://fake")
    store.add(session_id="s1", model="m1", input_tokens=2, output_tokens=3)
    store.add(session_id="s1", model="m2", input_tokens=1, output_tokens=1)

    snapshot = store.get("s1")
    assert snapshot.totals.input_tokens == 3
    assert snapshot.totals.output_tokens == 4
    assert snapshot.per_model["m1"].input_tokens == 2
    assert snapshot.per_model["m1"].output_tokens == 3
    assert snapshot.per_model["m2"].input_tokens == 1
    assert snapshot.per_model["m2"].output_tokens == 1


def test_build_store_uses_redis_when_available(monkeypatch) -> None:
    monkeypatch.setenv("ANYMIND_USAGE_REDIS_URL", "redis://fake")

    class FakeClient:
        def ping(self):
            return True

    class FakeRedisUsageStore:
        def __init__(self, url: str) -> None:
            self._client = FakeClient()

    monkeypatch.setattr(usage_store, "RedisUsageStore", FakeRedisUsageStore)
    store = usage_store._build_store()
    assert isinstance(store, FakeRedisUsageStore)
