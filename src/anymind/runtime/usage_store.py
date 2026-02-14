from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import structlog

from anymind.runtime.usage import UsageTotals

_logger = structlog.get_logger("anymind.usage_store")


@dataclass
class UsageSnapshot:
    totals: UsageTotals
    per_model: Dict[str, UsageTotals]


class UsageStore:
    def add(self, *, session_id: str, model: str, input_tokens: int, output_tokens: int) -> None:
        raise NotImplementedError

    def get(self, session_id: str) -> UsageSnapshot:
        raise NotImplementedError


class InMemoryUsageStore(UsageStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: Dict[str, UsageTotals] = {}
        self._per_model: Dict[str, Dict[str, UsageTotals]] = {}

    def add(self, *, session_id: str, model: str, input_tokens: int, output_tokens: int) -> None:
        if not session_id:
            return
        with self._lock:
            totals = self._totals.setdefault(session_id, UsageTotals())
            totals.add(input_tokens, output_tokens)
            model_totals = self._per_model.setdefault(session_id, {}).setdefault(
                model, UsageTotals()
            )
            model_totals.add(input_tokens, output_tokens)

    def get(self, session_id: str) -> UsageSnapshot:
        if not session_id:
            return UsageSnapshot(UsageTotals(), {})
        with self._lock:
            totals = self._totals.get(session_id, UsageTotals())
            per_model = self._per_model.get(session_id, {})
            # Copy to avoid mutation outside lock
            totals_copy = UsageTotals(
                input_tokens=totals.input_tokens,
                output_tokens=totals.output_tokens,
            )
            per_model_copy = {
                name: UsageTotals(
                    input_tokens=totals.input_tokens,
                    output_tokens=totals.output_tokens,
                )
                for name, totals in per_model.items()
            }
        return UsageSnapshot(totals_copy, per_model_copy)


class RedisUsageStore(UsageStore):
    def __init__(self, redis_url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _totals_keys(self, session_id: str) -> tuple[str, str]:
        return f"{session_id}_input", f"{session_id}_output"

    def _model_key(self, session_id: str) -> str:
        return f"{session_id}_models"

    def add(self, *, session_id: str, model: str, input_tokens: int, output_tokens: int) -> None:
        if not session_id:
            return
        input_key, output_key = self._totals_keys(session_id)
        model_key = self._model_key(session_id)
        pipe = self._client.pipeline()
        pipe.incrby(input_key, int(input_tokens or 0))
        pipe.incrby(output_key, int(output_tokens or 0))
        pipe.hincrby(model_key, f"{model}:input", int(input_tokens or 0))
        pipe.hincrby(model_key, f"{model}:output", int(output_tokens or 0))
        pipe.execute()

    def get(self, session_id: str) -> UsageSnapshot:
        if not session_id:
            return UsageSnapshot(UsageTotals(), {})
        input_key, output_key = self._totals_keys(session_id)
        model_key = self._model_key(session_id)
        pipe = self._client.pipeline()
        pipe.get(input_key)
        pipe.get(output_key)
        pipe.hgetall(model_key)
        input_val, output_val, model_vals = pipe.execute()
        totals = UsageTotals(
            input_tokens=int(input_val or 0),
            output_tokens=int(output_val or 0),
        )
        per_model: Dict[str, UsageTotals] = {}
        if isinstance(model_vals, dict):
            for key, value in model_vals.items():
                if not key:
                    continue
                if key.endswith(":input"):
                    name = key[:-6]
                    per_model.setdefault(name, UsageTotals()).input_tokens = int(value or 0)
                elif key.endswith(":output"):
                    name = key[:-7]
                    per_model.setdefault(name, UsageTotals()).output_tokens = int(value or 0)
        return UsageSnapshot(totals, per_model)


_STORE: Optional[UsageStore] = None


def _build_store() -> UsageStore:
    redis_url = os.environ.get("ANYMIND_USAGE_REDIS_URL") or ""
    if redis_url:
        try:
            store = RedisUsageStore(redis_url)
            store._client.ping()
            _logger.info("usage_store_backend", backend="redis", url=redis_url)
            return store
        except Exception as exc:
            _logger.warning("usage_store_redis_unavailable", error=str(exc))
    _logger.info("usage_store_backend", backend="memory")
    return InMemoryUsageStore()


def get_usage_store() -> UsageStore:
    global _STORE
    if _STORE is None:
        _STORE = _build_store()
    return _STORE
