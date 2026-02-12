from __future__ import annotations

from typing import Any, Iterable

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, message_to_dict
from langchain_core.outputs import LLMResult


def _serialize_messages(batch: Iterable[BaseMessage]) -> list[dict[str, Any]]:
    return [message_to_dict(message) for message in batch]


def _serialize_generations(result: LLMResult) -> list[list[dict[str, Any]]]:
    serialized: list[list[dict[str, Any]]] = []
    for batch in result.generations:
        row: list[dict[str, Any]] = []
        for generation in batch:
            message = getattr(generation, "message", None)
            if message is not None:
                row.append({"message": message_to_dict(message)})
            else:
                row.append({"text": getattr(generation, "text", "")})
        serialized.append(row)
    return serialized


class LLMRequestLogger(BaseCallbackHandler):
    def __init__(self, *, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        self._logger = structlog.get_logger("anymind.llm")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        payload = {
            "provider": self._provider,
            "model": self._model,
            "serialized": serialized,
            "messages": [_serialize_messages(batch) for batch in messages],
        }
        if kwargs:
            payload["params"] = kwargs
        self._logger.info("llm_request", **payload)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        payload: dict[str, Any] = {
            "provider": self._provider,
            "model": self._model,
            "response": _serialize_generations(response),
        }
        if response.llm_output:
            payload["llm_output"] = response.llm_output
        if kwargs:
            payload["params"] = kwargs
        self._logger.info("llm_response", **payload)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        payload: dict[str, Any] = {
            "provider": self._provider,
            "model": self._model,
            "error": repr(error),
        }
        if kwargs:
            payload["params"] = kwargs
        self._logger.error("llm_error", **payload)


def build_llm_logger_callback(*, provider: str, model: str) -> LLMRequestLogger:
    return LLMRequestLogger(provider=provider, model=model)
