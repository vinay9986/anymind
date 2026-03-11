from __future__ import annotations

from typing import Any, Iterable

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, message_to_dict
from langchain_core.outputs import LLMResult

from anymind.runtime.session_context import get_session_id
from anymind.runtime.usage_store import get_usage_store


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
        self._record_usage(response)

    def _record_usage(self, response: LLMResult) -> None:
        session_id = get_session_id()
        if not session_id:
            return
        token_usage = None
        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage")
        if token_usage is None:
            for batch in response.generations or []:
                for generation in batch:
                    message = getattr(generation, "message", None)
                    usage = (
                        getattr(message, "usage_metadata", None) if message else None
                    )
                    if usage:
                        token_usage = usage
                        break
                if token_usage:
                    break
        if not token_usage:
            return
        input_tokens = token_usage.get(
            "prompt_tokens", token_usage.get("input_tokens", 0)
        )
        output_tokens = token_usage.get(
            "completion_tokens", token_usage.get("output_tokens", 0)
        )
        get_usage_store().add(
            session_id=session_id,
            model=self._model,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
        )

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
