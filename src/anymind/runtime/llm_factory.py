from __future__ import annotations

from typing import Any, Dict, Tuple

from anymind.runtime.llm_logging import build_llm_logger_callback

from langchain.chat_models import init_chat_model

from anymind.config.schemas import ModelConfig


class LLMFactory:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, Tuple[Tuple[str, Any], ...]], Any] = {}

    def get(self, config: ModelConfig) -> Any:
        provider = config.model_provider or ""
        params = tuple(sorted(config.model_parameters.items()))
        key = (config.model, provider, params)
        if key in self._cache:
            return self._cache[key]
        model_parameters = dict(config.model_parameters)
        logger_callback = build_llm_logger_callback(
            provider=provider or "unknown", model=config.model
        )
        callbacks = model_parameters.get("callbacks")
        if callbacks is None:
            model_parameters["callbacks"] = [logger_callback]
        elif hasattr(callbacks, "add_handler"):
            callbacks.add_handler(logger_callback)
        elif isinstance(callbacks, (list, tuple)):
            model_parameters["callbacks"] = [*list(callbacks), logger_callback]
        else:
            model_parameters["callbacks"] = [callbacks, logger_callback]
        if provider:
            client = init_chat_model(
                model=config.model, model_provider=provider, **model_parameters
            )
        else:
            client = init_chat_model(model=config.model, **model_parameters)
        self._cache[key] = client
        return client
