from __future__ import annotations

from typing import Any, Dict, Tuple

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
        if provider:
            client = init_chat_model(
                model=config.model, model_provider=provider, **config.model_parameters
            )
        else:
            client = init_chat_model(model=config.model, **config.model_parameters)
        self._cache[key] = client
        return client
