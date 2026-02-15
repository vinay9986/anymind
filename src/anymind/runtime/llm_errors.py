from __future__ import annotations

import re
from typing import Any, Optional


_STATUS_PATTERN = re.compile(
    r"\b(?:HTTP|status code|status)\s*[:=]?\s*(\d{3})\b", re.IGNORECASE
)

_LLM_MODULE_HINTS = (
    "openai",
    "anthropic",
    "cohere",
    "mistral",
    "groq",
    "together",
    "vertexai",
    "google",
    "ai21",
    "bedrock",
    "botocore",
    "boto3",
    "azure",
    "litellm",
    "langchain_openai",
    "langchain_anthropic",
)

_LLM_CLASS_HINTS = (
    "APIStatus",
    "BadRequest",
    "RateLimit",
    "ServiceUnavailable",
    "InternalServer",
    "Authentication",
    "Permission",
    "Overload",
)


class LLMHTTPError(RuntimeError):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(f"LLM request failed with HTTP {status_code}: {message}")
        self.status_code = int(status_code)
        self.message = message


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_http_status(error: BaseException) -> Optional[int]:
    for attr in ("status_code", "http_status", "status"):
        code = _coerce_int(getattr(error, attr, None))
        if code is not None:
            return code

    response = getattr(error, "response", None)
    if response is not None:
        if hasattr(response, "status_code"):
            code = _coerce_int(getattr(response, "status_code", None))
            if code is not None:
                return code
        if isinstance(response, dict):
            code = _coerce_int(response.get("status_code"))
            if code is not None:
                return code
            meta = response.get("ResponseMetadata")
            if isinstance(meta, dict):
                code = _coerce_int(meta.get("HTTPStatusCode"))
                if code is not None:
                    return code

    message = str(error)
    match = _STATUS_PATTERN.search(message)
    if match:
        code = _coerce_int(match.group(1))
        if code is not None:
            return code

    return None


def _looks_like_llm_error(error: BaseException) -> bool:
    module = str(getattr(error.__class__, "__module__", "") or "").lower()
    if any(hint in module for hint in _LLM_MODULE_HINTS):
        return True
    name = str(getattr(error.__class__, "__name__", "") or "")
    if any(hint in name for hint in _LLM_CLASS_HINTS):
        return True
    return False


def raise_if_llm_http_error(error: BaseException, *, llm_only: bool = False) -> None:
    if isinstance(error, LLMHTTPError):
        raise error

    status = extract_http_status(error)
    if status is None:
        return
    if not (400 <= status <= 599):
        return
    if llm_only or _looks_like_llm_error(error):
        raise LLMHTTPError(status_code=status, message=str(error)) from error


async def safe_ainvoke(
    target: Any, *args: Any, llm_only: bool = True, **kwargs: Any
) -> Any:
    try:
        return await target.ainvoke(*args, **kwargs)
    except Exception as exc:
        raise_if_llm_http_error(exc, llm_only=llm_only)
        raise
