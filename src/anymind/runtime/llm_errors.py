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


def _message_content(m: Any) -> str:
    if isinstance(m, tuple) and len(m) >= 2:
        return str(m[1] or "")
    return str(getattr(m, "content", "") or "")


def _set_message_content(m: Any, content: str) -> Any:
    if isinstance(m, tuple) and len(m) >= 2:
        return (m[0], content)
    try:
        return m.__class__(content=content)
    except Exception:
        return m


async def _compress_messages(messages: list, max_chars: int, model_client: Any) -> list:
    """Compress the largest message(s) until total fits within max_chars."""
    from anymind.agents.iot_utils import compress_tool_feedback, sanitize_for_llm

    result = list(messages)
    for _ in range(len(result)):
        total = sum(len(_message_content(m)) for m in result)
        if total <= max_chars:
            break
        sizes = [(i, len(_message_content(result[i]))) for i in range(len(result))]
        idx, size = max(sizes, key=lambda x: x[1])
        if size <= 100:
            break
        other_total = total - size
        budget = max(500, max_chars - other_total)
        if budget >= size:
            break
        content = sanitize_for_llm(_message_content(result[idx]))
        compressed = await compress_tool_feedback(content, budget, model_client)
        result[idx] = _set_message_content(result[idx], compressed)
    return result


class ContextGuardedLLM:
    """Proxy that compresses messages before every ainvoke call.

    Wrapping the model_client with this ensures that LangGraph's internal
    calls (which bypass safe_ainvoke) still get compression applied before
    hitting the LLM API — preventing context-length explosions.
    """

    def __init__(self, inner: Any, max_chars: Optional[int] = None) -> None:
        self._inner = inner
        self._max_chars = max_chars

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        from anymind.agents.iot_utils import DEFAULT_FEEDBACK_MAX_CHARS

        limit = self._max_chars or DEFAULT_FEEDBACK_MAX_CHARS
        if isinstance(messages, list):
            total = sum(len(_message_content(m)) for m in messages)
            if total > limit:
                messages = await _compress_messages(messages, limit, self._inner)
        return await self._inner.ainvoke(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


async def safe_ainvoke(
    target: Any,
    *args: Any,
    llm_only: bool = True,
    _compressing: bool = False,
    model_client: Any = None,
    **kwargs: Any,
) -> Any:
    from anymind.agents.iot_utils import DEFAULT_FEEDBACK_MAX_CHARS, sanitize_for_llm

    if not _compressing and args:
        first = args[0]

        # Direct LLM call: target IS the model_client, args[0] is a list of messages
        if llm_only and isinstance(first, list):
            messages = first
            total = sum(len(_message_content(m)) for m in messages)
            if total > DEFAULT_FEEDBACK_MAX_CHARS:
                messages = await _compress_messages(
                    messages, DEFAULT_FEEDBACK_MAX_CHARS, target
                )
                args = (messages,) + args[1:]

        # Agent call: args[0] is a dict with a "messages" key
        elif not llm_only and isinstance(first, dict) and "messages" in first:
            messages = list(first["messages"])
            # Always sanitize (strips null bytes / invalid chars that cause HTTP 400)
            sanitized: list = []
            for m in messages:
                if isinstance(m, tuple) and len(m) >= 2:
                    sanitized.append((m[0], sanitize_for_llm(str(m[1] or ""))))
                else:
                    content = sanitize_for_llm(_message_content(m))
                    sanitized.append(_set_message_content(m, content))
            messages = sanitized
            # Compress if over limit and a model_client is available for the call
            client = model_client or target
            total = sum(len(_message_content(m)) for m in messages)
            if total > DEFAULT_FEEDBACK_MAX_CHARS:
                messages = await _compress_messages(
                    messages, DEFAULT_FEEDBACK_MAX_CHARS, client
                )
            args = ({**first, "messages": messages},) + args[1:]

    try:
        return await target.ainvoke(*args, **kwargs)
    except Exception as exc:
        raise_if_llm_http_error(exc, llm_only=llm_only)
        raise
