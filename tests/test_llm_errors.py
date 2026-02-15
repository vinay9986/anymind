import pytest

from anymind.runtime.llm_errors import (
    LLMHTTPError,
    _looks_like_llm_error,
    extract_http_status,
    raise_if_llm_http_error,
    safe_ainvoke,
)


class StatusError(Exception):
    def __init__(self, message: str, status_code=None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ResponseError(Exception):
    def __init__(self, message: str, response) -> None:
        super().__init__(message)
        self.response = response


class OpenAIError(Exception):
    __module__ = "openai.error"


class RateLimitError(Exception):
    pass


def test_extract_http_status_from_attrs() -> None:
    err = StatusError("boom", status_code="401")
    assert extract_http_status(err) == 401


def test_extract_http_status_from_response_dict() -> None:
    err = ResponseError("boom", {"status_code": 503})
    assert extract_http_status(err) == 503

    err = ResponseError("boom", {"ResponseMetadata": {"HTTPStatusCode": 429}})
    assert extract_http_status(err) == 429


def test_extract_http_status_from_message() -> None:
    err = StatusError("HTTP 404 while calling")
    assert extract_http_status(err) == 404


def test_looks_like_llm_error() -> None:
    assert _looks_like_llm_error(OpenAIError("boom")) is True
    assert _looks_like_llm_error(RateLimitError("boom")) is True
    assert _looks_like_llm_error(StatusError("boom")) is False


def test_raise_if_llm_http_error() -> None:
    with pytest.raises(LLMHTTPError):
        raise_if_llm_http_error(OpenAIError("HTTP 503"), llm_only=True)

    with pytest.raises(LLMHTTPError):
        raise_if_llm_http_error(StatusError("boom", status_code=500), llm_only=True)

    with pytest.raises(LLMHTTPError):
        raise_if_llm_http_error(RateLimitError("status code 429"), llm_only=False)

    raise_if_llm_http_error(StatusError("status 200", status_code=200), llm_only=True)


class AsyncTarget:
    def __init__(self, exc: Exception | None) -> None:
        self._exc = exc

    async def ainvoke(self, *args, **kwargs):
        if self._exc is not None:
            raise self._exc
        return {"ok": True}


@pytest.mark.anyio
async def test_safe_ainvoke_raises_llm_error() -> None:
    target = AsyncTarget(StatusError("HTTP 500", status_code=500))
    with pytest.raises(LLMHTTPError):
        await safe_ainvoke(target, llm_only=True)


@pytest.mark.anyio
async def test_safe_ainvoke_raises_original_error() -> None:
    target = AsyncTarget(StatusError("HTTP 500", status_code=500))
    with pytest.raises(StatusError):
        await safe_ainvoke(target, llm_only=False)


@pytest.mark.anyio
async def test_safe_ainvoke_success() -> None:
    target = AsyncTarget(None)
    result = await safe_ainvoke(target, llm_only=True)
    assert result == {"ok": True}
