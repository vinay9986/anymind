from __future__ import annotations

import pytest

from anymind.runtime.llm_errors import (
    LLMHTTPError,
    extract_http_status,
    raise_if_llm_http_error,
)


class HttpStatusError(Exception):
    def __init__(self, message: str, http_status=None) -> None:
        super().__init__(message)
        self.http_status = http_status


class ResponseObject:
    def __init__(self, status_code) -> None:
        self.status_code = status_code


class ResponseObjectError(Exception):
    def __init__(self, message: str, response) -> None:
        super().__init__(message)
        self.response = response


class PlainError(Exception):
    pass


def test_extract_http_status_additional_paths() -> None:
    assert extract_http_status(HttpStatusError("boom", http_status="418")) == 418
    assert (
        extract_http_status(
            ResponseObjectError("boom", ResponseObject(status_code="429"))
        )
        == 429
    )
    assert extract_http_status(PlainError("nothing here")) is None


def test_raise_if_llm_http_error_additional_paths() -> None:
    err = LLMHTTPError(status_code=500, message="boom")
    with pytest.raises(LLMHTTPError):
        raise_if_llm_http_error(err, llm_only=True)

    raise_if_llm_http_error(PlainError("status 500"), llm_only=False)
    raise_if_llm_http_error(PlainError("status 200"), llm_only=True)
