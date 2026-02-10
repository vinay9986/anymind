import json

import pytest

from anymind.runtime.json_parser import extract_json_from_response, parse_json_robust


def test_extract_json_from_response_fenced() -> None:
    text = 'Here is data: ```json {"a": 1}```'
    extracted = extract_json_from_response(text)
    assert json.loads(extracted) == {"a": 1}


def test_extract_json_from_response_single_quotes() -> None:
    text = "prefix {'a': 1, 'b': 'x'} suffix"
    extracted = extract_json_from_response(text)
    assert json.loads(extracted) == {"a": 1, "b": "x"}


def test_extract_json_from_response_no_braces() -> None:
    text = "  just text  "
    extracted = extract_json_from_response(text)
    assert extracted == "just text"


@pytest.mark.asyncio
async def test_parse_json_robust_handles_mixed_text() -> None:
    text = 'Result: {"a": 1, "b": [2,3]}'
    parsed = await parse_json_robust(text, context="test")
    assert parsed == {"a": 1, "b": [2, 3]}


@pytest.mark.asyncio
async def test_parse_json_robust_raw_decode() -> None:
    text = '{"a": 1} trailing'
    parsed = await parse_json_robust(text, context="test")
    assert parsed == {"a": 1}


@pytest.mark.asyncio
async def test_parse_json_robust_repair() -> None:
    text = "{'a': 1,}"
    parsed = await parse_json_robust(text, context="test")
    assert parsed == {"a": 1}


@pytest.mark.asyncio
async def test_parse_json_robust_returns_non_string() -> None:
    payload = {"a": 1}
    parsed = await parse_json_robust(payload, context="test")
    assert parsed is payload
