from __future__ import annotations

import json

import pytest

from anymind.runtime import json_parser


def test_normalize_json_quotes_handles_empty_valid_and_escaped_single_quotes() -> None:
    assert json_parser._normalize_json_quotes("") == ""
    assert json_parser._normalize_json_quotes('{"a": 1}') == '{"a": 1}'
    normalized = json_parser._normalize_json_quotes(r"{'quote': 'it\\'s fine'}")
    assert normalized == '{"quote": "it\\\\"s fine"}'


def test_raw_decode_first_value_empty_raises() -> None:
    with pytest.raises(json.JSONDecodeError, match="Empty response"):
        json_parser._raw_decode_first_value("   ")


@pytest.mark.asyncio
async def test_parse_json_robust_direct_success_and_extraction_raw_decode(
    monkeypatch,
) -> None:
    events = []
    monkeypatch.setattr(
        json_parser.log,
        "debug",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    monkeypatch.setattr(
        json_parser.log,
        "info",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    monkeypatch.setattr(
        json_parser.log,
        "warning",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    direct = await json_parser.parse_json_robust('{"a": 1}', context="direct")
    extracted = await json_parser.parse_json_robust(
        'prefix {"a": 1} trailing {"b": 2}',
        context="extraction_raw_decode",
    )

    assert direct == {"a": 1}
    assert extracted == {"a": 1}
    assert any(event == "parse_json_robust_direct_success" for event, _ in events)
    assert any(
        event == "parse_json_robust_extraction_raw_decode_success"
        for event, _ in events
    )


@pytest.mark.asyncio
async def test_parse_json_robust_repair_string_path(monkeypatch) -> None:
    monkeypatch.setattr(
        json_parser,
        "repair_json",
        lambda text, return_objects=False: '{"fixed": true}',
    )

    parsed = await json_parser.parse_json_robust(
        "prefix {'fixed': true, trailing",
        context="repair-string",
    )

    assert parsed == {"fixed": True}
