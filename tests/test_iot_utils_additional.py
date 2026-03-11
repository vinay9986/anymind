from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from anymind.agents import iot_utils
from anymind.runtime.evidence import EvidenceLedger, use_ledger
from anymind.runtime.usage import UsageTotals


class RoleFallbackMessage:
    def __init__(self, role: str, content: str) -> None:
        self.type = ""
        self.role = role
        self.content = content


@dataclass
class DummyMessage:
    usage_metadata: dict


def test_extract_user_input_tuple_and_dict_and_query_fallbacks() -> None:
    assert (
        iot_utils.extract_user_input(
            {"messages": [("assistant", "reply"), ("user", "tuple user")]}
        )
        == "tuple user"
    )
    assert (
        iot_utils.extract_user_input(
            {"messages": [{"role": "user", "content": "dict user"}]}
        )
        == "dict user"
    )
    assert iot_utils.extract_user_input({"query": "query fallback"}) == "query fallback"
    assert (
        iot_utils.extract_user_input({"message": "message fallback"})
        == "message fallback"
    )


def test_extract_conversation_messages_handles_role_fallback_and_ignores_blank() -> (
    None
):
    payload = {
        "messages": [
            ("", "skip"),
            RoleFallbackMessage("assistant", "reply"),
            {"type": "user", "content": "typed"},
            {"role": "", "type": "", "content": "skip"},
        ]
    }
    assert iot_utils.extract_conversation_messages(payload) == [
        ("assistant", "reply"),
        ("user", "typed"),
    ]


def test_build_conversation_query_empty_history_after_filtering() -> None:
    assert iot_utils.build_conversation_query([]) == ""
    assert (
        iot_utils.build_conversation_query([("system", "ignore"), ("user", "latest")])
        == "latest"
    )


@pytest.mark.asyncio
async def test_ensure_current_time_tool_supports_callable_coroutine_and_swallows_errors() -> (
    None
):
    calls: list[dict[str, str]] = []

    async def async_callable(payload):
        calls.append(payload)

    class Tool:
        name = "get_current_time"

        def __call__(self, payload):
            return async_callable(payload)

    class BrokenTool:
        name = "current_time"

        def invoke(self, payload):
            raise RuntimeError("boom")

    await iot_utils.ensure_current_time_tool([Tool()])
    await iot_utils.ensure_current_time_tool([BrokenTool()])
    await iot_utils.ensure_current_time_tool([object()])

    assert calls == [{"format": "iso", "timezone": "UTC"}]


def test_usage_counter_add_usage_none_and_budget_with_session(monkeypatch) -> None:
    counter = iot_utils.UsageCounter()
    counter.add_usage(None)
    counter.add_usage_list([])
    counter.add_from_messages(
        [DummyMessage({"prompt_tokens": 2, "completion_tokens": 3})]
    )

    monkeypatch.setattr(iot_utils, "get_session_id", lambda: "session-1")
    monkeypatch.setattr(
        iot_utils,
        "get_usage_store",
        lambda: SimpleNamespace(
            get=lambda session_id: SimpleNamespace(
                totals=UsageTotals(input_tokens=10, output_tokens=5)
            )
        ),
    )

    assert counter.total_tokens == 5
    assert iot_utils.budget_exhausted(counter, 10) is True
    assert iot_utils.budget_exhausted(counter, 20) is False


def test_tool_display_name_and_format_tool_catalog(monkeypatch) -> None:
    class NamedByToolName:
        tool_name = "tool-by-name"

    class NamedByDunder:
        __name__ = "tool-by-dunder"

    nameless = SimpleNamespace()
    multiline_tool = SimpleNamespace(name="search")
    single_tool = SimpleNamespace(name="time")

    monkeypatch.setattr(
        iot_utils,
        "require_tool_description",
        lambda tool, context: (
            "first line\nsecond line"
            if tool is multiline_tool
            else "short description" if tool is single_tool else ""
        ),
    )

    assert iot_utils._tool_display_name(NamedByToolName()) == "tool-by-name"
    assert iot_utils._tool_display_name(NamedByDunder()) == "tool-by-dunder"
    assert iot_utils._tool_display_name(nameless) == ""

    formatted = iot_utils._format_tool_catalog(
        [nameless, multiline_tool, single_tool],
        max_desc_chars=12,
    )
    truncated = iot_utils._format_tool_catalog(
        [single_tool],
        max_chars=10,
    )

    assert "Tool catalog:" in formatted
    assert "- search:" in formatted
    assert "first lin..." in formatted
    assert truncated.startswith("Tool catal") or truncated.startswith("- time:")


def test_tool_feedback_from_ledger_with_catalog_and_empty_records(monkeypatch) -> None:
    tool = SimpleNamespace(name="search")
    monkeypatch.setattr(
        iot_utils,
        "require_tool_description",
        lambda tool, context: "search docs",
    )

    ledger = EvidenceLedger()
    ledger.start_turn()
    ledger.add("search", {"q": 1}, "")

    with use_ledger(ledger):
        feedback = iot_utils.tool_feedback_from_ledger([tool], max_chars=50)

    assert "Tool catalog:" in feedback
    assert "No external tool results yet" in feedback


def test_try_load_embedder_success_and_similarity_edge_cases(
    monkeypatch, tmp_path
) -> None:
    model_path = tmp_path / "model.onnx"
    tokenizer_path = tmp_path / "tokenizer.json"
    model_path.write_text("model")
    tokenizer_path.write_text("tokenizer")

    created = {}

    class DummyEmbedder:
        def __init__(self, config):
            created["config"] = config

    monkeypatch.setenv("ONNX_MODEL_PATH", str(model_path))
    monkeypatch.setenv("ONNX_TOKENIZER_PATH", str(tokenizer_path))
    monkeypatch.setenv("ONNX_MAX_LENGTH", "64")
    monkeypatch.setattr(iot_utils, "OnnxSentenceEmbedder", DummyEmbedder)
    iot_utils.try_load_embedder.cache_clear()

    embedder = iot_utils.try_load_embedder()

    assert isinstance(embedder, DummyEmbedder)
    assert created["config"].max_length == 64
    assert iot_utils.pairwise_similarities(np.array([[1.0, 0.0]])) == []
    assert iot_utils.select_semantic_representative([], np.array([[1.0, 0.0]])) == ""
    assert (
        iot_utils.select_semantic_representative(["only"], np.array([[1.0, 0.0]]))
        == "only"
    )
