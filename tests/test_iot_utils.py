import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from anymind.agents import iot_utils
from anymind.runtime.evidence import EvidenceLedger, use_ledger


@dataclass
class DummyMessage:
    usage_metadata: dict


class DummyTypeMessage:
    def __init__(self, role: str, content: str) -> None:
        self.type = role
        self.content = content


@pytest.mark.asyncio
async def test_ensure_current_time_tool_calls_async() -> None:
    class Tool:
        def __init__(self) -> None:
            self.name = "current_time"
            self.calls = []

        async def ainvoke(self, payload):
            self.calls.append(payload)

    tool = Tool()
    await iot_utils.ensure_current_time_tool([tool])
    assert tool.calls == [{"format": "iso", "timezone": "UTC"}]


@pytest.mark.asyncio
async def test_ensure_current_time_tool_noop_when_missing() -> None:
    await iot_utils.ensure_current_time_tool([])


def test_message_text_handles_list_content() -> None:
    msg = SimpleNamespace(content=["a", "b"])
    assert iot_utils.message_text(msg) == "a\nb"


def test_extract_user_input_from_messages() -> None:
    payload = {
        "messages": [
            ("user", "first"),
            ("assistant", "reply"),
            {"role": "user", "content": "second"},
        ]
    }
    assert iot_utils.extract_user_input(payload) == "second"


def test_extract_user_input_from_message_object() -> None:
    payload = {"messages": [DummyTypeMessage("user", "hello")]}
    assert iot_utils.extract_user_input(payload) == "hello"


def test_extract_user_input_fallback_fields() -> None:
    payload = {"input": "fallback"}
    assert iot_utils.extract_user_input(payload) == "fallback"


def test_extract_conversation_messages() -> None:
    payload = {
        "messages": [
            ("user", "first"),
            DummyTypeMessage("assistant", "reply"),
            {"role": "user", "content": "second"},
        ]
    }
    parsed = iot_utils.extract_conversation_messages(payload)
    assert parsed == [
        ("user", "first"),
        ("assistant", "reply"),
        ("user", "second"),
    ]


def test_extract_conversation_messages_empty() -> None:
    assert iot_utils.extract_conversation_messages({"messages": "nope"}) == []


def test_build_conversation_query() -> None:
    messages = [
        ("user", "hello"),
        ("assistant", "hi"),
        ("user", "latest"),
    ]
    query = iot_utils.build_conversation_query(messages)
    assert "Conversation so far" in query
    assert "Latest user question" in query
    assert "latest" in query


def test_build_conversation_query_no_user() -> None:
    messages = [("assistant", "only assistant")]
    assert iot_utils.build_conversation_query(messages) == "only assistant"


@pytest.mark.asyncio
async def test_ensure_current_time_tool_calls_sync() -> None:
    class Tool:
        def __init__(self) -> None:
            self.name = "current_time"
            self.calls = []

        def invoke(self, payload):
            self.calls.append(payload)
            return None

    tool = Tool()
    await iot_utils.ensure_current_time_tool([tool])
    assert tool.calls == [{"format": "iso", "timezone": "UTC"}]


def test_usage_counter_add_and_budget() -> None:
    counter = iot_utils.UsageCounter()
    counter.add(3, 4)
    counter.add_usage({"input_tokens": 1, "output_tokens": 2})
    counter.add_usage_list([{"prompt_tokens": 5, "completion_tokens": 1}])
    counter.add_from_messages([DummyMessage({"input_tokens": 2, "output_tokens": 3})])
    assert counter.input_tokens == 3 + 1 + 5 + 2
    assert counter.output_tokens == 4 + 2 + 1 + 3
    assert counter.total_tokens == counter.input_tokens + counter.output_tokens
    assert iot_utils.budget_exhausted(counter, counter.total_tokens - 1)
    assert not iot_utils.budget_exhausted(counter, None)


def test_tool_feedback_from_ledger() -> None:
    ledger = EvidenceLedger()
    ledger.add("tool", {"q": 1}, "result")
    ledger.start_turn()
    ledger.add("tool2", {"q": 2}, "new result")
    with use_ledger(ledger):
        feedback = iot_utils.tool_feedback_from_ledger(max_chars=1000)
    assert "tool2" in feedback
    assert "new result" in feedback


def test_tool_feedback_from_ledger_no_records() -> None:
    ledger = EvidenceLedger()
    ledger.start_turn()
    with use_ledger(ledger):
        feedback = iot_utils.tool_feedback_from_ledger()
    assert "No external tool results yet" in feedback


def test_tool_feedback_from_ledger_no_ledger() -> None:
    assert "No external tool results yet" in iot_utils.tool_feedback_from_ledger()


def test_truncate_and_iteration_temperature() -> None:
    assert iot_utils.truncate_text("hello", 2) == "hell..."
    assert iot_utils.truncate_text("hello", 10) == "hello"
    assert iot_utils.truncate_text("hello", 0) == ""
    assert iot_utils.iteration_temperature(1) == 0.2
    assert iot_utils.iteration_temperature(5) > 0.2


def test_try_load_embedder_returns_none_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ONNX_MODEL_PATH", str(tmp_path / "missing.onnx"))
    monkeypatch.setenv("ONNX_TOKENIZER_PATH", str(tmp_path / "missing.json"))
    iot_utils.try_load_embedder.cache_clear()
    assert iot_utils.try_load_embedder() is None


def test_pairwise_similarities_and_representative() -> None:
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    sims = iot_utils.pairwise_similarities(embeddings)
    assert len(sims) == 3
    answer = iot_utils.select_semantic_representative(["a", "b", "c"], embeddings)
    assert answer in {"a", "b"}
