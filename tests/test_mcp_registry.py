from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from anymind.config.schemas import MCPConfig
from anymind.runtime.cache import InMemoryCache
from anymind.runtime.evidence import EvidenceLedger, use_ledger
from anymind.runtime.mcp_registry import (
    MCPRegistryFactory,
    bedrock_tool_interceptor,
    cache_tool_interceptor,
    confirm_tool_interceptor,
    _cache_key,
    _default_mcp_env,
    evidence_tool_interceptor,
    tool_call_logging_interceptor,
    tool_error_logging_interceptor,
    resolve_mcp_config,
    tool_result_to_text,
)


def test_resolve_mcp_config_resolves_paths_and_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KAGI_API_KEY", "kagi-key")
    monkeypatch.setenv("ONNX_MODEL_PATH", "models/model.onnx")

    raw = MCPConfig(
        servers={
            "test": {
                "transport": "stdio",
                "args": ["./script.py", "--flag"],
                "env": {"CUSTOM": "1"},
            }
        }
    )
    resolved = resolve_mcp_config(raw, tmp_path)
    args = resolved["test"]["args"]
    assert str(tmp_path) in args[0]
    env = resolved["test"]["env"]
    assert env["KAGI_API_KEY"] == "kagi-key"
    assert env["CUSTOM"] == "1"
    assert env["ONNX_MODEL_PATH"].startswith(str(tmp_path))


def test_resolve_mcp_config_includes_default_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KAGI_API_KEY", "kagi-key")
    raw = MCPConfig(
        servers={
            "test": {
                "transport": "stdio",
                "args": ["script.py"],
                "env": None,
            }
        }
    )
    resolved = resolve_mcp_config(raw, tmp_path)
    env = resolved["test"]["env"]
    assert env["KAGI_API_KEY"] == "kagi-key"


def test_default_mcp_env_filters_empty_values(monkeypatch) -> None:
    monkeypatch.setenv("KAGI_API_KEY", "kagi-key")
    monkeypatch.setenv("SCRAPFLY_API_KEY", "")
    env = _default_mcp_env()
    assert env["KAGI_API_KEY"] == "kagi-key"
    assert "SCRAPFLY_API_KEY" not in env


def test_resolve_mcp_config_ignores_non_dict_entries(tmp_path) -> None:
    raw = SimpleNamespace(
        servers={"good": {"transport": "stdio", "args": ["tool.py"]}, "bad": "nope"}
    )
    resolved = resolve_mcp_config(raw, tmp_path)
    assert "good" in resolved
    assert "bad" not in resolved


def test_resolve_mcp_config_preserves_non_stdio_env(tmp_path) -> None:
    raw = MCPConfig(
        servers={
            "http": {
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "env": {"ONNX_MODEL_PATH": "./model.onnx"},
            }
        }
    )
    resolved = resolve_mcp_config(raw, tmp_path)
    assert resolved["http"]["env"]["ONNX_MODEL_PATH"] == "./model.onnx"


def test_resolve_mcp_config_preserves_env_placeholders(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ONNX_MODEL_PATH", "$MODEL_PATH")
    raw = MCPConfig(
        servers={"test": {"transport": "stdio", "args": ["./tool.py"], "env": None}}
    )
    resolved = resolve_mcp_config(raw, tmp_path)
    assert resolved["test"]["env"]["ONNX_MODEL_PATH"] == "$MODEL_PATH"


def test_tool_result_to_text_handles_toolmessage() -> None:
    msg = ToolMessage(content=[{"type": "text", "text": "hello"}], tool_call_id="1")
    assert tool_result_to_text(msg) == "hello"


class DummyResult:
    def __init__(self, content) -> None:
        self.content = content


class DummyContentItem:
    def __init__(self, item_type: str, **kwargs) -> None:
        self.type = item_type
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_tool_result_to_text_preserves_concat_blob_payload() -> None:
    result = DummyResult(
        [
            {
                "type": "json",
                "json": {"concat_blob": "blob"},
            }
        ]
    )
    text = tool_result_to_text(result)
    assert '"concat_blob"' in text
    assert "blob" in text


def test_tool_result_to_text_preserves_matches_payload() -> None:
    result = DummyResult(
        [
            {
                "type": "json",
                "json": {
                    "matches": [
                        {"text": "alpha"},
                        {"snippet": "beta"},
                    ]
                },
            }
        ]
    )
    text = tool_result_to_text(result)
    assert "alpha" in text
    assert "beta" in text


def test_tool_result_to_text_handles_json_payload() -> None:
    result = DummyResult([{"type": "json", "json": {"value": 1}}])
    assert '"value": 1' in tool_result_to_text(result)


def test_tool_result_to_text_handles_custom_content_items() -> None:
    result = DummyResult(
        [
            DummyContentItem("text", text="hello"),
            DummyContentItem("json", json={"value": 2}),
            DummyContentItem("binary", data="blob"),
        ]
    )
    text = tool_result_to_text(result)
    assert '"type": "json"' in text
    assert '"value": 2' in text
    assert '"data": "blob"' in text


def test_tool_result_to_text_handles_toolmessage_dict_content() -> None:
    msg = ToolMessage(content="placeholder", tool_call_id="1")
    msg.content = {"value": 1}
    assert tool_result_to_text(msg) == '{"value": 1}'


def test_tool_result_to_text_handles_non_serializable_toolmessage_dict() -> None:
    msg = ToolMessage(content="placeholder", tool_call_id="1")
    msg.content = {"value": {1, 2}}
    text = tool_result_to_text(msg)
    assert "value" in text


def test_cache_key_is_stable() -> None:
    assert _cache_key("search", {"b": 2, "a": 1}) == _cache_key(
        "search", {"a": 1, "b": 2}
    )


@pytest.mark.asyncio
async def test_cache_tool_interceptor_caches_text() -> None:
    cache = InMemoryCache()
    handler_calls = {"count": 0}

    async def handler(_request):
        handler_calls["count"] += 1
        return ToolMessage(content="cached", tool_call_id="tool-id")

    request = SimpleNamespace(name="tool", args={"q": 1}, runtime=SimpleNamespace())
    interceptor = cache_tool_interceptor(cache, ttl_seconds=60)

    first = await interceptor(request, handler)
    second = await interceptor(request, handler)

    assert handler_calls["count"] == 1
    assert tool_result_to_text(first) == "cached"
    assert tool_result_to_text(second) == "cached"


@pytest.mark.asyncio
async def test_cache_tool_interceptor_skips_non_text() -> None:
    cache = InMemoryCache()
    handler_calls = {"count": 0}

    async def handler(_request):
        handler_calls["count"] += 1
        return ToolMessage(
            content=[{"type": "json", "json": {"value": 1}}], tool_call_id="id"
        )

    request = SimpleNamespace(name="tool", args={"q": 2}, runtime=SimpleNamespace())
    interceptor = cache_tool_interceptor(cache, ttl_seconds=60)
    await interceptor(request, handler)
    await interceptor(request, handler)
    assert handler_calls["count"] == 2


@pytest.mark.asyncio
async def test_evidence_tool_interceptor_records() -> None:
    ledger = EvidenceLedger()

    async def handler(_request):
        return ToolMessage(content="result", tool_call_id="tool-id")

    request = SimpleNamespace(name="tool", args={"q": 1}, runtime=SimpleNamespace())
    with use_ledger(ledger):
        await evidence_tool_interceptor(request, handler)

    records = ledger.all()
    assert len(records) == 1
    assert records[0].tool == "tool"
    assert records[0].content == "result"


@pytest.mark.asyncio
async def test_bedrock_tool_interceptor_formats_text() -> None:
    async def handler(_request):
        return ToolMessage(
            content=[{"type": "text", "text": "hello", "id": "ignore"}],
            tool_call_id="orig",
        )

    request = SimpleNamespace(
        name="tool", args={}, runtime=SimpleNamespace(tool_call_id="call-id")
    )
    result = await bedrock_tool_interceptor(request, handler)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-id"
    assert tool_result_to_text(result) == "hello"


@pytest.mark.asyncio
async def test_confirm_tool_interceptor_denies(monkeypatch) -> None:
    async def handler(_request):
        return ToolMessage(content="ok", tool_call_id="id")

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    request = SimpleNamespace(
        name="tool", args={}, runtime=SimpleNamespace(tool_call_id="id")
    )
    result = await confirm_tool_interceptor(request, handler)
    assert tool_result_to_text(result) == "Tool call denied by user."


@pytest.mark.asyncio
async def test_confirm_tool_interceptor_allows(monkeypatch) -> None:
    async def handler(_request):
        return ToolMessage(content="ok", tool_call_id="id")

    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    request = SimpleNamespace(
        name="tool", args={}, runtime=SimpleNamespace(tool_call_id="id")
    )
    result = await confirm_tool_interceptor(request, handler)
    assert tool_result_to_text(result) == "ok"


@pytest.mark.asyncio
async def test_confirm_tool_interceptor_handles_eof(monkeypatch) -> None:
    async def handler(_request):
        raise AssertionError("handler should not run")

    def _raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    request = SimpleNamespace(
        name="tool", args={}, runtime=SimpleNamespace(tool_call_id="id")
    )
    result = await confirm_tool_interceptor(request, handler)
    assert tool_result_to_text(result) == "Tool call denied by user."


@pytest.mark.asyncio
async def test_tool_error_logging_interceptor_logs_and_reraises(monkeypatch) -> None:
    events = []

    async def handler(_request):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "anymind.runtime.mcp_registry.logger.exception",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    request = SimpleNamespace(
        name="tool", args={"q": 1}, runtime=SimpleNamespace(tool_call_id="id")
    )
    with pytest.raises(RuntimeError, match="boom"):
        await tool_error_logging_interceptor(request, handler)
    assert events[0][0] == "tool_call_failed"
    assert events[0][1]["tool"] == "tool"


@pytest.mark.asyncio
async def test_tool_call_logging_interceptor_logs_success(monkeypatch) -> None:
    events = []

    async def handler(_request):
        return ToolMessage(content="done", tool_call_id="id")

    monkeypatch.setattr(
        "anymind.runtime.mcp_registry.logger.info",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    request = SimpleNamespace(
        name="tool", args={"q": 1}, runtime=SimpleNamespace(tool_call_id="id")
    )
    result = await tool_call_logging_interceptor(request, handler)
    assert tool_result_to_text(result) == "done"
    assert events[0][0] == "tool_call_start"
    assert events[1][0] == "tool_call_result"
    assert "duration_ms" in events[1][1]


@pytest.mark.asyncio
async def test_tool_call_logging_interceptor_logs_errors(monkeypatch) -> None:
    events = []

    async def handler(_request):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "anymind.runtime.mcp_registry.logger.info",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    monkeypatch.setattr(
        "anymind.runtime.mcp_registry.logger.exception",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    request = SimpleNamespace(
        name="tool", args={"q": 1}, runtime=SimpleNamespace(tool_call_id="id")
    )
    with pytest.raises(RuntimeError, match="boom"):
        await tool_call_logging_interceptor(request, handler)
    assert events[0][0] == "tool_call_start"
    assert events[1][0] == "tool_call_error"


@pytest.mark.asyncio
async def test_mcp_registry_factory_caches(monkeypatch) -> None:
    class DummyClient:
        def __init__(self, _config, tool_interceptors=None) -> None:
            self._tools = ["tool1"]

        async def get_tools(self):
            return self._tools

        async def close(self):
            return None

    monkeypatch.setattr(
        "anymind.runtime.mcp_registry.MultiServerMCPClient", DummyClient
    )
    factory = MCPRegistryFactory()
    raw = MCPConfig(servers={})
    registry1 = await factory.get(raw, base_dir=Path("."))
    registry2 = await factory.get(raw, base_dir=Path("."))
    assert registry1 is registry2
