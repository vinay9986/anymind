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
    evidence_tool_interceptor,
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


def test_tool_result_to_text_handles_toolmessage() -> None:
    msg = ToolMessage(content=[{"type": "text", "text": "hello"}], tool_call_id="1")
    assert tool_result_to_text(msg) == "hello"


class DummyResult:
    def __init__(self, content) -> None:
        self.content = content


def test_tool_result_to_text_prefers_concat_blob() -> None:
    result = DummyResult(
        [
            {
                "type": "json",
                "json": {"concat_blob": "blob"},
            }
        ]
    )
    assert tool_result_to_text(result) == "blob"


def test_tool_result_to_text_uses_matches() -> None:
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
    assert tool_result_to_text(result) == "alpha\nbeta"


def test_tool_result_to_text_handles_json_payload() -> None:
    result = DummyResult([{"type": "json", "json": {"value": 1}}])
    assert '"value": 1' in tool_result_to_text(result)


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
