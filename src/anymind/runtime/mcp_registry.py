from __future__ import annotations

from pathlib import Path
import os
import json
from typing import Any, Dict, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_core.messages import ToolMessage

from anymind.config.schemas import MCPConfig
from anymind.runtime.evidence import get_current_ledger
from anymind.runtime.cache import CacheBackend


_MCP_ENV_ALLOWLIST = {
    "GOOGLE_CSE_API_KEY",
    "GOOGLE_CSE_ENGINE_ID",
    "GOOGLE_CSE_SECRET_ARN",
    "GOOGLE_CSE_API_KEY_SECRET_ARN",
    "GOOGLE_CSE_ENGINE_ID_SECRET_ARN",
    "GOOGLE_CSE_CACHE_TTL_SECONDS",
    "GOOGLE_CSE_ENDPOINT",
    "GOOGLE_CSE_COUNTRY_REGION",
    "GOOGLE_CSE_GEOLOCATION",
    "GOOGLE_CSE_RESULT_LANGUAGE",
    "SCRAPFLY_API_KEY",
    "SCRAPFLY_API_KEY_SECRET_ARN",
    "SCRAPFLY_API_KEY_CACHE_TTL_SECONDS",
    "SCRAPFLY_COUNTRY",
    "SCRAPFLY_FORMAT",
    "AGENT_DATA_BUCKET",
    "AGENT_DATA_BASE_PREFIX",
    "BEDROCK_KNOWLEDGE_BASE_ID",
    "BEDROCK_KNOWLEDGE_BASE_MODEL_ARN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
}


def _default_mcp_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key in _MCP_ENV_ALLOWLIST:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def resolve_mcp_config(raw: MCPConfig, base_dir: Path) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for name, cfg in raw.servers.items():
        if not isinstance(cfg, dict):
            continue
        cfg_copy = dict(cfg)
        args = cfg_copy.get("args")
        if isinstance(args, list):
            resolved_args = []
            for arg in args:
                if isinstance(arg, str):
                    arg_path = Path(arg)
                    if not arg_path.is_absolute() and (
                        arg.startswith("./") or arg.endswith(".py")
                    ):
                        arg = str((base_dir / arg).resolve())
                resolved_args.append(arg)
            cfg_copy["args"] = resolved_args
        if cfg_copy.get("transport") == "stdio":
            env_cfg = cfg_copy.get("env")
            base_env = _default_mcp_env()
            if isinstance(env_cfg, dict):
                merged = dict(base_env)
                merged.update(
                    {str(k): str(v) for k, v in env_cfg.items() if v is not None}
                )
                cfg_copy["env"] = merged
            elif env_cfg is None:
                if base_env:
                    cfg_copy["env"] = base_env
        resolved[name] = cfg_copy
    return resolved


def tool_result_to_text(result: MCPToolCallResult) -> str:
    if isinstance(result, ToolMessage):
        content = result.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content).strip()

    text_parts = []
    for item in result.content:
        if getattr(item, "type", None) == "text":
            text_parts.append(getattr(item, "text", ""))
        else:
            text_parts.append(str(item))
    return "\n".join(text_parts).strip()


def _is_text_only_result(result: MCPToolCallResult) -> bool:
    if isinstance(result, ToolMessage):
        content = result.content
        if isinstance(content, list):
            return all(
                isinstance(item, dict) and item.get("type") == "text"
                for item in content
            )
        return True
    if hasattr(result, "content"):
        for item in result.content:
            if getattr(item, "type", None) != "text":
                return False
        return True
    return False


def _cache_key(tool_name: str, args: Any) -> str:
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)
    return f"tool:{payload}"


def cache_tool_interceptor(cache: CacheBackend, ttl_seconds: int) -> Any:
    async def _interceptor(request: MCPToolCallRequest, handler) -> MCPToolCallResult:
        key = _cache_key(request.name, request.args)
        cached = await cache.get(key)
        if cached is not None:
            tool_call_id = (
                getattr(request.runtime, "tool_call_id", None) or "tool_call_id"
            )
            return ToolMessage(
                content=cached.get("content", ""), tool_call_id=str(tool_call_id)
            )

        result = await handler(request)
        if _is_text_only_result(result):
            text = tool_result_to_text(result)
            if text:
                await cache.set(key, {"content": text}, ttl_seconds=ttl_seconds)
        return result

    return _interceptor


async def evidence_tool_interceptor(
    request: MCPToolCallRequest, handler
) -> MCPToolCallResult:
    result = await handler(request)
    ledger = get_current_ledger()
    if ledger is not None:
        text = tool_result_to_text(result)
        if text:
            ledger.add(request.name, request.args, text)
    return result


async def bedrock_tool_interceptor(
    request: MCPToolCallRequest, handler
) -> MCPToolCallResult:
    """Return a ToolMessage with text blocks that exclude text.id fields."""
    result = await handler(request)
    text = tool_result_to_text(result)
    tool_call_id = getattr(request.runtime, "tool_call_id", None) or "tool_call_id"
    content = []
    if text:
        content = [{"type": "text", "text": text}]
    return ToolMessage(content=content or "", tool_call_id=str(tool_call_id))


async def confirm_tool_interceptor(
    request: MCPToolCallRequest, handler
) -> MCPToolCallResult:
    prompt = f"Tool call requested: {request.name} args={request.args}. Allow? [y/N] "
    try:
        decision = input(prompt).strip().lower()  # nosec B322
    except EOFError:
        decision = ""
    if decision not in {"y", "yes"}:
        tool_call_id = getattr(request.runtime, "tool_call_id", None) or "tool_call_id"
        return ToolMessage(
            content="Tool call denied by user.", tool_call_id=str(tool_call_id)
        )
    return await handler(request)


class MCPToolRegistry:
    def __init__(self, client: MultiServerMCPClient, tools: List[Any]) -> None:
        self._client = client
        self._tools = tools

    @property
    def tools(self) -> List[Any]:
        return self._tools

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await self._client.close()


class MCPRegistryFactory:
    def __init__(self) -> None:
        self._cache: Dict[str, MCPToolRegistry] = {}

    async def get(
        self,
        raw_config: MCPConfig,
        base_dir: Path,
        interceptors: List[Any] | None = None,
    ) -> MCPToolRegistry:
        resolved = resolve_mcp_config(raw_config, base_dir)
        interceptor_fingerprint = tuple(id(item) for item in (interceptors or []))
        cache_key = f"{sorted(resolved.items())}:{interceptor_fingerprint}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        client = MultiServerMCPClient(resolved, tool_interceptors=interceptors or [])
        tools = await client.get_tools()
        registry = MCPToolRegistry(client, tools)
        self._cache[cache_key] = registry
        return registry
