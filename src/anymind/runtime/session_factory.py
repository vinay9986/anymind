from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from anymind.agents.base import AgentContext
from anymind.config.loader import (
    load_mcp_config,
    load_model_config,
    load_pricing_config,
)
from anymind.config.schemas import MCPConfig, ModelConfig, PricingConfig
from anymind.policies.tool_policy import resolve_tool_policy
from anymind.runtime.agent_registry import AgentRegistry
from anymind.runtime.cache import create_cache
from anymind.runtime.checkpoints import create_checkpointer
from anymind.runtime.llm_factory import LLMFactory
from anymind.runtime.mcp_registry import (
    MCPRegistryFactory,
    bedrock_tool_interceptor,
    cache_tool_interceptor,
    confirm_tool_interceptor,
    evidence_tool_interceptor,
    tool_call_logging_interceptor,
    tool_error_logging_interceptor,
)
from anymind.runtime.session import Session
from anymind.runtime.tool_validation import ensure_tools_have_descriptions
from anymind.runtime.usage import PricingTable


class SessionFactory:
    def __init__(
        self,
        *,
        llm_factory: Optional[LLMFactory] = None,
        mcp_factory: Optional[MCPRegistryFactory] = None,
        agents: Optional[AgentRegistry] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self._llm_factory = llm_factory or LLMFactory()
        self._mcp_factory = mcp_factory or MCPRegistryFactory()
        self._agents = agents or AgentRegistry()
        self._base_dir = base_dir or Path.cwd()

    def _apply_search_env(self, model_cfg: ModelConfig) -> None:
        search_cfg = model_cfg.search
        if search_cfg is None:
            return
        provider = str(search_cfg.provider or "").strip().lower() or "kagi"
        if provider != "kagi":
            return
        api_key = str(search_cfg.kagi_api_key or "").strip()
        if api_key:
            os.environ.setdefault("KAGI_API_KEY", api_key)
        endpoint = str(search_cfg.kagi_endpoint or "").strip()
        if endpoint:
            os.environ.setdefault("KAGI_API_ENDPOINT", endpoint)

    async def create_session(
        self,
        *,
        agent_name: str,
        model_config: Optional[ModelConfig] = None,
        pricing_config: Optional[PricingConfig] = None,
        mcp_config: Optional[MCPConfig] = None,
    ) -> Session:
        model_cfg = model_config or load_model_config()
        self._apply_search_env(model_cfg)
        pricing_cfg = pricing_config or load_pricing_config()
        mcp_cfg = mcp_config
        if mcp_cfg is None:
            try:
                mcp_cfg = load_mcp_config()
            except FileNotFoundError:
                mcp_cfg = MCPConfig()

        model_client = self._llm_factory.get(model_cfg)
        pricing = PricingTable(pricing_cfg)
        tool_policy_name = model_cfg.tools_policy
        if model_cfg.model_provider == "ollama" and tool_policy_name == "auto":
            tool_policy_name = "planner"
        tool_policy = resolve_tool_policy(tool_policy_name)

        interceptors: list[object] = []
        interceptors.append(tool_call_logging_interceptor)
        interceptors.append(tool_error_logging_interceptor)
        interceptors.append(evidence_tool_interceptor)
        cache = await create_cache(model_cfg.cache)
        if cache is not None and model_cfg.cache is not None:
            interceptors.append(
                cache_tool_interceptor(cache, model_cfg.cache.ttl_seconds)
            )
        if tool_policy.name == "confirm":
            interceptors.append(confirm_tool_interceptor)
        if model_cfg.model_provider == "bedrock":
            interceptors.append(bedrock_tool_interceptor)

        tools: list[object] = []
        if model_cfg.tools_enabled and tool_policy.name != "never" and mcp_cfg.servers:
            registry = await self._mcp_factory.get(
                mcp_cfg,
                base_dir=self._base_dir,
                interceptors=interceptors,
            )
            tools = registry.tools
            ensure_tools_have_descriptions(
                tools,
                context=f"session_factory:{model_cfg.model_provider or 'unknown'}",
            )

        checkpointer, conn = await create_checkpointer(
            model_cfg.state_dir, model_cfg.checkpoint
        )
        agent_instance = self._agents.get(agent_name)

        agent_with_tools = None
        if model_cfg.tools_enabled and tools:
            agent_with_tools = agent_instance.build(
                AgentContext(
                    model_config=model_cfg,
                    pricing=pricing_cfg,
                    tools=tools,
                    tool_policy=tool_policy,
                    model_client=model_client,
                    checkpointer=checkpointer,
                )
            )
        agent_no_tools = agent_instance.build(
            AgentContext(
                model_config=model_cfg,
                pricing=pricing_cfg,
                tools=[],
                tool_policy=tool_policy,
                model_client=model_client,
                checkpointer=checkpointer,
            )
        )

        return Session(
            agent_name=agent_name,
            model_config=model_cfg,
            pricing=pricing,
            tool_policy_name=tool_policy.name,
            model_client=model_client,
            tools=tools,
            agent_with_tools=agent_with_tools,
            agent_no_tools=agent_no_tools,
            checkpointer=checkpointer,
            checkpointer_conn=conn,
        )
