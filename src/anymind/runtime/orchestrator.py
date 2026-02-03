from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import BaseMessage

from anymind.agents.base import AgentContext
from anymind.config.loader import (
    load_mcp_config,
    load_model_config,
    load_pricing_config,
)
from anymind.config.schemas import MCPConfig, ModelConfig, PricingConfig
from anymind.policies.tool_policy import resolve_tool_policy
from anymind.runtime.agent_registry import AgentRegistry
from anymind.runtime.checkpoints import create_checkpointer
from anymind.runtime.llm_factory import LLMFactory
from anymind.runtime.cache import create_cache
from anymind.runtime.evidence import EvidenceLedger, EvidenceRecord, use_ledger
from anymind.runtime.mcp_registry import (
    MCPRegistryFactory,
    bedrock_tool_interceptor,
    cache_tool_interceptor,
    confirm_tool_interceptor,
    evidence_tool_interceptor,
)
from anymind.runtime.usage import PricingTable, UsageTotals, normalize_usage_metadata


@dataclass
class Session:
    agent_name: str
    model_config: ModelConfig
    pricing: PricingTable
    tool_policy_name: str
    model_client: Any
    tools: list[Any]
    agent_with_tools: Any
    agent_no_tools: Any
    checkpointer: Any
    checkpointer_conn: Any
    totals_by_model: Dict[str, UsageTotals] = field(default_factory=dict)
    agent_cache: Dict[tuple[str, ...], Any] = field(default_factory=dict)
    budget_exhausted: bool = False
    evidence_ledger: EvidenceLedger = field(default_factory=EvidenceLedger)

    async def close(self) -> None:
        if self.checkpointer_conn is not None:
            await self.checkpointer_conn.close()


class Orchestrator:
    def __init__(self) -> None:
        self._llm_factory = LLMFactory()
        self._mcp_factory = MCPRegistryFactory()
        self._agents = AgentRegistry()
        self._base_dir = Path.cwd()

    async def create_session(
        self,
        *,
        agent_name: str,
        model_config: Optional[ModelConfig] = None,
        pricing_config: Optional[PricingConfig] = None,
        mcp_config: Optional[MCPConfig] = None,
    ) -> Session:
        model_cfg = model_config or load_model_config()
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

        interceptors: list[Any] = []
        evidence_ledger = EvidenceLedger()
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

        tools: list[Any] = []
        if model_cfg.tools_enabled and tool_policy.name != "never" and mcp_cfg.servers:
            registry = await self._mcp_factory.get(
                mcp_cfg,
                base_dir=self._base_dir,
                interceptors=interceptors,
            )
            tools = registry.tools

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
            evidence_ledger=evidence_ledger,
        )

    async def run_turn(
        self,
        session: Session,
        *,
        user_input: str,
        thread_id: Optional[str] = None,
        pause_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        if session.budget_exhausted:
            raise BudgetExceededError("Token budget exceeded")
        policy = resolve_tool_policy(session.tool_policy_name)
        tools_enabled = bool(session.model_config.tools_enabled and session.tools)

        session.evidence_ledger.start_turn()

        with use_ledger(session.evidence_ledger):
            if pause_event is not None:
                await pause_event.wait()

            agent = (
                session.agent_with_tools if tools_enabled else session.agent_no_tools
            )
            if tools_enabled and policy.name == "planner":
                selections, planner_usage = await policy.select_tools(
                    user_input=user_input,
                    tools=session.tools,
                    model_client=session.model_client,
                    model_name=session.model_config.model,
                )
                self._apply_usage(session, planner_usage)
                if selections:
                    selected_tools = [
                        tool
                        for tool in session.tools
                        if getattr(tool, "name", "") in selections
                    ]
                    if selected_tools:
                        cache_key = tuple(
                            getattr(tool, "name", "") for tool in selected_tools
                        )
                        agent = session.agent_cache.get(cache_key)
                        if agent is None:
                            agent = self._agents.get(session.agent_name).build(
                                AgentContext(
                                    model_config=session.model_config,
                                    pricing=session.pricing,
                                    tools=selected_tools,
                                    tool_policy=policy,
                                    model_client=session.model_client,
                                    checkpointer=session.checkpointer,
                                )
                            )
                            session.agent_cache[cache_key] = agent
                    else:
                        agent = session.agent_no_tools
                else:
                    agent = session.agent_no_tools

            if pause_event is not None:
                await pause_event.wait()

            config = {
                "configurable": {
                    "thread_id": thread_id or session.model_config.thread_id
                }
            }
            result = await agent.ainvoke(
                {"messages": [("user", user_input)]}, config=config
            )
        messages = result.get("messages", [])
        response_text = _message_text(messages[-1]) if messages else ""

        evidence_records = session.evidence_ledger.recent()
        if evidence_records:
            if pause_event is not None:
                await pause_event.wait()
            response_text, citation_usage = await _render_with_citations(
                session.model_client,
                session.model_config.model,
                response_text,
                evidence_records,
            )
            self._apply_usage(session, citation_usage)

        usage_metadata = result.get("usage_metadata")
        if not usage_metadata:
            usage_metadata = normalize_usage_metadata(
                session.model_config.model, messages
            )

        self._apply_usage(session, usage_metadata)
        self._enforce_token_budget(session)
        costs = self._costs(session)

        return {
            "response": response_text,
            "usage": usage_metadata,
            "costs": costs,
            "evidence": [
                {"id": record.id, "tool": record.tool, "content": record.content}
                for record in evidence_records
            ],
        }

    def _apply_usage(
        self, session: Session, usage_metadata: Dict[str, Dict[str, int]]
    ) -> None:
        for model_name, usage in usage_metadata.items():
            model_totals = session.totals_by_model.setdefault(model_name, UsageTotals())
            model_totals.add(
                usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            )

    def _costs(self, session: Session) -> Dict[str, Dict[str, float]]:
        costs: Dict[str, Dict[str, float]] = {}
        for model_name, totals in session.totals_by_model.items():
            costs[model_name] = session.pricing.cost(model_name, totals)
        return costs

    def _total_tokens(self, session: Session) -> int:
        total = 0
        for totals in session.totals_by_model.values():
            total += totals.input_tokens + totals.output_tokens
        return total

    def _enforce_token_budget(self, session: Session) -> None:
        if session.model_config.budget_tokens is None:
            return
        if self._total_tokens(session) >= int(session.model_config.budget_tokens):
            session.budget_exhausted = True


class BudgetExceededError(RuntimeError):
    pass


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


async def _render_with_citations(
    model_client: Any,
    model_name: str,
    draft: str,
    evidence_records: list[EvidenceRecord],
) -> tuple[str, Dict[str, Dict[str, int]]]:
    summary = EvidenceLedger.summarize(evidence_records)
    prompt = (
        "You are a response editor. Rewrite the draft to include citations in square brackets "
        "after claims that rely on evidence. Use only the evidence IDs provided. "
        "Do not invent citations. Keep the answer concise.\n\n"
        f"Draft:\n{draft}\n\nEvidence ledger:\n{summary}"
    )
    message = await model_client.ainvoke(
        [("system", "Add citations."), ("user", prompt)]
    )
    return _message_text(message), normalize_usage_metadata(model_name, [message])
