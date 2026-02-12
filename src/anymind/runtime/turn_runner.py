from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional


from anymind.agents.base import AgentContext
from anymind.policies.tool_policy import resolve_tool_policy
from anymind.runtime.citations import render_with_citations
from anymind.runtime.evidence import use_ledger
from anymind.runtime.messages import message_text
from anymind.runtime.session import Session
from anymind.runtime.session_usage import (
    apply_usage,
    enforce_token_budget,
    token_totals,
)
from anymind.runtime.tool_selection import select_tools_for_policy
from anymind.runtime.usage import normalize_usage_metadata
from anymind.runtime.agent_registry import AgentRegistry


class BudgetExceededError(RuntimeError):
    pass


class TurnRunner:
    def __init__(self, agents: Optional[AgentRegistry] = None) -> None:
        self._agents = agents or AgentRegistry()

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

        async with session.lock:
            session.evidence_ledger.start_turn()

            with use_ledger(session.evidence_ledger):
                if pause_event is not None:
                    await pause_event.wait()

                input_messages: list[tuple[str, str]] = [("user", user_input)]
                if session.agent_name == "research_agent" and session.chat_history:
                    max_messages = 20
                    history = session.chat_history[-max_messages:]
                    input_messages = history + [("user", user_input)]

                agent = (
                    session.agent_with_tools
                    if tools_enabled
                    else session.agent_no_tools
                )

                if tools_enabled and policy.name == "planner":
                    selected_tools, planner_usage = await select_tools_for_policy(
                        policy=policy,
                        tools=session.tools,
                        user_input=user_input,
                        model_client=session.model_client,
                        model_name=session.model_config.model,
                    )
                    if planner_usage:
                        apply_usage(session, planner_usage)
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

                if pause_event is not None:
                    await pause_event.wait()

                config = {
                    "configurable": {
                        "thread_id": thread_id or session.model_config.thread_id
                    }
                }
                result = await agent.ainvoke(
                    {"messages": input_messages}, config=config
                )

            messages = result.get("messages", [])
            response_text = message_text(messages[-1]) if messages else ""

            if session.agent_name == "research_agent":
                session.chat_history.append(("user", user_input))
                session.chat_history.append(("assistant", response_text))
                if len(session.chat_history) > 40:
                    session.chat_history = session.chat_history[-40:]

            evidence_records = session.evidence_ledger.recent()
            if evidence_records:
                if pause_event is not None:
                    await pause_event.wait()
                response_text, citation_usage = await render_with_citations(
                    model_client=session.model_client,
                    model_name=session.model_config.model,
                    draft=response_text,
                    evidence_records=evidence_records,
                )
                apply_usage(session, citation_usage)

            usage_metadata = result.get("usage_metadata")
            if not usage_metadata:
                usage_metadata = normalize_usage_metadata(
                    session.model_config.model, messages
                )

            apply_usage(session, usage_metadata)
            enforce_token_budget(session)
            token_totals_out = token_totals(session)

            return {
                "response": response_text,
                "usage": usage_metadata,
                "tokens": token_totals_out,
                "evidence": [
                    {
                        "id": record.id,
                        "tool": record.tool,
                        "args": record.args,
                        "content": record.content,
                    }
                    for record in evidence_records
                ],
            }
