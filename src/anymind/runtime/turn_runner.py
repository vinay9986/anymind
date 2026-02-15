from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional


from anymind.agents.base import AgentContext
from anymind.policies.tool_policy import resolve_tool_policy
from anymind.runtime.citations import render_with_citations
from anymind.runtime.evidence import summarize_for_display, use_ledger
from anymind.runtime.messages import message_text
from anymind.runtime.session import Session
from anymind.runtime.session_context import reset_session_id, set_session_id
from anymind.runtime.session_usage import (
    enforce_token_budget,
    token_totals,
)
from anymind.runtime.usage_store import get_usage_store
from anymind.runtime.tool_selection import select_tools_for_policy
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

        token = set_session_id(session.session_id)
        try:
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

                    selected_tools = session.tools if tools_enabled else []
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
                    if selected_tools:
                        cache_key = tuple(
                            getattr(tool, "name", "") for tool in selected_tools
                        )
                        agent = session.agent_cache.get(cache_key)
                        if agent is None:
                            agent = self._agents.get(session.agent_name).build(
                                AgentContext(
                                    model_config=session.model_config,
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
                        },
                        "metadata": {"session_id": session.session_id},
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
            evidence_id_list = result.get("evidence_ids")
            if evidence_id_list:
                by_id = {record.id: record for record in session.evidence_ledger.all()}
                filtered: list[Any] = []
                for eid in evidence_id_list:
                    record = by_id.get(str(eid))
                    if record is not None:
                        filtered.append(record)
                if filtered:
                    evidence_records = filtered
            sop_cfg = (
                session.model_config.sop if session.agent_name == "sop_agent" else None
            )
            evidence_output_records = evidence_records
            if sop_cfg is not None and bool(getattr(sop_cfg, "include_evidence", True)):
                evidence_output_records = session.evidence_ledger.recent()
            if evidence_records:
                if pause_event is not None:
                    await pause_event.wait()
                response_text, citation_usage = await render_with_citations(
                    model_client=session.model_client,
                    model_name=session.model_config.model,
                    draft=response_text,
                    evidence_records=evidence_records,
                )

            snapshot = get_usage_store().get(session.session_id)
            if snapshot.per_model:
                usage_metadata = {
                    model: {
                        "input_tokens": totals.input_tokens,
                        "output_tokens": totals.output_tokens,
                    }
                    for model, totals in snapshot.per_model.items()
                }
            else:
                usage_metadata = {
                    session.model_config.model: {
                        "input_tokens": snapshot.totals.input_tokens,
                        "output_tokens": snapshot.totals.output_tokens,
                    }
                }

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
                        "content": summarize_for_display(record),
                    }
                    for record in evidence_output_records
                ],
            }
        finally:
            reset_session_id(token)
