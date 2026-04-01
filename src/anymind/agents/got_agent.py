from __future__ import annotations

import asyncio
import uuid
from typing import Any, Optional

import structlog
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from anymind.agents.base import AgentContext
from anymind.agents.agot.prompts import TASK_EXECUTION_SYS_PROMPT
from anymind.agents.bedrock_middleware import BedrockToolResultSanitizer
from anymind.agents.got.algorithm import GoTAlgorithm, GoTConfig as GoTAlgorithmConfig
from anymind.agents.iot_utils import (
    ensure_current_time_tool,
    extract_user_input,
    message_text,
    try_load_embedder,
)
from anymind.agents.tool_agent_pool import ToolAgentPool
from anymind.agents.usage_tracker import UsageBudgetTracker
from anymind.config.schemas import GoTConfig as GoTSettings
from anymind.runtime.validated_json import generate_validated_json
from anymind.runtime.llm_errors import safe_ainvoke


class GoTAgent:
    name = "got_agent"

    def build(self, context: AgentContext) -> Any:
        return _GoTRuntime(context)


class _GoTRuntime:
    def __init__(self, context: AgentContext) -> None:
        self._context = context
        self._model_name = context.model_config.model
        self._budget_tokens = context.model_config.budget_tokens
        self._settings = context.model_config.got or GoTSettings()
        self._algo_config = GoTAlgorithmConfig(**self._settings.model_dump())
        self._embedder = try_load_embedder()
        if self._embedder is not None:
            setattr(self._embedder, "_force_single", True)
        self._logger = structlog.get_logger("anymind.got")

        self._tool_pool: ToolAgentPool | None = None
        if context.tools:
            middleware = []
            if context.model_config.model_provider == "bedrock":
                middleware.append(BedrockToolResultSanitizer())
            pool_size = max(1, int(self._settings.max_concurrency))
            agents = [
                create_agent(
                    context.model_client,
                    context.tools,
                    system_prompt=TASK_EXECUTION_SYS_PROMPT,
                    middleware=middleware,
                    checkpointer=context.checkpointer,
                )
                for _ in range(pool_size)
            ]
            self._tool_pool = ToolAgentPool(agents)

        self._usage_tracker = UsageBudgetTracker(self._model_name, self._budget_tokens)
        self._budget_exhausted = False

    def _apply_usage_list(self, usage_list: list[dict[str, int]]) -> None:
        self._usage_tracker.add_usage_list(usage_list)
        if not self._budget_exhausted and self._usage_tracker.budget_exhausted():
            self._budget_exhausted = True
            self._logger.warning(
                "got_budget_exhausted",
                input_tokens=self._usage_tracker.input_tokens,
                output_tokens=self._usage_tracker.output_tokens,
                budget_tokens=self._budget_tokens,
            )

    async def _call_worker(
        self, *, user_prompt: str, config: Optional[dict[str, Any]]
    ) -> tuple[str, Optional[dict[str, int]]]:
        if self._tool_pool is None:
            message = await safe_ainvoke(
                self._context.model_client,
                [("system", TASK_EXECUTION_SYS_PROMPT), ("user", user_prompt)],
            )
            usage = getattr(message, "usage_metadata", None)
            return message_text(message), usage

        async with self._tool_pool.acquire() as agent:
            result = await safe_ainvoke(
                agent,
                {"messages": [("user", user_prompt)]},
                config=config,
                llm_only=False,
                model_client=self._context.model_client,
            )
            messages = result.get("messages", [])
            if not messages:
                return "", None
            last_message = messages[-1]
            response_text = message_text(last_message)
            return response_text, None

    async def _run_planner_json(
        self,
        role_name: str,
        system_prompt: str,
        user_prompt: str,
        validator: Any,
        task_context: str,
    ):
        result = await generate_validated_json(
            role_name=role_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=validator,
            model_client=self._context.model_client,
            max_reasks=3,
            original_task_context=task_context,
        )
        return result

    async def ainvoke(
        self, inputs: dict[str, Any], config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        self._usage_tracker = UsageBudgetTracker(self._model_name, self._budget_tokens)
        self._budget_exhausted = False

        query = extract_user_input(inputs)
        await ensure_current_time_tool(self._context.tools)
        self._logger.info(
            "got_start",
            query=query,
            budget_tokens=self._budget_tokens,
            max_layers=self._algo_config.max_layers,
            beam_width=self._algo_config.beam_width,
            children_per_expand=self._algo_config.children_per_expand,
            max_concurrency=self._algo_config.max_concurrency,
        )

        async def planner_runner(
            role_name: str,
            system_prompt: str,
            user_prompt: str,
            validator: Any,
            task_context: str,
        ):
            return await self._run_planner_json(
                role_name,
                system_prompt,
                user_prompt,
                validator,
                task_context,
            )

        async def solver_runner(user_prompt: str) -> str:
            fresh_config = {
                "configurable": {"thread_id": f"got-solver-{uuid.uuid4().hex}"}
            }
            response_text, usage = await self._call_worker(
                user_prompt=user_prompt, config=fresh_config
            )
            return response_text

        algorithm = GoTAlgorithm(
            config=self._algo_config,
            planner_runner=planner_runner,
            solver_runner=solver_runner,
            embedder=self._embedder,
            should_stop=lambda: self._budget_exhausted,
        )

        answer, graph = await algorithm.solve(query)
        usage_metadata = self._usage_tracker.usage_metadata()
        self._logger.info(
            "got_complete",
            budget_exhausted=self._budget_exhausted,
            nodes=len(graph.to_dict().get("nodes", [])),
            final_answer_len=len(answer),
        )
        return {
            "messages": [AIMessage(content=answer)],
            "usage_metadata": usage_metadata,
        }
