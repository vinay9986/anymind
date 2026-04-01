from __future__ import annotations

import asyncio
import uuid
from typing import Any, Optional

import structlog

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from anymind.agents.agot.algorithm import AGOTAlgorithm
from anymind.agents.agot.prompts import TASK_EXECUTION_SYS_PROMPT
from anymind.agents.agot.mappings import AGoTMappings
from anymind.agents.base import AgentContext
from anymind.agents.bedrock_middleware import BedrockToolResultSanitizer
from anymind.agents.iot_utils import (
    ensure_current_time_tool,
    extract_user_input,
    message_text,
)
from anymind.agents.tool_agent_pool import ToolAgentPool
from anymind.agents.usage_tracker import UsageBudgetTracker
from anymind.config.schemas import AGoTConfig
from anymind.runtime.validated_json import (
    ValidatedJsonResult,
    generate_validated_json_with_calls,
)
from anymind.runtime.llm_errors import safe_ainvoke


class AGoTAgent:
    name = "agot_agent"

    def build(self, context: AgentContext) -> Any:
        return _AGoTRuntime(context)


class _AGoTRuntime:
    def __init__(self, context: AgentContext) -> None:
        self._context = context
        self._model_name = context.model_config.model
        self._budget_tokens = context.model_config.budget_tokens
        self._settings = context.model_config.agot or AGoTConfig()
        self._logger = structlog.get_logger("anymind.agot")

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
                "agot_budget_exhausted",
                input_tokens=self._usage_tracker.input_tokens,
                output_tokens=self._usage_tracker.output_tokens,
                budget_tokens=self._budget_tokens,
            )

    async def _call_planner(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        message = await safe_ainvoke(
            self._context.model_client,
            [("system", system_prompt), ("user", user_prompt)],
        )
        usage = getattr(message, "usage_metadata", None)
        return message_text(message), usage

    async def _call_fix(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        message = await safe_ainvoke(
            self._context.model_client,
            [("system", system_prompt), ("user", user_prompt)],
        )
        usage = getattr(message, "usage_metadata", None)
        return message_text(message), usage

    async def _call_worker(
        self,
        *,
        user_prompt: str,
        config: Optional[dict[str, Any]],
    ) -> tuple[str, Optional[dict[str, int]]]:
        run_config: dict[str, Any] = {}
        if config:
            run_config = dict(config)
        cfg = dict(run_config.get("configurable", {}))
        cfg["thread_id"] = f"agot-worker-{uuid.uuid4().hex}"
        run_config["configurable"] = cfg

        async with self._tool_pool.acquire() as agent:
            result = await safe_ainvoke(
                agent,
                {"messages": [("user", user_prompt)]},
                config=run_config,
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
    ) -> ValidatedJsonResult:
        result = await generate_validated_json_with_calls(
            role_name=role_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=validator,
            call_model=self._call_planner,
            fix_model=self._call_fix,
            max_reasks=3,
            original_task_context=task_context,
        )
        return result

    async def _run_worker_json(
        self,
        role_name: str,
        system_prompt: str,
        user_prompt: str,
        validator: Any,
        task_context: str,
        config: Optional[dict[str, Any]],
    ) -> ValidatedJsonResult:
        async def _call(system_prompt: str, user_prompt: str):
            return await self._call_worker(user_prompt=user_prompt, config=config)

        result = await generate_validated_json_with_calls(
            role_name=role_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=validator,
            call_model=_call,
            fix_model=self._call_fix,
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
            "agot_start",
            query=query,
            budget_tokens=self._budget_tokens,
            d_max=self._settings.d_max,
            l_max=self._settings.l_max,
            n_max=self._settings.n_max,
            max_concurrency=self._settings.max_concurrency,
        )

        async def planner_runner(
            role_name: str,
            system_prompt: str,
            user_prompt: str,
            validator: Any,
            task_context: str,
        ) -> ValidatedJsonResult:
            return await self._run_planner_json(
                role_name,
                system_prompt,
                user_prompt,
                validator,
                task_context,
            )

        async def worker_runner(
            role_name: str,
            system_prompt: str,
            user_prompt: str,
            validator: Any,
            task_context: str,
        ) -> ValidatedJsonResult:
            return await self._run_worker_json(
                role_name,
                system_prompt,
                user_prompt,
                validator,
                task_context,
                config,
            )

        mappings = AGoTMappings(
            planner_runner=planner_runner,
            worker_runner=worker_runner,
            d_max=self._settings.d_max,
        )
        algorithm = AGOTAlgorithm(
            mappings=mappings,
            d_max=self._settings.d_max,
            l_max=self._settings.l_max,
            n_max=self._settings.n_max,
            max_concurrency=self._settings.max_concurrency,
            should_stop=lambda: self._budget_exhausted,
        )

        answer, graph = await algorithm.solve(query)
        usage_metadata = self._usage_tracker.usage_metadata()
        self._logger.info(
            "agot_complete",
            budget_exhausted=self._budget_exhausted,
            nodes=len(graph.nodes),
            layers=graph.get_layer_count(),
            final_answer_len=len(answer),
        )
        return {
            "messages": [AIMessage(content=answer)],
            "usage_metadata": usage_metadata,
        }
