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
    UsageCounter,
    budget_exhausted,
    ensure_current_time_tool,
    extract_user_input,
    message_text,
)
from anymind.config.schemas import AGoTConfig
from anymind.runtime.usage import extract_usage_from_messages
from anymind.runtime.validated_json import (
    ValidatedJsonResult,
    generate_validated_json_with_calls,
)


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
        self._tool_agents = [
            create_agent(
                context.model_client,
                context.tools,
                system_prompt=TASK_EXECUTION_SYS_PROMPT,
                middleware=middleware,
                checkpointer=context.checkpointer,
            )
            for _ in range(pool_size)
        ]
        self._agent_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=pool_size)
        for idx in range(pool_size):
            self._agent_queue.put_nowait(idx)

        self._usage_counter = UsageCounter()
        self._budget_exhausted = False

    def _apply_usage_list(self, usage_list: list[dict[str, int]]) -> None:
        self._usage_counter.add_usage_list(usage_list)
        if not self._budget_exhausted and budget_exhausted(
            self._usage_counter, self._budget_tokens
        ):
            self._budget_exhausted = True
            self._logger.warning(
                "agot_budget_exhausted",
                input_tokens=self._usage_counter.input_tokens,
                output_tokens=self._usage_counter.output_tokens,
                budget_tokens=self._budget_tokens,
            )

    async def _call_planner(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        message = await self._context.model_client.ainvoke(
            [("system", system_prompt), ("user", user_prompt)]
        )
        usage = getattr(message, "usage_metadata", None)
        return message_text(message), usage

    async def _call_fix(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        message = await self._context.model_client.ainvoke(
            [("system", system_prompt), ("user", user_prompt)]
        )
        usage = getattr(message, "usage_metadata", None)
        return message_text(message), usage

    async def _call_worker(
        self,
        *,
        user_prompt: str,
        config: Optional[dict[str, Any]],
    ) -> tuple[str, Optional[dict[str, int]]]:
        idx = await self._agent_queue.get()
        try:
            run_config: dict[str, Any] = {}
            if config:
                run_config = dict(config)
            cfg = dict(run_config.get("configurable", {}))
            cfg["thread_id"] = f"agot-worker-{uuid.uuid4().hex}"
            run_config["configurable"] = cfg

            result = await self._tool_agents[idx].ainvoke(
                {"messages": [("user", user_prompt)]}, config=run_config
            )
            messages = result.get("messages", [])
            if not messages:
                return "", None
            response_text = message_text(messages[-1])
            totals = extract_usage_from_messages(messages)
            usage = {
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
            }
            if totals.input_tokens or totals.output_tokens:
                return response_text, usage
            return response_text, None
        finally:
            self._agent_queue.put_nowait(idx)

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
        self._apply_usage_list(result.usage_metadata)
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
        self._apply_usage_list(result.usage_metadata)
        return result

    async def ainvoke(
        self, inputs: dict[str, Any], config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        self._usage_counter = UsageCounter()
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
        usage_metadata = {
            self._model_name: {
                "input_tokens": self._usage_counter.input_tokens,
                "output_tokens": self._usage_counter.output_tokens,
            }
        }
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
