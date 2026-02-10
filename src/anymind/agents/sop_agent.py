from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

import structlog
from langchain_core.messages import AIMessage

from anymind.agents.base import AgentContext
from anymind.agents.aiot_agent import AIoTAgent
from anymind.agents.giot_agent import GIoTAgent
from anymind.agents.agot_agent import AGoTAgent
from anymind.agents.got_agent import GoTAgent
from anymind.agents.iot_utils import (
    ensure_current_time_tool,
    extract_user_input,
    message_text,
)
from anymind.config.schemas import SopConfig
from anymind.runtime.evidence import get_current_ledger
from anymind.runtime.usage import UsageTotals, normalize_usage_metadata
from anymind.agents.sop.sop_executor import SopExecutionConfig, execute_sop
from anymind.agents.sop.sop_optimizer import optimize_sop
from anymind.agents.sop.sop_validation import get_optimize_flag, validate_sop_structure
from anymind.agents.sop.sop_runtime_updates import update_sop_with_node_result


class SopAgent:
    name = "sop_agent"

    def build(self, context: AgentContext) -> Any:
        return _SopRuntime(context)


class _SopUsageTracker:
    def __init__(self, budget_tokens: Optional[int]) -> None:
        self._budget_tokens = budget_tokens
        self._totals_by_model: dict[str, UsageTotals] = {}

    def add_usage_metadata(
        self, usage_metadata: dict[str, dict[str, int]] | None
    ) -> None:
        if not usage_metadata:
            return
        for model_name, usage in usage_metadata.items():
            totals = self._totals_by_model.setdefault(model_name, UsageTotals())
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            output_tokens = usage.get(
                "output_tokens", usage.get("completion_tokens", 0)
            )
            totals.add(input_tokens, output_tokens)

    def total_tokens(self) -> int:
        return sum(
            t.input_tokens + t.output_tokens for t in self._totals_by_model.values()
        )

    def budget_exhausted(self) -> bool:
        if self._budget_tokens is None:
            return False
        return self.total_tokens() >= int(self._budget_tokens)

    def remaining_budget(self) -> Optional[int]:
        if self._budget_tokens is None:
            return None
        remaining = int(self._budget_tokens) - self.total_tokens()
        return max(0, remaining)

    def usage_metadata(self) -> dict[str, dict[str, int]]:
        return {
            model: {
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
            }
            for model, totals in self._totals_by_model.items()
        }


class _SopRuntime:
    def __init__(self, context: AgentContext) -> None:
        self._context = context
        self._logger = structlog.get_logger("anymind.sop")
        self._settings = context.model_config.sop or SopConfig()
        self._agent_factories = {
            "aiot": AIoTAgent(),
            "giot": GIoTAgent(),
            "agot": AGoTAgent(),
            "got": GoTAgent(),
        }
        self._agent_cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def _load_sop(self, raw_input: str) -> dict[str, Any]:
        raw_input = (raw_input or "").strip()
        if not raw_input:
            raise ValueError("SOP input is empty")
        if raw_input.startswith("@"):
            path = Path(raw_input[1:]).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = json.loads(raw_input)
        if not isinstance(data, dict):
            raise ValueError("SOP input must be a JSON object")
        return data

    async def _select_tools(
        self, *, user_input: str, usage_tracker: _SopUsageTracker
    ) -> list[Any]:
        tools = list(self._context.tools or [])
        if not tools:
            return []

        policy = self._context.tool_policy
        if policy.name == "never":
            return []

        if policy.name == "planner":
            selections, usage = await policy.select_tools(
                user_input=user_input,
                tools=tools,
                model_client=self._context.model_client,
                model_name=self._context.model_config.model,
            )
            usage_tracker.add_usage_metadata(usage)
            if selections:
                return [
                    tool for tool in tools if getattr(tool, "name", "") in selections
                ]
            return []

        return tools

    def _get_agent_runtime(
        self,
        algorithm: str,
        tools: list[Any],
        remaining_budget: Optional[int],
    ) -> Any:
        agent = self._agent_factories.get(algorithm)
        if agent is None:
            agent = self._agent_factories["aiot"]
        tool_names = tuple(getattr(tool, "name", "") for tool in tools)

        if remaining_budget is None:
            cache_key = (algorithm, tool_names)
            cached = self._agent_cache.get(cache_key)
            if cached is not None:
                return cached

        model_config = self._context.model_config
        if remaining_budget is not None:
            model_config = model_config.model_copy(
                update={"budget_tokens": remaining_budget}
            )

        runtime = agent.build(
            AgentContext(
                model_config=model_config,
                pricing=self._context.pricing,
                tools=tools,
                tool_policy=self._context.tool_policy,
                model_client=self._context.model_client,
                checkpointer=self._context.checkpointer,
            )
        )

        if remaining_budget is None:
            self._agent_cache[(algorithm, tool_names)] = runtime
        return runtime

    async def _solve(
        self, algorithm: str, query: str, usage_tracker: _SopUsageTracker
    ) -> tuple[str, dict[str, dict[str, int]] | None, list[Any]]:
        tools = await self._select_tools(user_input=query, usage_tracker=usage_tracker)
        runtime = self._get_agent_runtime(
            algorithm, tools, usage_tracker.remaining_budget()
        )
        thread_id = f"sop-{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        result = await runtime.ainvoke({"messages": [("user", query)]}, config=config)
        messages = result.get("messages", [])
        response_text = message_text(messages[-1]) if messages else ""
        usage_metadata = result.get("usage_metadata")
        if not usage_metadata:
            usage_metadata = normalize_usage_metadata(
                self._context.model_config.model, messages
            )
        return response_text, usage_metadata, []

    async def ainvoke(
        self, inputs: dict[str, Any], config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        query = extract_user_input(inputs)
        await ensure_current_time_tool(self._context.tools)
        usage_tracker = _SopUsageTracker(self._context.model_config.budget_tokens)
        ledger = get_current_ledger()
        execution_id = uuid.uuid4().hex

        try:
            sop = self._load_sop(query)
        except Exception as exc:
            return {
                "messages": [AIMessage(content=f"Invalid SOP input: {exc}")],
                "usage_metadata": usage_tracker.usage_metadata(),
            }

        ok, errors = validate_sop_structure(sop)
        if not ok:
            return {
                "messages": [AIMessage(content="Invalid SOP: " + "; ".join(errors))],
                "usage_metadata": usage_tracker.usage_metadata(),
            }

        if get_optimize_flag(sop):
            try:
                optimized = await optimize_sop(
                    sop=sop, model_client=self._context.model_client
                )
            except Exception as exc:
                self._logger.warning("sop_optimize_failed", error=str(exc))
            else:
                sop = optimized.sop
                for usage in optimized.usage_metadata:
                    if usage:
                        usage_tracker.add_usage_metadata(
                            {self._context.model_config.model: usage}
                        )

        sop_config = SopExecutionConfig(
            max_concurrency=self._settings.max_concurrency,
            node_context_max_chars=self._settings.node_context_max_chars,
            node_output_preview_chars=self._settings.node_output_preview_chars,
            include_evidence=self._settings.include_evidence,
            refinement_enabled=self._settings.refinement_enabled,
            refinement_coverage_threshold=self._settings.refinement_coverage_threshold,
            evidence_max_chars=self._settings.evidence_max_chars,
            evidence_item_max_chars=self._settings.evidence_item_max_chars,
            trace_steps=self._settings.trace_steps,
            trace_max_chars=self._settings.trace_max_chars,
        )

        self._logger.info(
            "sop_start",
            execution_id=execution_id,
            nodes=len(sop.get("nodes", [])),
            budget_tokens=self._context.model_config.budget_tokens,
            tools=len(self._context.tools or []),
        )

        node_results, final_answer, metrics = await execute_sop(
            sop=sop,
            execution_id=execution_id,
            config=sop_config,
            solver=lambda algo, prompt: self._solve(algo, prompt, usage_tracker),
            record_usage=usage_tracker.add_usage_metadata,
            budget_exhausted=usage_tracker.budget_exhausted,
            ledger=ledger,
            model_client=self._context.model_client,
            model_name=self._context.model_config.model,
        )

        for node_id, result in node_results.items():
            update_sop_with_node_result(sop, node_id=node_id, result=result)

        metrics = dict(metrics or {})
        metrics.pop("execution_id", None)
        self._logger.info("sop_complete", execution_id=execution_id, **metrics)

        usage_metadata = usage_tracker.usage_metadata()
        return {
            "messages": [AIMessage(content=final_answer)],
            "usage_metadata": usage_metadata,
        }
