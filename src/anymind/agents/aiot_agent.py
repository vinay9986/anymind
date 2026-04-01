from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Optional

import structlog

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from anymind.agents.base import AgentContext
from anymind.agents.bedrock_middleware import BedrockToolResultSanitizer
from anymind.agents.iot_prompts import (
    BRAIN_SYSTEM_PROMPT,
    BRAIN_USER_PROMPT,
    LLM_FINAL_PROMPT,
    LLM_SYSTEM_PROMPT,
    LLM_USER_PROMPT,
)
from anymind.agents.iot_utils import (
    UsageCounter,
    budget_exhausted,
    ensure_current_time_tool,
    extract_user_input,
    message_text,
    _format_tool_catalog,
    select_semantic_representative,
    truncate_text,
    tool_feedback_from_ledger,
    try_load_embedder,
)
from anymind.config.schemas import AIoTConfig
from anymind.runtime.json_validation import JSONStructureValidator
from anymind.runtime.session_context import get_session_id
from anymind.runtime.usage_store import get_usage_store
from anymind.runtime.validated_json import (
    ValidatedJsonResult,
    generate_validated_json,
    generate_validated_json_with_calls,
)
from anymind.runtime.llm_errors import raise_if_llm_http_error, safe_ainvoke


class AIoTAgent:
    name = "aiot_agent"

    def build(self, context: AgentContext) -> Any:
        return _AIoTRuntime(context)


class _AIoTRuntime:
    def __init__(self, context: AgentContext) -> None:
        self._context = context
        self._model_name = context.model_config.model
        self._budget_tokens = context.model_config.budget_tokens
        self._settings = context.model_config.aiot or AIoTConfig()
        self._logger = structlog.get_logger("anymind.aiot")

        middleware = []
        if context.model_config.model_provider == "bedrock":
            middleware.append(BedrockToolResultSanitizer())
        self._tool_agent = create_agent(
            context.model_client,
            context.tools,
            system_prompt=LLM_SYSTEM_PROMPT,
            middleware=middleware,
            checkpointer=context.checkpointer,
        )

    async def _call_worker(
        self, *, user_prompt: str, config: Optional[dict[str, Any]]
    ) -> tuple[str, Optional[dict[str, int]]]:
        result = await safe_ainvoke(
            self._tool_agent,
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

    async def _call_fix(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        message = await safe_ainvoke(
            self._context.model_client,
            [("system", system_prompt), ("user", user_prompt)],
        )
        usage = getattr(message, "usage_metadata", None)
        return message_text(message), usage

    async def _run_validated_worker(
        self,
        *,
        user_prompt: str,
        validator: JSONStructureValidator,
        config: Optional[dict[str, Any]],
        role_name: str,
        task_context: str,
    ) -> ValidatedJsonResult:
        async def _call(system_prompt: str, user_prompt: str):
            return await self._call_worker(user_prompt=user_prompt, config=config)

        return await generate_validated_json_with_calls(
            role_name=role_name,
            system_prompt=LLM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            validator=validator,
            call_model=_call,
            fix_model=self._call_fix,
            max_reasks=3,
            original_task_context=task_context,
        )

    async def ainvoke(
        self, inputs: dict[str, Any], config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        query = extract_user_input(inputs)
        await ensure_current_time_tool(self._context.tools)
        usage_counter = UsageCounter()
        prompt_history = ""
        tool_feedback = tool_feedback_from_ledger()
        worker_tools_configuration = _format_tool_catalog(self._context.tools)
        iteration = 1
        last_worker_response = ""
        final_response: Optional[str] = None
        min_iterations = max(1, int(self._settings.min_iterations))
        self._logger.info(
            "aiot_start",
            query=query,
            budget_tokens=self._budget_tokens,
            tools=len(self._context.tools),
            min_iterations=min_iterations,
            self_consistency_samples=int(self._settings.self_consistency_samples),
        )

        brain_validator = JSONStructureValidator(
            {
                "self_thought": {
                    "type": str,
                    "required": True,
                    "description": "instructions for worker",
                },
                "iteration_stop": {
                    "type": bool,
                    "required": True,
                    "description": "stop flag",
                },
            },
            validator_name="brain-iteration-validator",
        )
        worker_validator = JSONStructureValidator(
            {
                "response": {
                    "type": str,
                    "required": True,
                    "description": "worker response",
                },
                "answer_to_query": {
                    "type": bool,
                    "required": True,
                    "description": "completeness flag",
                },
            },
            validator_name="llm-response-validator",
        )
        final_validator = JSONStructureValidator(
            {
                "response": {
                    "type": str,
                    "required": True,
                    "description": "final answer",
                },
                "explanation": {
                    "type": str,
                    "required": True,
                    "description": "reasoning summary",
                },
            },
            validator_name="llm-final-validator",
        )

        while True:
            if budget_exhausted(usage_counter, self._budget_tokens):
                break

            brain_user = BRAIN_USER_PROMPT.format(
                prompt_history=prompt_history,
                iteration=iteration,
                query=query,
                tool_feedback=tool_feedback,
                worker_tools_configuration=worker_tools_configuration,
            )
            self._logger.info("aiot_brain_start", iteration=iteration)
            brain_result = await generate_validated_json(
                role_name="brain",
                system_prompt=BRAIN_SYSTEM_PROMPT,
                user_prompt=brain_user,
                validator=brain_validator,
                model_client=self._context.model_client,
                max_reasks=3,
                original_task_context=f"AIoT brain iteration for query: {query}",
            )
            # budget tracking handled via session usage store
            brain_thought = str(brain_result.data.get("self_thought", "") or "")
            brain_payload = {
                "iteration": iteration,
                "iteration_stop": bool(brain_result.data.get("iteration_stop")),
                "thought_len": len(brain_thought),
            }
            if self._settings.trace_steps:
                brain_payload["brain_thought"] = truncate_text(
                    brain_thought, int(self._settings.trace_max_chars)
                )
            self._logger.info("aiot_brain_complete", **brain_payload)

            worker_user = LLM_USER_PROMPT.format(
                brain_thought=str(brain_result.data.get("self_thought", "")),
                prompt_history=prompt_history,
                query=query,
                tool_feedback=tool_feedback,
            )
            self._logger.info("aiot_worker_start", iteration=iteration)
            worker_result = await self._run_validated_worker(
                user_prompt=worker_user,
                validator=worker_validator,
                config=config,
                role_name="worker",
                task_context=f"AIoT worker iteration for query: {query}",
            )
            # budget tracking handled via session usage store

            tool_feedback = tool_feedback_from_ledger()
            last_worker_response = str(worker_result.data.get("response", "")).strip()
            worker_payload = {
                "iteration": iteration,
                "answer_to_query": bool(worker_result.data.get("answer_to_query")),
                "response_len": len(last_worker_response),
                "tool_feedback_len": len(tool_feedback),
            }
            if self._settings.trace_steps:
                worker_payload["worker_response"] = truncate_text(
                    last_worker_response, int(self._settings.trace_max_chars)
                )
            self._logger.info("aiot_worker_complete", **worker_payload)

            prompt_history += (
                f"Iter {iteration} brain:{brain_result.data.get('self_thought','')}\n"
                f"llm:{last_worker_response}\n\n"
            )

            if bool(worker_result.data.get("answer_to_query")) and (
                iteration >= min_iterations
            ):
                final_response = last_worker_response
                break

            iteration += 1

        if final_response is None:
            final_response = (
                last_worker_response or "No answer produced before budget exhaustion."
            )

        if self._settings.self_consistency_samples > 1 and not budget_exhausted(
            usage_counter, self._budget_tokens
        ):
            self._logger.info(
                "aiot_self_consistency_start",
                samples=int(self._settings.self_consistency_samples),
            )
            final_user = LLM_FINAL_PROMPT.format(
                prompt_history=prompt_history,
                query=query,
                tool_feedback=tool_feedback,
            )

            async def _sample(sample_idx: int) -> tuple[str, str, list[dict[str, int]]]:
                sample_result = await self._run_validated_worker(
                    user_prompt=final_user,
                    validator=final_validator,
                    config=config,
                    role_name=f"self_consistency_{sample_idx}",
                    task_context=f"AIoT self-consistency sample {sample_idx} for query: {query}",
                )
                response = str(sample_result.data.get("response", "")).strip()
                explanation = str(sample_result.data.get("explanation", "")).strip()
                if self._settings.trace_samples:
                    self._logger.info(
                        "aiot_self_consistency_sample",
                        sample=sample_idx,
                        response_len=len(response),
                        explanation_len=len(explanation),
                        response=truncate_text(
                            response, int(self._settings.trace_max_chars)
                        ),
                        explanation=truncate_text(
                            explanation, int(self._settings.trace_max_chars)
                        ),
                    )
                return response, explanation, sample_result.usage_metadata

            tasks = [
                _sample(i)
                for i in range(1, int(self._settings.self_consistency_samples))
            ]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                answers: list[str] = [final_response]
                explanations: list[str] = [""]
                for result in results:
                    if isinstance(result, BaseException):
                        raise_if_llm_http_error(result)
                        continue
                    response, explanation, usage_list = result
                    if response:
                        answers.append(response)
                        explanations.append(explanation)
                    # budget tracking handled via session usage store

                counts = Counter([a for a in answers if a])
                if counts:
                    most_common = counts.most_common()
                    selected = most_common[0][0]
                    tie = (
                        len(most_common) > 1 and most_common[0][1] == most_common[1][1]
                    )
                    if tie:
                        tied_answers = [
                            ans
                            for ans, votes in most_common
                            if votes == most_common[0][1]
                        ]
                        embedder = try_load_embedder()
                        if embedder is not None and all(
                            a.strip() for a in tied_answers
                        ):
                            try:
                                embeddings = embedder.embed(tied_answers)
                            except Exception as exc:  # pragma: no cover - defensive
                                self._logger.warning(
                                    "aiot_embedder_failed",
                                    error=str(exc),
                                )
                                selected = tied_answers[0]
                            else:
                                selected = select_semantic_representative(
                                    tied_answers, embeddings
                                )
                        else:
                            selected = tied_answers[0]
                    final_response = selected
            self._logger.info(
                "aiot_self_consistency_complete",
                final_len=len(final_response or ""),
            )

        session_id = get_session_id()
        if session_id:
            snapshot = get_usage_store().get(session_id)
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
                    self._model_name: {
                        "input_tokens": snapshot.totals.input_tokens,
                        "output_tokens": snapshot.totals.output_tokens,
                    }
                }
        else:
            usage_metadata = {
                self._model_name: {
                    "input_tokens": usage_counter.input_tokens,
                    "output_tokens": usage_counter.output_tokens,
                }
            }
        self._logger.info(
            "aiot_complete",
            iterations=iteration,
            budget_exhausted=budget_exhausted(usage_counter, self._budget_tokens),
        )
        return {
            "messages": [AIMessage(content=final_response)],
            "usage_metadata": usage_metadata,
        }
