from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Optional, Tuple

import structlog

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from anymind.agents.base import AgentContext
from anymind.agents.bedrock_middleware import BedrockToolResultSanitizer
from anymind.agents.iot_prompts import (
    BRAIN_SYSTEM_PROMPT,
    BRAIN_USER_PROMPT,
    FACILITATOR_SYSTEM_PROMPT,
    FACILITATOR_USER_PROMPT,
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
    pairwise_similarities,
    select_semantic_representative,
    truncate_text,
    tool_feedback_from_ledger,
    try_load_embedder,
)
from anymind.config.schemas import GIoTConfig
from anymind.runtime.json_validation import JSONStructureValidator
from anymind.runtime.usage import extract_usage_from_messages
from anymind.runtime.validated_json import (
    ValidatedJsonResult,
    generate_validated_json,
    generate_validated_json_with_calls,
)


class GIoTAgent:
    name = "giot_agent"

    def build(self, context: AgentContext) -> Any:
        return _GIoTRuntime(context)


class _GIoTRuntime:
    def __init__(self, context: AgentContext) -> None:
        self._context = context
        self._model_name = context.model_config.model
        self._budget_tokens = context.model_config.budget_tokens
        self._settings = context.model_config.giot or GIoTConfig()
        self._embedder = try_load_embedder()
        self._logger = structlog.get_logger("anymind.giot")

        middleware = []
        if context.model_config.model_provider == "bedrock":
            middleware.append(BedrockToolResultSanitizer())

        self._tool_agents = [
            create_agent(
                context.model_client,
                context.tools,
                system_prompt=LLM_SYSTEM_PROMPT,
                middleware=middleware,
                checkpointer=context.checkpointer,
            )
            for _ in range(self._settings.n_agents)
        ]

    async def _call_worker(
        self,
        *,
        agent_idx: int,
        user_prompt: str,
        config: Optional[dict[str, Any]],
    ) -> tuple[str, Optional[dict[str, int]]]:
        result = await self._tool_agents[agent_idx].ainvoke(
            {"messages": [("user", user_prompt)]}, config=config
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
        return response_text, (
            usage if totals.input_tokens or totals.output_tokens else None
        )

    async def _call_fix(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        message = await self._context.model_client.ainvoke(
            [("system", system_prompt), ("user", user_prompt)]
        )
        usage = getattr(message, "usage_metadata", None)
        return message_text(message), usage

    async def _run_validated_worker(
        self,
        *,
        agent_idx: int,
        user_prompt: str,
        validator: JSONStructureValidator,
        config: Optional[dict[str, Any]],
        role_name: str,
        task_context: str,
    ) -> ValidatedJsonResult:
        async def _call(system_prompt: str, user_prompt: str):
            return await self._call_worker(
                agent_idx=agent_idx, user_prompt=user_prompt, config=config
            )

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

    async def _run_one_iteration(
        self,
        *,
        query: str,
        state: dict[str, Any],
        agent_id: int,
        config: Optional[dict[str, Any]],
    ) -> Tuple[str, dict[str, Any], list[dict[str, int]]]:
        usage_list: list[dict[str, int]] = []

        if state.get("self_consistency_applied") is True:
            latest = str(state.get("latest_answer", "") or "").strip()
            if latest:
                return latest, {"response": latest, "answer_to_query": True}, usage_list

        iteration = int(state.get("iteration") or 1)
        tool_feedback = str(
            state.get("tool_scratchpad", "No external tool results yet.")
        ).strip()

        brain_user = BRAIN_USER_PROMPT.format(
            prompt_history=state.get("prompt_history", ""),
            iteration=iteration,
            query=query,
            tool_feedback=tool_feedback,
        )
        brain_result = await generate_validated_json(
            role_name=f"brain_agent_{agent_id}",
            system_prompt=BRAIN_SYSTEM_PROMPT,
            user_prompt=brain_user,
            validator=JSONStructureValidator(
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
            ),
            model_client=self._context.model_client,
            max_reasks=3,
            original_task_context=f"GIoT brain iteration for query: {query}",
        )
        usage_list.extend(brain_result.usage_metadata)

        worker_user = LLM_USER_PROMPT.format(
            brain_thought=str(brain_result.data.get("self_thought", "")),
            prompt_history=state.get("prompt_history", ""),
            query=query,
            tool_feedback=tool_feedback,
        )
        worker_result = await self._run_validated_worker(
            agent_idx=agent_id,
            user_prompt=worker_user,
            validator=JSONStructureValidator(
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
            ),
            config=config,
            role_name=f"worker_agent_{agent_id}",
            task_context=f"GIoT worker iteration for query: {query}",
        )
        usage_list.extend(worker_result.usage_metadata)

        answer = str(worker_result.data.get("response", "")).strip()
        state["latest_answer"] = answer
        state["iteration"] = iteration + 1
        state["prompt_history"] = str(state.get("prompt_history", "")) + (
            f"Iter {iteration} brain:{brain_result.data.get('self_thought','')}\n"
            f"llm:{answer}\n\n"
        )
        state["tool_scratchpad"] = tool_feedback_from_ledger()

        return (
            answer,
            {
                "response": answer,
                "answer_to_query": bool(
                    worker_result.data.get("answer_to_query", False)
                ),
            },
            usage_list,
        )

    async def _apply_self_consistency(
        self,
        *,
        agent_id: int,
        query: str,
        state: dict[str, Any],
        config: Optional[dict[str, Any]],
    ) -> tuple[str, list[dict[str, int]]]:
        prompt = LLM_FINAL_PROMPT.format(
            prompt_history=state.get("prompt_history", ""),
            query=query,
            tool_feedback=state.get("tool_scratchpad", "No external tool results yet."),
        )
        result = await self._run_validated_worker(
            agent_idx=agent_id,
            user_prompt=prompt,
            validator=JSONStructureValidator(
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
            ),
            config=config,
            role_name=f"self_consistency_{agent_id}",
            task_context=f"GIoT self-consistency for query: {query}",
        )
        response = str(result.data.get("response", "")).strip()
        return response, result.usage_metadata

    async def ainvoke(
        self, inputs: dict[str, Any], config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        query = extract_user_input(inputs)
        await ensure_current_time_tool(self._context.tools)
        usage_counter = UsageCounter()
        base_thread_id = None
        if config and isinstance(config, dict):
            base_thread_id = (config.get("configurable", {}) or {}).get("thread_id")
        if not base_thread_id:
            base_thread_id = self._context.model_config.thread_id
        agent_configs = [
            {"configurable": {"thread_id": f"{base_thread_id}::giot::{i}"}}
            for i in range(self._settings.n_agents)
        ]

        agent_states: list[dict[str, Any]] = [
            {
                "query": query,
                "prompt_history": "",
                "latest_answer": "No answer yet",
                "tool_scratchpad": tool_feedback_from_ledger(),
                "iteration": 1,
                "self_consistency_applied": False,
            }
            for _ in range(self._settings.n_agents)
        ]

        vote_k = (
            max(1, int(self._settings.vote_ratio * self._settings.n_agents))
            if self._settings.n_agents > 1
            else 1
        )

        self._logger.info(
            "giot_start",
            query=query,
            n_agents=self._settings.n_agents,
            vote_k=vote_k,
            budget_tokens=self._budget_tokens,
            tools=len(self._context.tools),
            embedder_loaded=self._embedder is not None,
        )

        round_num = 1
        while True:
            if budget_exhausted(usage_counter, self._budget_tokens):
                break

            self._logger.info("giot_round_start", round=round_num)
            tasks = [
                self._run_one_iteration(
                    query=query,
                    state=agent_states[i],
                    agent_id=i,
                    config=agent_configs[i],
                )
                for i in range(self._settings.n_agents)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            answers: list[str] = []
            agent_responses: list[dict[str, Any]] = []
            for result in results:
                if isinstance(result, BaseException):
                    answers.append("Error")
                    agent_responses.append(
                        {"response": "Error", "answer_to_query": False}
                    )
                    continue
                answer, response_obj, usage_list = result
                answers.append(answer)
                agent_responses.append(response_obj)
                usage_counter.add_usage_list(usage_list)

            if self._settings.trace_steps:
                self._logger.info(
                    "giot_round_answers",
                    round=round_num,
                    answers=[
                        truncate_text(a, int(self._settings.trace_max_chars))
                        for a in answers
                    ],
                )

            if budget_exhausted(usage_counter, self._budget_tokens):
                break

            complete_count = sum(
                1 for r in agent_responses if r.get("answer_to_query") is True
            )
            self._logger.info(
                "giot_round_complete",
                round=round_num,
                complete_count=complete_count,
            )
            if complete_count < vote_k:
                round_num += 1
                continue

            if self._settings.self_consistency_samples > 1:
                self._logger.info(
                    "giot_self_consistency_start",
                    round=round_num,
                    samples=int(self._settings.self_consistency_samples),
                )
                sc_tasks: list[asyncio.Task[tuple[str, list[dict[str, int]]]]] = []
                sc_indices: list[int] = []
                for i in range(self._settings.n_agents):
                    if agent_responses[i].get("answer_to_query") is not True:
                        continue
                    if agent_states[i].get("self_consistency_applied") is True:
                        continue
                    sc_indices.append(i)
                    sc_tasks.append(
                        asyncio.create_task(
                            self._apply_self_consistency(
                                agent_id=i,
                                query=query,
                                state=agent_states[i],
                                config=agent_configs[i],
                            )
                        )
                    )
                if sc_tasks:
                    sc_results = await asyncio.gather(*sc_tasks, return_exceptions=True)
                    for idx, result in zip(sc_indices, sc_results):
                        if isinstance(result, BaseException):
                            continue
                        refined, usage_list = result
                        if refined:
                            answers[idx] = refined
                            agent_states[idx]["latest_answer"] = refined
                            agent_responses[idx]["response"] = refined
                            agent_states[idx]["self_consistency_applied"] = True
                        usage_counter.add_usage_list(usage_list)
                self._logger.info(
                    "giot_self_consistency_complete",
                    round=round_num,
                )

            if self._embedder is not None and all(
                a.strip() and a != "Error" for a in answers
            ):
                try:
                    embeddings = self._embedder.embed(answers)
                except Exception as exc:  # pragma: no cover - defensive
                    self._logger.warning(
                        "giot_embedder_failed",
                        round=round_num,
                        error=str(exc),
                    )
                    self._embedder = None
                else:
                    similarities = pairwise_similarities(embeddings)
                    threshold = max(
                        self._settings.sim_min,
                        self._settings.sim_start
                        - self._settings.sim_decay * (round_num - 1),
                    )
                    semantic_consensus = bool(similarities) and all(
                        sim >= threshold for sim in similarities
                    )
                    if semantic_consensus:
                        final_answer = select_semantic_representative(
                            answers, embeddings
                        )
                        self._logger.info(
                            "giot_consensus_semantic",
                            round=round_num,
                            threshold=threshold,
                        )
                        return self._build_response(final_answer, usage_counter)

            formatted_answers = "\n".join(
                [f"Agent {i + 1}: {a}" for i, a in enumerate(answers)]
            )
            if self._settings.trace_steps:
                self._logger.info(
                    "giot_facilitator_prompt",
                    round=round_num,
                    prompt=truncate_text(
                        formatted_answers, int(self._settings.trace_max_chars)
                    ),
                )
            facilitator = await generate_validated_json(
                role_name="facilitator",
                system_prompt=FACILITATOR_SYSTEM_PROMPT,
                user_prompt=FACILITATOR_USER_PROMPT.format(answers=formatted_answers),
                validator=JSONStructureValidator(
                    {
                        "consensus": {
                            "type": bool,
                            "required": True,
                            "description": "consensus flag",
                        },
                        "explanation": {
                            "type": str,
                            "required": True,
                            "description": "explanation",
                        },
                    },
                    validator_name="facilitator-response-validator",
                ),
                model_client=self._context.model_client,
                max_reasks=3,
                original_task_context=f"GIoT facilitator consensus check for query: {query}",
            )
            usage_counter.add_usage_list(facilitator.usage_metadata)
            if self._settings.trace_steps:
                self._logger.info(
                    "giot_facilitator_decision",
                    round=round_num,
                    consensus=bool(facilitator.data.get("consensus")),
                    explanation=truncate_text(
                        str(facilitator.data.get("explanation", "") or ""),
                        int(self._settings.trace_max_chars),
                    ),
                )
            if bool(facilitator.data.get("consensus")):
                if self._embedder is not None and all(
                    a.strip() and a != "Error" for a in answers
                ):
                    try:
                        embeddings = self._embedder.embed(answers)
                    except Exception as exc:  # pragma: no cover - defensive
                        self._logger.warning(
                            "giot_embedder_failed",
                            round=round_num,
                            error=str(exc),
                        )
                        self._embedder = None
                        final_answer = Counter(
                            [a for a in answers if a != "Error"]
                        ).most_common(1)[0][0]
                    else:
                        final_answer = select_semantic_representative(
                            answers, embeddings
                        )
                else:
                    final_answer = Counter(
                        [a for a in answers if a != "Error"]
                    ).most_common(1)[0][0]
                self._logger.info(
                    "giot_consensus_facilitator",
                    round=round_num,
                )
                return self._build_response(final_answer, usage_counter)

            round_num += 1

        best_answer = self._select_best_answer(
            [state.get("latest_answer", "") for state in agent_states]
        )
        self._logger.info(
            "giot_complete",
            rounds=round_num,
            budget_exhausted=budget_exhausted(usage_counter, self._budget_tokens),
        )
        return self._build_response(best_answer, usage_counter)

    def _select_best_answer(self, answers: list[str]) -> str:
        valid = [a.strip() for a in answers if isinstance(a, str) and a.strip()]
        if not valid:
            return "No answer produced before budget exhaustion."
        if self._embedder is not None and len(valid) > 1:
            try:
                embeddings = self._embedder.embed(valid)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.warning("giot_embedder_failed", error=str(exc))
                self._embedder = None
            else:
                return select_semantic_representative(valid, embeddings)
        return Counter(valid).most_common(1)[0][0]

    def _build_response(
        self, final_answer: str, usage_counter: UsageCounter
    ) -> dict[str, Any]:
        usage_metadata = {
            self._model_name: {
                "input_tokens": usage_counter.input_tokens,
                "output_tokens": usage_counter.output_tokens,
            }
        }
        return {
            "messages": [AIMessage(content=final_answer)],
            "usage_metadata": usage_metadata,
        }
