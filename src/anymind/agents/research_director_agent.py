from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import structlog
from langchain_core.messages import AIMessage

from anymind.agents.aiot_agent import AIoTAgent
from anymind.agents.agot_agent import AGoTAgent
from anymind.agents.giot_agent import GIoTAgent
from anymind.agents.got_agent import GoTAgent
from anymind.agents.base import AgentContext
from anymind.agents.iot_utils import (
    UsageCounter,
    budget_exhausted,
    build_conversation_query,
    ensure_current_time_tool,
    extract_conversation_messages,
    extract_user_input,
    message_text,
    tool_feedback_from_ledger,
    truncate_text,
    try_load_embedder,
)
from anymind.agents.research_director.brain_selector import select_brain_for_question
from anymind.agents.research_director.prompts import (
    FINAL_SYSTEM_PROMPT,
    MANAGER_SYSTEM_PROMPT,
)
from anymind.config.schemas import ResearchDirectorConfig
from anymind.runtime.json_validation import JSONStructureValidator
from anymind.runtime.validated_json import (
    ValidatedJsonResult,
    generate_validated_json_with_calls,
)


@dataclass
class ProbeResult:
    probe_id: str
    gap_id: str
    probe_question: str
    strategy: str
    status: str
    answer: str


class ResearchDirectorAgent:
    name = "research_agent"

    def build(self, context: AgentContext) -> Any:
        return _ResearchDirectorRuntime(context)


class _ResearchDirectorRuntime:
    def __init__(self, context: AgentContext) -> None:
        self._context = context
        self._model_name = context.model_config.model
        self._budget_tokens = context.model_config.budget_tokens
        self._settings = (
            context.model_config.research_director or ResearchDirectorConfig()
        )
        self._logger = structlog.get_logger("anymind.research_agent")
        self._embedder = try_load_embedder()

        self._probe_runtimes: dict[str, Any] = {
            "aiot": AIoTAgent().build(context),
            "giot": GIoTAgent().build(context),
            "agot": AGoTAgent().build(context),
            "got": GoTAgent().build(context),
        }

    def _apply_usage_map(
        self, usage_metadata: Optional[dict[str, dict[str, int]]], counter: UsageCounter
    ) -> None:
        if not usage_metadata:
            return
        for usage in usage_metadata.values():
            counter.add_usage(usage)

    def _truncate(self, text: str) -> str:
        return truncate_text(text, int(self._settings.trace_max_chars))

    def _cosine(self, a: Any, b: Any) -> float:
        if a is None or b is None:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _validate_manager_decision(self, obj: dict[str, Any]) -> tuple[bool, str]:
        allowed_top = {"action", "gaps", "probes", "probe_batches", "rationale"}
        extra_top = [k for k in obj.keys() if k not in allowed_top]
        if extra_top:
            return False, f"Extra top-level key(s) not permitted: {extra_top}"

        action = obj.get("action")
        if action not in {"run_probes", "finalize"}:
            return False, "action must be 'run_probes' or 'finalize'"

        gaps = obj.get("gaps", []) or []
        if not isinstance(gaps, list):
            return False, "Field 'gaps' must be a list"

        probes = obj.get("probes", []) or []
        if not isinstance(probes, list):
            return False, "Field 'probes' must be a list"

        probe_batches = obj.get("probe_batches", []) or []
        if not isinstance(probe_batches, list):
            return False, "Field 'probe_batches' must be a list of lists"

        if (
            isinstance(probe_batches, list)
            and len(probe_batches) == 1
            and isinstance(probe_batches[0], list)
            and probe_batches[0]
            and all(isinstance(x, list) for x in probe_batches[0])
        ):
            probe_batches = probe_batches[0]
            obj["probe_batches"] = probe_batches

        normalized_batches: list[list[str]] = []
        for batch in probe_batches:
            if not isinstance(batch, list):
                normalized_batches = []
                break
            normalized_batch: list[str] = []
            for pid in batch:
                if isinstance(pid, str) and pid.strip():
                    normalized_batch.append(pid.strip())
                    continue
                if isinstance(pid, dict):
                    candidate = (
                        pid.get("probe_id") or pid.get("probeId") or pid.get("id")
                    )
                    if isinstance(candidate, str) and candidate.strip():
                        normalized_batch.append(candidate.strip())
                        continue
                normalized_batch.append("")
            normalized_batches.append(normalized_batch)
        if normalized_batches:
            obj["probe_batches"] = normalized_batches
            probe_batches = normalized_batches

        if action == "finalize":
            return True, ""

        if not gaps:
            return False, "action='run_probes' requires at least one gap"
        if not probes:
            return False, "action='run_probes' requires at least one probe"

        gap_ids: set[str] = set()
        for idx, gap in enumerate(gaps):
            if not isinstance(gap, dict):
                return False, f"Gap at index {idx} must be an object"
            allowed_gap = {"gap_id", "gap"}
            extra_gap = [k for k in gap.keys() if k not in allowed_gap]
            if extra_gap:
                return False, f"Extra key(s) in gaps[{idx}] not permitted: {extra_gap}"
            missing_gap = [k for k in ("gap_id", "gap") if k not in gap]
            if missing_gap:
                return False, f"Missing key(s) in gaps[{idx}]: {missing_gap}"
            gid = gap.get("gap_id")
            if not isinstance(gid, str) or not gid.strip():
                return False, f"Invalid gaps[{idx}].gap_id"
            if gid in gap_ids:
                return False, f"Duplicate gap_id: {gid!r}"
            if not isinstance(gap.get("gap"), str) or not str(gap["gap"]).strip():
                return False, f"Invalid gaps[{idx}].gap"
            gap_ids.add(gid)

        probe_ids: list[str] = []
        probe_by_id: dict[str, dict[str, Any]] = {}
        for idx, probe in enumerate(probes):
            if not isinstance(probe, dict):
                return False, f"Probe at index {idx} must be an object"
            allowed = {"probe_id", "gap_id", "probe_question"}
            extra = [k for k in probe.keys() if k not in allowed]
            if extra:
                return False, f"Extra key(s) in probes[{idx}] not permitted: {extra}"
            missing = [
                k for k in ("probe_id", "gap_id", "probe_question") if k not in probe
            ]
            if missing:
                return False, f"Missing key(s) in probes[{idx}]: {missing}"
            pid = probe.get("probe_id")
            if not isinstance(pid, str) or not pid.strip():
                return False, f"Invalid probes[{idx}].probe_id"
            pid = pid.strip()
            if pid in probe_by_id:
                return False, f"Duplicate probe_id: {pid!r}"
            gid_raw = probe.get("gap_id")
            if not isinstance(gid_raw, str) or not gid_raw.strip():
                return False, f"Invalid probes[{idx}].gap_id"
            gid = gid_raw.strip()
            if gid not in gap_ids:
                return False, f"Unknown probes[{idx}].gap_id: {gid_raw!r}"
            q = probe.get("probe_question")
            if not isinstance(q, str) or not q.strip():
                return False, f"Invalid probes[{idx}].probe_question"
            probe_by_id[pid] = probe
            probe_ids.append(pid)

        if not probe_batches:
            probe_batches = [probe_ids]
            obj["probe_batches"] = probe_batches

        if not probe_ids:
            return False, "No probe_ids found in probes[]"

        known_ids = set(probe_ids)
        seen: set[str] = set()
        for bidx, batch in enumerate(probe_batches):
            if not isinstance(batch, list):
                return (
                    False,
                    f"probe_batches[{bidx}] must be a list of probe_id strings",
                )
            if not batch:
                return False, f"probe_batches[{bidx}] cannot be empty"
            for pid in batch:
                if not isinstance(pid, str) or not pid.strip():
                    return False, f"probe_batches[{bidx}] contains invalid probe_id"
                if pid not in known_ids:
                    return (
                        False,
                        f"probe_batches[{bidx}] references unknown probe_id: {pid!r}",
                    )
                if pid in seen:
                    return False, f"probe_id appears in multiple batches: {pid!r}"
                seen.add(pid)

        missing_ids = sorted(known_ids - seen)
        if missing_ids:
            return False, f"probe_batches missing probe_id(s): {missing_ids}"

        return True, ""

    def _guard_probes_not_paraphrases(
        self,
        *,
        user_question: str,
        probes: list[dict[str, Any]],
        emb_cache: dict[str, Any],
    ) -> None:
        uq = (user_question or "").strip()
        if not uq:
            return
        seen: set[str] = set()
        for probe in probes:
            pid = str(probe.get("probe_id") or "").strip()
            pq = str(probe.get("probe_question") or "").strip()
            lowered = pq.lower()
            if lowered == uq.lower():
                raise ValueError(
                    f"Probe {pid or '<missing>'} repeats the user question"
                )
            if lowered in seen:
                raise ValueError(f"Duplicate probe question: {pid or '<missing>'}")
            seen.add(lowered)

        if self._embedder is None:
            return

        if uq in emb_cache:
            user_vec = emb_cache[uq]
        else:
            vecs = self._embedder.embed([uq])
            user_vec = vecs[0] if getattr(vecs, "shape", (0,))[0] == 1 else None
            emb_cache[uq] = user_vec

        threshold = float(self._settings.similarity_threshold)
        for probe in probes:
            pid = str(probe.get("probe_id") or "").strip()
            pq = str(probe.get("probe_question") or "").strip()
            if not pq:
                continue
            if pq in emb_cache:
                vec = emb_cache[pq]
            else:
                vecs = self._embedder.embed([pq])
                vec = vecs[0] if getattr(vecs, "shape", (0,))[0] == 1 else None
                emb_cache[pq] = vec
            sim = self._cosine(user_vec, vec)
            if sim >= threshold:
                raise ValueError(
                    f"Probe {pid or '<missing>'} too similar to user question (sim={sim:.3f})"
                )

    async def _call_llm(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        message = await self._context.model_client.ainvoke(
            [("system", system_prompt), ("user", user_prompt)]
        )
        usage = getattr(message, "usage_metadata", None)
        return message_text(message), usage

    async def _manager_decide(
        self,
        *,
        query: str,
        probe_history: list[ProbeResult],
        tool_names: list[str],
        feedback: str,
        usage_counter: UsageCounter,
    ) -> dict[str, Any]:
        manager_validator = JSONStructureValidator(
            {
                "action": {
                    "type": str,
                    "required": True,
                    "description": "run_probes|finalize",
                },
                "gaps": {"type": list, "required": False, "description": "gap list"},
                "probes": {
                    "type": list,
                    "required": False,
                    "description": "probe list",
                },
                "probe_batches": {
                    "type": list,
                    "required": False,
                    "description": "batch list",
                },
                "rationale": {
                    "type": str,
                    "required": False,
                    "description": "short rationale",
                },
            },
            validator_name="research-director-manager",
        )

        history_text = "[]"
        if probe_history:
            history_payload = [
                {
                    "probe_id": p.probe_id,
                    "gap_id": p.gap_id,
                    "strategy": p.strategy,
                    "status": p.status,
                    "probe_question": p.probe_question,
                    "answer_preview": truncate_text(
                        p.answer, int(self._settings.probe_answer_preview_chars)
                    ),
                }
                for p in reversed(probe_history[-12:])
            ]
            history_text = json.dumps(history_payload, ensure_ascii=False)

        tools_available = bool(tool_names)
        evidence_text = tool_feedback_from_ledger()

        user_prompt = (
            f"USER QUESTION:\n{query.strip()}\n\n"
            f"TOOLS AVAILABLE: {tools_available}\n"
            f"TOOL NAMES: {', '.join(tool_names) if tool_names else 'None'}\n\n"
            f"FEEDBACK (if any):\n{feedback.strip() or 'None'}\n\n"
            f"PROBE HISTORY (latest-first JSON):\n{history_text}\n\n"
            f"EVIDENCE SUMMARY:\n{evidence_text}\n\n"
            "Return the decision JSON now."
        )

        result = await generate_validated_json_with_calls(
            role_name="research_agent_manager",
            system_prompt=MANAGER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            validator=manager_validator,
            call_model=self._call_llm,
            fix_model=self._call_llm,
            max_reasks=3,
            original_task_context="Research director manager decision",
        )
        usage_counter.add_usage_list(result.usage_metadata)

        ok, err = self._validate_manager_decision(result.data)
        if not ok:
            raise ValueError(err)
        return result.data

    async def _run_probe(
        self,
        *,
        probe: dict[str, Any],
        iteration: int,
        base_thread_id: str,
        semaphore: asyncio.Semaphore,
        usage_counter: UsageCounter,
    ) -> ProbeResult:
        probe_id = str(probe.get("probe_id") or "").strip()
        gap_id = str(probe.get("gap_id") or "").strip()
        question = str(probe.get("probe_question") or "").strip()
        strategy, level, score, _features = select_brain_for_question(question)

        if self._settings.trace_steps:
            self._logger.info(
                "research_agent_probe_start",
                probe_id=probe_id,
                gap_id=gap_id,
                strategy=strategy,
                level=level,
                score=score,
            )

        async with semaphore:
            try:
                runtime = self._probe_runtimes[strategy]
                thread_id = f"{base_thread_id}::rd::{iteration}::{probe_id}"
                result = await runtime.ainvoke(
                    {"messages": [("user", question)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                messages = result.get("messages", [])
                answer = message_text(messages[-1]) if messages else ""
                self._apply_usage_map(result.get("usage_metadata"), usage_counter)
                status = "ok"
            except Exception as exc:
                answer = f"Error: {exc}"
                status = "error"

        if self._settings.trace_steps:
            self._logger.info(
                "research_agent_probe_end",
                probe_id=probe_id,
                status=status,
                answer_preview=self._truncate(answer),
            )

        return ProbeResult(
            probe_id=probe_id,
            gap_id=gap_id,
            probe_question=question,
            strategy=strategy,
            status=status,
            answer=answer,
        )

    async def _finalize(
        self,
        *,
        query: str,
        probe_history: list[ProbeResult],
        usage_counter: UsageCounter,
    ) -> str:
        final_validator = JSONStructureValidator(
            {
                "final_answer": {
                    "type": str,
                    "required": True,
                    "description": "final answer",
                }
            },
            validator_name="research-director-final",
        )

        summary_payload = [
            {
                "probe_id": p.probe_id,
                "gap_id": p.gap_id,
                "strategy": p.strategy,
                "status": p.status,
                "probe_question": p.probe_question,
                "answer": truncate_text(
                    p.answer, int(self._settings.probe_answer_preview_chars)
                ),
            }
            for p in probe_history
        ]
        summary_text = json.dumps(summary_payload, ensure_ascii=False)
        evidence_text = tool_feedback_from_ledger()

        user_prompt = (
            f"USER QUESTION:\n{query.strip()}\n\n"
            f"PROBE OUTPUTS (JSON):\n{summary_text}\n\n"
            f"EVIDENCE SUMMARY:\n{evidence_text}\n\n"
            "Return final answer JSON now."
        )

        result = await generate_validated_json_with_calls(
            role_name="research_agent_final",
            system_prompt=FINAL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            validator=final_validator,
            call_model=self._call_llm,
            fix_model=self._call_llm,
            max_reasks=3,
            original_task_context="Research director final synthesis",
        )
        usage_counter.add_usage_list(result.usage_metadata)
        return str(result.data.get("final_answer", "") or "").strip()

    async def ainvoke(
        self, inputs: dict[str, Any], config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        conversation = extract_conversation_messages(inputs)
        if conversation and len(conversation) > 1:
            query = build_conversation_query(conversation)
        else:
            query = extract_user_input(inputs)
        await ensure_current_time_tool(self._context.tools)
        usage_counter = UsageCounter()
        probe_history: list[ProbeResult] = []
        feedback = ""
        iteration = 1

        base_thread_id = None
        if config and isinstance(config, dict):
            base_thread_id = (config.get("configurable", {}) or {}).get("thread_id")
        if not base_thread_id:
            base_thread_id = self._context.model_config.thread_id

        tool_names = [
            str(getattr(tool, "name", "") or "") for tool in self._context.tools
        ]
        emb_cache: dict[str, Any] = {}

        self._logger.info(
            "research_agent_start",
            query=query,
            budget_tokens=self._budget_tokens,
            tools=len(self._context.tools),
            max_parallel_probes=self._settings.max_parallel_probes,
        )

        while True:
            if budget_exhausted(usage_counter, self._budget_tokens):
                break

            if self._settings.trace_steps:
                self._logger.info(
                    "research_agent_iteration_start",
                    iteration=iteration,
                )

            try:
                decision = await self._manager_decide(
                    query=query,
                    probe_history=probe_history,
                    tool_names=tool_names,
                    feedback=feedback,
                    usage_counter=usage_counter,
                )
                feedback = ""
            except Exception as exc:
                feedback = f"Manager validation error: {exc}"
                self._logger.warning(
                    "research_agent_manager_invalid",
                    iteration=iteration,
                    error=str(exc),
                )
                continue

            if budget_exhausted(usage_counter, self._budget_tokens):
                break

            action = str(decision.get("action") or "").strip()
            if action == "finalize":
                final_answer = await self._finalize(
                    query=query,
                    probe_history=probe_history,
                    usage_counter=usage_counter,
                )
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

            probes = decision.get("probes", []) or []
            probe_batches = decision.get("probe_batches", []) or []

            try:
                self._guard_probes_not_paraphrases(
                    user_question=query,
                    probes=probes,
                    emb_cache=emb_cache,
                )
            except Exception as exc:
                feedback = f"Probe guard failed: {exc}"
                self._logger.warning(
                    "research_agent_probe_guard_failed",
                    iteration=iteration,
                    error=str(exc),
                )
                continue

            max_probes = max(1, int(self._settings.max_probes_per_iteration))
            if len(probes) > max_probes:
                feedback = f"Too many probes ({len(probes)}). Limit to {max_probes} per iteration."
                probes = probes[:max_probes]
                probe_ids = {p.get("probe_id") for p in probes}
                probe_batches = [
                    [pid for pid in batch if pid in probe_ids]
                    for batch in probe_batches
                ]
                probe_batches = [batch for batch in probe_batches if batch]

            if not probe_batches:
                probe_batches = [
                    [p.get("probe_id") for p in probes if p.get("probe_id")]
                ]

            id_to_probe = {
                str(p.get("probe_id") or "").strip(): p
                for p in probes
                if p.get("probe_id")
            }
            batch = probe_batches[0]

            semaphore = asyncio.Semaphore(
                max(1, int(self._settings.max_parallel_probes))
            )
            tasks = []
            for pid in batch:
                probe = id_to_probe.get(str(pid))
                if not probe:
                    continue
                tasks.append(
                    self._run_probe(
                        probe=probe,
                        iteration=iteration,
                        base_thread_id=base_thread_id,
                        semaphore=semaphore,
                        usage_counter=usage_counter,
                    )
                )

            if tasks:
                results = await asyncio.gather(*tasks)
                probe_history.extend(results)

            if budget_exhausted(usage_counter, self._budget_tokens):
                break

            iteration += 1

        fallback = (
            "Budget exhausted before final synthesis. Unable to complete the request."
        )
        if probe_history:
            fallback = (
                "Budget exhausted before final synthesis. "
                "Collected probe results:\n"
                + "\n".join(
                    f"- {p.probe_id} ({p.strategy}): {truncate_text(p.answer, 300)}"
                    for p in probe_history[-5:]
                )
            )
        usage_metadata = {
            self._model_name: {
                "input_tokens": usage_counter.input_tokens,
                "output_tokens": usage_counter.output_tokens,
            }
        }
        return {
            "messages": [AIMessage(content=fallback)],
            "usage_metadata": usage_metadata,
        }
