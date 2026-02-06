from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

import structlog

from anymind.agents.agot.prompts import (
    COMPLEXITY_CHECK_SYS_PROMPT,
    COMPLEXITY_CHECK_USER_PROMPT,
    FINAL_TASK_EXECUTION_PROMPT,
    FINAL_TASK_SYS_PROMPT,
    FINAL_TASK_USER_PROMPT,
    SYSTEM_PROMPT,
    TASK_EXECUTION_SYS_PROMPT,
    TASK_EXECUTION_USER_PROMPT,
    USER_PROMPT_INITIAL_SUB_TASK,
    USER_PROMPT_INITIAL_TASK,
    USER_PROMPT_NEW_TASK,
)
from anymind.agents.agot.tasks import Task
from anymind.agents.agot.validators import (
    TaskListResponseValidator,
    make_validator_complexity,
    make_validator_eval,
    make_validator_final_answer,
    make_validator_final_task,
)
from anymind.agents.agot.heritage import Heritage
from anymind.agents.agot.nested_graph import NestedGraph
from anymind.agents.iot_utils import tool_feedback_from_ledger
from anymind.runtime.validated_json import ValidatedJsonResult

PlannerRunner = Callable[[str, str, str, Any, str], Awaitable[ValidatedJsonResult]]


class AGoTMappings:
    def __init__(
        self,
        *,
        planner_runner: PlannerRunner,
        worker_runner: PlannerRunner,
        d_max: int,
    ) -> None:
        self._planner_runner = planner_runner
        self._worker_runner = worker_runner
        self._d_max = int(d_max)
        self._log = structlog.get_logger("anymind.agot")

    async def t_empty(self, query: str, n_max: int) -> tuple[list[Task], str]:
        user_prompt = USER_PROMPT_INITIAL_TASK.format(
            question=_normalize_query_for_prompt(query),
            max_new_tasks=n_max,
            tool_feedback=self._current_tool_feedback(),
        )
        validator = TaskListResponseValidator(
            max_tasks=n_max, validator_name="agot_t_empty"
        )
        obj = await self._planner_runner(
            "agot_t_empty",
            SYSTEM_PROMPT,
            user_prompt,
            validator,
            f"AGoT T_empty for query: {query}",
        )

        tasks: list[Task] = []
        for i, item in enumerate(obj.data.get("tasks", [])[:n_max]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "")
            content = str(item.get("content", "") or "")
            tasks.append(
                Task(heritage=Heritage.create_root(0, i), title=title, content=content)
            )

        strategy = str(obj.data.get("strategy", "") or "")
        self._log.info("agot_T_empty_complete", n_tasks=len(tasks), strategy=strategy)
        return tasks, strategy

    async def t_0(
        self, query: str, parent_graph: NestedGraph, n_max: int
    ) -> tuple[list[Task], str]:
        context_summary = self._create_context_summary(parent_graph)

        if "\x1e" in query:
            task_title, task_content = query.split("\x1e", 1)
        else:
            task_title, task_content = "Initial Analysis", query

        user_prompt = USER_PROMPT_INITIAL_SUB_TASK.format(
            task_title=task_title,
            task_content=task_content,
            context_summary=context_summary,
            max_new_tasks=n_max,
            tool_feedback=self._current_tool_feedback(),
        )
        validator = TaskListResponseValidator(
            max_tasks=n_max, validator_name="agot_t_0"
        )
        obj = await self._planner_runner(
            "agot_t_0",
            SYSTEM_PROMPT,
            user_prompt,
            validator,
            f"AGoT T_0 for query: {query}",
        )

        tasks: list[Task] = []
        for i, item in enumerate(obj.data.get("tasks", [])[:n_max]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "")
            content = str(item.get("content", "") or "")
            heritage = parent_graph.heritage.append(0, i)
            tasks.append(Task(heritage=heritage, title=title, content=content))

        strategy = str(obj.data.get("strategy", "") or "")
        self._log.info(
            "agot_T_0_complete",
            n_tasks=len(tasks),
            strategy=strategy,
            parent_heritage=str(parent_graph.heritage),
        )
        return tasks, strategy

    async def t_e(
        self,
        query: str,
        graph: NestedGraph,
        n_max: int,
    ) -> tuple[list[Task], str, list[tuple[Heritage, Heritage]]]:
        context_summary = self._create_context_summary(graph)
        current_strategy = (
            graph.current_strategy
            or "Continue systematic analysis based on current findings"
        )
        user_prompt = USER_PROMPT_NEW_TASK.format(
            context_summary=context_summary,
            current_strategy=current_strategy,
            max_new_tasks=n_max,
            tool_feedback=self._current_tool_feedback(),
        )

        validator = TaskListResponseValidator(
            max_tasks=n_max, validator_name="agot_t_e"
        )
        obj = await self._planner_runner(
            "agot_t_e",
            SYSTEM_PROMPT,
            user_prompt,
            validator,
            f"AGoT T_e for query: {query}",
        )

        next_layer = int(graph.current_layer) + 1
        tasks: list[Task] = []
        for i, item in enumerate(obj.data.get("tasks", [])[:n_max]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "")
            content = str(item.get("content", "") or "")
            heritage = graph.heritage.append(next_layer, i)
            tasks.append(Task(heritage=heritage, title=title, content=content))

        edges: list[tuple[Heritage, Heritage]] = []
        if tasks:
            prev_layer = (
                max(0, int(graph.current_layer) - 1)
                if int(graph.current_layer) > 0
                else 0
            )
            for prev_heritage, _ in graph.get_nodes_by_layer(prev_layer):
                for task in tasks:
                    edges.append((prev_heritage, task.heritage))

        strategy = str(obj.data.get("strategy", "") or "")
        self._log.info(
            "agot_T_e_complete",
            n_tasks=len(tasks),
            n_edges=len(edges),
            strategy=strategy,
            graph_heritage=str(graph.heritage),
        )
        return tasks, strategy, edges

    async def c(self, thought: Task, graph: NestedGraph) -> bool:
        context_summary = self._create_context_summary(graph)
        current_depth = graph.heritage.depth() + 1
        user_prompt = COMPLEXITY_CHECK_USER_PROMPT.format(
            task_title=thought.title,
            task_content=thought.content,
            depth=current_depth,
            context_summary=context_summary,
            depth_threshold=self._d_max,
            tool_feedback=self._current_tool_feedback(),
        )
        validator = make_validator_complexity(validator_name="agot_complexity_check")
        obj = await self._planner_runner(
            "agot_complexity_check",
            COMPLEXITY_CHECK_SYS_PROMPT,
            user_prompt,
            validator,
            f"AGoT complexity check for: {thought.title}",
        )
        return bool(obj.data.get("is_complex", False))

    async def eval(self, thought: Task, graph: NestedGraph, query: str) -> str:
        context_summary = self._create_context_summary(graph)
        user_prompt = TASK_EXECUTION_USER_PROMPT.format(
            original_query=_normalize_query_for_prompt(query),
            task_title=thought.title,
            task_content=thought.content,
            context_summary=context_summary,
            tool_feedback=self._current_tool_feedback(),
        )

        validator = make_validator_eval(validator_name="agot_eval")
        obj = await self._worker_runner(
            "agot_eval",
            TASK_EXECUTION_SYS_PROMPT,
            user_prompt,
            validator,
            f"AGoT task evaluation: {thought.title}",
        )
        return _coerce_answer_to_text(obj.data.get("answer"))

    async def phi(self, graph: NestedGraph) -> Task:
        findings_summary = self._create_findings_summary(graph)
        user_prompt = FINAL_TASK_USER_PROMPT.format(findings_summary=findings_summary)

        validator = make_validator_final_task(validator_name="agot_final_task")
        obj = await self._planner_runner(
            "agot_final_task",
            FINAL_TASK_SYS_PROMPT,
            user_prompt,
            validator,
            "AGoT final synthesis task generation",
        )

        final_layer = graph.get_layer_count()
        final_heritage = graph.heritage.append(final_layer, 0)
        title = str(obj.data.get("title", "") or "")
        content = str(obj.data.get("content", "") or "")
        return Task(heritage=final_heritage, title=title, content=content)

    async def execute_final_task(
        self, final_task: Task, graph: NestedGraph
    ) -> dict[str, Any]:
        task_graph_snapshot = self.get_task_graph_snapshot(graph)
        task_graph_json = json.dumps(
            task_graph_snapshot, ensure_ascii=False, indent=2, default=str
        )

        user_prompt = FINAL_TASK_EXECUTION_PROMPT.format(
            final_task_title=final_task.title,
            final_task_content=final_task.content,
            task_graph=task_graph_json,
        )

        validator = make_validator_final_answer(validator_name="agot_final_answer")
        obj = await self._worker_runner(
            "agot_final_answer",
            TASK_EXECUTION_SYS_PROMPT,
            user_prompt,
            validator,
            "AGoT final synthesis execution",
        )

        if isinstance(obj.data.get("graph"), str):
            return obj.data
        obj.data["graph"] = task_graph_json
        return obj.data

    def get_task_graph_snapshot(self, graph: NestedGraph) -> dict[str, Any]:
        return self._build_task_graph_snapshot(graph)

    def _build_task_graph_snapshot(self, graph: NestedGraph) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for heritage, node in sorted(graph.nodes.items(), key=lambda item: item[0]):
            node_payload: dict[str, Any] = {
                "heritage": str(heritage),
                "title": node.thought.title,
                "content": node.thought.content,
                "answer": node.answer,
            }
            if node.nested_graph is not None:
                node_payload["nested_graph"] = self._build_task_graph_snapshot(
                    node.nested_graph
                )
            nodes.append(node_payload)

        snapshot: dict[str, Any] = {"heritage": str(graph.heritage), "nodes": nodes}
        if graph.final_answer is not None:
            snapshot["final_answer"] = graph.final_answer
        return snapshot

    def _create_findings_summary(self, graph: NestedGraph) -> str:
        findings: list[str] = []
        for node in graph.nodes.values():
            if node.answer and node.thought.title:
                findings.append(f"{node.thought.title}: {node.answer}")
        if not findings:
            return "No completed analysis available for synthesis."
        return "\n".join(findings)

    def _create_context_summary(self, graph: NestedGraph) -> str:
        completed_nodes: list[str] = []
        current_layer_nodes: list[str] = []
        current_layer = int(graph.current_layer)

        for heritage, node in graph.nodes.items():
            try:
                layer_num, _ = heritage.current_position()
            except ValueError:
                continue

            if layer_num < current_layer and node.answer:
                completed_nodes.append(f"- {node.thought.title}: {node.answer}")
            elif layer_num == current_layer:
                status = "completed" if node.answer else "in progress"
                current_layer_nodes.append(f"- {node.thought.title} ({status})")

        parts: list[str] = []
        if completed_nodes:
            parts.append("Completed analysis from earlier layers:")
            parts.extend(completed_nodes)
        if current_layer_nodes:
            parts.append("Current layer tasks:")
            parts.extend(current_layer_nodes)
        if not parts:
            parts.append("No prior graph context available.")

        tool_feedback = self._current_tool_feedback()
        if tool_feedback:
            parts.append("External tool findings:")
            parts.append(tool_feedback)

        return "\n".join(parts)

    def _current_tool_feedback(self) -> str:
        feedback = tool_feedback_from_ledger()
        if not feedback or "No external tool results yet." in feedback:
            return ""
        return feedback


def _normalize_query_for_prompt(query: str) -> str:
    return query.replace("\x1e", "\n")


def _coerce_answer_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)
