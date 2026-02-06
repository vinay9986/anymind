from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Callable

import structlog

from anymind.agents.agot.mappings import AGoTMappings
from anymind.agents.agot.tasks import Task
from anymind.agents.agot.heritage import Heritage
from anymind.agents.agot.nested_graph import (
    NestedGraph,
    create_empty_graph,
    create_root_graph,
)


def _coerce_final_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


class AGOTAlgorithm:
    """Recursive AGoT algorithm with nested task graph."""

    def __init__(
        self,
        *,
        mappings: AGoTMappings,
        d_max: int = 1,
        l_max: int = 3,
        n_max: int = 3,
        max_concurrency: int = 3,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.mappings = mappings
        self.d_max = max(0, int(d_max))
        self.l_max = max(1, int(l_max))
        self.n_max = max(1, int(n_max))
        self.max_concurrency = max(1, int(max_concurrency))
        self._should_stop = should_stop or (lambda: False)

        self.sem = asyncio.Semaphore(self.max_concurrency)
        self.log = structlog.get_logger("anymind.agot")

    async def solve(self, query: str) -> tuple[str, NestedGraph]:
        return await self.agot(
            query=query, heritage=Heritage.create_empty(), parent_graph=None
        )

    def _halt(self) -> bool:
        return bool(self._should_stop())

    async def agot(
        self,
        *,
        query: str,
        heritage: Heritage,
        parent_graph: Optional[NestedGraph],
    ) -> tuple[str, NestedGraph]:
        self.log.info(
            "agot_recursive_start",
            heritage=str(heritage),
            depth=heritage.depth(),
            d_max=self.d_max,
            l_max=self.l_max,
            n_max=self.n_max,
            max_concurrency=self.max_concurrency,
            has_parent=parent_graph is not None,
        )

        graph = (
            create_root_graph() if heritage.is_empty() else create_empty_graph(heritage)
        )
        current_depth = heritage.depth()

        if self._halt():
            graph.final_answer = self._create_fallback_answer(
                graph, budget_exhausted=True
            )
            return graph.final_answer or "Budget exhausted before completion.", graph

        for layer in range(self.l_max):
            if self._halt():
                break
            self.log.info(
                "agot_layer_start",
                heritage=str(heritage),
                layer=layer,
                depth=current_depth,
            )
            early = await self._process_layer(
                layer=layer,
                query=query,
                graph=graph,
                heritage=heritage,
                parent_graph=parent_graph,
                current_depth=current_depth,
            )
            if early:
                self.log.info(
                    "agot_early_termination",
                    heritage=str(heritage),
                    layer=layer,
                    reason="final_task_detected",
                )
                break

        if graph.final_answer is None and not self._halt():
            try:
                final_task = await self.mappings.phi(graph)
                final_obj = await self.mappings.execute_final_task(final_task, graph)
                graph.final_answer = _coerce_final_answer(final_obj.get("final_answer"))
            except Exception as exc:
                self.log.warning(
                    "agot_final_extraction_failed",
                    heritage=str(heritage),
                    error=str(exc),
                )
                graph.final_answer = self._create_fallback_answer(graph)

        if graph.final_answer is None:
            graph.final_answer = self._create_fallback_answer(
                graph, budget_exhausted=self._halt()
            )

        final_answer = graph.final_answer or "No solution found"
        self.log.info(
            "agot_recursive_complete",
            heritage=str(heritage),
            depth=current_depth,
            final_answer_len=len(final_answer),
            nodes=len(graph.nodes),
            layers=graph.get_layer_count(),
            budget_exhausted=self._halt(),
        )
        return final_answer, graph

    async def _process_layer(
        self,
        *,
        layer: int,
        query: str,
        graph: NestedGraph,
        heritage: Heritage,
        parent_graph: Optional[NestedGraph],
        current_depth: int,
    ) -> bool:
        if self._halt():
            return True

        if layer == 0:
            if heritage.is_empty():
                tasks, strategy = await self.mappings.t_empty(query, self.n_max)
                edges: list[tuple[Heritage, Heritage]] = []
            else:
                tasks, strategy = await self.mappings.t_0(
                    query, parent_graph or graph, self.n_max
                )
                edges = []
        else:
            tasks, strategy, edges = await self.mappings.t_e(query, graph, self.n_max)
            if not tasks and strategy:
                return True

        graph.current_layer = layer
        graph.current_strategy = strategy

        if tasks:
            await asyncio.gather(
                *[
                    graph.add_node_thread_safe(task.heritage, task, strategy)
                    for task in tasks
                ],
                return_exceptions=True,
            )

        if edges:
            await asyncio.gather(
                *[graph.add_edge_thread_safe(src, dst) for (src, dst) in edges],
                return_exceptions=True,
            )

        await self._process_layer_nodes(
            tasks=tasks, query=query, graph=graph, current_depth=current_depth
        )
        return False

    async def _process_layer_nodes(
        self, *, tasks: list[Task], query: str, graph: NestedGraph, current_depth: int
    ) -> None:
        if not tasks or self._halt():
            return
        await asyncio.gather(
            *[
                self._process_node(
                    thought=t, query=query, graph=graph, current_depth=current_depth
                )
                for t in tasks
            ],
            return_exceptions=True,
        )

    async def _process_node(
        self, *, thought: Task, query: str, graph: NestedGraph, current_depth: int
    ) -> None:
        if self._halt():
            return
        self.log.debug(
            "agot_node_start",
            thought_heritage=str(thought.heritage),
            title=thought.title,
            current_depth=current_depth,
        )
        try:
            async with self.sem:
                if self._halt():
                    return
                is_complex = await self.mappings.c(thought, graph)

            if is_complex and current_depth < self.d_max:
                nested_heritage = thought.heritage
                combined_query = f"{thought.title}\x1e{thought.content}"
                nested_answer, nested_graph = await self.agot(
                    query=combined_query,
                    heritage=nested_heritage,
                    parent_graph=graph,
                )

                node = graph.get_node(thought.heritage)
                if node is not None:
                    node.nested_graph = nested_graph
                    node.answer = nested_answer
            else:
                async with self.sem:
                    if self._halt():
                        return
                    answer = await self.mappings.eval(thought, graph, query)
                graph.set_node_answer(thought.heritage, answer)
        except Exception as exc:
            self.log.error(
                "agot_node_error",
                thought_heritage=str(thought.heritage),
                error=str(exc),
            )
            try:
                graph.set_node_answer(thought.heritage, f"Processing error: {exc!s}")
            except Exception as record_exc:
                self.log.debug(
                    "agot_node_error_record_failed",
                    thought_heritage=str(thought.heritage),
                    error=str(record_exc),
                )
        finally:
            node = graph.get_node(thought.heritage)
            self.log.debug(
                "agot_node_complete",
                thought_heritage=str(thought.heritage),
                has_answer=bool(node and node.answer),
            )

    def _create_fallback_answer(
        self, graph: NestedGraph, budget_exhausted: bool = False
    ) -> str:
        answers: list[str] = []
        for node in graph.nodes.values():
            if node.answer:
                answers.append(node.answer.strip())
        if not answers:
            if budget_exhausted:
                return "Budget exhausted before completion."
            return "No solution could be determined."
        joined = " ".join(a for a in answers if a)
        if budget_exhausted:
            return f"Budget exhausted before completion. Partial analysis: {joined}"
        return f"Based on the available analysis: {joined}"
