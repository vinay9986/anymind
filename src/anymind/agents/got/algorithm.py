from __future__ import annotations

import asyncio
import heapq
import json
import random
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Awaitable, Callable, Optional

import numpy as np
import structlog

from anymind.agents.got.prompts import (
    SOLVER_USER_PROMPT,
    SOLVER_USER_PROMPT_WITH_CONTEXT,
    SYSTEM_PROMPT,
    USER_PROMPT_EXPAND,
    USER_PROMPT_FINALISE,
    USER_PROMPT_INITIAL_TASK,
    USER_PROMPT_REFLECT,
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_PROMPT,
)
from anymind.agents.got.validators import (
    ReflectionScoreValidator,
    TaskListResponseValidator,
    make_validator_finalise,
    make_validator_verifier,
)
from anymind.agents.iot_utils import truncate_text
from anymind.runtime.evidence import get_current_ledger
from anymind.runtime.onnx_embedder import OnnxSentenceEmbedder
from anymind.runtime.validated_json import ValidatedJsonResult

PlannerRunner = Callable[[str, str, str, Any, str], Awaitable[ValidatedJsonResult]]
SolverRunner = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class GoTConfig:
    max_layers: int = 3
    beam_width: int = 2
    children_per_expand: int = 3
    max_concurrency: int = 6

    reflection: bool = True
    reflection_weight: float = 0.5

    diversity_lambda: float = 0.1
    relevance_weight: float = 0.2

    prune_delta: float = 0.1
    success_threshold: float = 0.8

    verify: bool = True
    verify_threshold: float = 0.8
    verify_max_retries: int = 2

    max_diversity_samples: int = 20
    tool_feedback_max_chars: int = 8000
    context_max_chars: int = 1800
    final_top_k: int = 12


@dataclass
class GoTNode:
    id: int
    title: str
    content: str
    depth: int
    parent_ids: list[int] = field(default_factory=list)

    answer: Optional[str] = None
    base_score: float = 0.0
    reflection_score: float = 0.0
    diversity_penalty: float = 0.0
    relevance_score: float = 0.0
    total_score: float = 0.0


class GoTGraph:
    def __init__(self) -> None:
        self._nodes: dict[int, GoTNode] = {}
        self._children: dict[int, list[int]] = {}

    def add_node(self, node: GoTNode) -> None:
        self._nodes[node.id] = node
        for parent_id in node.parent_ids:
            self._children.setdefault(parent_id, []).append(node.id)
            self._children.setdefault(node.id, [])

    def get_node(self, node_id: int) -> Optional[GoTNode]:
        return self._nodes.get(int(node_id))

    def leaf_nodes(self) -> list[GoTNode]:
        leaves: list[GoTNode] = []
        for node_id, node in self._nodes.items():
            if len(self._children.get(node_id, [])) == 0:
                leaves.append(node)
        return leaves

    def path_to_root(self, node: GoTNode) -> list[GoTNode]:
        path = [node]
        current = node
        while current.parent_ids:
            parent = self._nodes.get(current.parent_ids[0])
            if parent is None:
                break
            path.insert(0, parent)
            current = parent
        return path

    def to_dict(self) -> dict[str, Any]:
        edges: list[tuple[int, int]] = []
        for parent_id, child_ids in self._children.items():
            for child_id in child_ids:
                edges.append((parent_id, child_id))

        nodes_payload = [
            {
                "id": n.id,
                "title": n.title,
                "content": n.content,
                "depth": n.depth,
                "parent_ids": list(n.parent_ids),
                "answer": n.answer,
                "base_score": round(float(n.base_score), 6),
                "reflection_score": round(float(n.reflection_score), 6),
                "diversity_penalty": round(float(n.diversity_penalty), 6),
                "relevance_score": round(float(n.relevance_score), 6),
                "total_score": round(float(n.total_score), 6),
            }
            for n in self._nodes.values()
        ]

        return {"nodes": nodes_payload, "edges": edges}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


class GoTAlgorithm:
    def __init__(
        self,
        *,
        config: GoTConfig,
        planner_runner: PlannerRunner,
        solver_runner: SolverRunner,
        embedder: Optional[OnnxSentenceEmbedder] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.cfg = config
        self._planner_runner = planner_runner
        self._solver_runner = solver_runner
        self._embedder = embedder
        self._should_stop = should_stop or (lambda: False)

        self.sem = asyncio.Semaphore(max(1, int(self.cfg.max_concurrency)))
        self.graph = GoTGraph()
        self.open_q: dict[int, list[tuple[float, int, GoTNode]]] = {}

        self._id = count()
        self._counter = count()
        self._emb_cache: dict[str, np.ndarray] = {}
        self._base_question: str = ""

        self.log = structlog.get_logger("anymind.got")

    async def solve(self, query: str) -> tuple[str, GoTGraph]:
        self._base_question = (query or "").strip()
        await self._bootstrap()

        if self._halt():
            return (
                self._fallback_answer("Budget exhausted before completion."),
                self.graph,
            )

        layer = 0
        iteration = 0
        while not self._terminate(layer):
            if self._halt():
                break
            iteration += 1
            tau = min(0.2 + 0.05 * layer, 0.7)
            beam = self._pop_beam(layer)

            self.log.info(
                "got_iteration_start",
                iteration=iteration,
                layer=layer,
                beam_size=len(beam),
                open_queues={d: len(q) for d, q in self.open_q.items() if q},
                leaf_count=len(self.graph.leaf_nodes()),
            )

            if beam:
                await asyncio.gather(*(self._expand(node, tau) for node in beam))

            if self.cfg.reflection:
                await self._reflect_paths(layer)

            pruned = self._prune_open_queues()
            if pruned:
                self.log.info("got_pruned_open_queues", pruned=pruned)

            await self._solve_unsolved_leaves()

            if not self.open_q.get(layer):
                layer += 1

        if self._halt():
            return (
                self._fallback_answer("Budget exhausted before completion."),
                self.graph,
            )

        result_obj = await self._finalise(self._compose_question_with_tools())
        if self._halt():
            return (
                self._fallback_answer("Budget exhausted before completion."),
                self.graph,
            )

        if self.cfg.verify:
            verified = await self._verify_answer(
                self._compose_question_with_tools(),
                _coerce_text(result_obj.get("final_answer")),
            )
            retries_left = int(self.cfg.verify_max_retries)
            while not verified and retries_left > 0 and not self._halt():
                self.log.warning(
                    "got_verifier_rejected_answer",
                    retries_left=retries_left,
                    threshold=float(self.cfg.verify_threshold),
                )
                attempt_idx = int(self.cfg.verify_max_retries) - retries_left + 1
                await self._reflect_paths(self.cfg.max_layers)
                await self._solve_unsolved_leaves()
                result_obj = await self._finalise(self._compose_question_with_tools())
                verified = await self._verify_answer(
                    self._compose_question_with_tools(),
                    _coerce_text(result_obj.get("final_answer")),
                )
                retries_left -= 1

        final_answer = _coerce_text(result_obj.get("final_answer")).strip()
        return final_answer or self._fallback_answer("No solution found."), self.graph

    def _halt(self) -> bool:
        return bool(self._should_stop())

    async def _bootstrap(self) -> None:
        tool_feedback = self._tool_feedback()
        user_prompt = USER_PROMPT_INITIAL_TASK.format(
            question=self._base_question,
            max_new_tasks=self.cfg.children_per_expand,
            tool_feedback=tool_feedback,
        )
        validator = TaskListResponseValidator(
            max_tasks=self.cfg.children_per_expand,
            validator_name="got_initial_tasks",
        )
        result = await self._planner_runner(
            "got_initial_tasks",
            SYSTEM_PROMPT,
            user_prompt,
            validator,
            f"GoT bootstrap for query: {self._base_question}",
        )
        obj = result.data

        self.open_q[0] = []
        tasks = obj.get("tasks", [])
        for task in tasks if isinstance(tasks, list) else []:
            if not isinstance(task, dict):
                continue
            node = await self._make_node(
                title=str(task.get("title", "") or ""),
                content=str(task.get("content", "") or ""),
                depth=0,
                parents=[],
            )
            self._push_open(node)

        self.log.info(
            "got_bootstrap_complete",
            n_tasks=len(tasks) if isinstance(tasks, list) else 0,
            strategy=str(obj.get("strategy", "")),
        )

    async def _expand(self, node: GoTNode, tau: float) -> None:
        async with self.sem:
            if self._halt():
                return
            context_summary = self._context_summary(node.depth)
            tool_feedback = self._tool_feedback()
            user_prompt = USER_PROMPT_EXPAND.format(
                original_question=self._base_question,
                depth=node.depth + 1,
                parent_content=node.content,
                k=self.cfg.children_per_expand,
                context_summary=context_summary,
                tool_feedback=tool_feedback,
            )
            validator = TaskListResponseValidator(
                max_tasks=self.cfg.children_per_expand,
                validator_name="got_expand",
            )
            result = await self._planner_runner(
                f"got_expand_node_{node.id}",
                SYSTEM_PROMPT,
                user_prompt,
                validator,
                f"GoT expand node {node.id}",
            )
            obj = result.data

        tasks = obj.get("tasks", [])
        created = 0
        for task in tasks if isinstance(tasks, list) else []:
            if not isinstance(task, dict):
                continue
            child = await self._make_node(
                title=str(task.get("title", "") or ""),
                content=str(task.get("content", "") or ""),
                depth=node.depth + 1,
                parents=[node.id],
            )
            self._push_open(child)
            created += 1

        self.log.info(
            "got_expand_complete",
            node_id=node.id,
            depth=node.depth,
            created_children=created,
        )

    def _completed_paths(self, layer: int) -> list[list[GoTNode]]:
        completed: list[list[GoTNode]] = []
        for leaf in self.graph.leaf_nodes():
            if leaf.depth == layer and leaf.answer is not None:
                completed.append(self.graph.path_to_root(leaf))
        return completed

    async def _reflect_paths(self, layer: int) -> None:
        if self._halt():
            return
        completed = self._completed_paths(layer)
        if not completed:
            return

        self.log.info(
            "got_reflection_start",
            layer=layer,
            n_paths=len(completed),
            max_concurrency=self.cfg.max_concurrency,
        )

        async def _reflect_one(path: list[GoTNode]) -> None:
            async with self.sem:
                if self._halt():
                    return
                await self._reflect(path)

        await asyncio.gather(*(_reflect_one(path) for path in completed))

    async def _reflect(self, path: list[GoTNode]) -> None:
        pretty = "\n".join(
            f"{i + 1}. {n.title}: {n.answer if n.answer is not None else 'N/A'}"
            for i, n in enumerate(path)
        )
        user_prompt = USER_PROMPT_REFLECT.format(pretty_path=pretty)
        validator = ReflectionScoreValidator(validator_name="got_reflection")
        result = await self._planner_runner(
            "got_reflection",
            SYSTEM_PROMPT,
            user_prompt,
            validator,
            "GoT reflection",
        )
        obj = result.data

        score = float(obj.get("score", 0.5))
        new_tasks = obj.get("new_tasks")

        for node in path:
            node.reflection_score = score
            node.total_score = await self._comp_score(node)
            self._reheap(node)

        if isinstance(new_tasks, list) and new_tasks:
            leaf = path[-1]
            inserted = 0
            for task in new_tasks:
                if not isinstance(task, dict):
                    continue
                child = await self._make_node(
                    title=str(task.get("title", "") or ""),
                    content=str(task.get("content", "") or ""),
                    depth=leaf.depth + 1,
                    parents=[leaf.id],
                )
                self._push_open(child)
                inserted += 1
            if inserted:
                self.log.info(
                    "got_reflection_new_tasks",
                    leaf_id=leaf.id,
                    inserted=inserted,
                    reflection_score=score,
                )

    async def _solve_unsolved_leaves(self) -> None:
        if self._halt():
            return
        leaves_to_solve = [n for n in self.graph.leaf_nodes() if n.answer is None]
        if not leaves_to_solve:
            return

        async def _solve(node: GoTNode) -> None:
            async with self.sem:
                if self._halt():
                    return
                context = self._solver_context(node)
                if context:
                    user = SOLVER_USER_PROMPT_WITH_CONTEXT.format(
                        context=context, sub_task_content=node.content
                    )
                else:
                    user = SOLVER_USER_PROMPT.format(sub_task_content=node.content)
                response = await self._solver_runner(user)
                node.answer = response.strip()
                self.log.info(
                    "got_leaf_solved",
                    node_id=node.id,
                    depth=node.depth,
                    answer_len=len(node.answer or ""),
                )

        await asyncio.gather(*(_solve(node) for node in leaves_to_solve))

    async def _finalise(self, question: str) -> dict[str, Any]:
        leaves = self.graph.leaf_nodes()
        top_k = max(1, int(self.cfg.final_top_k))
        if len(leaves) > top_k:
            leaves = sorted(leaves, key=lambda n: n.total_score, reverse=True)[:top_k]
        summary = "\n".join(
            f"- [{n.id}] {n.title}: {n.answer if n.answer is not None else 'N/A'}"
            for n in leaves
        )
        user_prompt = USER_PROMPT_FINALISE.format(
            original_question=question,
            leaf_summary_list=summary,
        )
        validator = make_validator_finalise(validator_name="got_finalise")
        result = await self._planner_runner(
            "got_finalise",
            SYSTEM_PROMPT,
            user_prompt,
            validator,
            "GoT finalise",
        )
        obj = result.data
        self.log.info(
            "got_finalise_complete",
            final_answer_len=len(_coerce_text(obj.get("final_answer"))),
            leaf_count=len(leaves),
        )
        return obj

    async def _verify_answer(self, question: str, answer: str) -> bool:
        user_prompt = VERIFIER_USER_PROMPT.format(question=question, answer=answer)
        validator = make_validator_verifier(validator_name="got_verifier")
        result = await self._planner_runner(
            "got_verifier",
            VERIFIER_SYSTEM_PROMPT,
            user_prompt,
            validator,
            "GoT verifier",
        )
        obj = result.data
        try:
            score = float(obj.get("score", 0.0))
        except Exception:
            score = 0.0
        passed = score >= float(self.cfg.verify_threshold)
        self.log.info(
            "got_verifier_score",
            score=score,
            threshold=float(self.cfg.verify_threshold),
            passed=passed,
        )
        return passed

    async def _make_node(
        self, *, title: str, content: str, depth: int, parents: list[int]
    ) -> GoTNode:
        node_id = next(self._id)
        node = GoTNode(
            id=node_id,
            title=title,
            content=content,
            depth=depth,
            parent_ids=list(parents),
        )
        node.base_score = self._base_score(node)
        node.total_score = await self._comp_score(node)
        self.graph.add_node(node)
        return node

    def _base_score(self, node: GoTNode) -> float:
        max_depth = float(max(1, int(self.cfg.max_layers)))
        return (max_depth - node.depth + 1.0) / max_depth

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _embed_many(self, texts: list[str]) -> list[np.ndarray]:
        if self._embedder is None:
            return [np.zeros((0,), dtype=np.float32) for _ in texts]

        missing = [t for t in texts if t not in self._emb_cache]
        if missing:
            batch = self._embedder.embed(missing)
            for idx, text in enumerate(missing):
                self._emb_cache[text] = batch[idx]
        return [self._emb_cache[t] for t in texts]

    async def _diversity(self, node: GoTNode) -> float:
        if self._embedder is None:
            return 0.0
        siblings = [entry[2] for entry in self.open_q.get(node.depth, [])]
        if not siblings:
            return 0.0
        sampled = siblings
        if len(sampled) > self.cfg.max_diversity_samples:
            sampled = random.sample(sampled, self.cfg.max_diversity_samples)
        texts = [node.content] + [s.content for s in sampled]
        vectors = self._embed_many(texts)
        node_vec = vectors[0]
        sib_vecs = vectors[1:]
        sims = [self._cosine(node_vec, v) for v in sib_vecs if v.size]
        return max(sims) if sims else 0.0

    async def _relevance(self, node: GoTNode) -> float:
        if self._embedder is None or not self._base_question:
            node.relevance_score = 1.0
            return 1.0
        node_text = f"{node.title}: {node.content}"
        vectors = self._embed_many([node_text, self._base_question])
        sim = (
            self._cosine(vectors[0], vectors[1])
            if vectors[0].size and vectors[1].size
            else 0.0
        )
        relevance = max(0.0, (sim + 1.0) / 2.0)
        node.relevance_score = relevance
        return relevance

    async def _comp_score(self, node: GoTNode) -> float:
        node.diversity_penalty = await self._diversity(node)
        relevance_score = await self._relevance(node)
        alpha = float(self.cfg.reflection_weight)
        beta = float(self.cfg.relevance_weight)
        lam = float(self.cfg.diversity_lambda)
        return (
            (1.0 - alpha - beta) * node.base_score
            + alpha * node.reflection_score
            + beta * relevance_score
            - lam * node.diversity_penalty
        )

    def _push_open(self, node: GoTNode) -> None:
        depth = node.depth
        self.open_q.setdefault(depth, [])
        heapq.heappush(
            self.open_q[depth], (-node.total_score, next(self._counter), node)
        )

    def _pop_beam(self, layer: int) -> list[GoTNode]:
        beam: list[GoTNode] = []
        q = self.open_q.get(layer, [])
        while q and len(beam) < int(self.cfg.beam_width):
            beam.append(heapq.heappop(q)[2])
        return beam

    def _reheap(self, node: GoTNode) -> None:
        q = self.open_q.get(node.depth, [])
        for i, (_, counter_value, existing) in enumerate(q):
            if existing.id == node.id:
                q[i] = (-node.total_score, counter_value, node)
                heapq.heapify(q)
                break

    def _prune_open_queues(self) -> int:
        delta = float(self.cfg.prune_delta)
        if delta <= 0:
            return 0

        best: Optional[float] = None
        for q in self.open_q.values():
            if q:
                score = -q[0][0]
                best = score if best is None else max(best, score)

        if best is None:
            return 0

        threshold = best - delta
        total_pruned = 0
        for depth, q in list(self.open_q.items()):
            original_size = len(q)
            new_q = [entry for entry in q if -entry[0] >= threshold]
            pruned = original_size - len(new_q)
            total_pruned += pruned
            if pruned:
                heapq.heapify(new_q)
                self.open_q[depth] = new_q
        return total_pruned

    def _terminate(self, layer: int) -> bool:
        if self._halt():
            return True

        max_layers = int(self.cfg.max_layers)
        if max_layers > 0 and layer >= max_layers:
            return True

        thr = float(self.cfg.success_threshold)
        if thr <= 1.0:
            for leaf in self.graph.leaf_nodes():
                if leaf.answer is None or float(leaf.reflection_score) < thr:
                    continue
                path_nodes = self.graph.path_to_root(leaf)
                if all(n.answer is not None for n in path_nodes):
                    return True

        open_nodes_count = sum(len(q) for q in self.open_q.values())
        if open_nodes_count > 0:
            return False

        unsolved = [n for n in self.graph.leaf_nodes() if n.answer is None]
        return len(unsolved) == 0

    def _context_summary(self, depth: int) -> str:
        if depth <= 0:
            return "No prior solved context."
        solved = [n for n in self.graph.leaf_nodes() if n.answer and n.depth < depth]
        if not solved:
            return "No prior solved context."
        solved.sort(key=lambda n: (n.depth, -n.total_score))
        lines: list[str] = []
        for node in solved[:8]:
            ans = truncate_text(str(node.answer or ""), 240)
            lines.append(f"- [L{node.depth}] {node.title}: {ans}")
        summary = "\n".join(lines)
        return truncate_text(summary, self.cfg.context_max_chars)

    def _tool_feedback(self) -> str:
        blob = self._tool_blob()
        if blob:
            return blob
        return ""

    def _tool_blob(self) -> str:
        ledger = get_current_ledger()
        max_chars = int(self.cfg.tool_feedback_max_chars)
        if ledger is not None:
            records = ledger.recent()
            if records:
                parts: list[str] = []
                total = 0
                for record in records:
                    block = f"[{record.id}] {record.tool}: {record.content.strip()}"
                    if total + len(block) > max_chars:
                        remaining = max_chars - total
                        if remaining > 0:
                            parts.append(block[:remaining].rstrip())
                        break
                    parts.append(block)
                    total += len(block)
                return "\n".join(parts).strip()

        return ""

    def _solver_context(self, node: GoTNode) -> str:
        parts: list[str] = []
        if self._base_question:
            parts.append(f"Original question: {self._base_question}")
        tool_feedback = self._tool_feedback()
        if tool_feedback:
            parts.append(f"External tool findings:\n{tool_feedback}")
        context_summary = self._context_summary(node.depth)
        if context_summary and "No prior solved context" not in context_summary:
            parts.append(f"Prior solved context:\n{context_summary}")
        return "\n\n".join(parts).strip()

    def _compose_question_with_tools(self) -> str:
        base_question = self._base_question
        tool_feedback = self._tool_feedback()
        if not tool_feedback:
            return base_question
        return f"{base_question}\n\nExternal tool findings:\n{tool_feedback}"

    def _fallback_answer(self, message: str) -> str:
        leaves = [n for n in self.graph.leaf_nodes() if n.answer]
        if leaves:
            best = max(leaves, key=lambda n: n.total_score)
            return best.answer or message
        return message
