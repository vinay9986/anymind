from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import structlog

from anymind.agents.agot.heritage import Heritage
from anymind.agents.agot.tasks import Task


@dataclass(slots=True)
class Node:
    thought: Task
    strategy: str = ""
    answer: Optional[str] = None
    nested_graph: Optional["NestedGraph"] = None


@dataclass(slots=True)
class NestedGraph:
    """Nested graph container for AGoT: G_h = (V_h, E_h, F_h)."""

    heritage: Heritage
    nodes: dict[Heritage, Node] = field(default_factory=dict)
    edges: list[tuple[Heritage, Heritage]] = field(default_factory=list)
    final_answer: Optional[str] = None

    current_layer: int = 0
    current_strategy: str = ""

    _heritage_locks: dict[Heritage, asyncio.Lock] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock_creation_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _edge_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )

    def add_node(self, heritage: Heritage, thought: Task, strategy: str = "") -> Node:
        if heritage in self.nodes:
            raise ValueError(f"Node with heritage {heritage} already exists")
        node = Node(thought=thought, strategy=strategy)
        self.nodes[heritage] = node
        return node

    async def add_node_thread_safe(
        self, heritage: Heritage, thought: Task, strategy: str = ""
    ) -> Node:
        log = structlog.get_logger("anymind.agot")

        async with self._lock_creation_lock:
            lock = self._heritage_locks.get(heritage)
            if lock is None:
                lock = asyncio.Lock()
                self._heritage_locks[heritage] = lock

        async with lock:
            existing = self.nodes.get(heritage)
            if existing is not None:
                return existing
            node = Node(thought=thought, strategy=strategy)
            self.nodes[heritage] = node
            log.debug(
                "agot_node_created",
                graph_heritage=str(self.heritage),
                heritage=str(heritage),
                title=thought.title,
            )
            return node

    def add_edge(self, source_heritage: Heritage, target_heritage: Heritage) -> None:
        if source_heritage not in self.nodes:
            raise ValueError(f"Source node {source_heritage} not found")
        if target_heritage not in self.nodes:
            raise ValueError(f"Target node {target_heritage} not found")
        edge = (source_heritage, target_heritage)
        if edge not in self.edges:
            self.edges.append(edge)

    async def add_edge_thread_safe(
        self, source_heritage: Heritage, target_heritage: Heritage
    ) -> bool:
        async with self._edge_lock:
            if source_heritage not in self.nodes:
                raise ValueError(f"Source node {source_heritage} not found")
            if target_heritage not in self.nodes:
                raise ValueError(f"Target node {target_heritage} not found")

            edge = (source_heritage, target_heritage)
            if edge in self.edges:
                return False
            self.edges.append(edge)
            return True

    def get_node(self, heritage: Heritage) -> Node | None:
        return self.nodes.get(heritage)

    def get_nodes_by_layer(self, layer: int) -> list[tuple[Heritage, Node]]:
        layer = int(layer)
        out: list[tuple[Heritage, Node]] = []
        for heritage, node in self.nodes.items():
            try:
                node_layer, _ = heritage.current_position()
            except ValueError:
                continue
            if node_layer == layer:
                out.append((heritage, node))
        return out

    def get_layer_count(self) -> int:
        if not self.nodes:
            return 0
        max_layer = -1
        for heritage in self.nodes.keys():
            try:
                layer, _ = heritage.current_position()
            except ValueError:
                continue
            max_layer = max(max_layer, layer)
        return max_layer + 1

    def is_layer_complete(self, layer: int) -> bool:
        layer_nodes = self.get_nodes_by_layer(layer)
        if not layer_nodes:
            return True
        for _, node in layer_nodes:
            if node.answer is None:
                if node.nested_graph is None or not node.nested_graph.is_complete():
                    return False
        return True

    def get_leaf_nodes(self) -> list[tuple[Heritage, Node]]:
        sources = {source for source, _ in self.edges}
        return [(h, n) for h, n in self.nodes.items() if h not in sources]

    def is_complete(self) -> bool:
        if self.final_answer is not None:
            return True
        leaf_nodes = self.get_leaf_nodes()
        if not leaf_nodes:
            return False
        for _, node in leaf_nodes:
            if node.answer is None:
                if node.nested_graph is None or not node.nested_graph.is_complete():
                    return False
        return True

    def set_node_answer(self, heritage: Heritage, answer: str) -> None:
        node = self.nodes.get(heritage)
        if node is None:
            raise ValueError(f"Node {heritage} not found")
        node.answer = answer

    def to_string(self, *, include_nested: bool = True, indent: int = 0) -> str:
        prefix = "  " * indent
        lines: list[str] = [f"{prefix}Graph {self.heritage}:"]
        for layer in range(self.get_layer_count()):
            layer_nodes = self.get_nodes_by_layer(layer)
            if not layer_nodes:
                continue
            lines.append(f"{prefix}  Layer {layer}:")
            for heritage, node in sorted(layer_nodes, key=lambda x: x[0]):
                answer_preview = "(no answer)"
                if node.answer:
                    answer_preview = (
                        f"-> {node.answer[:80]}{'...' if len(node.answer) > 80 else ''}"
                    )
                lines.append(
                    f"{prefix}    {heritage}: {node.thought.title} {answer_preview}"
                )
                if include_nested and node.nested_graph:
                    lines.append(f"{prefix}      Nested:")
                    lines.append(
                        node.nested_graph.to_string(
                            include_nested=include_nested, indent=indent + 3
                        )
                    )
        if self.final_answer:
            lines.append(
                f"{prefix}  Final: {self.final_answer[:120]}{'...' if len(self.final_answer) > 120 else ''}"
            )
        return "\n".join(lines)


def create_empty_graph(heritage: Heritage) -> NestedGraph:
    return NestedGraph(heritage=heritage)


def create_root_graph() -> NestedGraph:
    return NestedGraph(heritage=Heritage.create_empty())
