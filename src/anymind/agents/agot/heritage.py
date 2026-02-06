from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

EMPTY_HERITAGE = "empty"


@dataclass(frozen=True, slots=True)
class Heritage:
    """Tracks a node's position across nested AGoT graphs.

    A heritage is a sequence of (layer, node) tuples.
    - Empty heritage: empty
    - Top-level node: ((0, 1))
    - Nested node: ((0, 0), (1, 2))
    """

    sequence: tuple[tuple[int, int], ...] = ()

    @classmethod
    def create_empty(cls) -> "Heritage":
        return cls(sequence=())

    @classmethod
    def create_root(cls, layer: int, node: int) -> "Heritage":
        return cls(sequence=((int(layer), int(node)),))

    def append(self, layer: int, node: int) -> "Heritage":
        return Heritage(sequence=(*self.sequence, (int(layer), int(node))))

    def parent_heritage(self) -> "Heritage":
        if not self.sequence:
            raise ValueError("Empty heritage has no parent")
        return Heritage(sequence=self.sequence[:-1])

    def depth(self) -> int:
        return len(self.sequence)

    def is_empty(self) -> bool:
        return not self.sequence

    def current_position(self) -> tuple[int, int]:
        if not self.sequence:
            raise ValueError("Empty heritage has no current position")
        return self.sequence[-1]

    def graph_heritage(self) -> "Heritage":
        if len(self.sequence) <= 1:
            return Heritage.create_empty()
        return Heritage(sequence=self.sequence[:-1])

    def __str__(self) -> str:
        if not self.sequence:
            return EMPTY_HERITAGE
        inner = ",".join(f"({layer},{node})" for layer, node in self.sequence)
        return f"({inner})"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Heritage):
            return NotImplemented
        return self.sequence < other.sequence


def parse_heritage_string(heritage_str: str) -> Heritage:
    heritage_str = heritage_str.strip()
    if not heritage_str or heritage_str.lower() == EMPTY_HERITAGE:
        return Heritage.create_empty()

    if not heritage_str.startswith("(") or not heritage_str.endswith(")"):
        raise ValueError(f"Invalid heritage string format: {heritage_str}")

    inner = heritage_str[1:-1].strip()
    if not inner:
        return Heritage.create_empty()

    if not inner.startswith("(") or not inner.endswith(")"):
        raise ValueError(f"Invalid heritage string format: {heritage_str}")

    parts = inner.split("),(")
    positions: list[tuple[int, int]] = []
    for idx, part in enumerate(parts):
        if idx == 0:
            part = part.lstrip("(")
        if idx == len(parts) - 1:
            part = part.rstrip(")")
        if "," not in part:
            raise ValueError(f"Invalid position format in heritage string: {part}")
        raw_layer, raw_node = part.split(",", 1)
        try:
            layer = int(raw_layer.strip())
            node = int(raw_node.strip())
        except ValueError as exc:
            raise ValueError(f"Non-numeric value in heritage string: {part}") from exc
        positions.append((layer, node))

    return Heritage(sequence=tuple(positions))


def create_initial_heritage(layer: int, node: int) -> Heritage:
    return Heritage.create_root(layer, node)


def create_nested_heritage(parent: Heritage, layer: int, node: int) -> Heritage:
    return parent.append(layer, node)


def heritage_depth(heritage: Heritage | str) -> int:
    if isinstance(heritage, str):
        heritage = parse_heritage_string(heritage)
    return heritage.depth()


def heritage_to_tuple(heritage: Heritage) -> tuple[tuple[int, int], ...]:
    return heritage.sequence


def heritage_from_iterable(items: Iterable[tuple[int, int]]) -> Heritage:
    return Heritage(sequence=tuple((int(layer), int(node)) for layer, node in items))
