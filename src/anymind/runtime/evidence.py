from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List
from contextvars import ContextVar


@dataclass
class EvidenceRecord:
    id: str
    tool: str
    args: Any
    content: str
    created_at: float = field(default_factory=time.time)


class EvidenceLedger:
    def __init__(self) -> None:
        self._records: List[EvidenceRecord] = []
        self._counter = 0
        self._turn_start = 0

    def start_turn(self) -> None:
        self._turn_start = len(self._records)

    def add(self, tool: str, args: Any, content: str) -> EvidenceRecord:
        self._counter += 1
        record = EvidenceRecord(
            id=f"E{self._counter}",
            tool=tool,
            args=args,
            content=content,
        )
        self._records.append(record)
        return record

    def recent(self) -> List[EvidenceRecord]:
        return list(self._records[self._turn_start :])

    def all(self) -> List[EvidenceRecord]:
        return list(self._records)

    @staticmethod
    def summarize(
        records: Iterable[EvidenceRecord], max_chars: int | None = None
    ) -> str:
        lines: List[str] = []
        for record in records:
            content = record.content.strip()
            line = f"[{record.id}] {record.tool}: {content}"
            lines.append(line)
        return "\n".join(lines)


_CURRENT_LEDGER: ContextVar[EvidenceLedger | None] = ContextVar(
    "anymind_current_ledger", default=None
)


@contextlib.contextmanager
def use_ledger(ledger: EvidenceLedger):
    token = _CURRENT_LEDGER.set(ledger)
    try:
        yield
    finally:
        _CURRENT_LEDGER.reset(token)


def get_current_ledger() -> EvidenceLedger | None:
    return _CURRENT_LEDGER.get()
