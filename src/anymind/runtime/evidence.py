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
    def summarize(records: Iterable[EvidenceRecord], max_chars: int = 2000) -> str:
        lines: List[str] = []
        total = 0
        for record in records:
            snippet = record.content.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            line = f"[{record.id}] {record.tool}: {snippet}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
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
