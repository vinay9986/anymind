from __future__ import annotations

import contextlib
import json
import re
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
        summary = "\n".join(lines)
        if max_chars is None or max_chars <= 0:
            return summary
        if max_chars <= 3:
            return "..."[:max_chars]
        if len(summary) <= max_chars - 3:
            return f"{summary}..."
        return f"{summary[: max_chars - 3]}..."


def summarize_for_display(record: EvidenceRecord) -> str:
    content = (record.content or "").strip()
    if not content:
        return ""

    def _extract_value(data: Any) -> str:
        if isinstance(data, dict):
            if "url" in data:
                return str(data.get("url", "")).strip()
            if "timestamp" in data:
                return str(data.get("timestamp", "")).strip()
            results = data.get("results")
            if isinstance(results, list):
                for item in results:
                    value = _extract_value(item)
                    if value:
                        return value
        if isinstance(data, list):
            for item in data:
                value = _extract_value(item)
                if value:
                    return value
        return ""

    try:
        data = json.loads(content)
    except Exception:
        data = None

    extracted = _extract_value(data) if data is not None else ""
    if extracted:
        return extracted

    url_match = re.search(r"https?://[^\s\"']+", content)
    if url_match:
        return url_match.group(0)
    ts_match = re.search(r"\d{4}-\d{2}-\d{2}T[0-9:.+-]+Z?", content)
    if ts_match:
        return ts_match.group(0)
    return ""


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
