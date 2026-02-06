from __future__ import annotations

from dataclasses import dataclass

from anymind.agents.agot.heritage import Heritage


@dataclass(frozen=True, slots=True)
class Task:
    heritage: Heritage
    title: str
    content: str
