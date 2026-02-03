from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver

try:  # Optional; depends on langgraph checkpoint extras
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AsyncSqliteSaver = None  # type: ignore[assignment]

from anymind.config.schemas import CheckpointConfig


def resolve_state_dir(state_dir: Optional[Path] = None) -> Path:
    if state_dir is None:
        env_value = os.getenv("AM_STATE_DIR")
        if env_value:
            state_dir = Path(env_value).expanduser()
        else:
            state_dir = Path.home() / ".local" / "share" / "anymind"
    else:
        state_dir = Path(state_dir).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


async def create_checkpointer(
    state_dir: Optional[Path] = None,
    checkpoint: Optional[CheckpointConfig] = None,
) -> tuple[object, aiosqlite.Connection | None]:
    backend = (checkpoint.backend if checkpoint else "sqlite").lower()
    if backend == "memory":
        return MemorySaver(), None
    if backend == "redis":
        raise RuntimeError(
            "Redis checkpointer is not configured. "
            "Use backend=sqlite or backend=memory."
        )

    if AsyncSqliteSaver is None:
        return MemorySaver(), None

    db_path: Path
    if checkpoint and checkpoint.path:
        db_path = Path(checkpoint.path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        root = resolve_state_dir(state_dir)
        db_path = root / "checkpoints.sqlite"

    conn = await aiosqlite.connect(str(db_path))
    return AsyncSqliteSaver(conn), conn
