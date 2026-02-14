from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

_SESSION_ID: ContextVar[Optional[str]] = ContextVar("anymind_session_id", default=None)


def set_session_id(session_id: Optional[str]) -> Token:
    return _SESSION_ID.set(session_id)


def reset_session_id(token: Token) -> None:
    _SESSION_ID.reset(token)


def get_session_id() -> Optional[str]:
    return _SESSION_ID.get()
