from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

import structlog
from structlog import stdlib as structlog_stdlib


def _sanitize_filename(name: str) -> str:
    if not name:
        return "anymind"
    sanitized = name.replace(os.sep, "_").replace("/", "_")
    return sanitized or "anymind"


def _resolve_log_path(log_path: str | None, run_id: str | None) -> str:
    file_name = _sanitize_filename(run_id or f"anymind-{uuid.uuid4().hex}")
    if not file_name.endswith(".log"):
        file_name = f"{file_name}.log"

    if log_path:
        candidate = Path(log_path)
        if candidate.is_dir() or str(log_path).endswith(os.sep):
            path = candidate / file_name
        else:
            path = candidate
    else:
        log_dir = Path(os.getenv("ANYMIND_LOG_DIR", tempfile.gettempdir()))
        path = log_dir / file_name

    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = True,
    log_path: str | None = None,
    run_id: str | None = None,
) -> str:
    resolved_path = _resolve_log_path(log_path, run_id)
    os.environ["ANYMIND_LOG_PATH"] = resolved_path
    os.environ["ANYMIND_LOG_LEVEL"] = level.upper()
    log_level = getattr(logging, level.upper(), logging.INFO)
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.processors.KeyValueRenderer()
    )
    pre_chain = [
        structlog_stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    formatter = structlog_stdlib.ProcessorFormatter(
        processors=pre_chain + [renderer],
        foreign_pre_chain=pre_chain,
    )
    handler = logging.FileHandler(resolved_path, encoding="utf-8")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    structlog.configure(
        processors=[
            structlog_stdlib.add_log_level,
            structlog_stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog_stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog_stdlib.LoggerFactory(),
        wrapper_class=structlog_stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return resolved_path
