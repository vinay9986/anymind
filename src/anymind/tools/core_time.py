from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import structlog

logger = structlog.get_logger(__name__)


def _handle_current_time(event: Mapping[str, Any]) -> dict[str, Any]:
    logger.info("tool_current_time_input", request=dict(event))
    fmt = str(event.get("format", "iso") or "iso").strip().lower()
    timezone_name = str(event.get("timezone", "UTC") or "UTC").strip().upper()

    now_utc = datetime.now(timezone.utc)

    if fmt == "unix":
        payload: dict[str, Any] = {
            "timestamp": int(now_utc.timestamp()),
            "format": "unix",
            "timezone": timezone_name,
        }
    else:
        payload = {
            "timestamp": now_utc.isoformat(),
            "format": "iso",
            "timezone": timezone_name,
        }

    payload["source"] = "core_tools_lambda"
    logger.info("tool_current_time_output", response=payload)
    return payload


def current_time(format: str = "iso", timezone: str = "UTC") -> dict[str, Any]:
    """Return current time in ISO or unix format.

    WHEN TO USE (ALWAYS USE FIRST in these scenarios):
    - User query contains "today", "this week", "this month", "this year", "currently", "now", "recent"
    - Before calling internet_search with any time-relative query
    - When you need to calculate time differences or deadlines
    - User asks "what time is it" or "what's the date"
    - You need to timestamp events or determine recency

    WHY TO USE: Provides accurate current time to resolve ambiguous time references.
    Prevents errors from guessing dates.

    WHEN NOT TO USE:
    - Query is about historical/past dates with explicit years (e.g., "2024 Olympics")
    - Query is about future dates explicitly stated (e.g., "June 2027")
    - No time component in the query

    CRITICAL WORKFLOW:
    User: "What happened in AI this year?"
    1. Call current_time first -> get "2026"
    2. Then call internet_search with "AI developments 2026"

    EXAMPLES:
    - OK: Query "latest iPhone" -> call current_time first, then search "iPhone 2026"
    - OK: Query "today's weather in NYC" -> call current_time first
    - OK: Query "How old is a person born in 1990?" -> call current_time to get current year
    - NO: Query "When was WWII?" -> do not use (historical fact with no relative time)
    """
    payload: dict[str, Any] = {"format": format, "timezone": timezone}
    return _handle_current_time(payload)
