"""Deterministic built-in node functions for the SOP executor.

Builtin nodes skip the LLM entirely — they run Python code and return
a result string directly. Register new builtins with @register_builtin.

Node config shape:
    {
        "id": "...",
        "type": "builtin",
        "function": "<registered name>",
        ... (any extra fields are passed to the function via the node dict)
    }
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Callable

_BUILTIN_REGISTRY: dict[str, Callable[[dict[str, Any], dict[str, Any]], str]] = {}


def register_builtin(name: str) -> Callable:
    """Decorator — registers a function under the given builtin name."""

    def decorator(fn: Callable) -> Callable:
        _BUILTIN_REGISTRY[name] = fn
        return fn

    return decorator


def get_builtin(name: str) -> Callable[[dict[str, Any], dict[str, Any]], str] | None:
    return _BUILTIN_REGISTRY.get(name)


# ---------------------------------------------------------------------------
# compute_time_window
# ---------------------------------------------------------------------------


@register_builtin("compute_time_window")
def compute_time_window(node: dict[str, Any], node_results: dict[str, Any]) -> str:
    """Return a JSON time-window object for the requested window type.

    Node config fields:
        window (str, optional): one of the supported window types below.
                                Defaults to "last_full_week".

    Supported window types
    ----------------------
    last_full_week   Most recently completed Sun–Sat week (newsletter default).
    last_7_days      Rolling 7 days ending yesterday.
    last_30_days     Rolling 30 days ending yesterday.
    last_month       Previous full calendar month.
    last_quarter     Previous full calendar quarter.

    Returns a JSON object with: today, week_start, week_end, label, window.
    """
    today = datetime.date.today()
    window = str(node.get("window", "") or "last_full_week").strip()

    if window == "last_full_week":
        # Find the most recently completed Saturday.
        # weekday(): Mon=0 … Sat=5, Sun=6
        days_since_sat = (today.weekday() - 5) % 7
        if days_since_sat == 0:
            days_since_sat = 7  # today IS Saturday — use the previous one
        week_end = today - datetime.timedelta(days=days_since_sat)
        week_start = week_end - datetime.timedelta(days=6)
        label = (
            f"Week of {week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}"
        )

    elif window == "last_7_days":
        week_end = today - datetime.timedelta(days=1)
        week_start = today - datetime.timedelta(days=7)
        label = f"Last 7 days: {week_start} – {week_end}"

    elif window == "last_30_days":
        week_end = today - datetime.timedelta(days=1)
        week_start = today - datetime.timedelta(days=30)
        label = f"Last 30 days: {week_start} – {week_end}"

    elif window == "last_month":
        first_of_this_month = today.replace(day=1)
        week_end = first_of_this_month - datetime.timedelta(days=1)
        week_start = week_end.replace(day=1)
        label = week_start.strftime("%B %Y")

    elif window == "last_quarter":
        # Current quarter starts on month 1, 4, 7, or 10.
        current_q_start_month = ((today.month - 1) // 3) * 3 + 1
        first_of_current_q = today.replace(month=current_q_start_month, day=1)
        week_end = first_of_current_q - datetime.timedelta(days=1)
        prev_q_start_month = ((week_end.month - 1) // 3) * 3 + 1
        week_start = week_end.replace(month=prev_q_start_month, day=1)
        q_num = (week_start.month - 1) // 3 + 1
        label = f"Q{q_num} {week_start.year}"

    else:
        supported = (
            "last_full_week, last_7_days, last_30_days, last_month, last_quarter"
        )
        raise ValueError(f"Unknown window type {window!r}. Supported: {supported}")

    return json.dumps(
        {
            "today": str(today),
            "week_start": str(week_start),
            "week_end": str(week_end),
            "label": label,
            "window": window,
        },
        ensure_ascii=False,
    )
