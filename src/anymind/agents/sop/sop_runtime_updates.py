"""Helpers for mutating SOP dicts during execution."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    new_value: dict[str, Any] = {}
    parent[key] = new_value
    return new_value


def _find_node_dict(sop: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    nodes = sop.get("nodes")
    if not isinstance(nodes, list):
        return None
    for nd in nodes:
        if isinstance(nd, dict) and nd.get("id") == node_id:
            return nd
    return None


def update_sop_with_node_result(
    sop: dict[str, Any],
    *,
    node_id: str,
    result: dict[str, Any],
    pattern: str | None = None,
    timestamp: float | None = None,
) -> None:
    """Record a node's execution result onto the SOP dict (in-place)."""

    if not isinstance(sop, dict) or not node_id:
        return

    ts = float(timestamp if timestamp is not None else time.time())

    meta = _ensure_dict(sop, "metadata")
    execution = _ensure_dict(meta, "execution")
    execution["updated_at"] = ts
    if pattern:
        execution.setdefault("team_pattern", pattern)

    node_results = _ensure_dict(execution, "node_results")
    node_results[node_id] = deepcopy(result)

    node_dict = _find_node_dict(sop, node_id)
    if node_dict is None:
        return

    node_exec = _ensure_dict(node_dict, "execution")
    node_exec["updated_at"] = ts
    node_exec.update(
        {
            "status": result.get("status"),
            "error": result.get("error"),
            "algorithms": result.get("algorithms"),
            "metrics": result.get("metrics"),
        }
    )
