from __future__ import annotations

from typing import Any


def get_optimize_flag(sop: dict[str, Any]) -> bool:
    optimize = sop.get("optimize")
    if optimize is None:
        return False
    return bool(optimize)


def validate_sop_structure(sop: Any) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not isinstance(sop, dict):
        return False, [f"SOP must be a JSON object, got {type(sop).__name__}"]

    optimize = sop.get("optimize")
    if optimize is not None and not isinstance(optimize, bool):
        errors.append(
            f"Field 'optimize' must be boolean when present, got {type(optimize).__name__}"
        )

    nodes = sop.get("nodes")
    edges = sop.get("edges")

    if not isinstance(nodes, list):
        errors.append("Missing or invalid 'nodes' list")
        nodes = []
    if not isinstance(edges, list):
        errors.append("Missing or invalid 'edges' list")
        edges = []

    if isinstance(nodes, list) and len(nodes) == 0:
        errors.append("SOP must contain at least one node")

    node_ids: set[str] = set()
    for nd in nodes:
        if not isinstance(nd, dict):
            errors.append(f"Node must be an object, got {type(nd).__name__}")
            continue
        nid = nd.get("id")
        if not isinstance(nid, str) or not nid.strip():
            errors.append("Node missing string 'id'")
            continue
        if nid in node_ids:
            errors.append(f"Duplicate node id: {nid}")
        node_ids.add(nid)

    for ed in edges:
        if not isinstance(ed, dict):
            errors.append(f"Edge must be an object, got {type(ed).__name__}")
            continue
        src = ed.get("source") or ed.get("src") or ed.get("from")
        dst = ed.get("target") or ed.get("dst") or ed.get("to")
        if src not in node_ids:
            errors.append(f"Edge source not found: {src}")
        if dst not in node_ids:
            errors.append(f"Edge target not found: {dst}")
        if src == dst and src is not None:
            errors.append(f"Self-loop edge detected on: {src}")

    return (len(errors) == 0), errors


def get_node_question(node: dict[str, Any], *, allow_fallback: bool = True) -> str:
    """Extract a node's question/prompt text.

    - Prefers explicit `inputs.value` / `parameters.value` (or `.question`).
    - When `allow_fallback=False`, returns only explicit values (used for control/input nodes).
    """
    for container_key in ("inputs", "parameters"):
        params = node.get(container_key)
        if isinstance(params, dict):
            q = params.get("value")
            if isinstance(q, str) and q.strip():
                return q.strip()
            q = params.get("question")
            if isinstance(q, str) and q.strip():
                return q.strip()
        elif isinstance(params, str) and params.strip():
            return params.strip()

    if not allow_fallback:
        return ""

    for key in ("operation", "description", "name", "type", "id"):
        val = node.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""
