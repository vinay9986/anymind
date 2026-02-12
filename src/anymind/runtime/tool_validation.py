from __future__ import annotations

from typing import Any, Iterable


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function", {}) if isinstance(tool.get("function"), dict) else {}
        name = fn.get("name") or tool.get("name")
        return str(name or "").strip()
    for attr in ("name", "tool_name", "__name__"):
        value = getattr(tool, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def extract_tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        desc = tool.get("description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        fn = tool.get("function", {}) if isinstance(tool.get("function"), dict) else {}
        desc = fn.get("description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        return ""

    value = getattr(tool, "description", None)
    if isinstance(value, str) and value.strip():
        return value.strip()

    metadata = getattr(tool, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("description", "tool_description"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        tool_spec = metadata.get("tool_spec") or metadata.get("toolSpec")
        if isinstance(tool_spec, dict):
            value = tool_spec.get("description") or tool_spec.get("summary")
            if isinstance(value, str) and value.strip():
                return value.strip()

    tool_spec = getattr(tool, "tool_spec", None) or getattr(tool, "toolSpec", None)
    if isinstance(tool_spec, dict):
        value = tool_spec.get("description") or tool_spec.get("summary")
        if isinstance(value, str) and value.strip():
            return value.strip()

    doc = getattr(tool, "__doc__", None)
    if isinstance(doc, str) and doc.strip():
        return doc.strip()

    return ""


def require_tool_description(tool: Any, *, context: str = "unknown") -> str:
    description = extract_tool_description(tool)
    if description:
        return description
    name = _tool_name(tool)
    raise ValueError(f"Tool '{name}' is missing a description (context: {context}).")


def ensure_tools_have_descriptions(
    tools: Iterable[Any], *, context: str = "unknown"
) -> None:
    for tool in tools:
        require_tool_description(tool, context=context)
