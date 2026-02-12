from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence, Tuple

from langchain_core.messages import BaseMessage

from anymind.runtime.messages import message_text
from anymind.runtime.usage import normalize_usage_metadata
from anymind.runtime.tool_validation import require_tool_description


class ToolPolicy(Protocol):
    name: str

    async def select_tools(
        self,
        *,
        user_input: str,
        tools: Sequence[Any],
        model_client: Any,
        model_name: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, int]]]: ...


@dataclass
class AutoToolPolicy:
    name: str = "auto"

    async def select_tools(
        self,
        *,
        user_input: str,
        tools: Sequence[Any],
        model_client: Any,
        model_name: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        return [getattr(tool, "name", "") for tool in tools], {}


@dataclass
class NeverToolPolicy:
    name: str = "never"

    async def select_tools(
        self,
        *,
        user_input: str,
        tools: Sequence[Any],
        model_client: Any,
        model_name: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        return [], {}


@dataclass
class ConfirmToolPolicy:
    name: str = "confirm"

    async def select_tools(
        self,
        *,
        user_input: str,
        tools: Sequence[Any],
        model_client: Any,
        model_name: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        return [getattr(tool, "name", "") for tool in tools], {}


@dataclass
class PlannerToolPolicy:
    name: str = "planner"

    _system_prompt: str = (
        "You are a tool-selection planner. "
        "Return ONLY JSON with a top-level key 'tools' that is a list. "
        "Each item must include a 'name' and an 'arguments' object. "
        "Return an empty list when no tools are helpful. "
        "Use ONLY the tool names provided. "
        "If the user greets or makes small talk, return []. "
        "If the user explicitly asks for current time/date, choose the time tool. "
    )

    async def select_tools(
        self,
        *,
        user_input: str,
        tools: Sequence[Any],
        model_client: Any,
        model_name: str,
    ) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        if not tools:
            return [], {}

        if _is_greeting(user_input):
            return [], {}

        fallback = _fallback_tool_selection(user_input, tools)
        if fallback:
            return fallback, {}

        tool_names = {getattr(tool, "name", "") for tool in tools}
        prompt = (
            f"User request:\n{user_input}\n\n"
            f"Available tools (JSON):\n{_tool_catalog(tools)}\n\n"
            "Respond with JSON."
        )

        message: BaseMessage = await model_client.ainvoke(
            [("system", self._system_prompt), ("user", prompt)]
        )
        usage_metadata = normalize_usage_metadata(model_name, [message])
        raw_output = message_text(message)
        selections = _parse_tool_plan(raw_output, tool_names)
        if selections:
            return selections, usage_metadata

        return _fallback_tool_selection(user_input, tools), usage_metadata


def _tool_catalog(tools: Sequence[Any]) -> str:
    payload = []
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
        description = require_tool_description(tool, context="tool_policy")
        parameters: dict[str, Any] = {}
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                parameters = args_schema.model_json_schema()
            except Exception:
                try:
                    parameters = args_schema.schema()
                except Exception:
                    parameters = {}
        payload.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters or {},
            }
        )
    return json.dumps(payload, indent=2, sort_keys=True)


def _extract_json_block(text: str) -> str:
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text


def _parse_tool_plan(text: str, tool_names: set[str]) -> List[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = json.loads(_extract_json_block(text))
        except Exception:
            return []

    if isinstance(payload, dict):
        tools_payload = payload.get("tools", [])
    else:
        tools_payload = payload

    selected: list[str] = []
    if isinstance(tools_payload, list):
        for item in tools_payload:
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
            else:
                continue
            if name and name in tool_names:
                selected.append(name)
    return selected


def _is_greeting(text: str) -> bool:
    return bool(
        re.search(
            r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
            text.lower(),
        )
    )


def _fallback_tool_selection(user_input: str, tools: Sequence[Any]) -> List[str]:
    tool_names = {getattr(tool, "name", "") for tool in tools}
    time_tool_candidates = {"get_current_time", "current_time"}
    time_tool = next(iter(tool_names & time_tool_candidates), None)
    if time_tool:
        if re.search(
            r"\b(what'?s the time|what time is it|current time|time now)\b",
            user_input.lower(),
        ):
            return [time_tool]
    return []


def resolve_tool_policy(name: str) -> ToolPolicy:
    normalized = (name or "auto").lower()
    if normalized == "planner":
        return PlannerToolPolicy()
    if normalized == "confirm":
        return ConfirmToolPolicy()
    if normalized == "never":
        return NeverToolPolicy()
    return AutoToolPolicy()
