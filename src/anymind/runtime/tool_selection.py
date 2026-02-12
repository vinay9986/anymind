from __future__ import annotations

from typing import Any, Sequence, Tuple

from anymind.policies.tool_policy import ToolPolicy


async def select_tools_for_policy(
    *,
    policy: ToolPolicy,
    tools: Sequence[Any],
    user_input: str,
    model_client: Any,
    model_name: str,
) -> Tuple[list[Any], dict[str, dict[str, int]]]:
    if not tools or policy.name == "never":
        return [], {}

    if policy.name != "planner":
        return list(tools), {}

    selections, usage_metadata = await policy.select_tools(
        user_input=user_input,
        tools=tools,
        model_client=model_client,
        model_name=model_name,
    )
    if not selections:
        return [], usage_metadata

    selected_tools = [tool for tool in tools if getattr(tool, "name", "") in selections]
    return selected_tools, usage_metadata
