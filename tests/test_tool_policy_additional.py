from __future__ import annotations

from types import SimpleNamespace

import pytest

from anymind.policies.tool_policy import (
    ConfirmToolPolicy,
    NeverToolPolicy,
    PlannerToolPolicy,
    _extract_json_block,
    _fallback_tool_selection,
    _parse_tool_plan,
    resolve_tool_policy,
)


class DummyMessage:
    def __init__(self, content, usage_metadata=None) -> None:
        self.content = content
        self.usage_metadata = usage_metadata or {}


class DummyModel:
    def __init__(self, message):
        self.message = message
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        return self.message


@pytest.mark.asyncio
async def test_never_and_confirm_policies() -> None:
    tools = [SimpleNamespace(name="search"), SimpleNamespace(name="time")]
    never_selected, never_usage = await NeverToolPolicy().select_tools(
        user_input="anything",
        tools=tools,
        model_client=None,
        model_name="x",
    )
    confirm_selected, confirm_usage = await ConfirmToolPolicy().select_tools(
        user_input="anything",
        tools=tools,
        model_client=None,
        model_name="x",
    )

    assert never_selected == []
    assert never_usage == {}
    assert confirm_selected == ["search", "time"]
    assert confirm_usage == {}


def test_extract_json_block_and_parse_tool_plan_edge_cases() -> None:
    assert _extract_json_block('prefix {"a": 1} suffix [1, 2]') == '{"a": 1}'
    assert _extract_json_block("no delimiters") == "no delimiters"

    assert _parse_tool_plan(
        '{"tools": ["search", {"name": "time"}, 1]}', {"search", "time"}
    ) == [
        "search",
        "time",
    ]
    assert _parse_tool_plan('{"tools": "search"}', {"search"}) == []
    assert _parse_tool_plan("not json", {"search"}) == []


def test_fallback_tool_selection_and_resolve_tool_policy() -> None:
    tools = [SimpleNamespace(name="search")]
    assert _fallback_tool_selection("what time is it", tools) == []
    assert _fallback_tool_selection(
        "time now", [SimpleNamespace(name="get_current_time")]
    ) == ["get_current_time"]
    assert isinstance(resolve_tool_policy("planner"), PlannerToolPolicy)
    assert isinstance(resolve_tool_policy("confirm"), ConfirmToolPolicy)
    assert isinstance(resolve_tool_policy("never"), NeverToolPolicy)
    assert resolve_tool_policy("unknown").name == "auto"


def test_tool_catalog_uses_empty_parameters_when_schema_methods_fail() -> None:
    from anymind.policies import tool_policy

    class BadSchema:
        def model_json_schema(self):
            raise RuntimeError("nope")

        def schema(self):
            raise RuntimeError("still nope")

    tool = SimpleNamespace(name="search", description="desc", args_schema=BadSchema())
    catalog = tool_policy._tool_catalog([tool])

    assert '"parameters": {}' in catalog


@pytest.mark.asyncio
async def test_planner_tool_policy_no_tools_and_invalid_model_output() -> None:
    policy = PlannerToolPolicy()
    empty_selected, empty_usage = await policy.select_tools(
        user_input="find stuff",
        tools=[],
        model_client=None,
        model_name="x",
    )
    assert empty_selected == []
    assert empty_usage == {}

    model = DummyModel(DummyMessage('{"tools": [{"name": "unknown"}]}'))
    selected, usage = await policy.select_tools(
        user_input="find stuff",
        tools=[SimpleNamespace(name="search")],
        model_client=model,
        model_name="model-x",
    )

    assert selected == []
    assert usage == {"model-x": {"input_tokens": 0, "output_tokens": 0}}
    assert model.calls == 1
