from types import SimpleNamespace

import pytest

from anymind.policies.tool_policy import (
    AutoToolPolicy,
    PlannerToolPolicy,
    _extract_json_block,
    _fallback_tool_selection,
    _is_greeting,
    _parse_tool_plan,
    _tool_catalog,
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


def test_is_greeting_and_fallback() -> None:
    assert _is_greeting("Hello there")
    tool = SimpleNamespace(name="current_time")
    selection = _fallback_tool_selection("what time is it", [tool])
    assert selection == ["current_time"]


def test_parse_tool_plan_handles_json_and_text() -> None:
    text = 'Response: {"tools": [{"name": "search"}]}'
    selected = _parse_tool_plan(text, {"search"})
    assert selected == ["search"]
    selected_list = _parse_tool_plan('["search", "other"]', {"search"})
    assert selected_list == ["search"]
    assert _extract_json_block("prefix [1,2] suffix") == "[1,2]"


def test_tool_catalog_handles_args_schema() -> None:
    class DummySchema:
        def model_json_schema(self):
            raise RuntimeError("no")

        def schema(self):
            return {"type": "object", "properties": {"q": {"type": "string"}}}

    tool = SimpleNamespace(name="search", description="desc", args_schema=DummySchema())
    catalog = _tool_catalog([tool])
    assert "search" in catalog
    assert "properties" in catalog


@pytest.mark.asyncio
async def test_planner_tool_policy_short_circuits_greeting() -> None:
    policy = PlannerToolPolicy()
    model = DummyModel(DummyMessage("{}"))
    tools = [SimpleNamespace(name="search")]
    selected, usage = await policy.select_tools(
        user_input="hi", tools=tools, model_client=model, model_name="test"
    )
    assert selected == []
    assert usage == {}
    assert model.calls == 0


@pytest.mark.asyncio
async def test_planner_tool_policy_fallback_time() -> None:
    policy = PlannerToolPolicy()
    model = DummyModel(DummyMessage("{}"))
    tools = [SimpleNamespace(name="current_time")]
    selected, usage = await policy.select_tools(
        user_input="what time is it", tools=tools, model_client=model, model_name="x"
    )
    assert selected == ["current_time"]
    assert usage == {}
    assert model.calls == 0


@pytest.mark.asyncio
async def test_planner_tool_policy_uses_model() -> None:
    policy = PlannerToolPolicy()
    model = DummyModel(DummyMessage('{"tools": [{"name": "search"}]}'))
    tools = [SimpleNamespace(name="search")]
    selected, usage = await policy.select_tools(
        user_input="find stuff", tools=tools, model_client=model, model_name="x"
    )
    assert selected == ["search"]
    assert usage == {"x": {"input_tokens": 0, "output_tokens": 0}}
    assert model.calls == 1


@pytest.mark.asyncio
async def test_auto_tool_policy_selects_all() -> None:
    policy = AutoToolPolicy()
    tools = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    selected, usage = await policy.select_tools(
        user_input="anything", tools=tools, model_client=None, model_name="x"
    )
    assert selected == ["a", "b"]
    assert usage == {}
