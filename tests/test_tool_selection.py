import pytest

from anymind.runtime.tool_selection import select_tools_for_policy


class DummyTool:
    def __init__(self, name: str) -> None:
        self.name = name


class DummyPlanner:
    name = "planner"

    def __init__(self, selections, usage=None) -> None:
        self._selections = selections
        self._usage = usage or {}

    async def select_tools(self, **kwargs):
        return self._selections, self._usage


@pytest.mark.asyncio
async def test_select_tools_never_policy() -> None:
    tools = [DummyTool("a"), DummyTool("b")]
    policy = DummyPlanner(selections=["a"])
    policy.name = "never"
    selected, usage = await select_tools_for_policy(
        policy=policy,
        tools=tools,
        user_input="test",
        model_client=None,
        model_name="gpt",
    )
    assert selected == []
    assert usage == {}


@pytest.mark.asyncio
async def test_select_tools_auto_policy() -> None:
    tools = [DummyTool("a"), DummyTool("b")]
    policy = DummyPlanner(selections=["a"])
    policy.name = "auto"
    selected, usage = await select_tools_for_policy(
        policy=policy,
        tools=tools,
        user_input="test",
        model_client=None,
        model_name="gpt",
    )
    assert [tool.name for tool in selected] == ["a", "b"]
    assert usage == {}


@pytest.mark.asyncio
async def test_select_tools_planner_empty() -> None:
    tools = [DummyTool("a"), DummyTool("b")]
    policy = DummyPlanner(selections=[], usage={"gpt": {"input_tokens": 1}})
    selected, usage = await select_tools_for_policy(
        policy=policy,
        tools=tools,
        user_input="test",
        model_client=None,
        model_name="gpt",
    )
    assert selected == []
    assert usage == {"gpt": {"input_tokens": 1}}


@pytest.mark.asyncio
async def test_select_tools_planner_subset() -> None:
    tools = [DummyTool("a"), DummyTool("b"), DummyTool("c")]
    policy = DummyPlanner(selections=["b"])
    selected, usage = await select_tools_for_policy(
        policy=policy,
        tools=tools,
        user_input="test",
        model_client=None,
        model_name="gpt",
    )
    assert [tool.name for tool in selected] == ["b"]
    assert usage == {}
