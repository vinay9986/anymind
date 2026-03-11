from __future__ import annotations

from types import SimpleNamespace

import pytest

from anymind.config.schemas import ModelConfig
from anymind.runtime.session import Session


@pytest.mark.asyncio
async def test_session_close_closes_checkpointer_connection() -> None:
    closed = {"value": False}

    class DummyConn:
        async def close(self) -> None:
            closed["value"] = True

    session = Session(
        session_id="session-1",
        agent_name="agent",
        model_config=ModelConfig(model="gpt-test", model_provider="openai"),
        tool_policy_name="default",
        model_client=SimpleNamespace(),
        tools=[],
        agent_with_tools=None,
        agent_no_tools=None,
        checkpointer=None,
        checkpointer_conn=DummyConn(),
    )

    await session.close()
    assert closed["value"] is True
