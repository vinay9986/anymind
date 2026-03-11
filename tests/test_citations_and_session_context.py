from __future__ import annotations

from types import SimpleNamespace

import pytest

from anymind.runtime.citations import render_with_citations
from anymind.runtime.evidence import EvidenceRecord
from anymind.runtime.session_context import (
    get_session_id,
    reset_session_id,
    set_session_id,
)


def test_session_context_round_trip() -> None:
    token = set_session_id("session-1")
    assert get_session_id() == "session-1"
    reset_session_id(token)
    assert get_session_id() is None


@pytest.mark.asyncio
async def test_render_with_citations_uses_ledger_summary(monkeypatch) -> None:
    calls = {}

    async def fake_safe_ainvoke(model_client, messages):
        calls["model_client"] = model_client
        calls["messages"] = messages
        return SimpleNamespace(content="draft [E1]", usage_metadata={"input_tokens": 3})

    monkeypatch.setattr("anymind.runtime.citations.safe_ainvoke", fake_safe_ainvoke)

    evidence_records = [
        EvidenceRecord(
            id="E1",
            tool="internet_search",
            args={"q": "pricing"},
            content="OpenAI pricing page",
        ),
    ]

    text, usage = await render_with_citations(
        model_client="model-client",
        model_name="model-x",
        draft="Pricing changed.",
        evidence_records=evidence_records,
    )

    assert text == "draft [E1]"
    assert usage == {"model-x": {"input_tokens": 3, "output_tokens": 0}}
    assert calls["model_client"] == "model-client"
    assert calls["messages"][0] == ("system", "Add citations.")
    assert "Pricing changed." in calls["messages"][1][1]
    assert "[E1] internet_search: OpenAI pricing page" in calls["messages"][1][1]
