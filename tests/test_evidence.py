import pytest

from anymind.runtime.evidence import EvidenceLedger, get_current_ledger, use_ledger


def test_ledger_add_recent_all_summarize() -> None:
    ledger = EvidenceLedger()
    ledger.add("tool_a", {"q": 1}, "first")
    ledger.start_turn()
    ledger.add("tool_b", {}, "second")
    ledger.add("tool_c", {}, "third")

    recent = ledger.recent()
    assert [record.tool for record in recent] == ["tool_b", "tool_c"]
    assert len(ledger.all()) == 3

    summary = EvidenceLedger.summarize(recent, max_chars=200)
    assert "tool_b" in summary
    assert "tool_c" in summary


def test_ledger_summarize_truncates() -> None:
    ledger = EvidenceLedger()
    content = "x" * 400
    record = ledger.add("tool", {}, content)
    summary = EvidenceLedger.summarize([record], max_chars=500)
    assert summary.endswith("...")
    assert len(summary) <= 500


def test_use_ledger_context_manager() -> None:
    ledger = EvidenceLedger()
    assert get_current_ledger() is None
    with use_ledger(ledger):
        assert get_current_ledger() is ledger
    assert get_current_ledger() is None
