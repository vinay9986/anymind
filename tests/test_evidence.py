import pytest

from anymind.runtime.evidence import (
    EvidenceLedger,
    get_current_ledger,
    summarize_for_display,
    use_ledger,
)


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


def test_ledger_summarize_short_max_chars() -> None:
    ledger = EvidenceLedger()
    record = ledger.add("tool", {}, "content")
    summary = EvidenceLedger.summarize([record], max_chars=2)
    assert summary == ".."


def test_use_ledger_context_manager() -> None:
    ledger = EvidenceLedger()
    assert get_current_ledger() is None
    with use_ledger(ledger):
        assert get_current_ledger() is ledger
    assert get_current_ledger() is None


def test_summarize_for_display_prefers_timestamp() -> None:
    ledger = EvidenceLedger()
    record = ledger.add(
        "current_time",
        {},
        '{"timestamp":"2026-02-13T00:00:00+00:00","format":"iso"}',
    )
    assert summarize_for_display(record) == "2026-02-13T00:00:00+00:00"


def test_summarize_for_display_extracts_url_from_results() -> None:
    ledger = EvidenceLedger()
    record = ledger.add(
        "internet_search", {}, '{"results":[{"url":"https://example.com"}]}'
    )
    assert summarize_for_display(record) == "https://example.com"


def test_summarize_for_display_falls_back_to_regex() -> None:
    ledger = EvidenceLedger()
    record = ledger.add(
        "internet_search",
        {},
        "See https://example.com at 2026-02-13T01:02:03Z.",
    )
    assert summarize_for_display(record) == "2026-02-13T01:02:03Z"


def test_summarize_for_display_handles_list_payload() -> None:
    ledger = EvidenceLedger()
    record = ledger.add("internet_search", {}, '[{"url":"https://example.com/list"}]')
    assert summarize_for_display(record) == "https://example.com/list"


def test_summarize_for_display_falls_back_to_url() -> None:
    ledger = EvidenceLedger()
    record = ledger.add(
        "internet_search", {}, "Visit https://example.com/docs for details."
    )
    assert summarize_for_display(record) == "https://example.com/docs"


def test_summarize_for_display_empty_content() -> None:
    ledger = EvidenceLedger()
    record = ledger.add("internet_search", {}, "")
    assert summarize_for_display(record) == ""
