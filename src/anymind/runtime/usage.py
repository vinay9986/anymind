from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from langchain_core.messages import BaseMessage

@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)


def extract_usage_from_messages(messages: Iterable[BaseMessage]) -> UsageTotals:
    totals = UsageTotals()
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        totals.add(input_tokens, output_tokens)
    return totals


def normalize_usage_metadata(
    model_name: str, messages: Iterable[BaseMessage]
) -> Dict[str, Dict[str, int]]:
    totals = extract_usage_from_messages(messages)
    return {
        str(model_name): {
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
        }
    }
