from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from langchain_core.messages import BaseMessage

from anymind.config.schemas import PricingConfig


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)


class PricingTable:
    def __init__(self, config: PricingConfig) -> None:
        self.currency = config.currency
        self.prices = config.prices_per_1k_tokens
        self.default = config.default

    def _entry_for_model(self, model_name: str) -> Dict[str, float]:
        if model_name in self.prices:
            entry = self.prices[model_name]
        else:
            entry = None
            for key, value in self.prices.items():
                if model_name.startswith(key):
                    entry = value
                    break
            if entry is None:
                entry = self.default
        return {
            "input": float(entry.get("input", 0.0)),
            "output": float(entry.get("output", 0.0)),
        }

    def cost(self, model_name: str, totals: UsageTotals) -> Dict[str, float]:
        entry = self._entry_for_model(model_name)
        input_cost = (totals.input_tokens / 1000.0) * entry["input"]
        output_cost = (totals.output_tokens / 1000.0) * entry["output"]
        total_cost = input_cost + output_cost
        return {
            "input": input_cost,
            "output": output_cost,
            "total": total_cost,
        }


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
