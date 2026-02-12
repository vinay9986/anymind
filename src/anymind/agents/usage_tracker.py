from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from anymind.runtime.usage import UsageTotals


@dataclass
class UsageBudgetTracker:
    model_name: str
    budget_tokens: Optional[int] = None
    totals_by_model: dict[str, UsageTotals] = field(default_factory=dict)

    def add_usage(
        self, usage: Optional[dict[str, int]], *, model_name: Optional[str] = None
    ) -> None:
        if not usage:
            return
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        name = model_name or self.model_name
        totals = self.totals_by_model.setdefault(name, UsageTotals())
        totals.add(input_tokens, output_tokens)

    def add_usage_list(
        self,
        usage_list: Iterable[dict[str, int]],
        *,
        model_name: Optional[str] = None,
    ) -> None:
        for usage in usage_list:
            self.add_usage(usage, model_name=model_name)

    def add_usage_metadata(
        self, usage_metadata: Optional[dict[str, dict[str, int]]]
    ) -> None:
        if not usage_metadata:
            return
        for model, usage in usage_metadata.items():
            self.add_usage(usage, model_name=model)

    @property
    def input_tokens(self) -> int:
        return sum(totals.input_tokens for totals in self.totals_by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(totals.output_tokens for totals in self.totals_by_model.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def budget_exhausted(self) -> bool:
        if self.budget_tokens is None:
            return False
        return self.total_tokens >= int(self.budget_tokens)

    def remaining_budget(self) -> Optional[int]:
        if self.budget_tokens is None:
            return None
        remaining = int(self.budget_tokens) - self.total_tokens
        return max(0, remaining)

    def usage_metadata(self) -> dict[str, dict[str, int]]:
        if not self.totals_by_model:
            return {
                self.model_name: {
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            }
        return {
            model: {
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
            }
            for model, totals in self.totals_by_model.items()
        }
