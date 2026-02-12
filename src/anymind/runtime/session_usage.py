from __future__ import annotations

from typing import Any, Dict

from anymind.runtime.session import Session
from anymind.runtime.usage import PricingTable, UsageTotals


def apply_usage(session: Session, usage_metadata: Dict[str, Dict[str, int]]) -> None:
    for model_name, usage in usage_metadata.items():
        model_totals = session.totals_by_model.setdefault(model_name, UsageTotals())
        model_totals.add(usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def token_totals(session: Session) -> Dict[str, Dict[str, int]]:
    totals_out: Dict[str, Dict[str, int]] = {}
    for model_name, totals in session.totals_by_model.items():
        totals_out[model_name] = {
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
            "total_tokens": totals.input_tokens + totals.output_tokens,
        }
    return totals_out


def _total_tokens(session: Session) -> int:
    return sum(
        totals.input_tokens + totals.output_tokens
        for totals in session.totals_by_model.values()
    )


def enforce_token_budget(session: Session) -> None:
    if session.model_config.budget_tokens is None:
        return
    if _total_tokens(session) >= int(session.model_config.budget_tokens):
        session.budget_exhausted = True


def session_summary(session: Session) -> Dict[str, Any]:
    models: Dict[str, Dict[str, float]] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_input_cost = 0.0
    total_output_cost = 0.0
    provider = (session.model_config.model_provider or "").lower()
    costs_free = provider == "ollama"

    pricing: PricingTable = session.pricing

    for model_name, totals in session.totals_by_model.items():
        costs = (
            {"input": 0.0, "output": 0.0, "total": 0.0}
            if costs_free
            else pricing.cost(model_name, totals)
        )
        model_input = totals.input_tokens
        model_output = totals.output_tokens
        models[model_name] = {
            "input_tokens": model_input,
            "output_tokens": model_output,
            "total_tokens": model_input + model_output,
            "input_cost": costs["input"],
            "output_cost": costs["output"],
            "total_cost": costs["total"],
        }
        total_input_tokens += model_input
        total_output_tokens += model_output
        total_input_cost += costs["input"]
        total_output_cost += costs["output"]

    total = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "input_cost": total_input_cost,
        "output_cost": total_output_cost,
        "total_cost": total_input_cost + total_output_cost,
    }

    return {
        "currency": pricing.currency,
        "models": models,
        "total": total,
    }
