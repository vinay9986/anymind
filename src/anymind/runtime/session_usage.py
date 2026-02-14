from __future__ import annotations

from typing import Any, Dict

from anymind.runtime.session import Session
from anymind.runtime.usage import UsageTotals
from anymind.runtime.usage_store import get_usage_store


def apply_usage(session: Session, usage_metadata: Dict[str, Dict[str, int]]) -> None:
    for model_name, usage in usage_metadata.items():
        model_totals = session.totals_by_model.setdefault(model_name, UsageTotals())
        model_totals.add(usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def token_totals(session: Session) -> Dict[str, Dict[str, int]]:
    snapshot = get_usage_store().get(session.session_id)
    totals_out: Dict[str, Dict[str, int]] = {}
    if snapshot.per_model:
        for model_name, totals in snapshot.per_model.items():
            totals_out[model_name] = {
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
                "total_tokens": totals.input_tokens + totals.output_tokens,
            }
        return totals_out
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
    totals = get_usage_store().get(session.session_id).totals
    if (totals.input_tokens + totals.output_tokens) >= int(
        session.model_config.budget_tokens
    ):
        session.budget_exhausted = True


def session_summary(session: Session) -> Dict[str, Any]:
    models: Dict[str, Dict[str, float]] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    usage_snapshot = get_usage_store().get(session.session_id)
    totals_by_model = usage_snapshot.per_model or session.totals_by_model

    for model_name, totals in totals_by_model.items():
        model_input = totals.input_tokens
        model_output = totals.output_tokens
        models[model_name] = {
            "input_tokens": model_input,
            "output_tokens": model_output,
            "total_tokens": model_input + model_output,
        }
        total_input_tokens += model_input
        total_output_tokens += model_output

    if usage_snapshot.per_model:
        total = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
    else:
        totals = usage_snapshot.totals
        total = {
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
            "total_tokens": totals.input_tokens + totals.output_tokens,
        }

    return {
        "models": models,
        "total": total,
    }
