from __future__ import annotations

from typing import Any

from anymind.runtime.json_validation import JSONStructureValidator
from anymind.runtime.validated_json import generate_validated_json

_COVERAGE_GATE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Score whether an answer fully addresses the question.\n"
    "Return ONLY minified JSON with the required fields."
)

_COVERAGE_GATE_USER_PROMPT = """Evaluate if this answer adequately addresses the question.

Question:
{question}

Answer:
{answer}

Context (may be empty):
{context}

Evaluate:
1. Completeness (coverage 0.0-1.0): Does it fully answer the question?
2. Sufficiency: "stop" (adequate) or "refine" (needs improvement)
3. Gaps: List missing information
4. Refined follow-up: If gaps exist, suggest a refined question
5. Rationale: brief justification

Return JSON: {{
  "coverage": float,
  "sufficiency": str,
  "gaps": list[str],
  "refined_follow_up": str,
  "rationale": str
}}"""


def _gate_validator() -> JSONStructureValidator:
    return JSONStructureValidator(
        {
            "coverage": {
                "type": (int, float),
                "required": True,
                "description": "0.0-1.0 coverage score",
            },
            "sufficiency": {
                "type": str,
                "required": True,
                "description": "stop|refine",
            },
            "gaps": {
                "type": list,
                "required": True,
                "description": "missing info list",
            },
            "refined_follow_up": {
                "type": str,
                "required": True,
                "description": "next question",
            },
            "rationale": {
                "type": str,
                "required": True,
                "description": "brief rationale",
            },
        },
        validator_name="sop-coverage-gate",
    )


async def evaluate_answer(
    *,
    question: str,
    answer: str,
    context: dict[str, Any] | None,
    model_client: Any,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    ctx = context or {}
    user_prompt = _COVERAGE_GATE_USER_PROMPT.format(
        question=(question or "").strip(),
        answer=(answer or "").strip(),
        context=str(ctx).strip(),
    )

    result = await generate_validated_json(
        role_name="sop_coverage_gate",
        system_prompt=_COVERAGE_GATE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        validator=_gate_validator(),
        model_client=model_client,
        max_reasks=2,
        original_task_context="SOP coverage gate evaluation",
    )

    gate = dict(result.data)
    coverage_raw = gate.get("coverage", 0.0)
    try:
        coverage = float(coverage_raw)
    except Exception:
        coverage = 0.0
    coverage = max(0.0, min(1.0, coverage))

    suff = str(gate.get("sufficiency", "refine") or "refine").strip().lower()
    if suff not in {"stop", "refine"}:
        suff = "refine"

    gaps_raw = gate.get("gaps")
    gaps: list[str] = []
    if isinstance(gaps_raw, list):
        gaps = [str(x).strip() for x in gaps_raw if str(x).strip()]

    refined = str(gate.get("refined_follow_up", "") or "").strip()
    rationale = str(gate.get("rationale", "") or "").strip()

    return (
        {
            "coverage": coverage,
            "sufficiency": suff,
            "gaps": gaps,
            "refined_follow_up": refined,
            "rationale": rationale,
        },
        result.usage_metadata,
    )
