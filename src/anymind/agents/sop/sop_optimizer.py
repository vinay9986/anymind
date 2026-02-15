from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from anymind.runtime.json_parser import parse_json_robust
from anymind.agents.sop.sop_validation import validate_sop_structure
from anymind.runtime.llm_errors import safe_ainvoke


@dataclass
class SopOptimizeResult:
    sop: dict[str, Any]
    usage_metadata: list[dict[str, int]]


_SYSTEM_PROMPT = (
    "You are an SOP optimizer. You will be given a JSON SOP workflow with nodes and edges.\n"
    "Your job:\n"
    "1) Preserve the intent/objective.\n"
    "2) Improve for execution: remove redundancy, make steps self-contained, and increase safe parallelism.\n"
    "3) Ensure the result is VALID and executable.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Reply with exactly ONE minified JSON object.\n"
    "- No markdown, no code fences, no comments, no extra prose.\n"
    "- Required keys: nodes (list), edges (list). Optional keys: id,name,version,description,metadata,optimize.\n"
    "- Node ids must be unique; edges must reference existing node ids; no self-loops.\n"
)


def _render_user_prompt(sop: dict[str, Any]) -> str:
    return (
        "Optimize this SOP JSON. Keep the same general structure and fields.\n"
        "If the input has `optimize: true`, set `optimize` to false in the output and "
        "add metadata.optimization.applied=true.\n\n"
        f"SOP JSON:\n{json.dumps(sop, ensure_ascii=False, indent=2, default=str)}"
    )


async def _call_model(
    model_client: Any, system_prompt: str, user_prompt: str
) -> tuple[str, dict[str, int] | None]:
    message = await safe_ainvoke(
        model_client, [("system", system_prompt), ("user", user_prompt)]
    )
    usage = getattr(message, "usage_metadata", None)
    content = getattr(message, "content", "")
    if isinstance(content, list):
        text = "\n".join(str(part) for part in content)
    else:
        text = str(content)
    return text, usage


async def optimize_sop(
    *,
    sop: dict[str, Any],
    model_client: Any,
    max_reasks: int = 2,
) -> SopOptimizeResult:
    """Optimize a SOP using an LLM, returning a structurally valid SOP dict."""
    usage_list: list[dict[str, int]] = []

    raw, usage = await _call_model(
        model_client, _SYSTEM_PROMPT, _render_user_prompt(sop)
    )
    if usage:
        usage_list.append(usage)

    for attempt in range(max_reasks + 1):
        try:
            parsed = await parse_json_robust(
                raw, context=f"sop_optimize_attempt_{attempt}"
            )
        except json.JSONDecodeError as exc:
            fix_user = (
                "Your previous output was not valid JSON.\n"
                f"Error: {exc}\n"
                "Return ONLY a single minified JSON object with keys nodes/edges (+ optional metadata).\n"
            )
            raw, usage = await _call_model(model_client, _SYSTEM_PROMPT, fix_user)
            if usage:
                usage_list.append(usage)
            continue

        ok, errors = validate_sop_structure(parsed)
        if ok:
            optimized = dict(parsed)
            if optimized.get("optimize") is True:
                optimized["optimize"] = False
            meta = optimized.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                optimized["metadata"] = meta
            opt_meta = meta.get("optimization")
            if not isinstance(opt_meta, dict):
                opt_meta = {}
                meta["optimization"] = opt_meta
            opt_meta.setdefault("applied", True)
            return SopOptimizeResult(sop=optimized, usage_metadata=usage_list)

        fix_user = (
            "Your previous output JSON failed SOP validation.\n"
            f"Validation errors: {errors}\n"
            "Fix the JSON so it validates. Return ONLY a single minified JSON object.\n"
        )
        raw, usage = await _call_model(model_client, _SYSTEM_PROMPT, fix_user)
        if usage:
            usage_list.append(usage)

    raise RuntimeError(f"SOP optimization failed validation after {max_reasks} reasks.")
