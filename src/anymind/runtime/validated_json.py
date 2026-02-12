from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import BaseMessage

from anymind.runtime.json_parser import parse_json_robust
from anymind.runtime.json_validation import JSONStructureValidator
from anymind.runtime.validation_prompts import (
    create_json_parsing_fix_prompt,
    create_validation_fix_prompt,
    extract_keep_unchanged_fields,
    extract_keep_unchanged_sections,
)

import structlog

log = structlog.get_logger(__name__)

CallModel = Callable[[str, str], Awaitable[tuple[str, Optional[dict[str, int]]]]]


@dataclass
class ValidatedJsonResult:
    data: dict[str, Any]
    raw: str
    attempts: int
    usage_metadata: list[dict[str, int]]


class SessionManager:
    """Manages hierarchical session IDs for validation tracking."""

    def __init__(self, algorithm_name: str) -> None:
        self.algorithm_name = (algorithm_name or "validation").lower()
        self.base_session_id = (
            f"{self.algorithm_name}_session_{int(time.time() * 1000) % 1000000}"
        )
        self.sub_session_counter = 0

    def get_session_id(self) -> str:
        return self.base_session_id

    def create_sub_session(self, context: str = "default") -> str:
        self.sub_session_counter += 1
        normalized_context = (context or "default").strip().replace(" ", "_")
        return f"{self.base_session_id}_{normalized_context}_{self.sub_session_counter}"


def _extract_assistant_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, BaseMessage):
        content = getattr(message, "content", "")
        if isinstance(content, list):
            return "\n".join(str(part) for part in content)
        return str(content)
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text = str(block.get("text") or "").strip()
                    if text:
                        texts.append(text)
            if texts:
                return "\n".join(texts)
        if "text" in message:
            return str(message.get("text") or "")
        return json.dumps(message, ensure_ascii=False)
    return str(message)


async def _call_model(
    model_client: Any, system_prompt: str, user_prompt: str
) -> tuple[str, Optional[dict[str, int]]]:
    model_name = (
        getattr(model_client, "model_name", None)
        or getattr(model_client, "model", None)
        or getattr(model_client, "name", None)
        or "unknown"
    )
    message = await model_client.ainvoke(
        [("system", system_prompt), ("user", user_prompt)]
    )
    usage = getattr(message, "usage_metadata", None)
    return _extract_assistant_text(message), usage


def _classify_syntax_error(error_message: str) -> str:
    error_msg_lower = (error_message or "").lower()

    if "expecting ',' delimiter" in error_msg_lower:
        return "missing_comma"
    if "expecting ':' delimiter" in error_msg_lower:
        return "missing_colon"
    if "expecting property name" in error_msg_lower:
        return "missing_property_name"
    if "unterminated string" in error_msg_lower:
        return "unterminated_string"
    if "expecting '}'" in error_msg_lower:
        return "missing_closing_brace"
    if "expecting ']'" in error_msg_lower:
        return "missing_closing_bracket"
    if "invalid character" in error_msg_lower:
        return "invalid_character"
    if "trailing comma" in error_msg_lower:
        return "trailing_comma"
    return "unknown_syntax_error"


def _determine_fixing_strategy(error_classification: str) -> str:
    strategy_map = {
        "missing_comma": "Add missing comma between elements",
        "missing_colon": "Add missing colon after property name",
        "missing_property_name": "Add missing property name in quotes",
        "unterminated_string": "Add missing closing quote",
        "missing_closing_brace": "Add missing closing brace }",
        "missing_closing_bracket": "Add missing closing bracket ]",
        "invalid_character": "Remove or escape invalid character",
        "trailing_comma": "Remove trailing comma",
        "unknown_syntax_error": "Analyze and fix syntax error",
    }
    return strategy_map.get(error_classification, "Generic syntax fixing")


def _assess_error_severity(error_classification: str) -> str:
    high_severity = {
        "missing_closing_brace",
        "missing_closing_bracket",
        "unterminated_string",
    }
    medium_severity = {
        "missing_comma",
        "missing_colon",
        "invalid_character",
    }

    if error_classification in high_severity:
        return "high"
    if error_classification in medium_severity:
        return "medium"
    return "low"


async def generate_validated_json_with_calls(
    *,
    role_name: str,
    system_prompt: str,
    user_prompt: str,
    validator: JSONStructureValidator,
    call_model: CallModel,
    fix_model: CallModel | None = None,
    max_reasks: int = 3,
    original_task_context: str,
) -> ValidatedJsonResult:
    """Generate a validated JSON object using custom callables."""

    session_manager = SessionManager(role_name)
    usage_list: list[dict[str, int]] = []
    fix_model = fix_model or call_model

    raw, usage = await call_model(system_prompt, user_prompt)
    if usage:
        usage_list.append(usage)

    original_output = raw
    current_json = raw

    async def _syntax_validate_and_fix(json_text: str) -> tuple[Any, str, int]:
        try:
            parsed_obj = await parse_json_robust(
                json_text,
                context=session_manager.create_sub_session("syntax_parse_initial"),
            )
            return (
                parsed_obj,
                json.dumps(parsed_obj, ensure_ascii=False, separators=(",", ":")),
                0,
            )
        except json.JSONDecodeError as exc:
            last_exc: json.JSONDecodeError = exc
            current = json_text
            for attempt_idx in range(1, max_reasks + 1):
                error_position = getattr(last_exc, "pos", 0)
                error_message = getattr(last_exc, "msg", str(last_exc))
                error_classification = _classify_syntax_error(error_message)
                log.warning(
                    "json_syntax_errors_detected",
                    role=role_name,
                    session_id=session_manager.get_session_id(),
                    attempt=attempt_idx,
                    error_position=error_position,
                    error_message=error_message,
                    error_classification=error_classification,
                    fixing_strategy=_determine_fixing_strategy(error_classification),
                    severity=_assess_error_severity(error_classification),
                )
                keep = extract_keep_unchanged_sections(
                    current, getattr(last_exc, "pos", 0)
                )
                fix_user = create_json_parsing_fix_prompt(
                    original_task_context=original_task_context,
                    raw_llm_output=original_output,
                    json_to_fix=current,
                    parsing_error_details=str(last_exc),
                    error_position=getattr(last_exc, "pos", 0),
                    error_context=str(getattr(last_exc, "msg", "")),
                    keep_unchanged_sections=keep,
                )
                current, usage = await fix_model(system_prompt, fix_user)
                if usage:
                    usage_list.append(usage)
                try:
                    parsed_obj = await parse_json_robust(
                        current,
                        context=session_manager.create_sub_session(
                            f"syntax_parse_attempt_{attempt_idx}"
                        ),
                    )
                    return (
                        parsed_obj,
                        json.dumps(
                            parsed_obj, ensure_ascii=False, separators=(",", ":")
                        ),
                        attempt_idx,
                    )
                except json.JSONDecodeError as next_exc:
                    last_exc = next_exc
            raise last_exc

    parsed, current_json, syntax_attempts = await _syntax_validate_and_fix(current_json)

    attempts = syntax_attempts
    for attempt_idx in range(0, max_reasks + 1):
        validation = validator.validate(parsed, original_text=current_json)
        if validation.validation_passed:
            return ValidatedJsonResult(
                data=validation.validated_output,
                raw=current_json,
                attempts=attempts + attempt_idx,
                usage_metadata=usage_list,
            )

        keep_fields = extract_keep_unchanged_fields(
            parsed, validation.validation_errors or []
        )
        fix_user = create_validation_fix_prompt(
            original_task_context=original_task_context,
            raw_llm_output=original_output,
            serialized_json_object=json.dumps(
                validation.validated_output, ensure_ascii=False, indent=2, default=str
            ),
            validation_error_details=validation.error_message or "",
            keep_unchanged_fields=keep_fields,
        )
        current_json, usage = await fix_model(system_prompt, fix_user)
        if usage:
            usage_list.append(usage)
        parsed, current_json, extra_attempts = await _syntax_validate_and_fix(
            current_json
        )
        attempts += extra_attempts

    return ValidatedJsonResult(
        data=parsed,
        raw=current_json,
        attempts=attempts,
        usage_metadata=usage_list,
    )


async def generate_validated_json(
    *,
    role_name: str,
    system_prompt: str,
    user_prompt: str,
    validator: JSONStructureValidator,
    model_client: Any,
    max_reasks: int = 3,
    original_task_context: str,
) -> ValidatedJsonResult:
    """Generate a validated JSON object with focused two-phase correction loops."""

    async def _call(
        system_prompt: str, user_prompt: str
    ) -> tuple[str, Optional[dict[str, int]]]:
        return await _call_model(model_client, system_prompt, user_prompt)

    return await generate_validated_json_with_calls(
        role_name=role_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        validator=validator,
        call_model=_call,
        fix_model=_call,
        max_reasks=max_reasks,
        original_task_context=original_task_context,
    )
