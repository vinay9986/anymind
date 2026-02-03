"""Robust JSON parsing utilities with extraction + repair fallbacks."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

import structlog

try:
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover

    def repair_json(text: str, return_objects: bool = False):  # type: ignore[no-redef]
        return text


log = structlog.get_logger(__name__)


def extract_json_from_response(text: str) -> str:
    """Extract a JSON object from mixed response text."""
    fenced_patterns = [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]

    for pattern in fenced_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            return _normalize_json_quotes(extracted)

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        extracted = text[first_brace : last_brace + 1]
        extracted = extracted.encode("utf-8").decode("utf-8-sig").strip()
        return _normalize_json_quotes(extracted)

    cleaned = text.encode("utf-8").decode("utf-8-sig").strip()
    return _normalize_json_quotes(cleaned)


def _normalize_json_quotes(json_str: str) -> str:
    if not json_str.strip():
        return json_str

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    escaped_singles: list[str] = []
    temp_marker = "___ESCAPED_SINGLE___"

    escaped_pattern = r"\\'"
    for i, match in enumerate(re.finditer(escaped_pattern, json_str)):
        marker = f"{temp_marker}{i}"
        escaped_singles.append(marker)
        json_str = json_str[: match.start()] + marker + json_str[match.end() :]

    result = json_str.replace("'", '"')

    for i, marker in enumerate(escaped_singles):
        result = result.replace(marker, '\\"')

    return result


def _raw_decode_first_value(text: str) -> Any:
    candidate = text.lstrip()
    if not candidate:
        raise json.JSONDecodeError("Empty response", text, 0)

    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(candidate)
    return value


async def parse_json_robust(text: Any, *, context: str = "unknown") -> Any:
    """Robust JSON parsing utility: text → JSON object."""
    if not isinstance(text, str):
        return text

    original_text = text.strip()
    extracted_json = original_text
    repaired_payload: Any | None = None
    start = perf_counter()
    success = False
    success_method: str | None = None
    parsed_type: str | None = None

    try:
        try:
            parsed = json.loads(original_text)
        except json.JSONDecodeError as exc:
            log.warning(
                "parse_json_robust_direct_failed",
                context=context,
                length=len(original_text),
                error=str(exc),
            )
            try:
                parsed = _raw_decode_first_value(original_text)
            except json.JSONDecodeError:
                parsed = None
            else:
                log.debug(
                    "parse_json_robust_raw_decode_success",
                    context=context,
                    length=len(original_text),
                )
                success = True
                success_method = "raw_decode"
                parsed_type = type(parsed).__name__
                return parsed
        else:
            log.debug(
                "parse_json_robust_direct_success",
                context=context,
                length=len(original_text),
            )
            success = True
            success_method = "direct"
            parsed_type = type(parsed).__name__
            return parsed

        extracted_json = extract_json_from_response(original_text)
        try:
            parsed = json.loads(extracted_json)
        except json.JSONDecodeError as exc:
            log.warning(
                "parse_json_robust_extraction_failed",
                context=context,
                original_length=len(original_text),
                extracted_length=len(extracted_json),
                error=str(exc),
            )
            try:
                parsed = _raw_decode_first_value(extracted_json)
            except json.JSONDecodeError:
                parsed = None
            else:
                log.debug(
                    "parse_json_robust_extraction_raw_decode_success",
                    context=context,
                    original_length=len(original_text),
                    extracted_length=len(extracted_json),
                )
                success = True
                success_method = "extraction_raw_decode"
                parsed_type = type(parsed).__name__
                return parsed
        else:
            log.debug(
                "parse_json_robust_extraction_success",
                context=context,
                original_length=len(original_text),
                extracted_length=len(extracted_json),
            )
            success = True
            success_method = "extraction"
            parsed_type = type(parsed).__name__
            return parsed

        try:
            repaired_payload = repair_json(extracted_json, return_objects=True)
            if isinstance(repaired_payload, str):
                try:
                    parsed = json.loads(repaired_payload)
                except json.JSONDecodeError:
                    parsed = _raw_decode_first_value(repaired_payload)
            else:
                parsed = repaired_payload
        except Exception as exc:  # pragma: no cover
            log.error(
                "parse_json_robust_repair_failed",
                context=context,
                original_length=len(original_text),
                extracted_length=len(extracted_json),
                error=str(exc),
            )
            doc_source = (
                repaired_payload
                if isinstance(repaired_payload, str)
                else extracted_json
            )
            raise json.JSONDecodeError(
                msg=f"JSON parsing failed: {exc!s}", doc=str(doc_source), pos=0
            ) from exc
        else:
            log.debug(
                "parse_json_robust_repair_success",
                context=context,
                original_length=len(original_text),
                extracted_length=len(extracted_json),
                result_type=type(parsed).__name__,
            )
            success = True
            success_method = "repair"
            parsed_type = type(parsed).__name__
            return parsed
    finally:
        duration_ms = (perf_counter() - start) * 1000
        if success:
            log.info(
                "parse_json_robust_success",
                context=context,
                method=success_method,
                original_length=len(original_text),
                extracted_length=len(extracted_json),
                result_type=parsed_type,
                duration_ms=duration_ms,
            )
        log.debug(
            "parse_json_robust_done",
            context=context,
            duration_ms=duration_ms,
            success=success,
        )
