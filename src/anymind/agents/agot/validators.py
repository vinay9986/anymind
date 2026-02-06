from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anymind.runtime.json_validation import ValidationResult


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class StrictJSONValidator:
    """Structured validator compatible with validated JSON helpers."""

    expected_keys: tuple[str, ...]
    required_types: dict[str, type]
    allow_additional_keys: bool = False
    validator_name: str = "strict-json-validator"

    def validate(self, json_obj: Any, *, original_text: str) -> ValidationResult:
        del original_text
        if not isinstance(json_obj, dict):
            return ValidationResult(
                validation_passed=False,
                validated_output=json_obj,
                error_message=f"Expected JSON object, got {type(json_obj).__name__}",
                validation_errors=[
                    {
                        "field": None,
                        "issue": "wrong_type",
                        "expected": "dict",
                        "got": type(json_obj).__name__,
                    },
                ],
            )

        errors: list[str] = []
        validation_errors: list[dict[str, Any]] = []

        expected = set(self.expected_keys)
        actual = set(json_obj.keys())

        missing = [k for k in self.expected_keys if k not in json_obj]
        if missing:
            for k in missing:
                errors.append(f"Missing required field: '{k}'")
                validation_errors.append({"field": k, "issue": "missing_required"})

        if not self.allow_additional_keys:
            extra = sorted(actual - expected)
            if extra:
                errors.append(f"Unexpected extra field(s): {', '.join(extra)}")
                for k in extra:
                    validation_errors.append({"field": k, "issue": "unexpected_field"})

        for field_name, expected_type in self.required_types.items():
            if field_name not in json_obj:
                continue
            value = json_obj[field_name]
            if not isinstance(value, expected_type):
                errors.append(
                    f"Field '{field_name}' must be {expected_type.__name__}, got {type(value).__name__}"
                )
                validation_errors.append(
                    {
                        "field": field_name,
                        "issue": "wrong_type",
                        "expected": expected_type.__name__,
                        "got": type(value).__name__,
                    }
                )

        if errors:
            return ValidationResult(
                validation_passed=False,
                validated_output=json_obj,
                error_message="; ".join(errors),
                validation_errors=validation_errors,
            )

        return ValidationResult(validation_passed=True, validated_output=json_obj)


class TaskListResponseValidator:
    def __init__(self, *, max_tasks: int, validator_name: str) -> None:
        self._max_tasks = int(max_tasks)
        self.validator_name = validator_name
        self._base = StrictJSONValidator(
            expected_keys=("tasks", "strategy"),
            required_types={"tasks": list, "strategy": str},
            allow_additional_keys=False,
            validator_name=validator_name,
        )

    def validate(self, json_obj: Any, *, original_text: str) -> ValidationResult:
        base = self._base.validate(json_obj, original_text=original_text)
        if not base.validation_passed:
            return base

        assert isinstance(json_obj, dict)
        tasks = json_obj.get("tasks")
        strategy = json_obj.get("strategy")

        errors: list[str] = []
        validation_errors: list[dict[str, Any]] = []

        if not _is_non_empty_str(strategy):
            errors.append("Field 'strategy' must be a non-empty string")
            validation_errors.append({"field": "strategy", "issue": "empty_string"})

        if not isinstance(tasks, list):
            errors.append("Field 'tasks' must be a list")
            validation_errors.append(
                {
                    "field": "tasks",
                    "issue": "wrong_type",
                    "expected": "list",
                    "got": type(tasks).__name__,
                }
            )
        else:
            if len(tasks) > self._max_tasks:
                errors.append(
                    f"Field 'tasks' must have <={self._max_tasks} items, got {len(tasks)}"
                )
                validation_errors.append(
                    {
                        "field": "tasks",
                        "issue": "too_many_items",
                        "max": self._max_tasks,
                        "got": len(tasks),
                    }
                )

            for idx, item in enumerate(tasks):
                field_prefix = f"tasks[{idx}]"
                if not isinstance(item, dict):
                    errors.append(f"Field '{field_prefix}' must be an object")
                    validation_errors.append(
                        {
                            "field": field_prefix,
                            "issue": "wrong_type",
                            "expected": "dict",
                            "got": type(item).__name__,
                        }
                    )
                    continue

                expected_keys = {"title", "content"}
                actual_keys = set(item.keys())
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)

                for k in missing:
                    errors.append(f"Missing required field '{field_prefix}.{k}'")
                    validation_errors.append(
                        {"field": f"{field_prefix}.{k}", "issue": "missing_required"}
                    )
                for k in extra:
                    errors.append(f"Unexpected extra field '{field_prefix}.{k}'")
                    validation_errors.append(
                        {"field": f"{field_prefix}.{k}", "issue": "unexpected_field"}
                    )

                title = item.get("title")
                content = item.get("content")
                if not _is_non_empty_str(title):
                    errors.append(
                        f"Field '{field_prefix}.title' must be a non-empty string"
                    )
                    validation_errors.append(
                        {"field": f"{field_prefix}.title", "issue": "empty_string"}
                    )
                if not _is_non_empty_str(content):
                    errors.append(
                        f"Field '{field_prefix}.content' must be a non-empty string"
                    )
                    validation_errors.append(
                        {"field": f"{field_prefix}.content", "issue": "empty_string"}
                    )

        if errors:
            return ValidationResult(
                validation_passed=False,
                validated_output=json_obj,
                error_message="; ".join(errors),
                validation_errors=validation_errors,
            )

        return ValidationResult(validation_passed=True, validated_output=json_obj)


def make_validator_complexity(*, validator_name: str) -> StrictJSONValidator:
    return StrictJSONValidator(
        expected_keys=("is_complex", "justification"),
        required_types={"is_complex": bool, "justification": str},
        allow_additional_keys=False,
        validator_name=validator_name,
    )


def make_validator_eval(*, validator_name: str) -> StrictJSONValidator:
    return StrictJSONValidator(
        expected_keys=("answer",),
        required_types={},
        allow_additional_keys=False,
        validator_name=validator_name,
    )


def make_validator_final_task(*, validator_name: str) -> StrictJSONValidator:
    return StrictJSONValidator(
        expected_keys=("title", "content"),
        required_types={"title": str, "content": str},
        allow_additional_keys=False,
        validator_name=validator_name,
    )


def make_validator_final_answer(*, validator_name: str) -> StrictJSONValidator:
    return StrictJSONValidator(
        expected_keys=("final_answer", "graph"),
        required_types={"graph": str},
        allow_additional_keys=False,
        validator_name=validator_name,
    )
