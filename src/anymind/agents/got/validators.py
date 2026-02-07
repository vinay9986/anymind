from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anymind.runtime.json_validation import ValidationResult


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class StrictJSONValidator:
    expected_keys: tuple[str, ...]
    required_types: dict[str, type]
    allow_additional_keys: bool = False
    validator_name: str = "got-strict-json"

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
                    }
                ],
            )

        errors: list[str] = []
        validation_errors: list[dict[str, Any]] = []

        expected = set(self.expected_keys)
        actual = set(json_obj.keys())

        missing = [k for k in self.expected_keys if k not in json_obj]
        if missing:
            for key in missing:
                errors.append(f"Missing required field: '{key}'")
                validation_errors.append({"field": key, "issue": "missing_required"})

        if not self.allow_additional_keys:
            extra = sorted(actual - expected)
            if extra:
                errors.append(f"Unexpected extra field(s): {', '.join(extra)}")
                for key in extra:
                    validation_errors.append(
                        {"field": key, "issue": "unexpected_field"}
                    )

        for field_name, expected_type in self.required_types.items():
            if field_name not in json_obj:
                continue
            value = json_obj[field_name]
            if not isinstance(value, expected_type):
                if isinstance(expected_type, tuple):
                    expected_name = ", ".join(
                        [getattr(t, "__name__", str(t)) for t in expected_type]
                    )
                else:
                    expected_name = getattr(
                        expected_type, "__name__", str(expected_type)
                    )
                errors.append(
                    f"Field '{field_name}' must be {expected_name}, got {type(value).__name__}"
                )
                validation_errors.append(
                    {
                        "field": field_name,
                        "issue": "wrong_type",
                        "expected": expected_name,
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
                    f"Field 'tasks' must have ≤{self._max_tasks} items, got {len(tasks)}"
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

                for key in missing:
                    errors.append(f"Missing required field '{field_prefix}.{key}'")
                    validation_errors.append(
                        {"field": f"{field_prefix}.{key}", "issue": "missing_required"}
                    )
                for key in extra:
                    errors.append(f"Unexpected extra field '{field_prefix}.{key}'")
                    validation_errors.append(
                        {"field": f"{field_prefix}.{key}", "issue": "unexpected_field"}
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


class ReflectionScoreValidator:
    def __init__(self, *, validator_name: str) -> None:
        self.validator_name = validator_name
        self._base = StrictJSONValidator(
            expected_keys=("score", "rationale"),
            required_types={"score": (int, float), "rationale": str},  # type: ignore[arg-type]
            allow_additional_keys=True,
            validator_name=validator_name,
        )

    def validate(self, json_obj: Any, *, original_text: str) -> ValidationResult:
        base = self._base.validate(json_obj, original_text=original_text)
        if not base.validation_passed:
            return base

        assert isinstance(json_obj, dict)
        allowed_keys = {"score", "rationale", "new_tasks"}
        extra_keys = sorted(set(json_obj.keys()) - allowed_keys)
        if extra_keys:
            return ValidationResult(
                validation_passed=False,
                validated_output=json_obj,
                error_message=f"Unexpected extra field(s): {', '.join(extra_keys)}",
                validation_errors=[
                    {"field": key, "issue": "unexpected_field"} for key in extra_keys
                ],
            )

        score = json_obj.get("score")
        rationale = json_obj.get("rationale")
        new_tasks = json_obj.get("new_tasks")

        errors: list[str] = []
        validation_errors: list[dict[str, Any]] = []

        try:
            score_f = float(score)
        except Exception:
            score_f = -1.0

        if not (0.0 <= score_f <= 1.0):
            errors.append("Field 'score' must be between 0.0 and 1.0")
            validation_errors.append(
                {
                    "field": "score",
                    "issue": "out_of_range",
                    "min": 0.0,
                    "max": 1.0,
                    "got": score,
                }
            )

        if not _is_non_empty_str(rationale):
            errors.append("Field 'rationale' must be a non-empty string")
            validation_errors.append({"field": "rationale", "issue": "empty_string"})

        if new_tasks is not None:
            if not isinstance(new_tasks, list):
                errors.append("Field 'new_tasks' must be a list when present")
                validation_errors.append(
                    {
                        "field": "new_tasks",
                        "issue": "wrong_type",
                        "expected": "list",
                        "got": type(new_tasks).__name__,
                    }
                )
            else:
                for idx, item in enumerate(new_tasks):
                    field_prefix = f"new_tasks[{idx}]"
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
                    for key in missing:
                        errors.append(f"Missing required field '{field_prefix}.{key}'")
                        validation_errors.append(
                            {
                                "field": f"{field_prefix}.{key}",
                                "issue": "missing_required",
                            }
                        )
                    for key in extra:
                        errors.append(f"Unexpected extra field '{field_prefix}.{key}'")
                        validation_errors.append(
                            {
                                "field": f"{field_prefix}.{key}",
                                "issue": "unexpected_field",
                            }
                        )
                    if not _is_non_empty_str(item.get("title")):
                        errors.append(
                            f"Field '{field_prefix}.title' must be a non-empty string"
                        )
                        validation_errors.append(
                            {"field": f"{field_prefix}.title", "issue": "empty_string"}
                        )
                    if not _is_non_empty_str(item.get("content")):
                        errors.append(
                            f"Field '{field_prefix}.content' must be a non-empty string"
                        )
                        validation_errors.append(
                            {
                                "field": f"{field_prefix}.content",
                                "issue": "empty_string",
                            }
                        )

        if errors:
            return ValidationResult(
                validation_passed=False,
                validated_output=json_obj,
                error_message="; ".join(errors),
                validation_errors=validation_errors,
            )

        json_obj["score"] = score_f
        return ValidationResult(validation_passed=True, validated_output=json_obj)


def make_validator_finalise(*, validator_name: str) -> StrictJSONValidator:
    return StrictJSONValidator(
        expected_keys=("final_answer", "graph"),
        required_types={"graph": str},
        allow_additional_keys=False,
        validator_name=validator_name,
    )


def make_validator_verifier(*, validator_name: str) -> StrictJSONValidator:
    return StrictJSONValidator(
        expected_keys=("score",),
        required_types={"score": (int, float)},  # type: ignore[arg-type]
        allow_additional_keys=False,
        validator_name=validator_name,
    )


class ToolPlanValidator:
    def __init__(
        self,
        *,
        max_tools: int,
        allowed_tool_names: set[str],
        validator_name: str,
    ) -> None:
        self._max_tools = max(0, int(max_tools))
        self._allowed_tool_names = {
            name.strip() for name in allowed_tool_names if name and name.strip()
        }
        self.validator_name = validator_name
        self._base = StrictJSONValidator(
            expected_keys=("tools",),
            required_types={"tools": list},
            allow_additional_keys=False,
            validator_name=validator_name,
        )

    def validate(self, json_obj: Any, *, original_text: str) -> ValidationResult:
        base = self._base.validate(json_obj, original_text=original_text)
        if not base.validation_passed:
            return base

        assert isinstance(json_obj, dict)
        tools = json_obj.get("tools")

        errors: list[str] = []
        validation_errors: list[dict[str, Any]] = []

        if not isinstance(tools, list):
            errors.append("Field 'tools' must be a list")
            validation_errors.append(
                {
                    "field": "tools",
                    "issue": "wrong_type",
                    "expected": "list",
                    "got": type(tools).__name__,
                }
            )
        else:
            if self._max_tools and len(tools) > self._max_tools:
                errors.append(
                    f"Field 'tools' must have ≤{self._max_tools} items, got {len(tools)}"
                )
                validation_errors.append(
                    {
                        "field": "tools",
                        "issue": "too_many_items",
                        "max": self._max_tools,
                        "got": len(tools),
                    }
                )

            for idx, item in enumerate(tools):
                field_prefix = f"tools[{idx}]"
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

                allowed_keys = {"name", "arguments", "reason"}
                extra = sorted(set(item.keys()) - allowed_keys)
                for key in extra:
                    errors.append(f"Unexpected extra field '{field_prefix}.{key}'")
                    validation_errors.append(
                        {"field": f"{field_prefix}.{key}", "issue": "unexpected_field"}
                    )

                name = item.get("name")
                if not _is_non_empty_str(name):
                    errors.append(
                        f"Field '{field_prefix}.name' must be a non-empty string"
                    )
                    validation_errors.append(
                        {"field": f"{field_prefix}.name", "issue": "empty_string"}
                    )
                else:
                    normalized = str(name).strip()
                    if (
                        self._allowed_tool_names
                        and normalized not in self._allowed_tool_names
                    ):
                        errors.append(
                            f"Field '{field_prefix}.name' must be one of the allowed tools"
                        )
                        validation_errors.append(
                            {
                                "field": f"{field_prefix}.name",
                                "issue": "invalid_tool_name",
                                "got": normalized,
                            }
                        )

                arguments = item.get("arguments")
                if arguments is not None and not isinstance(arguments, dict):
                    errors.append(
                        f"Field '{field_prefix}.arguments' must be an object when present"
                    )
                    validation_errors.append(
                        {
                            "field": f"{field_prefix}.arguments",
                            "issue": "wrong_type",
                            "expected": "dict",
                            "got": type(arguments).__name__,
                        }
                    )

                reason = item.get("reason")
                if reason is not None and not isinstance(reason, str):
                    errors.append(
                        f"Field '{field_prefix}.reason' must be a string when present"
                    )
                    validation_errors.append(
                        {
                            "field": f"{field_prefix}.reason",
                            "issue": "wrong_type",
                            "expected": "str",
                            "got": type(reason).__name__,
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
