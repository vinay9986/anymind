from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationError:
    field: str | None
    issue: str
    expected: str | None = None
    got: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    validation_passed: bool
    validated_output: Any
    error_message: str | None = None
    validation_errors: list[dict[str, Any]] | None = None


class JSONStructureValidator:
    """Generic JSON structure validator with field checking."""

    def __init__(self, schema: dict[str, Any], *, validator_name: str) -> None:
        self.schema = schema
        self.validator_name = validator_name

    @staticmethod
    def _expected_type_name(expected_type: Any) -> str:
        if isinstance(expected_type, tuple):
            return "|".join(
                JSONStructureValidator._expected_type_name(t) for t in expected_type
            )
        if isinstance(expected_type, type):
            return expected_type.__name__
        return str(expected_type)

    def validate(self, json_obj: Any, *, original_text: str) -> ValidationResult:
        if not isinstance(json_obj, dict):
            return ValidationResult(
                validation_passed=False,
                validated_output=json_obj,
                error_message=(
                    f"Expected JSON object (dictionary), got {type(json_obj).__name__}"
                ),
                validation_errors=[
                    {
                        "issue": "wrong_type",
                        "expected": "dict",
                        "got": type(json_obj).__name__,
                    }
                ],
            )

        errors: list[str] = []
        validation_errors: list[dict[str, Any]] = []

        for field_name, field_spec in self.schema.items():
            required = field_spec.get("required", True)
            expected_type = field_spec.get("type")
            description = field_spec.get("description", field_name)

            if required and field_name not in json_obj:
                errors.append(f"Missing required field: '{field_name}' ({description})")
                validation_errors.append(
                    {
                        "field": field_name,
                        "issue": "missing_required",
                        "description": description,
                    }
                )
                continue

            if field_name in json_obj and expected_type:
                actual_value = json_obj[field_name]
                if not isinstance(actual_value, expected_type):
                    expected_name = self._expected_type_name(expected_type)
                    errors.append(
                        f"Field '{field_name}' must be {expected_name}, got "
                        f"{type(actual_value).__name__}"
                    )
                    validation_errors.append(
                        {
                            "field": field_name,
                            "issue": "wrong_type",
                            "expected": expected_name,
                            "got": type(actual_value).__name__,
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

    def example_format(self) -> str:
        example: dict[str, Any] = {}
        for field_name, field_spec in self.schema.items():
            field_type = field_spec.get("type", str)
            if isinstance(field_type, tuple):
                field_type = field_type[0] if field_type else str
            if field_type == str:
                example[field_name] = "string_value"
            elif field_type == int:
                example[field_name] = 0
            elif field_type == bool:
                example[field_name] = True
            elif field_type == list:
                example[field_name] = []
            elif field_type == dict:
                example[field_name] = {}
            else:
                example[field_name] = "value"
        return json.dumps(example, indent=2)
