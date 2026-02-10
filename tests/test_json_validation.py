import json

from anymind.runtime.json_validation import JSONStructureValidator


def test_validation_missing_and_wrong_type() -> None:
    schema = {
        "name": {"type": str, "required": True},
        "count": {"type": int, "required": True},
    }
    validator = JSONStructureValidator(schema, validator_name="test")
    result = validator.validate({"name": 123}, original_text="{}")
    assert not result.validation_passed
    assert result.validation_errors
    issues = {err.get("issue") for err in result.validation_errors or []}
    assert "missing_required" in issues
    assert "wrong_type" in issues


def test_validation_non_dict_input() -> None:
    schema = {"name": {"type": str}}
    validator = JSONStructureValidator(schema, validator_name="test")
    result = validator.validate(["not-a-dict"], original_text="[]")
    assert not result.validation_passed
    assert result.validation_errors


def test_validation_success_and_example() -> None:
    schema = {
        "name": {"type": str},
        "enabled": {"type": bool},
        "items": {"type": list},
        "meta": {"type": dict},
    }
    validator = JSONStructureValidator(schema, validator_name="test")
    result = validator.validate(
        {"name": "ok", "enabled": True, "items": [], "meta": {}},
        original_text="{}",
    )
    assert result.validation_passed

    example = validator.example_format()
    payload = json.loads(example)
    assert set(payload.keys()) == {"name", "enabled", "items", "meta"}


def test_validation_tuple_type() -> None:
    schema = {"value": {"type": (str, int)}}
    validator = JSONStructureValidator(schema, validator_name="test")
    result = validator.validate({"value": []}, original_text="{}")
    assert not result.validation_passed
