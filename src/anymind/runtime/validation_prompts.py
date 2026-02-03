"""Prompt templates for focused JSON parsing + validation retries."""

# JSON Parsing Fix Prompt Template
JSON_PARSING_FIX_PROMPT = """Your previous JSON response has syntax errors. Please fix ONLY the JSON syntax issues while preserving all content.

**Original Task Context:**
{original_task_context}

**Raw LLM Output:**
{raw_llm_output}

**JSON to Fix:**
{json_to_fix}

**Specific JSON Error:**
{parsing_error_details}

**Error Location:**
Position {error_position}: "{error_context}"

**KEEP UNCHANGED (These parts are correctly formatted):**
{keep_unchanged_sections}

**CRITICAL INSTRUCTIONS:**
- Fix ONLY the JSON syntax errors (missing commas, quotes, braces, brackets, etc.)
- Do NOT change the content, values, or meaning of any data
- Keep all correctly formatted sections exactly as they appear in "KEEP UNCHANGED"
- Focus specifically on the error at position {error_position}
- Provide ONLY the corrected JSON object - no explanations, no markdown, no additional text
- Ensure the output is valid JSON that can be parsed by json.loads()

**Expected Output:**
A single, valid JSON object or array with syntax errors fixed."""


# Validation Fix Prompt Template
VALIDATION_FIX_PROMPT = """Your JSON response has validation errors. Please fix ONLY the validation issues while preserving all correct data.

**Original Task Context:**
{original_task_context}

**Raw LLM Output:**
{raw_llm_output}

**Parsed JSON Object:**
{serialized_json_object}

**Validation Errors:**
{validation_error_details}

**KEEP UNCHANGED (These fields are correctly validated):**
{keep_unchanged_fields}

**CRITICAL INSTRUCTIONS:**
- Fix ONLY the validation errors listed above
- Keep all correctly validated fields from "KEEP UNCHANGED" exactly as they are
- Add missing required fields with appropriate values
- Fix incorrect field types while preserving the intended meaning
- Do NOT change or remove any correctly validated data
- Provide ONLY the corrected JSON object - no explanations, no markdown, no additional text
- Ensure the output maintains the same overall structure and intent

**Expected Output:**
A single, valid JSON object that passes all validation requirements."""


def create_json_parsing_fix_prompt(
    original_task_context: str,
    raw_llm_output: str,
    json_to_fix: str,
    parsing_error_details: str,
    error_position: int,
    error_context: str,
    keep_unchanged_sections: str,
) -> str:
    return JSON_PARSING_FIX_PROMPT.format(
        original_task_context=original_task_context,
        raw_llm_output=raw_llm_output,
        json_to_fix=json_to_fix,
        parsing_error_details=parsing_error_details,
        error_position=error_position,
        error_context=error_context,
        keep_unchanged_sections=keep_unchanged_sections,
    )


def create_validation_fix_prompt(
    original_task_context: str,
    raw_llm_output: str,
    serialized_json_object: str,
    validation_error_details: str,
    keep_unchanged_fields: str,
) -> str:
    return VALIDATION_FIX_PROMPT.format(
        original_task_context=original_task_context,
        raw_llm_output=raw_llm_output,
        serialized_json_object=serialized_json_object,
        validation_error_details=validation_error_details,
        keep_unchanged_fields=keep_unchanged_fields,
    )


def extract_keep_unchanged_sections(
    json_to_fix: str,
    error_position: int,
    context_size: int = 50,
) -> str:
    if not json_to_fix or error_position < 0:
        return "No correctly formatted sections identified."

    before_error = json_to_fix[: max(0, error_position - context_size)]
    after_error = json_to_fix[min(len(json_to_fix), error_position + context_size) :]

    sections: list[str] = []

    if before_error.strip() and any(char in before_error for char in ["{", "[", '"']):
        sections.append(f"Section before error: {before_error.strip()}")

    if after_error.strip() and any(char in after_error for char in ["}", "]", '"']):
        sections.append(f"Section after error: {after_error.strip()}")

    if not sections:
        return (
            "Focus only on fixing the syntax error - preserve all existing content "
            "structure."
        )

    return "\n".join(sections)


def extract_keep_unchanged_fields(
    json_obj: dict | list | str | int | float | bool | None,
    validation_errors: list,
) -> str:
    if not isinstance(json_obj, dict):
        return "Preserve the overall structure type (object/array/value)."

    error_fields = set()
    for error in validation_errors:
        if isinstance(error, dict) and "field" in error:
            error_fields.add(error["field"])

    correct_fields = {k: v for k, v in json_obj.items() if k not in error_fields}

    if not correct_fields:
        return "No fields are correctly validated - fix all validation errors."

    field_examples = []
    for field_name, field_value in correct_fields.items():
        display_value = str(field_value)
        if len(display_value) > 50:
            display_value = display_value[:47] + "..."
        field_examples.append(f'"{field_name}": {display_value}')

    return "Keep these correctly validated fields exactly as they are:\n" + "\n".join(
        field_examples
    )
