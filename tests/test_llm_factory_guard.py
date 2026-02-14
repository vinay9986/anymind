from __future__ import annotations

from pathlib import Path


def _iter_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if p.is_file()]


def test_no_direct_llm_client_construction_outside_factory() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src" / "anymind"
    factory_path = src_root / "runtime" / "llm_factory.py"

    banned_tokens = {
        "init_chat_model",
        "ChatOpenAI",
        "ChatBedrock",
        "Anthropic",
        "OpenAI",
    }

    violations: list[tuple[str, str]] = []

    for path in _iter_py_files(src_root):
        if path == factory_path:
            continue
        content = path.read_text(encoding="utf-8")
        for token in banned_tokens:
            if token in content:
                violations.append((str(path), token))

    assert not violations, (
        "Direct LLM client construction is only allowed in llm_factory.py. "
        f"Found: {violations}"
    )
