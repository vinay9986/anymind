from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from langchain_core.messages import BaseMessage

from anymind.runtime.evidence import get_current_ledger
from anymind.runtime.onnx_embedder import OnnxEmbedderConfig, OnnxSentenceEmbedder
from anymind.runtime.usage import UsageTotals, extract_usage_from_messages


def message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def extract_user_input(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "user":
                return str(item[1])
            if hasattr(item, "type") and getattr(item, "type") in {"human", "user"}:
                return str(getattr(item, "content", ""))
            if isinstance(item, dict) and item.get("role") == "user":
                return str(item.get("content", ""))
    return str(
        payload.get("input") or payload.get("query") or payload.get("message") or ""
    )


@dataclass
class UsageCounter:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def add_usage(self, usage: Optional[dict[str, int]]) -> None:
        if not usage:
            return
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        self.add(input_tokens, output_tokens)

    def add_usage_list(self, usage_list: Iterable[dict[str, int]]) -> None:
        for usage in usage_list:
            self.add_usage(usage)

    def add_from_messages(self, messages: Iterable[BaseMessage]) -> None:
        totals: UsageTotals = extract_usage_from_messages(messages)
        self.add(totals.input_tokens, totals.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def budget_exhausted(counter: UsageCounter, budget_tokens: Optional[int]) -> bool:
    if budget_tokens is None:
        return False
    return counter.total_tokens >= int(budget_tokens)


def tool_feedback_from_ledger(max_chars: int = 8000) -> str:
    ledger = get_current_ledger()
    if ledger is None:
        return "Tools are available. No external tool results yet."
    records = ledger.recent()
    if not records:
        return "Tools are available. No external tool results yet."
    max_chars = int(max_chars or 0)
    if max_chars <= 0:
        max_chars = 8000
    parts: list[str] = []
    total = 0
    for record in records:
        content = str(record.content or "").strip()
        if not content:
            continue
        block = f"[{record.id}] {record.tool}: {content}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(block[:remaining].rstrip())
            break
        parts.append(block)
        total += len(block)
    summary = "\n".join(parts).strip()
    return summary or "Tools are available. No external tool results yet."


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def iteration_temperature(iteration: int) -> float:
    return min(0.2 + 0.05 * (iteration - 1), 1.0)


@lru_cache(maxsize=1)
def try_load_embedder() -> Optional[OnnxSentenceEmbedder]:
    model_path = Path(os.getenv("ONNX_MODEL_PATH") or "onnx_assets_out/model.onnx")
    tokenizer_path = Path(
        os.getenv("ONNX_TOKENIZER_PATH") or "onnx_assets_out/tokenizer.json"
    )
    max_length = int(os.getenv("ONNX_MAX_LENGTH") or 256)
    if not model_path.exists() or not tokenizer_path.exists():
        return None
    config = OnnxEmbedderConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
    )
    return OnnxSentenceEmbedder(config)


def pairwise_similarities(embeddings: np.ndarray) -> list[float]:
    n = int(embeddings.shape[0])
    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(np.dot(embeddings[i], embeddings[j])))
    return sims


def select_semantic_representative(answers: list[str], embeddings: np.ndarray) -> str:
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, 0.0)
    scores = sims.mean(axis=1)
    idx = int(np.argmax(scores))
    return answers[idx]
