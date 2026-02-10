from __future__ import annotations

import asyncio
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


def extract_conversation_messages(payload: dict[str, Any]) -> list[tuple[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    parsed: list[tuple[str, str]] = []
    for item in messages:
        if isinstance(item, tuple) and len(item) == 2:
            role, content = item
            role_name = str(role or "").strip().lower()
            if not role_name:
                continue
            parsed.append((role_name, str(content)))
            continue
        if hasattr(item, "type") and hasattr(item, "content"):
            role_name = str(getattr(item, "type", "") or "").strip().lower()
            if not role_name:
                role_name = str(getattr(item, "role", "") or "").strip().lower()
            if role_name:
                parsed.append((role_name, str(getattr(item, "content", ""))))
            continue
        if isinstance(item, dict):
            role_name = str(item.get("role") or item.get("type") or "").strip().lower()
            if role_name:
                parsed.append((role_name, str(item.get("content", ""))))
    return parsed


def build_conversation_query(messages: list[tuple[str, str]]) -> str:
    if not messages:
        return ""
    last_user_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx][0] == "user":
            last_user_idx = idx
            break
    if last_user_idx is None:
        return str(messages[-1][1])
    latest = str(messages[last_user_idx][1])
    history_pairs = messages[:last_user_idx]
    if not history_pairs:
        return latest
    lines: list[str] = []
    for role, content in history_pairs:
        if role not in {"user", "assistant"}:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    history = "\n".join(lines).strip()
    if not history:
        return latest
    return f"Conversation so far:\n{history}\n\nLatest user question:\n{latest}"


async def ensure_current_time_tool(tools: Iterable[Any]) -> None:
    tool = None
    for candidate in tools or []:
        name = str(getattr(candidate, "name", "") or "")
        if name in {"current_time", "get_current_time"}:
            tool = candidate
            break
    if tool is None:
        return
    payload = {"format": "iso", "timezone": "UTC"}
    try:
        if hasattr(tool, "ainvoke"):
            await tool.ainvoke(payload)
            return
        if hasattr(tool, "invoke"):
            result = tool.invoke(payload)
        elif callable(tool):
            result = tool(payload)
        else:
            return
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        return


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
