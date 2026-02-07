from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import threading
from typing import Iterable

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


@dataclass(frozen=True)
class OnnxEmbedderConfig:
    model_path: Path
    tokenizer_path: Path
    max_length: int = 256


class OnnxSentenceEmbedder:
    """Sentence embedding via ONNXRuntime + tokenizers."""

    def __init__(self, config: OnnxEmbedderConfig) -> None:
        self._config = config
        self._tokenizer = _load_tokenizer(config.tokenizer_path, config.max_length)
        self._force_single = False
        self._lock = threading.Lock()
        options = ort.SessionOptions()
        # Avoid buffer reuse issues with dynamic shapes in some ORT builds.
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(config.model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        batch = [t if isinstance(t, str) else str(t) for t in texts]
        if not batch:
            return np.zeros((0, 0), dtype=np.float32)

        if self._force_single and len(batch) > 1:
            return self._embed_singletons(batch)

        try:
            return self._embed_batch(batch)
        except Exception:
            if len(batch) == 1:
                raise
            self._force_single = True
            return self._embed_singletons(batch)

    def _embed_singletons(self, batch: list[str]) -> np.ndarray:
        vectors = [self._embed_batch([text]) for text in batch]
        return np.vstack(vectors)

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(batch)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
            feed["token_type_ids"] = token_type_ids

        with self._lock:
            (last_hidden,) = self._session.run(None, feed)

        mask = attention_mask.astype(np.float32)[..., None]
        summed = (last_hidden * mask).sum(axis=1)
        denom = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / denom

        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norm, 1e-9, None)
        return pooled.astype(np.float32)


@lru_cache(maxsize=1)
def _load_tokenizer(path: Path, max_length: int) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length=max_length)
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    tokenizer.enable_padding(direction="right", pad_id=pad_id, pad_token="[PAD]")
    return tokenizer
