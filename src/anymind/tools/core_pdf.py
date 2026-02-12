from __future__ import annotations

import base64
import io
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import boto3
import structlog
from pypdf import PdfReader

from anymind.tools.core_http import (
    _DEFAULT_USER_AGENT,
    _SafeRedirectHandler,
    _read_limited_bytes,
    _validate_public_https_url,
)
from anymind.tools.core_text import _slice_around_query

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "into",
    "over",
    "under",
    "between",
    "about",
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
    "how",
    "many",
    "most",
    "recent",
    "latest",
    "find",
    "identify",
    "determine",
    "extract",
    "list",
    "show",
    "give",
    "provide",
    "summarize",
}


def _normalize_queries(query: Any, queries: Any) -> list[str]:
    normalized: list[str] = []
    if isinstance(query, str) and query.strip():
        normalized.append(query.strip())

    if isinstance(queries, Sequence) and not isinstance(
        queries, (str, bytes, bytearray)
    ):
        for item in queries:
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip())

    return list(dict.fromkeys(normalized))


def _find_matches(
    *,
    text: str,
    query: str,
    context_chars: int,
    max_hits: int,
) -> list[str]:
    if not text or not query or max_hits <= 0:
        return []
    lower_text = text.lower()
    needle = query.lower()

    hits: list[str] = []
    start_idx = 0
    while len(hits) < max_hits:
        idx = lower_text.find(needle, start_idx)
        if idx < 0:
            break

        half = max(20, context_chars // 2)
        left = max(0, idx - half)
        right = min(len(text), idx + len(needle) + half)
        snippet = " ".join(text[left:right].split())
        if snippet:
            hits.append(snippet)
        start_idx = idx + max(1, len(needle))

    return hits


def _is_question_like(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if "?" in lowered:
        return True
    starters = (
        "how ",
        "what ",
        "which ",
        "who ",
        "where ",
        "when ",
        "why ",
        "find ",
        "identify ",
        "determine ",
        "extract ",
        "list ",
        "show ",
        "give ",
        "provide ",
        "summarize ",
    )
    return lowered.startswith(starters)


def _query_tokens(query_text: str) -> set[str]:
    tokens = {t for t in _TOKEN_RE.findall((query_text or "").lower())}
    return {t for t in tokens if t not in _STOPWORDS}


def _best_snippet(*, page_text: str, query_text: str, max_chars: int) -> str:
    cleaned_lines = [
        line.strip() for line in (page_text or "").splitlines() if line.strip()
    ]
    if not cleaned_lines:
        return ""

    tokens = _query_tokens(query_text)

    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(cleaned_lines):
        lower = line.lower()
        token_hits = sum(1 for tok in tokens if tok in lower)
        digit_bonus = 2 if any(ch.isdigit() for ch in line) else 0
        score = token_hits + digit_bonus
        scored.append((score, idx, line))

    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    chosen = [line for score, _, line in scored[:6] if score > 0 and line.strip()]
    if not chosen:
        chosen = cleaned_lines[:4]

    snippet = " ".join(chosen)
    snippet = " ".join(snippet.split())
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max(0, max_chars - 3)].rstrip() + "..."


try:  # Optional; used for semantic PDF search
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # Optional; used for semantic PDF search
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore[assignment]

try:  # Optional; used for semantic PDF search
    from tokenizers import Tokenizer  # type: ignore
except Exception:  # pragma: no cover
    Tokenizer = None  # type: ignore[assignment]


class _OnnxSentenceEmbedder:
    """Sentence embedding via ONNXRuntime + tokenizers.

    Export is expected to produce a transformer that outputs `last_hidden_state`.
    This embedder applies mean pooling over attention_mask and L2 normalizes.
    """

    def __init__(
        self, *, model_path: Path, tokenizer_path: Path, max_length: int = 256
    ) -> None:
        if np is None or ort is None or Tokenizer is None:  # pragma: no cover
            raise RuntimeError("onnx semantic search dependencies are not available")
        self._tokenizer = _load_tokenizer(tokenizer_path, max_length)
        self._lock = threading.Lock()
        options = ort.SessionOptions()  # type: ignore[union-attr]
        # Dynamic batch sizes can trigger buffer re-use warnings in some ORT builds.
        # Disable mem pattern/reuse to avoid shape mismatch warnings.
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(  # type: ignore[union-attr]
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._force_single = False
        self._fixed_batch = False
        outputs = self._session.get_outputs()
        if outputs:
            shape = outputs[0].shape
            if shape and isinstance(shape[0], int) and shape[0] == 1:
                self._fixed_batch = True
                self._force_single = True

    def embed(self, texts: list[str]) -> "np.ndarray":
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy unavailable")
        batch = [t if isinstance(t, str) else str(t) for t in texts]
        if not batch:
            return np.zeros((0, 0), dtype=np.float32)

        # NOTE: This is used in multiple tools (PDF + internet search). Large batches can cause
        # Lambda OOM because ONNX returns `last_hidden_state` with shape:
        #   (batch, max_length, hidden_size)
        # So we embed in small batches to cap peak memory.
        try:
            batch_size = int(os.environ.get("ONNX_EMBED_BATCH_SIZE", "32") or "32")
        except Exception:  # pragma: no cover - defensive
            batch_size = 32
        batch_size = max(1, min(batch_size, 256))

        def _embed_batch(texts_batch: list[str]) -> "np.ndarray":
            pooled_batches: list["np.ndarray"] = []
            for start in range(0, len(texts_batch), batch_size):
                slice_texts = texts_batch[start : start + batch_size]
                encodings = self._tokenizer.encode_batch(slice_texts)

                input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
                attention_mask = np.array(
                    [e.attention_mask for e in encodings], dtype=np.int64
                )

                feed: dict[str, "np.ndarray"] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if "token_type_ids" in self._input_names:
                    token_type_ids = np.array(
                        [e.type_ids for e in encodings], dtype=np.int64
                    )
                    feed["token_type_ids"] = token_type_ids

                with self._lock:
                    (last_hidden,) = self._session.run(None, feed)

                mask = attention_mask.astype(np.float32)[..., None]
                summed = (last_hidden * mask).sum(axis=1)
                denom = np.clip(mask.sum(axis=1), 1e-9, None)
                pooled = summed / denom

                norm = np.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = pooled / np.clip(norm, 1e-9, None)
                pooled_batches.append(pooled.astype(np.float32))

            if not pooled_batches:
                return np.zeros((0, 0), dtype=np.float32)
            if len(pooled_batches) == 1:
                return pooled_batches[0]
            return np.vstack(pooled_batches)

        if self._force_single or len(batch) == 1:
            if len(batch) == 1:
                return _embed_batch(batch)
            return np.vstack([_embed_batch([text]) for text in batch])

        try:
            return _embed_batch(batch)
        except Exception:
            self._force_single = True
            return np.vstack([_embed_batch([text]) for text in batch])


@lru_cache(maxsize=1)
def _load_tokenizer(path: Path, max_length: int) -> Any:
    if Tokenizer is None:  # pragma: no cover
        raise RuntimeError("tokenizers is not available")
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=max_length)
    pad_id = tok.token_to_id("[PAD]") or 0
    tok.enable_padding(
        direction="right",
        pad_id=pad_id,
        pad_token="[PAD]",
    )
    return tok


@lru_cache(maxsize=1)
def _try_load_pdf_embedder() -> Any:
    """Return an ONNX embedder if deps + assets are available, else None."""
    if np is None or ort is None or Tokenizer is None:
        return None

    def _env(name: str) -> str | None:
        value = os.getenv(name)
        if value is None:
            return None
        value = value.strip()
        return value or None

    model_path = Path(
        _env("PDF_ONNX_MODEL_PATH")
        or _env("ONNX_MODEL_PATH")
        or "/opt/onnx_assets_out/model.onnx"
    )
    tokenizer_path = Path(
        _env("PDF_ONNX_TOKENIZER_PATH")
        or _env("ONNX_TOKENIZER_PATH")
        or "/opt/onnx_assets_out/tokenizer.json"
    )
    max_length = int(_env("PDF_ONNX_MAX_LENGTH") or _env("ONNX_MAX_LENGTH") or "256")
    if not model_path.exists() or not tokenizer_path.exists():
        return None
    return _OnnxSentenceEmbedder(
        model_path=model_path, tokenizer_path=tokenizer_path, max_length=max_length
    )


def _semantic_search_pages(
    *,
    page_numbers: list[int],
    page_texts: list[str],
    query_text: str,
    max_matches: int,
    context_chars: int,
    min_similarity: float,
) -> tuple[list[dict[str, Any]], str]:
    embedder = _try_load_pdf_embedder()
    if np is None or embedder is None:
        return (
            [],
            "Semantic search unavailable (missing ONNX assets/dependencies); use keyword search instead.",
        )

    query_text = (query_text or "").strip()
    if not query_text:
        return ([], "Semantic search requires a non-empty query.")

    cleaned_pages: list[tuple[int, str]] = []
    for page, text in zip(page_numbers, page_texts):
        cleaned = " ".join((text or "").split())
        if cleaned:
            cleaned_pages.append((page, cleaned))

    if not cleaned_pages:
        return ([], "No extractable text detected in scanned pages.")

    try:
        vectors = embedder.embed([query_text] + [t for _, t in cleaned_pages])
        query_vec = vectors[0]
        page_vecs = vectors[1:]
        scores = page_vecs @ query_vec
    except Exception as exc:  # pragma: no cover - defensive
        return ([], f"Semantic embedding failed: {exc}")

    ranked = sorted(
        range(len(cleaned_pages)), key=lambda i: float(scores[i]), reverse=True
    )

    matches: list[dict[str, Any]] = []
    for idx in ranked:
        if len(matches) >= max_matches:
            break
        score = float(scores[idx])
        if score < float(min_similarity):
            continue
        page_num, page_text = cleaned_pages[idx]
        snippet = _slice_around_query(
            text=page_text, query_text=query_text, max_chars=context_chars
        )
        if not snippet:
            snippet = _best_snippet(
                page_text=page_text, query_text=query_text, max_chars=context_chars
            )
        matches.append(
            {
                "page": page_num,
                "query": query_text,
                "snippet": snippet,
                "score": round(score, 4),
            }
        )

    return (matches, "")


@dataclass(frozen=True, slots=True)
class PdfExtractRequest:
    source_type: str
    s3_key: str | None
    url: str | None
    pdf_base64: str | None
    queries: list[str]
    search_mode: str
    max_matches: int
    context_chars: int
    start_page: int
    max_pages: int
    min_similarity: float
    max_bytes: int
    timeout_seconds: float


def _parse_pdf_request(event: Mapping[str, Any]) -> PdfExtractRequest:
    url = str(event.get("url", "") or "").strip()
    s3_key = str(event.get("s3_key", "") or "").strip()
    pdf_base64 = str(event.get("pdf_base64", "") or "").strip()

    provided = [
        ("url", url),
        ("s3_key", s3_key),
        ("pdf_base64", pdf_base64),
    ]
    active = [name for name, value in provided if value]
    if len(active) != 1:
        raise ValueError("Provide exactly one of: url, s3_key, pdf_base64")

    source_type = active[0].replace("pdf_base64", "base64")
    queries = _normalize_queries(event.get("query"), event.get("queries"))

    search_mode = str(event.get("search_mode", "auto") or "auto").strip().lower()
    if search_mode not in {"auto", "keyword", "semantic"}:
        search_mode = "auto"

    max_matches = int(event.get("max_matches", 10) or 10)
    max_matches = max(1, min(max_matches, 25))

    context_chars = int(event.get("context_chars", 1000) or 1000)
    context_chars = max(200, min(context_chars, 4000))

    start_page = int(event.get("start_page", 1) or 1)
    start_page = max(1, start_page)

    max_pages = int(event.get("max_pages", 200) or 200)
    max_pages = max(1, min(max_pages, 1000))

    min_similarity = float(event.get("min_similarity", 0.0) or 0.0)
    min_similarity = max(0.0, min(min_similarity, 1.0))

    max_bytes = int(event.get("max_bytes", 100 * 1024 * 1024) or 100 * 1024 * 1024)
    max_bytes = max(1, min(max_bytes, 100 * 1024 * 1024))

    timeout_seconds = float(event.get("timeout_seconds", 30) or 30)
    timeout_seconds = max(1.0, min(timeout_seconds, 120.0))

    return PdfExtractRequest(
        source_type=source_type,
        s3_key=s3_key or None,
        url=url or None,
        pdf_base64=pdf_base64 or None,
        queries=queries,
        search_mode=search_mode,
        max_matches=max_matches,
        context_chars=context_chars,
        start_page=start_page,
        max_pages=max_pages,
        min_similarity=min_similarity,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )


def _get_s3_bucket_and_base_prefix() -> tuple[str, str]:
    bucket = str(os.environ.get("AGENT_DATA_BUCKET", "") or "").strip()
    base_prefix = str(os.environ.get("AGENT_DATA_BASE_PREFIX", "") or "").strip()
    if base_prefix and not base_prefix.endswith("/"):
        base_prefix += "/"
    if not bucket:
        raise RuntimeError("Missing required environment variable: AGENT_DATA_BUCKET")
    return bucket, base_prefix


def _looks_like_pdf(pdf_bytes: bytes, content_type: str) -> bool:
    if not pdf_bytes:
        return False
    head = pdf_bytes[:1024]
    if b"%PDF" in head:
        return True
    return "pdf" in (content_type or "").lower()


def _download_pdf_from_url(
    *, url: str, timeout_seconds: float, max_bytes: int
) -> tuple[bytes, str, str]:
    validated = _validate_public_https_url(url)
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    req = urllib.request.Request(
        validated,
        method="GET",
        headers={
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        },
    )

    logger.info(
        "pdf_download_start",
        url=validated,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
    )
    try:
        with opener.open(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None) or resp.getcode() or 0
            if int(status) >= 400:
                raise ValueError(f"HTTP {int(status)} while fetching PDF")
            final_url = resp.geturl() or validated
            content_type = str(resp.headers.get("Content-Type", "") or "")
            pdf_bytes = _read_limited_bytes(resp, max_bytes=max_bytes)
    except urllib.error.HTTPError as exc:
        logger.info(
            "pdf_download_error",
            url=validated,
            error=f"HTTP {exc.code} while fetching PDF",
        )
        raise ValueError(f"HTTP {exc.code} while fetching PDF") from exc
    except urllib.error.URLError as exc:
        logger.info(
            "pdf_download_error",
            url=validated,
            error=f"Failed to download PDF: {exc.reason}",
        )
        raise ValueError(f"Failed to download PDF: {exc.reason}") from exc
    except Exception as exc:
        logger.info(
            "pdf_download_error",
            url=validated,
            error=f"Failed to download PDF: {exc}",
        )
        raise

    if not pdf_bytes:
        raise ValueError("Downloaded document was empty")

    if not _looks_like_pdf(pdf_bytes, content_type):
        raise ValueError(
            f"URL did not appear to return a PDF (content-type={content_type or 'unknown'})"
        )

    logger.info(
        "pdf_download_complete",
        url=final_url,
        content_type=content_type,
        bytes=len(pdf_bytes),
    )

    return pdf_bytes, content_type, final_url


def _decode_pdf_base64(*, pdf_base64: str, max_bytes: int) -> bytes:
    cleaned = "".join(pdf_base64.split())
    try:
        pdf_bytes = base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc
    if not pdf_bytes:
        raise ValueError("Decoded PDF payload was empty")
    if len(pdf_bytes) > max_bytes:
        raise ValueError("PDF exceeds max_bytes limit")
    if not _looks_like_pdf(pdf_bytes, ""):
        raise ValueError("Decoded payload does not appear to be a valid PDF")
    return pdf_bytes


def _handle_pdf_extract_text(event: Mapping[str, Any]) -> dict[str, Any]:
    req = _parse_pdf_request(event)
    logger.info(
        "pdf_extract_request",
        source_type=req.source_type,
        url=req.url,
        s3_key=req.s3_key,
        pdf_base64_len=len(req.pdf_base64 or ""),
        queries=req.queries,
        search_mode=req.search_mode,
        max_matches=req.max_matches,
        context_chars=req.context_chars,
        start_page=req.start_page,
        max_pages=req.max_pages,
        min_similarity=req.min_similarity,
        max_bytes=req.max_bytes,
        timeout_seconds=req.timeout_seconds,
    )
    source: dict[str, Any] = {"source_type": req.source_type}

    if req.source_type == "s3":
        bucket, base_prefix = _get_s3_bucket_and_base_prefix()
        if not req.s3_key:
            raise ValueError("Missing required parameter: s3_key")
        if base_prefix and not req.s3_key.startswith(base_prefix):
            raise ValueError(
                f"s3_key must start with base prefix '{base_prefix}' for this environment"
            )

        s3 = boto3.client("s3")
        head = s3.head_object(Bucket=bucket, Key=req.s3_key)
        size = int(head.get("ContentLength") or 0)
        if size <= 0:
            raise ValueError("S3 object appears to be empty")
        if size > req.max_bytes:
            raise ValueError(f"PDF exceeds max_bytes limit ({size} > {req.max_bytes})")

        obj = s3.get_object(Bucket=bucket, Key=req.s3_key)
        pdf_bytes = obj["Body"].read()
        content_type = str(head.get("ContentType") or "")

        if not _looks_like_pdf(pdf_bytes, content_type):
            raise ValueError("Object does not appear to be a valid PDF")

        source.update(
            {
                "bucket": bucket,
                "key": req.s3_key,
                "content_type": content_type,
                "bytes": len(pdf_bytes),
            }
        )
    elif req.source_type == "url":
        if not req.url:
            raise ValueError("Missing required parameter: url")
        pdf_bytes, content_type, final_url = _download_pdf_from_url(
            url=req.url,
            timeout_seconds=req.timeout_seconds,
            max_bytes=req.max_bytes,
        )
        source.update(
            {
                "url": final_url,
                "content_type": content_type,
                "bytes": len(pdf_bytes),
            }
        )
    elif req.source_type == "base64":
        if not req.pdf_base64:
            raise ValueError("Missing required parameter: pdf_base64")
        pdf_bytes = _decode_pdf_base64(
            pdf_base64=req.pdf_base64, max_bytes=req.max_bytes
        )
        source.update(
            {
                "bytes": len(pdf_bytes),
            }
        )
    else:
        raise ValueError("Invalid PDF source_type")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    start_index = max(0, req.start_page - 1)
    end_index = min(total_pages, start_index + req.max_pages)
    scanned_pages = max(0, end_index - start_index)

    page_numbers: list[int] = []
    page_texts: list[str] = []
    any_text = False

    for idx in range(start_index, end_index):
        try:
            page_text = reader.pages[idx].extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            any_text = True
        page_numbers.append(idx + 1)
        page_texts.append(page_text)

    matches: list[dict[str, Any]] = []
    effective_mode = req.search_mode
    semantic_note = ""
    if effective_mode == "auto" and req.queries:
        candidate = req.queries[0] if len(req.queries) == 1 else " ".join(req.queries)
        if _is_question_like(candidate) or len(candidate.split()) >= 6:
            effective_mode = "semantic"
        else:
            effective_mode = "keyword"

    if effective_mode == "semantic" and req.queries:
        query_text = req.queries[0] if len(req.queries) == 1 else " ".join(req.queries)
        matches, semantic_note = _semantic_search_pages(
            page_numbers=page_numbers,
            page_texts=page_texts,
            query_text=query_text,
            max_matches=req.max_matches,
            context_chars=req.context_chars,
            min_similarity=req.min_similarity,
        )
        if semantic_note:
            effective_mode = "keyword"

    if effective_mode == "keyword" and req.queries and not matches:
        for page_num, page_text in zip(page_numbers, page_texts):
            remaining = max(0, req.max_matches - len(matches))
            if remaining <= 0:
                break

            per_query = max(1, remaining // max(1, len(req.queries)))
            for query in req.queries:
                if len(matches) >= req.max_matches:
                    break
                hits = _find_matches(
                    text=page_text,
                    query=query,
                    context_chars=req.context_chars,
                    max_hits=per_query,
                )
                for snippet in hits:
                    if len(matches) >= req.max_matches:
                        break
                    matches.append(
                        {
                            "page": page_num,
                            "query": query,
                            "snippet": snippet,
                        }
                    )

    notes = ""
    if not any_text:
        notes = "No extractable text detected in scanned pages; OCR may be required."
    elif semantic_note:
        notes = semantic_note
    elif req.queries and not matches and scanned_pages < total_pages:
        notes = (
            "No matches found in the scanned page range "
            f"({req.start_page}-{req.start_page + max(0, scanned_pages - 1)} of {total_pages}). "
            "Increase max_pages or adjust start_page to scan further."
        )
    elif req.queries and not matches:
        notes = (
            "No matches found for the provided query/queries in the scanned pages. "
            "Try different query terms."
        )
    elif not req.queries:
        notes = "Provide query/queries to return matching snippets with page numbers."

    resp: dict[str, Any] = {
        "matches": matches,
        "page_count": total_pages,
        "scanned_pages": scanned_pages,
        "scanned_page_range": {
            "start_page": req.start_page,
            "end_page": req.start_page + max(0, scanned_pages - 1),
        },
        "searched_queries": req.queries,
        "search_mode": req.search_mode,
        "search_mode_used": effective_mode,
        "min_similarity": req.min_similarity,
        "notes": notes,
        "source": source,
    }

    if req.source_type == "s3":
        resp["bucket"] = source.get("bucket")
        resp["key"] = source.get("key")
    if req.source_type == "url":
        resp["url"] = source.get("url")

    logger.info(
        "pdf_extract_response",
        response=resp,
    )
    return resp


def pdf_extract_text(
    *,
    url: str | None = None,
    s3_key: str | None = None,
    pdf_base64: str | None = None,
    query: str | None = None,
    queries: Sequence[str] | None = None,
    search_mode: str = "auto",
    max_matches: int = 10,
    context_chars: int = 1000,
    start_page: int = 1,
    max_pages: int = 200,
    min_similarity: float = 0.0,
    max_bytes: int = 100 * 1024 * 1024,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Extract text snippets from PDFs (url, s3, or base64) with keyword/semantic search.

    WHEN TO USE:
    - User provides a PDF URL and asks questions about its content
    - You need to find specific information within a PDF document
    - User asks to "search", "find", "locate", or "extract" content from a PDF
    - You need to verify claims by checking PDF source documents
    - User uploads or references a PDF and wants analysis

    WHY TO USE: Extracts and searches PDF content with page numbers, supporting
    both exact text matching and semantic/meaning-based search.

    WHEN NOT TO USE:
    - User asks about PDF format or structure (not content)
    - File is not a PDF (use appropriate tool for other formats)
    - PDF is behind authentication/login (tool cannot access)
    - User wants to create or modify a PDF (read-only)

    SEARCH MODE DECISION:
    - Use "keyword" when looking for exact terms, names, numbers, or phrases
    - Use "semantic" when looking for concepts or when exact wording is unknown
    - Use "auto" when unsure which is better

    EXAMPLES:
    - OK: "Find all mentions of revenue in this financial report PDF" -> keyword mode
    - OK: "What does this research paper say about climate change?" -> semantic mode
    - OK: "Extract the executive summary from page 1-5" -> start_page=1, max_pages=5
    - NO: "How do I create a PDF?" -> do not use (not for PDF creation)
    """
    payload: dict[str, Any] = {
        "search_mode": search_mode,
        "max_matches": max_matches,
        "context_chars": context_chars,
        "start_page": start_page,
        "max_pages": max_pages,
        "min_similarity": min_similarity,
        "max_bytes": max_bytes,
        "timeout_seconds": timeout_seconds,
    }
    if url:
        payload["url"] = url
    if s3_key:
        payload["s3_key"] = s3_key
    if pdf_base64:
        payload["pdf_base64"] = pdf_base64
    if query:
        payload["query"] = query
    if queries:
        payload["queries"] = list(queries)
    return _handle_pdf_extract_text(payload)
