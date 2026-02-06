from __future__ import annotations

import base64
from functools import lru_cache
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import boto3
from pypdf import PdfReader


_DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; Strands-CoreTools/0.1)"
_GOOGLE_CSE_DEFAULT_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
_GOOGLE_CSE_DEFAULT_CACHE_TTL_SECONDS = 300
_SEARCH_MODES = ("auto", "keyword", "semantic")
_SCRAPFLY_SCRAPE_API_ENDPOINT = "https://api.scrapfly.io/scrape"
_SCRAPFLY_DEFAULT_CACHE_TTL_SECONDS = 300
_SCRAPFLY_DEFAULT_COUNTRY = "us"
_SCRAPFLY_DEFAULT_FORMAT = "text:only_content,no_links,no_images"
_SCRAPFLY_MIN_CONTENT_CHARS = 200
_SCRAPFLY_MAX_CONTENT_CHARS = 250_000

_BEDROCK_KNOWLEDGE_BASE_ID_ENV = "BEDROCK_KNOWLEDGE_BASE_ID"
_BEDROCK_KNOWLEDGE_BASE_MODEL_ARN_ENV = "BEDROCK_KNOWLEDGE_BASE_MODEL_ARN"


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


_ANCHOR_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z0-9]{3,}", re.IGNORECASE)


def _find_query_anchor(*, text: str, query_text: str) -> tuple[int, int] | None:
    if not text or not query_text:
        return None

    haystack = text.lower()
    needle = " ".join((query_text or "").split()).lower()
    if needle:
        idx = haystack.find(needle)
        if idx >= 0:
            return (idx, len(needle))

    raw_tokens = [
        t.strip().lower()
        for t in _ANCHOR_TOKEN_RE.findall(query_text.lower())
        if t.strip()
    ]
    if not raw_tokens:
        return None

    seen: set[str] = set()
    tokens: list[str] = []
    for tok in raw_tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tok.isalpha() and tok in _STOPWORDS:
            continue
        tokens.append(tok)

    for tok in sorted(tokens, key=len, reverse=True):
        idx = haystack.find(tok)
        if idx >= 0:
            return (idx, len(tok))

    return None


def _slice_around_center(*, text: str, center: int, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""

    max_chars = int(max_chars or 0)
    if max_chars <= 0:
        return cleaned
    if len(cleaned) <= max_chars:
        return cleaned

    center = max(0, min(int(center or 0), len(cleaned) - 1))
    half = max_chars // 2
    start = max(0, center - half)
    end = start + max_chars
    if end > len(cleaned):
        end = len(cleaned)
        start = max(0, end - max_chars)
    return cleaned[start:end].strip()


def _slice_around_query(*, text: str, query_text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""

    max_chars_int = int(max_chars or 0)
    if max_chars_int <= 0 or len(cleaned) <= max_chars_int:
        return cleaned

    anchor = _find_query_anchor(text=cleaned, query_text=query_text)
    if anchor is not None:
        start, length = anchor
        center = start + max(0, length // 2)
    else:
        center = len(cleaned) // 2
    return _slice_around_center(text=cleaned, center=center, max_chars=max_chars_int)


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
        self._session = ort.InferenceSession(  # type: ignore[union-attr]
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._force_single = False

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
        (name, value)
        for name, value in (
            ("url", url),
            ("s3_key", s3_key),
            ("pdf_base64", pdf_base64),
        )
        if value
    ]
    if len(provided) != 1:
        raise ValueError("Provide exactly one of: url, s3_key, pdf_base64")

    source_field = provided[0][0]
    if source_field == "s3_key":
        source_type = "s3"
    elif source_field == "url":
        source_type = "url"
    else:
        source_type = "base64"

    queries = _normalize_queries(event.get("query"), event.get("queries"))

    search_mode = str(event.get("search_mode", "auto") or "auto").strip().lower()
    if search_mode not in _SEARCH_MODES:
        search_mode = "auto"

    max_matches = int(event.get("max_matches", 10) or 10)
    max_matches = max(1, min(max_matches, 50))

    context_chars = int(event.get("context_chars", 1000) or 1000)
    context_chars = max(1000, min(context_chars, 2000))

    start_page = int(event.get("start_page", 1) or 1)
    start_page = max(1, start_page)

    max_pages = int(event.get("max_pages", 200) or 200)
    max_pages = max(1, min(max_pages, 500))

    min_similarity = float(event.get("min_similarity", 0.0) or 0.0)
    min_similarity = max(-1.0, min(min_similarity, 1.0))

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


def _handle_current_time(event: Mapping[str, Any]) -> dict[str, Any]:
    fmt = str(event.get("format", "iso") or "iso").strip().lower()
    timezone_name = str(event.get("timezone", "UTC") or "UTC").strip().upper()

    now_utc = datetime.now(timezone.utc)

    if fmt == "unix":
        payload: dict[str, Any] = {
            "timestamp": int(now_utc.timestamp()),
            "format": "unix",
            "timezone": timezone_name,
        }
    else:
        payload = {
            "timestamp": now_utc.isoformat(),
            "format": "iso",
            "timezone": timezone_name,
        }

    payload["source"] = "core_tools_lambda"
    return payload


_google_cse_cached_at: float = 0.0
_google_cse_cached: dict[str, str] | None = None

_scrapfly_cached_at: float = 0.0
_scrapfly_cached: str | None = None


def _get_secret_string(secret_arn: str) -> str:
    if not secret_arn:
        raise RuntimeError("Missing secret ARN")
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get("SecretString")
    if isinstance(secret_string, str) and secret_string.strip():
        return secret_string.strip()
    secret_binary = response.get("SecretBinary")
    if secret_binary:
        try:
            decoded = base64.b64decode(secret_binary).decode("utf-8")
        except Exception as exc:
            raise RuntimeError(
                "Secret binary value was not valid base64/utf-8"
            ) from exc
        if decoded.strip():
            return decoded.strip()
    raise RuntimeError(
        f"Secrets Manager secret {secret_arn} did not contain a usable SecretString."
    )


def _get_google_cse_secret_value(secret_arn: str) -> str:
    if not secret_arn:
        raise RuntimeError("Missing Google CSE secret ARN")
    return _get_secret_string(secret_arn)


def _normalize_scrapfly_key(raw_key: str) -> str:
    cleaned = str(raw_key or "").strip()
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith("scp-") else f"scp-live-{cleaned}"


def _get_scrapfly_api_key() -> str:
    global _scrapfly_cached_at, _scrapfly_cached

    direct = str(os.environ.get("SCRAPFLY_API_KEY", "") or "").strip()
    if direct:
        return _normalize_scrapfly_key(direct)

    secret_arn = str(os.environ.get("SCRAPFLY_API_KEY_SECRET_ARN", "") or "").strip()
    if not secret_arn:
        raise RuntimeError(
            "Missing required env var: SCRAPFLY_API_KEY (or SCRAPFLY_API_KEY_SECRET_ARN)"
        )

    ttl = int(
        os.environ.get(
            "SCRAPFLY_API_KEY_CACHE_TTL_SECONDS", _SCRAPFLY_DEFAULT_CACHE_TTL_SECONDS
        )
    )
    ttl = max(0, ttl)
    now = time.time()
    if _scrapfly_cached is not None and ttl > 0 and (now - _scrapfly_cached_at) < ttl:
        return _scrapfly_cached

    key = _normalize_scrapfly_key(_get_secret_string(secret_arn))
    if not key:
        raise RuntimeError(
            f"Secrets Manager secret {secret_arn} did not contain a Scrapfly API key."
        )

    _scrapfly_cached = key
    _scrapfly_cached_at = now
    return key


def _parse_google_cse_credentials_from_json(payload: str) -> tuple[str, str]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Google CSE secret must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("Google CSE secret JSON must be an object")

    normalized = {str(k).strip().lower(): v for k, v in raw.items()}

    api_key = (
        normalized.get("api_key") or normalized.get("apikey") or normalized.get("key")
    )
    engine_id = (
        normalized.get("engine_id")
        or normalized.get("engineid")
        or normalized.get("cx")
    )

    api_key_str = str(api_key or "").strip()
    engine_id_str = str(engine_id or "").strip()
    if not api_key_str or not engine_id_str:
        raise ValueError(
            "Google CSE secret JSON must contain api_key and engine_id (or key/cx)"
        )

    return api_key_str, engine_id_str


def _get_google_cse_credentials() -> tuple[str, str]:
    global _google_cse_cached_at, _google_cse_cached

    ttl = int(
        os.environ.get(
            "GOOGLE_CSE_CACHE_TTL_SECONDS", _GOOGLE_CSE_DEFAULT_CACHE_TTL_SECONDS
        )
    )
    ttl = max(0, ttl)
    now = time.time()

    if (
        _google_cse_cached is not None
        and ttl > 0
        and (now - _google_cse_cached_at) < ttl
    ):
        return _google_cse_cached["api_key"], _google_cse_cached["engine_id"]

    secret_arn = str(os.environ.get("GOOGLE_CSE_SECRET_ARN", "") or "").strip()
    api_key_secret_arn = str(
        os.environ.get("GOOGLE_CSE_API_KEY_SECRET_ARN", "") or ""
    ).strip()
    engine_id_secret_arn = str(
        os.environ.get("GOOGLE_CSE_ENGINE_ID_SECRET_ARN", "") or ""
    ).strip()

    if secret_arn:
        secret_string = _get_google_cse_secret_value(secret_arn)
        api_key, engine_id = _parse_google_cse_credentials_from_json(secret_string)
    elif api_key_secret_arn and engine_id_secret_arn:
        api_key = _get_google_cse_secret_value(api_key_secret_arn)
        engine_id = _get_google_cse_secret_value(engine_id_secret_arn)
    else:
        api_key = str(os.environ.get("GOOGLE_CSE_API_KEY", "") or "").strip()
        engine_id = str(os.environ.get("GOOGLE_CSE_ENGINE_ID", "") or "").strip()

    if not api_key or not engine_id:
        raise RuntimeError(
            "Missing Google CSE credentials. Set GOOGLE_CSE_SECRET_ARN (JSON containing api_key/engine_id), "
            "or GOOGLE_CSE_API_KEY_SECRET_ARN + GOOGLE_CSE_ENGINE_ID_SECRET_ARN, "
            "or GOOGLE_CSE_API_KEY + GOOGLE_CSE_ENGINE_ID."
        )

    _google_cse_cached = {"api_key": api_key, "engine_id": engine_id}
    _google_cse_cached_at = now
    return api_key, engine_id


def _clean_google_snippet(value: Any) -> str:
    snippet = str(value or "")
    return snippet.replace("\xa0", " ").strip()


def _google_cse_request(
    *,
    endpoint: str,
    api_key: str,
    engine_id: str,
    search_term: str,
    country_region: str | None,
    geolocation: str | None,
    result_language: str | None,
    result_num: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "key": api_key,
        "cx": engine_id,
        "q": search_term,
        "num": result_num,
        "fields": "items(title,link,snippet)",
    }
    if country_region:
        params["cr"] = country_region
    if geolocation:
        params["gl"] = geolocation
    if result_language:
        params["lr"] = result_language

    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status = int(resp.getcode() or 0)
            body = resp.read() or b""
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        message = ""
        try:
            parsed = json.loads(body.decode("utf-8") or "{}")
            message = str(parsed.get("error", {}).get("message") or "").strip()
        except Exception:
            message = ""
        detail = f": {message}" if message else ""
        raise ValueError(f"Google CSE API error (HTTP {exc.code}){detail}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to call Google CSE API: {exc.reason}") from exc

    if status >= 400:
        raise ValueError(f"Google CSE API returned HTTP {status}")
    try:
        parsed = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Google CSE API returned invalid JSON") from exc
    if isinstance(parsed, dict) and "error" in parsed:
        message = str(parsed.get("error", {}).get("message") or "").strip()
        detail = f": {message}" if message else ""
        raise ValueError(f"Google CSE API error{detail}")
    if not isinstance(parsed, dict):
        raise ValueError("Google CSE API returned unexpected payload")
    return parsed


def _handle_google_search(event: Mapping[str, Any]) -> dict[str, Any]:
    search_term = str(event.get("search_term", "") or "").strip()
    if not search_term:
        raise ValueError("Missing required parameter: search_term")

    result_num = int(event.get("result_num", 10) or 10)
    result_num = max(1, min(result_num, 10))

    def _opt(name: str) -> str | None:
        val = str(event.get(name, "") or "").strip()
        return val or None

    country_region = (
        _opt("country_region")
        or str(os.environ.get("GOOGLE_CSE_COUNTRY_REGION", "") or "").strip()
        or None
    )
    geolocation = (
        _opt("geolocation")
        or str(os.environ.get("GOOGLE_CSE_GEOLOCATION", "us") or "us").strip()
        or None
    )
    result_language = (
        _opt("result_language")
        or str(
            os.environ.get("GOOGLE_CSE_RESULT_LANGUAGE", "lang_en") or "lang_en"
        ).strip()
        or None
    )

    timeout_seconds = float(event.get("timeout_seconds", 10) or 10)
    timeout_seconds = max(1.0, min(timeout_seconds, 30.0))

    api_key, engine_id = _get_google_cse_credentials()
    endpoint = str(
        os.environ.get("GOOGLE_CSE_ENDPOINT", _GOOGLE_CSE_DEFAULT_ENDPOINT)
        or _GOOGLE_CSE_DEFAULT_ENDPOINT
    ).strip()

    response = _google_cse_request(
        endpoint=endpoint,
        api_key=api_key,
        engine_id=engine_id,
        search_term=search_term,
        country_region=country_region,
        geolocation=geolocation,
        result_language=result_language,
        result_num=result_num,
        timeout_seconds=timeout_seconds,
    )

    items = response.get("items", [])
    results: list[dict[str, str]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "").strip()
            link = str(item.get("link", "") or "").strip()
            snippet = _clean_google_snippet(item.get("snippet"))
            if title or link or snippet:
                results.append(
                    {
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                    }
                )

    return {
        "query": search_term,
        "results": results,
        "count": len(results),
        "source": "google_cse",
    }


def _is_forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _strip_wrapping_quotes(value: str) -> str:
    cleaned = (value or "").strip()
    for _ in range(2):
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()
            continue
        break
    return cleaned


def _validate_public_https_url(url: str) -> str:
    url = _strip_wrapping_quotes(url)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError("Only https URLs are supported")
    if parsed.username or parsed.password:
        raise ValueError("URL must not include credentials")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a hostname")
    lowered = hostname.lower().strip(".")
    if lowered in {"localhost"} or lowered.endswith(".local"):
        raise ValueError("URL hostname is not allowed")

    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        ip = None

    if ip is not None:
        if _is_forbidden_ip(ip):
            raise ValueError("URL hostname resolves to a forbidden IP")
        return parsed.geturl()

    try:
        infos = socket.getaddrinfo(
            hostname,
            parsed.port or 443,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except Exception as exc:
        raise ValueError(f"Failed to resolve hostname: {exc}") from exc

    resolved_ips: set[str] = set()
    for _, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        resolved_ips.add(ip_str)

    if not resolved_ips:
        raise ValueError("Hostname did not resolve to any IP addresses")

    for ip_str in resolved_ips:
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_forbidden_ip(ip_obj):
            raise ValueError("URL hostname resolves to a forbidden IP")

    return parsed.geturl()


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        _validate_public_https_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


def _read_limited_bytes(response: Any, *, max_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            length = int(content_length)
        except (TypeError, ValueError):
            length = None
        if length is not None and length > max_bytes:
            raise ValueError("PDF exceeds max_bytes limit")

    buf = bytearray()
    chunk_size = 1024 * 1024
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise ValueError("PDF exceeds max_bytes limit")
    return bytes(buf)


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

    try:
        with opener.open(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None) or resp.getcode() or 0
            if int(status) >= 400:
                raise ValueError(f"HTTP {int(status)} while fetching PDF")
            final_url = resp.geturl() or validated
            content_type = str(resp.headers.get("Content-Type", "") or "")
            pdf_bytes = _read_limited_bytes(resp, max_bytes=max_bytes)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code} while fetching PDF") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to download PDF: {exc.reason}") from exc

    if not pdf_bytes:
        raise ValueError("Downloaded document was empty")

    if not _looks_like_pdf(pdf_bytes, content_type):
        raise ValueError(
            f"URL did not appear to return a PDF (content-type={content_type or 'unknown'})"
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

    return resp


def _extract_text_from_html(html_bytes: bytes) -> str:
    """Extract plain text from HTML, removing tags and scripts."""
    try:
        text = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = str(html_bytes, errors="ignore")

    # Remove script and style tags
    text = re.sub(
        r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def _split_text_into_chunks(text: str, max_chunk_size: int = 512) -> list[str]:
    """Split text into chunks for semantic search."""
    if not text:
        return []

    # Split by paragraphs first (double newline or period followed by space)
    paragraphs = []
    current = []
    words = text.split()

    for word in words:
        current.append(word)
        chunk_text = " ".join(current)
        if len(chunk_text) >= max_chunk_size:
            if current:
                paragraphs.append(chunk_text)
                current = []

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def _semantic_search_text(
    *,
    text: str,
    query_text: str,
    max_matches: int,
    context_chars: int,
    min_similarity: float,
) -> tuple[list[dict[str, Any]], str]:
    """Semantic search within text content, similar to _semantic_search_pages."""
    embedder = _try_load_pdf_embedder()
    if np is None or embedder is None:
        return (
            [],
            "Semantic search unavailable; using keyword fallback.",
        )

    query_text = (query_text or "").strip()
    if not query_text:
        return ([], "Semantic search requires a non-empty query.")

    cleaned_text = " ".join((text or "").split())
    chunks = _split_text_into_chunks(cleaned_text, max_chunk_size=1024)
    if not chunks:
        return ([], "No content found in the page.")

    chunk_starts: list[int] = []
    pos = 0
    for chunk in chunks:
        chunk_starts.append(pos)
        pos += len(chunk) + 1

    try:
        vectors = embedder.embed([query_text] + chunks)
        query_vec = vectors[0]
        chunk_vecs = vectors[1:]
        scores = chunk_vecs @ query_vec
    except Exception as exc:
        return ([], f"Semantic embedding failed: {exc}")

    ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)

    matches: list[dict[str, Any]] = []
    for idx in ranked:
        if len(matches) >= max_matches:
            break
        score = float(scores[idx])
        if score < float(min_similarity):
            continue

        chunk_text = chunks[idx]
        anchor = _find_query_anchor(text=chunk_text, query_text=query_text)
        if anchor is not None:
            anchor_start, anchor_len = anchor
            global_start = chunk_starts[idx] + anchor_start
            center = global_start + max(0, anchor_len // 2)
        else:
            center = chunk_starts[idx] + (len(chunk_text) // 2)
        snippet_text = _slice_around_center(
            text=cleaned_text, center=center, max_chars=context_chars
        )

        matches.append(
            {
                "text": snippet_text,
                "score": round(score, 4),
            }
        )

    return (matches, "")


def _download_html_from_url(
    *, url: str, timeout_seconds: float, max_bytes: int
) -> tuple[bytes, str, str]:
    """Download HTML content from a URL."""
    validated = _validate_public_https_url(url)
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    req = urllib.request.Request(
        validated,
        method="GET",
        headers={
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    try:
        with opener.open(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None) or resp.getcode() or 0
            if int(status) >= 400:
                raise ValueError(f"HTTP {int(status)}")
            final_url = resp.geturl() or validated
            content_type = str(resp.headers.get("Content-Type", "") or "")
            html_bytes = _read_limited_bytes(resp, max_bytes=max_bytes)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to download: {exc.reason}") from exc
    except Exception as exc:
        raise ValueError(f"Download error: {exc}") from exc

    if not html_bytes:
        raise ValueError("Downloaded content was empty")

    return html_bytes, content_type, final_url


def _scrapfly_request(
    *,
    api_key: str,
    url: str,
    format: str,
    country: str | None,
    asp: bool,
    render_js: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    timeout_ms = int(max(1000.0, min(timeout_seconds * 1000.0, 120_000.0)))
    params: dict[str, str] = {
        "key": api_key,
        "url": url,
        "format": format,
        "country": country or _SCRAPFLY_DEFAULT_COUNTRY,
        "asp": "true" if asp else "false",
        "render_js": "true" if render_js else "false",
        "retry": "false",
        "timeout": str(timeout_ms),
    }

    request_url = (
        f"{_SCRAPFLY_SCRAPE_API_ENDPOINT}?{urllib.parse.urlencode(params, safe=':,')}"
    )
    req = urllib.request.Request(
        request_url,
        method="GET",
        headers={
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "application/json",
        },
    )

    # Client-side timeout should be slightly higher than the tool timeout to account for network overhead.
    client_timeout = max(5.0, min(timeout_seconds + 10.0, 180.0))

    try:
        with urllib.request.urlopen(req, timeout=client_timeout) as resp:
            status = int(resp.getcode() or 0)
            body = resp.read() or b""
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        body = exc.read() if hasattr(exc, "read") else b""
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to call Scrapfly API: {exc.reason}") from exc

    try:
        parsed = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Scrapfly API returned invalid JSON") from exc

    if status >= 400:
        message = (
            str(parsed.get("message", "") or "").strip()
            if isinstance(parsed, dict)
            else ""
        )
        code = (
            str(parsed.get("code", "") or "").strip()
            if isinstance(parsed, dict)
            else ""
        )
        detail = f": {code} {message}".strip() if (code or message) else ""
        raise ValueError(f"Scrapfly API error (HTTP {status}){detail}")

    if not isinstance(parsed, dict):
        raise ValueError("Scrapfly API returned unexpected payload")

    return parsed


def _scrapfly_download_large_object(
    *, download_url: str, api_key: str, timeout_seconds: float
) -> bytes:
    parsed_url = urllib.parse.urlparse(download_url)
    query = urllib.parse.parse_qs(parsed_url.query)
    if "key" not in query:
        sep = "&" if parsed_url.query else "?"
        download_url = f"{download_url}{sep}{urllib.parse.urlencode({'key': api_key})}"

    req = urllib.request.Request(
        download_url,
        method="GET",
        headers={"User-Agent": _DEFAULT_USER_AGENT},
    )
    with urllib.request.urlopen(
        req, timeout=max(5.0, min(timeout_seconds, 180.0))
    ) as resp:
        return resp.read() or b""


def _scrapfly_extract_text(
    *,
    response: Mapping[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    config = response.get("config") if isinstance(response.get("config"), dict) else {}
    context = (
        response.get("context") if isinstance(response.get("context"), dict) else {}
    )
    result = response.get("result") if isinstance(response.get("result"), dict) else {}

    proxy = context.get("proxy") if isinstance(context.get("proxy"), dict) else {}
    cost = context.get("cost") if isinstance(context.get("cost"), dict) else {}

    meta: dict[str, Any] = {
        "scrapfly_uuid": response.get("uuid"),
        "scrapfly_request_id": config.get("request_id"),
        "proxy_country": proxy.get("country"),
        "proxy_pool": proxy.get("pool"),
        "api_cost": cost.get("total"),
        "target_status_code": result.get("status_code"),
        "result_format": result.get("format"),
    }

    fmt = str(result.get("format", "") or "").strip().lower()
    content = result.get("content")

    text: str
    if (
        fmt in {"clob", "blob"}
        and isinstance(content, str)
        and content.strip().startswith("http")
    ):
        raw = _scrapfly_download_large_object(
            download_url=content.strip(),
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        if fmt == "blob":
            raise ValueError(
                "Scrapfly returned a BLOB; binary content is not supported for internet_search."
            )
        text = raw.decode("utf-8", errors="ignore")
    elif isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    else:
        text = json.dumps(content, ensure_ascii=False, default=str)

    text = (text or "").strip()
    if len(text) > _SCRAPFLY_MAX_CONTENT_CHARS:
        text = text[: _SCRAPFLY_MAX_CONTENT_CHARS - 3] + "..."

    return text, meta


def _scrapfly_fetch_text(
    *, url: str, timeout_seconds: float
) -> tuple[str, dict[str, Any]]:
    api_key = _get_scrapfly_api_key()
    fmt = str(
        os.environ.get("SCRAPFLY_FORMAT", _SCRAPFLY_DEFAULT_FORMAT)
        or _SCRAPFLY_DEFAULT_FORMAT
    ).strip()
    if not fmt:
        fmt = _SCRAPFLY_DEFAULT_FORMAT

    country = str(
        os.environ.get("SCRAPFLY_COUNTRY", _SCRAPFLY_DEFAULT_COUNTRY)
        or _SCRAPFLY_DEFAULT_COUNTRY
    ).strip()
    country = country or _SCRAPFLY_DEFAULT_COUNTRY

    # Prefer defaults; escalate only on failures.
    attempts = [
        {"asp": False, "render_js": False, "mode": "default"},
        {"asp": True, "render_js": True, "mode": "asp_render"},
    ]

    last_error: str | None = None
    for attempt in attempts:
        try:
            resp = _scrapfly_request(
                api_key=api_key,
                url=url,
                format=fmt,
                country=country,
                asp=bool(attempt["asp"]),
                render_js=bool(attempt["render_js"]),
                timeout_seconds=timeout_seconds,
            )
            text, meta = _scrapfly_extract_text(
                response=resp, api_key=api_key, timeout_seconds=timeout_seconds
            )
            meta["scrape_mode"] = attempt["mode"]
            if text and len(text) >= _SCRAPFLY_MIN_CONTENT_CHARS:
                return text, meta
            last_error = "Scraped content was empty or too short."
        except Exception as exc:
            last_error = str(exc)
            continue

    raise ValueError(last_error or "Scrapfly scrape failed.")


def _url_looks_like_pdf(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    return (parsed.path or "").lower().endswith(".pdf")


def _looks_like_base64_pdf(text: str) -> bool:
    # Base64-encoded PDFs typically start with "%PDF-" which is "JVBERi0" in base64.
    cleaned = "".join((text or "").split())
    return cleaned.startswith("JVBERi0")


def _pdf_matches_to_snippets(matches: Any, *, max_chars: int) -> list[dict[str, Any]]:
    if not isinstance(matches, list):
        return []
    max_chars = int(max_chars or 0)
    if max_chars <= 0:
        max_chars = 1000

    snippets: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        snippet_text = str(match.get("snippet", "") or "").strip()
        if not snippet_text:
            continue
        if len(snippet_text) > max_chars:
            snippet_text = _slice_around_query(
                text=snippet_text,
                query_text=str(match.get("query", "") or ""),
                max_chars=max_chars,
            )
        snippets.append(
            {
                "text": snippet_text,
                "score": float(match.get("score", 0.0) or 0.0),
                "page": match.get("page"),
            }
        )
    return snippets


def _handle_internet_search(event: Mapping[str, Any]) -> dict[str, Any]:
    """
    Web search with robust scraping + semantic filtering.

    Uses Google Custom Search Engine for URL discovery, Scrapfly for fetching page content, and
    ONNX-based semantic filtering to extract relevant snippets.

    Output includes a concatenated blob with citations for the top results:
    - Each result includes url, title, target_status_code, snippet, score, and optional error.
    - PDFs are routed through pdf_extract_text (semantic mode).
    """
    query = str(event.get("query", "") or "").strip()
    if not query:
        raise ValueError("Missing required parameter: query")

    max_results = int(event.get("max_results", 5) or 5)
    max_results = max(1, min(max_results, 5))

    max_snippets_per_url = int(event.get("max_snippets", 1) or 1)
    max_snippets_per_url = max(1, min(max_snippets_per_url, 5))

    context_chars = int(event.get("context_chars", 1200) or 1200)
    context_chars = max(800, min(context_chars, 2000))
    max_snippet_chars = context_chars

    min_similarity = float(event.get("min_similarity", 0.3) or 0.3)
    min_similarity = max(0.0, min(min_similarity, 1.0))

    timeout_seconds = float(event.get("timeout_seconds", 30) or 30)
    timeout_seconds = max(5.0, min(timeout_seconds, 120.0))

    google_results = _handle_google_search(
        {
            "search_term": query,
            "result_num": max_results,
        }
    )

    google_items = google_results.get("results")
    if not isinstance(google_items, list) or not google_items:
        return {
            "query": query,
            "results": [],
            "urls_attempted": 0,
            "urls_succeeded": 0,
            "notes": "Google Search returned no results.",
            "concat_blob": "",
            "citations": [],
        }

    if _try_load_pdf_embedder() is None:
        return {
            "query": query,
            "results": [],
            "urls_attempted": 0,
            "urls_succeeded": 0,
            "notes": "Semantic search unavailable (missing ONNX assets/dependencies).",
            "concat_blob": "",
            "citations": [],
        }

    attempted = 0
    errors: list[str] = []
    results: list[dict[str, Any]] = []

    candidates: list[dict[str, Any]] = []
    for rank, item in enumerate(google_items[:max_results]):
        if not isinstance(item, dict):
            continue
        url = str(item.get("link", "") or "").strip()
        if not url:
            continue
        candidates.append(
            {
                "url": url,
                "title": str(item.get("title", "") or "").strip(),
                "google_snippet": _clean_google_snippet(item.get("snippet")),
                "source_rank": rank + 1,
            }
        )

    if not candidates:
        return {
            "query": query,
            "results": [],
            "urls_attempted": 0,
            "urls_succeeded": 0,
            "notes": "Google Search returned no usable results.",
            "concat_blob": "",
            "citations": [],
        }

    def _truncate_snippet(text: str) -> str:
        snippet = str(text or "").strip()
        if len(snippet) <= max_snippet_chars:
            return snippet
        return _slice_around_query(
            text=snippet, query_text=query, max_chars=max_snippet_chars
        )

    for candidate in candidates:
        attempted += 1
        url = str(candidate.get("url", "") or "").strip()
        title = str(candidate.get("title", "") or "").strip()

        try:
            if _url_looks_like_pdf(url):
                pdf_extract = _handle_pdf_extract_text(
                    {
                        "url": url,
                        "query": query,
                        "search_mode": "semantic",
                        "max_matches": max_snippets_per_url,
                        "context_chars": context_chars,
                        "min_similarity": min_similarity,
                        "timeout_seconds": min(timeout_seconds, 120.0),
                    }
                )
                if pdf_extract.get("search_mode_used") != "semantic":
                    raise ValueError(
                        pdf_extract.get("notes") or "Semantic PDF search unavailable."
                    )
                matches = _pdf_matches_to_snippets(
                    pdf_extract.get("matches"), max_chars=max_snippet_chars
                )
                if not matches:
                    raise ValueError(
                        pdf_extract.get("notes")
                        or "No semantic matches found in PDF content."
                    )
                best = matches[0]
                snippet_text = str(best.get("text", "") or "").strip()
                if not snippet_text:
                    raise ValueError("No snippet extracted from PDF.")
                page = best.get("page")
                if page is not None:
                    snippet_text = f"(page {page}) {snippet_text}"
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "target_status_code": 200,
                        "snippet": _truncate_snippet(snippet_text),
                        "score": float(best.get("score", 0.0) or 0.0),
                        "source": "semantic_pdf",
                    }
                )
                continue

            page_text, scrape_meta = _scrapfly_fetch_text(
                url=url, timeout_seconds=timeout_seconds
            )
            if not page_text or len(page_text) < _SCRAPFLY_MIN_CONTENT_CHARS:
                raise ValueError(
                    "Scrapfly returned insufficient content for semantic search."
                )

            snippets, note = _semantic_search_text(
                text=page_text,
                query_text=query,
                max_matches=max_snippets_per_url,
                context_chars=context_chars,
                min_similarity=min_similarity,
            )
            if not snippets:
                raise ValueError(note or "No semantic matches found in page content.")
            best = snippets[0]
            snippet_text = str(best.get("text", "") or "").strip()
            if not snippet_text:
                raise ValueError("No snippet extracted from semantic search.")
            results.append(
                {
                    "url": url,
                    "title": title,
                    "target_status_code": int(
                        scrape_meta.get("target_status_code", 0) or 0
                    ),
                    "snippet": _truncate_snippet(snippet_text),
                    "score": float(best.get("score", 0.0) or 0.0),
                    "source": "semantic_html",
                }
            )
        except Exception as exc:
            error_msg = str(exc)
            errors.append(f"{url}: {error_msg}")
            results.append(
                {
                    "url": url,
                    "title": title,
                    "error": error_msg,
                    "source": "error",
                }
            )

    citations: list[dict[str, Any]] = []
    concat_lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        url = str(result.get("url", "") or "").strip()
        title = str(result.get("title", "") or "").strip()
        snippet = str(result.get("snippet", "") or "").strip()
        error = str(result.get("error", "") or "").strip()
        if snippet:
            concat_lines.append(f"[{idx}] {title} — {url}\n{snippet}")
            citations.append(
                {
                    "id": idx,
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "score": result.get("score"),
                }
            )
        else:
            concat_lines.append(
                f"[{idx}] {title} — {url}\n[error] {error or 'No snippet available.'}"
            )
            citations.append({"id": idx, "url": url, "title": title, "error": error})

    succeeded = len([r for r in results if r.get("snippet")])
    notes = ""
    if succeeded == 0:
        notes = "No semantic snippets could be extracted from the top results."
        if errors:
            notes += f" Errors: {'; '.join(errors[:3])}"
    elif errors:
        notes = (
            f"Some URLs failed to load or match semantically: {'; '.join(errors[:2])}"
        )

    return {
        "query": query,
        "results": results,
        "urls_attempted": attempted,
        "urls_succeeded": succeeded,
        "notes": notes,
        "concat_blob": "\n\n".join(concat_lines).strip(),
        "citations": citations,
    }


def _get_bedrock_kb_config() -> tuple[str, str]:
    kb_id = str(os.environ.get(_BEDROCK_KNOWLEDGE_BASE_ID_ENV, "") or "").strip()
    model_arn = str(
        os.environ.get(_BEDROCK_KNOWLEDGE_BASE_MODEL_ARN_ENV, "") or ""
    ).strip()

    if not kb_id:
        raise RuntimeError(
            f"Missing required env var: {_BEDROCK_KNOWLEDGE_BASE_ID_ENV}"
        )
    if not model_arn:
        raise RuntimeError(
            f"Missing required env var: {_BEDROCK_KNOWLEDGE_BASE_MODEL_ARN_ENV}"
        )

    return kb_id, model_arn


def _simplify_bedrock_kb_location(location: Any) -> dict[str, Any]:
    if not isinstance(location, dict):
        return {}

    loc_type = str(location.get("type", "") or "").strip()
    uri: str | None = None

    if isinstance(location.get("webLocation"), dict):
        uri = str(location["webLocation"].get("url", "") or "").strip() or None
    elif isinstance(location.get("s3Location"), dict):
        uri = str(location["s3Location"].get("uri", "") or "").strip() or None
    elif isinstance(location.get("confluenceLocation"), dict):
        uri = str(location["confluenceLocation"].get("url", "") or "").strip() or None
    elif isinstance(location.get("sharePointLocation"), dict):
        uri = str(location["sharePointLocation"].get("url", "") or "").strip() or None
    elif isinstance(location.get("salesforceLocation"), dict):
        uri = str(location["salesforceLocation"].get("url", "") or "").strip() or None
    elif isinstance(location.get("kendraDocumentLocation"), dict):
        uri = (
            str(location["kendraDocumentLocation"].get("uri", "") or "").strip() or None
        )
    elif isinstance(location.get("sqlLocation"), dict):
        uri = str(location["sqlLocation"].get("query", "") or "").strip() or None
    elif isinstance(location.get("customDocumentLocation"), dict):
        uri = (
            str(location["customDocumentLocation"].get("id", "") or "").strip() or None
        )

    payload: dict[str, Any] = {}
    if loc_type:
        payload["type"] = loc_type
    if uri:
        payload["uri"] = uri
    return payload


def _handle_retrieve_and_generate(event: Mapping[str, Any]) -> dict[str, Any]:
    query = str(event.get("query", "") or "").strip()
    if not query:
        raise ValueError("Missing required parameter: query")

    number_of_results = int(event.get("number_of_results", 5) or 5)
    number_of_results = max(1, min(number_of_results, 10))

    context_chars = int(event.get("context_chars", 1000) or 1000)
    context_chars = max(1000, min(context_chars, 2000))

    knowledge_base_id, model_arn = _get_bedrock_kb_config()

    client = boto3.client("bedrock-agent-runtime")
    response = client.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": knowledge_base_id,
                "modelArn": model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": number_of_results,
                    }
                },
            },
        },
    )

    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(output, dict):
        output = {}
    answer = str(output.get("text", "") or "").strip()

    citations_out: list[dict[str, Any]] = []
    citations = response.get("citations", []) if isinstance(response, dict) else []
    if isinstance(citations, list):
        for citation in citations[:10]:
            if not isinstance(citation, dict):
                continue

            generated = citation.get("generatedResponsePart")
            cited_text = ""
            span_out: dict[str, int] = {}
            if isinstance(generated, dict):
                text_part = generated.get("textResponsePart")
                if isinstance(text_part, dict):
                    cited_text = str(text_part.get("text", "") or "").strip()
                    span = text_part.get("span")
                    if isinstance(span, dict):
                        try:
                            start = int(span.get("start", 0) or 0)
                            end = int(span.get("end", 0) or 0)
                            span_out = {"start": start, "end": end}
                        except Exception:
                            span_out = {}

            refs_out: list[dict[str, Any]] = []
            retrieved_refs = citation.get("retrievedReferences", [])
            if isinstance(retrieved_refs, list):
                for ref in retrieved_refs[:5]:
                    if not isinstance(ref, dict):
                        continue
                    content = ref.get("content")
                    ref_text = ""
                    if isinstance(content, dict):
                        ref_text = str(content.get("text", "") or "").strip()
                    if ref_text and len(ref_text) > context_chars:
                        ref_text = _slice_around_query(
                            text=ref_text, query_text=query, max_chars=context_chars
                        )
                    location = _simplify_bedrock_kb_location(ref.get("location"))
                    refs_out.append(
                        {
                            "location": location,
                            "snippet": ref_text,
                        }
                    )

            if cited_text or span_out or refs_out:
                citations_out.append(
                    {
                        "text": cited_text,
                        "span": span_out,
                        "references": refs_out,
                    }
                )

    session_id = (
        str(response.get("sessionId", "") or "").strip()
        if isinstance(response, dict)
        else ""
    )

    result: dict[str, Any] = {
        "query": query,
        "answer": answer,
        "knowledge_base_id": knowledge_base_id,
        "model_arn": model_arn,
        "number_of_results": number_of_results,
        "citations": citations_out,
    }
    if session_id:
        result["session_id"] = session_id
    return result


def current_time(format: str = "iso", timezone: str = "UTC") -> dict[str, Any]:
    """Return current time in ISO or unix format."""
    payload: dict[str, Any] = {"format": format, "timezone": timezone}
    return _handle_current_time(payload)


def google_search(
    search_term: str,
    result_num: int = 10,
    country_region: str | None = None,
    geolocation: str | None = None,
    result_language: str | None = None,
    timeout_seconds: float = 10,
) -> dict[str, Any]:
    """Search the web using Google Custom Search Engine."""
    payload: dict[str, Any] = {
        "search_term": search_term,
        "result_num": result_num,
        "timeout_seconds": timeout_seconds,
    }
    if country_region is not None:
        payload["country_region"] = country_region
    if geolocation is not None:
        payload["geolocation"] = geolocation
    if result_language is not None:
        payload["result_language"] = result_language
    return _handle_google_search(payload)


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
    """Extract text from PDFs (url, s3, or base64) with keyword/semantic search."""
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


def internet_search(
    query: str,
    max_results: int = 5,
    max_snippets: int = 1,
    context_chars: int = 1200,
    min_similarity: float = 0.3,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Web search with scraping + semantic filtering, returning concatenated snippets."""
    payload: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "max_snippets": max_snippets,
        "context_chars": context_chars,
        "min_similarity": min_similarity,
        "timeout_seconds": timeout_seconds,
    }
    return _handle_internet_search(payload)


def retrieve_and_generate(
    query: str,
    number_of_results: int = 5,
    context_chars: int = 1000,
) -> dict[str, Any]:
    """Retrieve and generate using Bedrock Knowledge Base."""
    payload: dict[str, Any] = {
        "query": query,
        "number_of_results": number_of_results,
        "context_chars": context_chars,
    }
    return _handle_retrieve_and_generate(payload)


def register_core_tools(mcp: Any) -> None:
    """Register core tools on a FastMCP server."""
    mcp.tool()(current_time)
    mcp.tool()(internet_search)
