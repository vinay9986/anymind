from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

import boto3
import structlog

from anymind.tools.core_http import _DEFAULT_USER_AGENT
from anymind.tools.core_pdf import _handle_pdf_extract_text, _try_load_pdf_embedder
from anymind.tools.core_text import (
    _find_query_anchor,
    _slice_around_center,
    _slice_around_query,
)

_KAGI_DEFAULT_ENDPOINT = "https://kagi.com/api/v0/search"
_KAGI_DEFAULT_CACHE_TTL_SECONDS = 300
_SEARCH_MODES = ("auto", "keyword", "semantic")
_SCRAPFLY_SCRAPE_API_ENDPOINT = "https://api.scrapfly.io/scrape"
_SCRAPFLY_DEFAULT_CACHE_TTL_SECONDS = 300
_SCRAPFLY_DEFAULT_COUNTRY = "us"
_SCRAPFLY_DEFAULT_FORMAT = "text:only_content,no_links,no_images"
_SCRAPFLY_MIN_CONTENT_CHARS = 200
_SCRAPFLY_MAX_CONTENT_CHARS = 250_000
_INTERNET_SEARCH_DEFAULT_MAX_RESULTS = 3
_INTERNET_SEARCH_DEFAULT_MAX_SNIPPETS = 5
_INTERNET_SEARCH_DEFAULT_CONTEXT_CHARS = 4000
_INTERNET_SEARCH_DEFAULT_MIN_SIMILARITY = 0.3
_INTERNET_SEARCH_DEFAULT_TIMEOUT_SECONDS = 60.0
_SCRAPFLY_ALLOWED_EXTRACTION_MODELS = {
    "article",
    "event",
    "food_recipe",
    "hotel",
    "hotel_listing",
    "job_listing",
    "job_posting",
    "organization",
    "product",
    "product_listing",
    "real_estate_property",
    "real_estate_property_listing",
    "review_list",
    "search_engine_results",
    "social_media_post",
    "software",
    "stock",
    "vehicle_ad",
}

logger = structlog.get_logger(__name__)

_kagi_cached_at: float = 0.0
_kagi_cached_key: str | None = None

_scrapfly_cached_at: float = 0.0
_scrapfly_cached: str | None = None


try:  # Optional; used for semantic search
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


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


def _normalize_scrapfly_key(raw_key: str) -> str:
    cleaned = str(raw_key or "").strip()
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith("scp-") else f"scp-live-{cleaned}"


def _debug_search_enabled() -> bool:
    return True


def _get_scrapfly_api_key() -> str:
    global _scrapfly_cached_at, _scrapfly_cached

    ttl = int(
        os.environ.get(
            "SCRAPFLY_API_KEY_CACHE_TTL_SECONDS", _SCRAPFLY_DEFAULT_CACHE_TTL_SECONDS
        )
        or _SCRAPFLY_DEFAULT_CACHE_TTL_SECONDS
    )
    now = time.time()
    if _scrapfly_cached and (now - _scrapfly_cached_at) < ttl:
        return _scrapfly_cached

    direct = str(os.environ.get("SCRAPFLY_API_KEY", "") or "").strip()
    if direct:
        key = _normalize_scrapfly_key(direct)
        _scrapfly_cached = key
        _scrapfly_cached_at = now
        return key

    secret_arn = str(os.environ.get("SCRAPFLY_API_KEY_SECRET_ARN", "") or "").strip()
    if not secret_arn:
        raise RuntimeError(
            "Missing Scrapfly API key. Set SCRAPFLY_API_KEY or SCRAPFLY_API_KEY_SECRET_ARN."
        )

    secret_string = _get_secret_string(secret_arn)
    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError:
        key = _normalize_scrapfly_key(secret_string)
    else:
        key = _normalize_scrapfly_key(str(payload.get("api_key", "") or ""))
    if not key:
        raise RuntimeError(
            f"Secrets Manager secret {secret_arn} did not contain a Scrapfly API key."
        )

    _scrapfly_cached = key
    _scrapfly_cached_at = now
    return key


def _get_kagi_api_key() -> str:
    global _kagi_cached_at, _kagi_cached_key
    ttl = int(
        os.environ.get(
            "KAGI_API_KEY_CACHE_TTL_SECONDS", _KAGI_DEFAULT_CACHE_TTL_SECONDS
        )
        or _KAGI_DEFAULT_CACHE_TTL_SECONDS
    )
    now = time.time()
    if _kagi_cached_key and (now - _kagi_cached_at) < ttl:
        return _kagi_cached_key

    api_key = str(os.environ.get("KAGI_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing Kagi API key. Set KAGI_API_KEY or provide search.kagi_api_key in config."
        )
    _kagi_cached_key = api_key
    _kagi_cached_at = now
    return api_key


def _clean_search_snippet(value: Any) -> str:
    snippet = str(value or "")
    return snippet.replace("\xa0", " ").strip()


def _kagi_request(
    *,
    endpoint: str,
    api_key: str,
    query: str,
    limit: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": query,
        "limit": limit,
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "application/json",
            "Authorization": f"Bot {api_key}",
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
            if isinstance(parsed, dict):
                errors = parsed.get("error")
                if isinstance(errors, list) and errors:
                    message = str(errors[0].get("msg", "") or "").strip()
        except Exception:
            message = ""
        detail = f": {message}" if message else ""
        raise ValueError(f"Kagi API error (HTTP {exc.code}){detail}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to call Kagi API: {exc.reason}") from exc

    if status >= 400:
        raise ValueError(f"Kagi API returned HTTP {status}")
    try:
        parsed = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Kagi API returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Kagi API returned unexpected payload")

    return parsed


def _handle_kagi_search(event: Mapping[str, Any]) -> dict[str, Any]:
    search_term = str(event.get("search_term", "") or "").strip()
    if not search_term:
        raise ValueError("Missing required parameter: search_term")

    result_num = int(event.get("result_num", 10) or 10)
    result_num = max(1, min(result_num, 10))

    timeout_seconds = float(event.get("timeout_seconds", 10) or 10)
    timeout_seconds = max(1.0, min(timeout_seconds, 30.0))

    api_key = _get_kagi_api_key()
    endpoint = str(
        os.environ.get("KAGI_API_ENDPOINT", _KAGI_DEFAULT_ENDPOINT)
        or _KAGI_DEFAULT_ENDPOINT
    ).strip()

    response = _kagi_request(
        endpoint=endpoint,
        api_key=api_key,
        query=search_term,
        limit=result_num,
        timeout_seconds=timeout_seconds,
    )
    if _debug_search_enabled():
        logger.info(
            "internet_search_kagi_response",
            query=search_term,
            result_num=result_num,
            response=response,
        )

    data = response.get("data", [])
    results: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            t_val = item.get("t")
            if t_val not in (0, "0"):
                continue
            title = str(item.get("title", "") or "").strip()
            link = str(item.get("url", "") or "").strip()
            snippet = _clean_search_snippet(item.get("snippet"))
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
        "source": "kagi",
    }


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
    if embedder is None or np is None:
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


_BLOCK_SIGNALS = ("429", "403", "captcha", "blocked", "access denied", "forbidden")


def _is_block_error(err: str) -> bool:
    """Return True when the error string looks like a site-level block or rate-limit."""
    low = err.lower()
    return any(s in low for s in _BLOCK_SIGNALS)


def _scrapfly_request(
    *,
    api_key: str,
    url: str,
    format: str,
    country: str | None,
    asp: bool,
    render_js: bool,
    timeout_seconds: float,
    extraction_model: str | None = None,
    proxy_pool: str | None = None,
) -> dict[str, Any]:
    timeout_seconds = float(timeout_seconds or 0.0)
    if asp:
        timeout_seconds = max(30.0, min(timeout_seconds, 150.0))
    else:
        timeout_seconds = max(1.0, min(timeout_seconds, 300.0))
    timeout_ms = int(timeout_seconds * 1000.0)
    params: dict[str, str] = {
        "key": api_key,
        "url": url,
        "format": format,
        "country": country or _SCRAPFLY_DEFAULT_COUNTRY,
        "asp": "true" if asp else "false",
        "render_js": "true" if render_js else "false",
        "retry": "false",
        "timeout": str(timeout_ms),
        "proxy_pool": proxy_pool or "public_datacenter_pool",
    }
    if extraction_model:
        params["extraction_model"] = extraction_model

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
        if _debug_search_enabled():
            logger.info(
                "internet_search_scrapfly_error",
                url=url,
                status=status,
                response=parsed,
                asp=bool(asp),
                render_js=bool(render_js),
            )
        detail = f": {code} {message}".strip() if (code or message) else ""
        raise ValueError(f"Scrapfly API error (HTTP {status}){detail}")

    if not isinstance(parsed, dict):
        raise ValueError("Scrapfly API returned unexpected payload")

    if _debug_search_enabled():
        logger.info(
            "internet_search_scrapfly_response",
            url=url,
            response=parsed,
            asp=bool(asp),
            render_js=bool(render_js),
        )

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
    if not _debug_search_enabled():
        if len(text) > _SCRAPFLY_MAX_CONTENT_CHARS:
            text = text[: _SCRAPFLY_MAX_CONTENT_CHARS - 3] + "..."

    return text, meta


def _scrapfly_extract_ai_data(
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
        "extraction_status": result.get("status"),
        "extraction_error": result.get("error"),
    }

    extracted = result.get("extracted_data")
    data = extracted.get("data") if isinstance(extracted, dict) else None
    if data is None:
        return "", meta

    def _safe_strip(value: object) -> str:
        return str(value).strip()

    def _join_corpus(items: list[object]) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            content = _safe_strip(item.get("content", ""))
            if not content:
                continue
            if content.startswith("http://") or content.startswith("https://"):
                continue
            if content in seen:
                continue
            seen.add(content)
            parts.append(content)
        return "\n".join(parts).strip()

    payload: object = data
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = payload

    text = ""
    if isinstance(payload, dict):
        article_body = _safe_strip(payload.get("article_body", ""))
        corpus_text = ""
        corpus = payload.get("corpus")
        if isinstance(corpus, list):
            corpus_text = _join_corpus(corpus)
        if article_body and corpus_text:
            text = f"{article_body}\n\n{corpus_text}"
        elif article_body:
            text = article_body
        elif corpus_text:
            text = corpus_text

    if not text:
        if isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload, ensure_ascii=False, default=str)

    return (text or "").strip(), meta


def _scrapfly_fetch_ai_data(
    *, url: str, timeout_seconds: float, extraction_model: str | None = None
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

    if extraction_model is None:
        extraction_model = "article"
    extraction_model = str(extraction_model or "").strip()
    if not extraction_model:
        raise ValueError("Missing Scrapfly extraction_model for AI extraction.")
    if extraction_model not in _SCRAPFLY_ALLOWED_EXTRACTION_MODELS:
        allowed = ", ".join(sorted(_SCRAPFLY_ALLOWED_EXTRACTION_MODELS))
        raise ValueError(
            f"Unsupported extraction_model '{extraction_model}'. Allowed: {allowed}"
        )

    # Attempt order:
    #   1. asp, no JS, datacenter  — fast and cheap
    #   2. asp, JS,    datacenter  — JS fallback for dynamic pages
    #   3. asp, JS,    residential — last resort when site blocks datacenter IPs (429/403)
    attempts = [
        {
            "asp": True,
            "render_js": False,
            "proxy_pool": "public_datacenter_pool",
            "mode": "asp",
        },
        {
            "asp": True,
            "render_js": True,
            "proxy_pool": "public_datacenter_pool",
            "mode": "asp+js",
        },
        {
            "asp": True,
            "render_js": True,
            "proxy_pool": "residential_pool",
            "mode": "asp+js+residential",
        },
    ]

    last_error: str | None = None
    skip_to_residential = False
    for attempt in attempts:
        # If a prior attempt hit a block/rate-limit, skip the remaining datacenter
        # attempts and go straight to residential.
        if skip_to_residential and attempt["proxy_pool"] != "residential_pool":
            continue
        try:
            resp = _scrapfly_request(
                api_key=api_key,
                url=url,
                format=fmt,
                country=country,
                asp=bool(attempt["asp"]),
                render_js=bool(attempt["render_js"]),
                timeout_seconds=timeout_seconds,
                extraction_model=extraction_model,
                proxy_pool=str(attempt["proxy_pool"]),
            )
            text, meta = _scrapfly_extract_ai_data(
                response=resp, api_key=api_key, timeout_seconds=timeout_seconds
            )
            meta["scrape_mode"] = attempt["mode"]
            meta["extraction_model"] = extraction_model
            if _debug_search_enabled():
                logger.info(
                    "internet_search_scrapfly_ai_text",
                    url=url,
                    scrape_meta=meta,
                    text=text,
                )
            if text:
                return text, meta
            last_error = "Scrapfly AI extraction returned empty content."
        except Exception as exc:
            last_error = str(exc)
            if _is_block_error(last_error):
                skip_to_residential = True
            continue

    raise ValueError(last_error or "Scrapfly AI extraction failed.")


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

    # Single attempt (no retries).
    attempts = [
        {"asp": True, "render_js": False, "mode": "asp"},
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
            if _debug_search_enabled():
                logger.info(
                    "internet_search_scrapfly_text",
                    url=url,
                    scrape_meta=meta,
                    text=text,
                )
            if text and len(text) >= _SCRAPFLY_MIN_CONTENT_CHARS:
                return text, meta
            last_error = "Scraped content was empty or too short."
        except Exception as exc:
            last_error = str(exc)
            continue

    raise ValueError(last_error or "Scrapfly scrape failed.")


# URL path patterns that indicate listing/index pages — heavy, slow, rarely useful.
# Individual article/abstract pages from the same domain work fine.
def _url_is_listing_page(url: str) -> bool:
    """Return True if the URL is clearly a feed/syndication endpoint (never article content)."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    path = (parsed.path or "").lower().rstrip("/")
    return path.endswith("/feed") or path.endswith("/rss") or path.endswith("/atom")


def _url_looks_like_pdf(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    return (parsed.path or "").lower().endswith(".pdf")


def _normalize_url_host(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:
        host = host.split(":", 1)[0]
    return host


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
        # Do not truncate tool output; preserve the snippet as returned.
        snippets.append(
            {
                "text": snippet_text,
                "score": float(match.get("score", 0.0) or 0.0),
                "page": match.get("page"),
            }
        )
    return snippets


def _handle_internet_search(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Web search with robust scraping + semantic filtering.

    Uses Kagi Search for URL discovery, Scrapfly AI extraction to fetch structured
    content, and ONNX-based semantic filtering to extract relevant snippets.

    Returns a list of result objects (url, title, target_status_code, snippet, score).
    PDFs are routed through pdf_extract_text (semantic mode).
    """
    query = str(event.get("query", "") or "").strip()
    if not query:
        raise ValueError("Missing required parameter: query")

    max_results = _INTERNET_SEARCH_DEFAULT_MAX_RESULTS
    desired_results = _INTERNET_SEARCH_DEFAULT_MAX_RESULTS

    max_snippets_per_url = _INTERNET_SEARCH_DEFAULT_MAX_SNIPPETS
    max_snippets_per_url = max(1, min(max_snippets_per_url, 5))

    context_chars = _INTERNET_SEARCH_DEFAULT_CONTEXT_CHARS
    max_snippet_chars = context_chars

    min_similarity = _INTERNET_SEARCH_DEFAULT_MIN_SIMILARITY
    min_similarity = max(0.0, min(min_similarity, 1.0))

    timeout_seconds = _INTERNET_SEARCH_DEFAULT_TIMEOUT_SECONDS
    timeout_seconds = max(5.0, min(timeout_seconds, 60.0))
    extraction_model = str(event.get("extraction_model", "") or "article").strip()
    if not extraction_model:
        extraction_model = "article"

    search_limit = min(10, max(desired_results * 4, desired_results * 2))
    if search_limit < desired_results:
        search_limit = desired_results

    search_results = _handle_kagi_search(
        {
            "search_term": query,
            "result_num": search_limit,
        }
    )

    search_items = search_results.get("results")
    if not isinstance(search_items, list) or not search_items:
        return []

    if _try_load_pdf_embedder() is None:
        return []

    attempted = 0
    errors: list[str] = []
    results: list[dict[str, Any]] = []

    candidates: list[dict[str, Any]] = []
    seen_success_hosts: set[str] = set()
    seen_failed_hosts: set[str] = set()
    dropped_duplicates = 0
    used_secondary_search = False

    def _collect_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for rank, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            url = str(item.get("link", "") or "").strip()
            if not url:
                continue
            if _url_is_listing_page(url):
                logger.info("internet_search_skip_listing_page", url=url)
                continue
            collected.append(
                {
                    "url": url,
                    "title": str(item.get("title", "") or "").strip(),
                    "search_snippet": _clean_search_snippet(item.get("snippet")),
                    "source_rank": rank + 1,
                    "host": _normalize_url_host(url),
                }
            )
        return collected

    candidates = _collect_candidates(search_items)

    if not candidates:
        return []

    def _process_candidates(items: list[dict[str, Any]]) -> None:
        nonlocal attempted, dropped_duplicates
        for candidate in items:
            if len(results) >= desired_results:
                break
            url = str(candidate.get("url", "") or "").strip()
            title = str(candidate.get("title", "") or "").strip()
            host = str(candidate.get("host", "") or "").strip()

            if host and (host in seen_success_hosts or host in seen_failed_hosts):
                dropped_duplicates += 1
                continue

            attempted += 1
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
                            "timeout_seconds": min(timeout_seconds, 60.0),
                        }
                    )
                    if _debug_search_enabled():
                        logger.info(
                            "internet_search_pdf_extract",
                            url=url,
                            result=pdf_extract,
                        )
                    if pdf_extract.get("search_mode_used") != "semantic":
                        raise ValueError(
                            pdf_extract.get("notes")
                            or "Semantic PDF search unavailable."
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
                            "snippet": snippet_text,
                            "score": float(best.get("score", 0.0) or 0.0),
                            "source": "semantic_pdf",
                        }
                    )
                    if host:
                        seen_success_hosts.add(host)
                    continue

                ai_text, scrape_meta = _scrapfly_fetch_ai_data(
                    url=url,
                    timeout_seconds=timeout_seconds,
                    extraction_model=extraction_model,
                )
                if not ai_text:
                    raise ValueError("Scrapfly AI extraction returned empty content.")

                snippets, note = _semantic_search_text(
                    text=ai_text,
                    query_text=query,
                    max_matches=max_snippets_per_url,
                    context_chars=context_chars,
                    min_similarity=min_similarity,
                )
                if not snippets:
                    raise ValueError(
                        note or "No semantic matches found in extracted AI content."
                    )
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
                        "snippet": snippet_text,
                        "score": float(best.get("score", 0.0) or 0.0),
                        "source": "semantic_ai",
                    }
                )
                if host:
                    seen_success_hosts.add(host)
            except Exception as exc:
                error_msg = str(exc)
                errors.append(f"{url}: {error_msg}")
                if host:
                    seen_failed_hosts.add(host)

    _process_candidates(candidates)

    if len(results) < desired_results:
        exclusion_hosts = sorted(seen_success_hosts | seen_failed_hosts)
        exclusion_terms = " ".join(f"-site:{host}" for host in exclusion_hosts if host)
        if exclusion_terms:
            secondary_query = f"{query} {exclusion_terms}".strip()
            secondary_results = _handle_kagi_search(
                {
                    "search_term": secondary_query,
                    "result_num": search_limit,
                }
            )
            secondary_items = secondary_results.get("results")
            if isinstance(secondary_items, list) and secondary_items:
                used_secondary_search = True
                secondary_candidates = _collect_candidates(secondary_items)
                _process_candidates(secondary_candidates)

    if _debug_search_enabled():
        logger.info(
            "internet_search_final_output",
            query=query,
            results=results,
            attempted=attempted,
            errors=errors,
            dropped_duplicates=dropped_duplicates,
            used_secondary_search=used_secondary_search,
        )
    return results


def internet_search(
    query: str,
    extraction_model: str = "article",
) -> list[dict[str, Any]]:
    """Evidence-driven web search with semantic relevance ranking.

    WHEN TO USE:
    - You need current information not in your training data
    - User asks about recent events, news, or real-time data
    - You need to verify facts that may have changed
    - User requests information from specific websites or sources
    - Query contains time-sensitive terms like "latest", "current", "recent", "today"

    WHY TO USE: Provides verified, sourced information from the public web with
    semantic relevance ranking. Results are limited to the top 3 successful
    unique domains using Scrapfly AI extraction.

    WHEN NOT TO USE:
    - Information is well-established and unlikely to change (historical facts, constants)
    - User is asking for analysis or opinion rather than external facts
    - Query is about personal/private data not on public web

    CRITICAL WORKFLOW:
    1. If query contains relative time ("this year", "today"), call current_time FIRST
    2. Then call this tool with resolved dates in the query

    GOOD SEARCH QUERY GUIDANCE (IMPORTANT):
    - Use specific, descriptive keywords (prefer proper nouns and domain terminology).
      Add context if needed (location, product, industry, timeframe).
    - Keep queries concise: start with essential content words (often 2-6 terms).
      Avoid filler like "how do I", "list of", etc.
    - Think like the source: use terms an expert page would use; try synonyms.
    - Avoid stuffing the query with long lists of keywords; overly long queries reduce relevance.
    - ALWAYS start with a broad query (no domain filters). Let Kagi surface a diverse mix of
      sources — news sites, blogs, papers, documentation — then refine if needed.
    - NEVER use site: filters. They collapse results to a single domain's pages, which are
      often dynamic/JS-rendered and fail to scrape. Broad queries surface individual article
      and abstract pages that scrape reliably.
    - Use operators when helpful (syntax matters: no spaces after colons):
      - Quotes for exact phrases: "admission requirements"
      - Exclude terms: jaguar -car
      - File type filters: cybersecurity report filetype:pdf
      - Alternatives: (college OR university) "admission requirements"   (OR must be capitalized)
      - Wildcard for unknown words: "the * of money"
      - Numeric ranges: Olympics 2000..2010
      - Date filters (when supported): electric car innovations after:2020 before:2023
    - Iterate gradually: start broad, inspect results, then refine.

    EXAMPLES:
    - OK: "What is the current GDP of Japan?" -> use this tool
    - OK: "Latest news about AI regulation" -> use this tool
    - NO: "What is the Pythagorean theorem?" -> do not use (timeless fact)
    - NO: "What do you think about climate policy?" -> do not use (asking for opinion)

    EXTRACTION MODEL (required; defaults to "article" if omitted):
    Provide `extraction_model` to choose a Scrapfly AI extraction schema.
    Use ONLY the allowed values below. Unsupported values will fail the tool.

    Allowed values (choose the closest fit):
    - article: News/blog content pages (use this for weather pages)
    - event: Events with dates/locations
    - food_recipe: Recipe pages
    - hotel / hotel_listing: Single hotel or hotel list pages
    - job_posting / job_listing: Individual job posts or job lists
    - organization: Company/organization profiles
    - product / product_listing: Product pages or listings
    - real_estate_property / real_estate_property_listing: Real estate details or lists
    - review_list: Review aggregator pages
    - social_media_post: Individual social posts
    - software: Software product pages
    - stock: Stock/market pages
    - vehicle_ad: Vehicle listings/ads
    - search_engine_results: SERP-like pages

    IMPORTANT:
    - Do NOT invent new extraction_model values. If unsure, use "article".
    - All other tool settings (result count, similarity thresholds, timeouts) are
      fixed defaults controlled by the tool implementation.
    """
    payload: dict[str, Any] = {
        "query": query,
        "extraction_model": extraction_model or "article",
    }
    return _handle_internet_search(payload)
