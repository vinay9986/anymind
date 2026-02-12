from __future__ import annotations

import re

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

_ANCHOR_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z0-9]{3,}", re.IGNORECASE)


def _query_tokens(query_text: str) -> set[str]:
    tokens = {t for t in _TOKEN_RE.findall((query_text or "").lower())}
    return {t for t in tokens if t not in _STOPWORDS}


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
        t.strip().lower() for t in _ANCHOR_TOKEN_RE.findall(query_text) if t.strip()
    ]
    if not raw_tokens:
        return None

    best: tuple[int, int] | None = None
    for token in raw_tokens:
        idx = haystack.find(token)
        if idx >= 0:
            best = (idx, len(token))
            break

    return best


def _slice_around_center(*, text: str, center: int, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""

    max_chars = int(max_chars or 0)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned

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
