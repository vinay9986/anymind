#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from anymind.tools.core_tools import internet_search


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test internet_search tool directly.")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--max-snippets", type=int, default=1)
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON output instead of a formatted summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = internet_search(
            query=args.query,
            max_results=args.max_results,
            max_snippets=args.max_snippets,
            context_chars=args.context_chars,
            min_similarity=args.min_similarity,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    print("== internet_search ==")
    print(f"query: {result.get('query')}")
    print(f"attempted: {result.get('urls_attempted')}")
    print(f"succeeded: {result.get('urls_succeeded')}")
    notes = result.get("notes")
    if notes:
        print(f"notes: {notes}")
    print()

    print("== concat_blob ==")
    print(result.get("concat_blob") or "")
    print()

    print("== citations ==")
    citations = result.get("citations") or []
    for citation in citations:
        cid = citation.get("id")
        title = citation.get("title")
        url = citation.get("url")
        score = citation.get("score")
        print(f"[{cid}] {title} — {url} (score={score})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
