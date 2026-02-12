#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
import time
from pathlib import Path

from anymind.tools.core_tools import internet_search
from anymind.runtime.logging import configure_logging

_ROOT = Path(__file__).resolve().parents[1]
_ONNX_DIR = _ROOT / "onnx_assets_out"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test internet_search tool directly.")
    parser.add_argument("query", help="Search query text")
    parser.add_argument(
        "--extraction-model",
        default="article",
        help="Scrapfly AI extraction schema (defaults to article).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON output instead of a formatted summary.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging for internet_search stages.",
    )
    return parser.parse_args()


def main() -> int:
    if _ONNX_DIR.exists():
        os.environ.setdefault("ONNX_MODEL_PATH", str(_ONNX_DIR / "model.onnx"))
        os.environ.setdefault("ONNX_TOKENIZER_PATH", str(_ONNX_DIR / "tokenizer.json"))

    args = _parse_args()
    if args.debug:
        os.environ["ANYMIND_DEBUG_SEARCH"] = "1"
        run_id = f"internet_search-{uuid.uuid4().hex}"
        log_path = os.environ.get("ANYMIND_LOG_PATH")
        resolved = configure_logging("INFO", log_path=log_path, run_id=run_id)
        print(f"[debug] internet_search logs: {resolved}")
    start_time = time.perf_counter()
    try:
        result = internet_search(
            query=args.query,
            extraction_model=args.extraction_model,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        print(f"[timing] internet_search_total_seconds={elapsed:.2f}", file=sys.stderr)
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start_time
    print(f"[timing] internet_search_total_seconds={elapsed:.2f}", file=sys.stderr)

    if args.raw:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    print("== internet_search ==")
    print(f"total_seconds: {elapsed:.2f}")
    if isinstance(result, dict):
        print(f"query: {result.get('query')}")
        print(f"attempted: {result.get('urls_attempted')}")
        print(f"succeeded: {result.get('urls_succeeded')}")
        notes = result.get("notes")
        if notes:
            print(f"notes: {notes}")
        results = result.get("results") or []
    else:
        results = result
        print(f"results: {len(results)}")
    print()

    print("== results ==")
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            print(f"[{idx}] {item}")
            continue
        title = item.get("title") or ""
        url = item.get("url") or ""
        score = item.get("score")
        print(f"[{idx}] {title} — {url} (score={score})")
        snippet = item.get("snippet")
        if snippet:
            print(snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
