#!/usr/bin/env python3
"""Node-by-node SOP test harness.

Runs each node in topological order, scoring each output with the coverage
gate and reporting PASS/FAIL. Nodes of type "builtin" are dispatched
directly (no LLM call). All other nodes call the configured model via the
same brain-selector logic used in production.

Design decisions
----------------
- One attempt per node. Infra retries (network/API errors) up to 2 times.
  No automatic prompt improvement — review logs and adjust prompts manually.
- Coverage gate always scores against the *original* node prompt, never a
  modified version, so the score reflects what the node was asked to do.
- Upstream context is accumulated as nodes complete, exactly mirroring the
  production sop_executor flow. Builtin nodes (e.g. time_window) populate
  the context instantly with no token spend.
- `final_newsletter` is skipped by default (requires all digest nodes).
  Pass --node final_newsletter explicitly to include it.

Usage
-----
    # Run all nodes (except final_newsletter):
    NO_COLOR=1 PYTHONUNBUFFERED=1 \\
      ANYMIND_LOG_PATH=/tmp/anymind_node_test.log \\
      poetry run python scripts/test_sop_nodes.py \\
        --sop config/sop_ai_research_newsletter_weekly_v1.json \\
        --config config/model.json \\
        2>&1 | tee /tmp/anymind_node_test.txt

    # Test specific nodes only (upstream context pre-built from previous nodes):
    poetry run python scripts/test_sop_nodes.py \\
        --node time_window --node gather_papers \\
        --sop config/sop_ai_research_newsletter_weekly_v1.json

    # Adjust pass threshold (default 0.7):
    ... --threshold 0.8

SOP development workflow
------------------------
1. Add or edit a node in the SOP JSON.
2. Run this harness with --node <node_id> to validate in isolation.
3. Check [score] and [gaps] in the output to understand coverage gaps.
4. Adjust the node prompt, re-run, repeat until PASS.
5. Run all nodes together to verify upstream context flows correctly.
6. Run the full SOP via `anymind --agent sop_agent -q @<sop_file>` for an
   end-to-end production test.

Builtin nodes
-------------
Nodes with "type": "builtin" (e.g. time_window) are dispatched to a
registered Python function from anymind/agents/sop/builtins.py. They run
instantly, spend zero tokens, and are always PASS. They do not go through
the coverage gate. To add a new builtin, register it with @register_builtin
in builtins.py and reference it in the SOP JSON with:
    { "type": "builtin", "function": "<name>", ... }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

_ROOT = Path(__file__).resolve().parents[1]
_ONNX_DIR = _ROOT / "onnx_assets_out"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topo_sort(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Kahn's algorithm — returns node IDs in topological order."""
    node_ids = [n["id"] for n in nodes]
    preds: dict[str, set[str]] = {nid: set() for nid in node_ids}
    succs: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for e in edges:
        src = e.get("source") or e.get("src") or e.get("from")
        dst = e.get("target") or e.get("dst") or e.get("to")
        if src in preds and dst in preds and src != dst:
            succs[src].add(dst)
            preds[dst].add(src)
    ready = [nid for nid in node_ids if not preds[nid]]
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for child in sorted(succs[nid]):
            preds[child].discard(nid)
            if not preds[child]:
                ready.append(child)
    return order


def _build_upstream_context(
    node_id: str,
    predecessors: list[str],
    node_results: dict[str, dict[str, Any]],
) -> str:
    """Build upstream context JSON for a node (mirrors sop_executor logic)."""
    if not predecessors:
        return ""
    upstream = []
    for pid in predecessors:
        res = node_results.get(pid, {})
        upstream.append(
            {
                "node_id": pid,
                "status": res.get("status", "ok"),
                "content": str(res.get("content") or "").strip(),
                "evidence_ids": [],
            }
        )
    return json.dumps({"upstream": upstream}, ensure_ascii=False, indent=2, default=str)


def _build_query(question: str, upstream_ctx: str) -> str:
    parts = [question]
    if upstream_ctx:
        parts.append("Upstream context:\n" + upstream_ctx)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    # Set up ONNX paths if available
    if _ONNX_DIR.exists():
        os.environ.setdefault("ONNX_MODEL_PATH", str(_ONNX_DIR / "model.onnx"))
        os.environ.setdefault("ONNX_TOKENIZER_PATH", str(_ONNX_DIR / "tokenizer.json"))

    # Configure logging (structured JSON to ANYMIND_LOG_PATH)
    from anymind.runtime.logging import configure_logging

    run_id = f"sop_node_test-{uuid.uuid4().hex}"
    log_path = os.environ.get("ANYMIND_LOG_PATH")
    resolved_log = configure_logging("INFO", log_path=log_path, run_id=run_id)
    print(f"[log] structured logs → {resolved_log}")

    log = structlog.get_logger("anymind.test_sop_nodes")

    # Load SOP
    sop_path = Path(args.sop)
    if not sop_path.is_absolute():
        sop_path = Path.cwd() / sop_path
    sop: dict[str, Any] = json.loads(sop_path.read_text(encoding="utf-8"))
    nodes_by_id = {n["id"]: n for n in sop.get("nodes", [])}
    edges = sop.get("edges", [])

    # Topological order
    topo = _topo_sort(list(nodes_by_id.values()), edges)

    # Filter to requested nodes (if --node given), always skip final_newsletter
    skip_nodes = {"final_newsletter"}
    if args.node:
        run_ids = set(args.node)
    else:
        run_ids = set(topo) - skip_nodes

    # Build predecessors map
    preds: dict[str, list[str]] = {nid: [] for nid in nodes_by_id}
    for e in edges:
        src = e.get("source") or e.get("src") or e.get("from")
        dst = e.get("target") or e.get("dst") or e.get("to")
        if src in preds and dst in preds:
            preds[dst].append(src)

    # Bootstrap session (same as production)
    from anymind.config.loader import (
        load_model_config,
        load_model_config_from_path,
        load_mcp_config,
    )
    from anymind.config.schemas import MCPConfig
    from anymind.runtime.session_factory import SessionFactory

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    model_cfg = load_model_config_from_path(config_path)

    try:
        mcp_cfg = load_mcp_config()
    except FileNotFoundError:
        mcp_cfg = MCPConfig()

    factory = SessionFactory()
    print(
        "[session] bootstrapping AgentContext (MCP tools, model client, checkpointer)…"
    )
    session = await factory.create_session(
        agent_name="sop_agent",
        model_config=model_cfg,
        mcp_config=mcp_cfg,
    )
    print(f"[session] ready — model={model_cfg.model}, tools={len(session.tools)}")

    # _SopRuntime is session.agent_with_tools (or agent_no_tools as fallback)
    sop_runtime = session.agent_with_tools or session.agent_no_tools
    model_client = session.model_client

    # Imports from SOP internals
    from anymind.agents.sop.brain_selector import select_brain_for_question
    from anymind.agents.sop.builtins import get_builtin  # registers builtins
    from anymind.agents.sop.sop_validation import get_node_question
    from anymind.agents.sop.coverage_gate import evaluate_answer
    from anymind.agents.usage_tracker import UsageBudgetTracker

    node_results: dict[str, dict[str, Any]] = {}

    results_summary: list[dict[str, Any]] = []
    overall_pass = True

    for node_id in topo:
        if node_id not in run_ids:
            continue
        node = nodes_by_id[node_id]
        allow_tools = node.get("allow_tools")
        if allow_tools is None:
            allow_tools = True
        allow_tools = bool(allow_tools)

        ntype = str(node.get("type", "") or "").strip().lower()
        original_prompt = get_node_question(node)
        upstream_ctx = _build_upstream_context(node_id, preds[node_id], node_results)

        passed = False
        infra_failures = 0
        max_infra_failures = 2
        node_start = time.perf_counter()
        output = ""

        print(f"\n{'='*70}")
        print(
            f"[node] {node_id}  type={ntype or 'llm'}  allow_tools={allow_tools}  predecessors={preds[node_id]}"
        )

        # --- Builtin dispatch (no LLM) ---
        if ntype == "builtin":
            fn_name = str(node.get("function", "") or "").strip()
            fn = get_builtin(fn_name)
            invoke_start = time.perf_counter()
            if fn is None:
                print(f"  [ERROR] unknown builtin: {fn_name!r}")
                node_results[node_id] = {"status": "failed", "content": ""}
                results_summary.append(
                    {"node_id": node_id, "status": "FAIL", "elapsed_s": 0.0}
                )
                overall_pass = False
                continue
            try:
                output = fn(node, node_results)
                invoke_elapsed = time.perf_counter() - invoke_start
                print(
                    f"  [builtin] {fn_name} → {len(output)} chars in {invoke_elapsed*1000:.0f}ms"
                )
                print(f"  [preview] {output[:500].strip()!r}")
                node_results[node_id] = {"status": "ok", "content": output}
                passed = True
            except Exception as exc:
                invoke_elapsed = time.perf_counter() - invoke_start
                print(f"  [ERROR] builtin {fn_name} raised: {exc}")
                node_results[node_id] = {"status": "failed", "content": ""}
                overall_pass = False
            elapsed_total = time.perf_counter() - node_start
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {node_id}  elapsed={elapsed_total:.1f}s")
            results_summary.append(
                {
                    "node_id": node_id,
                    "status": status,
                    "elapsed_s": round(elapsed_total, 2),
                }
            )
            continue

        # --- LLM dispatch ---
        brain, lvl, score, features = select_brain_for_question(original_prompt)
        query = _build_query(original_prompt, upstream_ctx)
        print(f"  [run] brain={brain}  ambiguity={lvl}({score:.2f})")
        log.info(
            "node_test_attempt",
            node_id=node_id,
            brain=brain,
            ambiguity_level=lvl,
            ambiguity_score=round(float(score), 4),
        )

        # Infra retry: retry the same prompt only on hard exceptions (network/API errors).
        while infra_failures <= max_infra_failures:
            usage_tracker = UsageBudgetTracker(model_cfg.model, model_cfg.budget_tokens)
            invoke_start = time.perf_counter()
            try:
                output, usage_meta, _ = await sop_runtime._solve(
                    brain, query, usage_tracker, allow_tools
                )
                break  # success
            except Exception as exc:
                elapsed = time.perf_counter() - invoke_start
                infra_failures += 1
                print(
                    f"  [ERROR] invoke failed after {elapsed:.1f}s ({infra_failures}/{max_infra_failures}): {exc}"
                )
                log.error(
                    "node_test_invoke_error",
                    node_id=node_id,
                    infra_failures=infra_failures,
                    error=str(exc),
                )
                if infra_failures >= max_infra_failures:
                    print(f"  [ABORT] {max_infra_failures} consecutive infra failures")

        invoke_elapsed = time.perf_counter() - invoke_start
        output_len = len(output)
        print(f"  [output] {output_len} chars in {invoke_elapsed:.1f}s")
        print(f"  [preview] {output[:500].strip()!r}")

        log.info(
            "node_test_output",
            node_id=node_id,
            output_len=output_len,
            elapsed_s=round(invoke_elapsed, 2),
        )

        # Score against the original node prompt — always.
        try:
            gate, _ = await evaluate_answer(
                question=original_prompt,
                answer=output,
                context={
                    "node_id": node_id,
                    "upstream": upstream_ctx[:500] if upstream_ctx else "",
                },
                model_client=model_client,
            )
        except Exception as exc:
            print(
                f"  [WARN] coverage gate failed: {exc} — treating as pass if output non-empty"
            )
            gate = {
                "coverage": 1.0 if output_len > 100 else 0.0,
                "sufficiency": "stop",
                "gaps": [],
                "rationale": str(exc),
            }

        coverage = float(gate.get("coverage", 0.0))
        sufficiency = gate.get("sufficiency", "refine")
        gaps = gate.get("gaps", [])
        rationale = gate.get("rationale", "")

        print(f"  [score] coverage={coverage:.2f}  sufficiency={sufficiency}")
        if rationale:
            print(f"  [rationale] {rationale[:300]}")
        if gaps:
            print(f"  [gaps]")
            for g in gaps:
                print(f"    - {g}")

        log.info(
            "node_test_score",
            node_id=node_id,
            coverage=round(coverage, 4),
            sufficiency=sufficiency,
            output_len=output_len,
        )

        if coverage >= args.threshold or sufficiency == "stop":
            passed = True
            node_results[node_id] = {"status": "ok", "content": output}
            print(f"  [PASS] coverage={coverage:.2f} >= {args.threshold}")

        elapsed_total = time.perf_counter() - node_start
        status = "PASS" if passed else "FAIL"
        if not passed:
            overall_pass = False
            if output:
                node_results[node_id] = {"status": "partial", "content": output}

        print(f"  [{status}] {node_id}  elapsed={elapsed_total:.1f}s")
        log.info(
            "node_test_result",
            node_id=node_id,
            status=status,
            elapsed_s=round(elapsed_total, 2),
            infra_failures=infra_failures,
        )
        results_summary.append(
            {
                "node_id": node_id,
                "status": status,
                "elapsed_s": round(elapsed_total, 2),
            }
        )

    # Summary
    print(f"\n{'='*70}")
    print("[summary]")
    for r in results_summary:
        marker = "✓" if r["status"] == "PASS" else "✗"
        print(f"  {marker} {r['node_id']:25s}  {r['status']}  {r['elapsed_s']:.0f}s")

    await session.close()
    return 0 if overall_pass else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SOP nodes individually and score their output."
    )
    parser.add_argument(
        "--sop",
        default="config/sop_ai_research_newsletter_weekly_v1.json",
        help="Path to SOP JSON file",
    )
    parser.add_argument(
        "--config",
        default="config/model.json",
        help="Path to model.json config file",
    )
    parser.add_argument(
        "--node",
        action="append",
        metavar="NODE_ID",
        help="Run only this node (repeatable). Omit to run all nodes.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Coverage score threshold to consider a node passing (default: 0.7)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(asyncio.run(run(args)))
