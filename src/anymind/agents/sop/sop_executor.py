from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import structlog

from anymind.runtime.evidence import EvidenceLedger
from anymind.agents.sop.brain_selector import select_brain_for_question
from anymind.agents.sop.coverage_gate import evaluate_answer
from anymind.agents.sop.sop_validation import get_node_question


@dataclass(frozen=True)
class SopExecutionConfig:
    max_concurrency: int = 3
    node_context_max_chars: int = 4000
    node_output_preview_chars: int = 2000
    include_evidence: bool = True
    refinement_enabled: bool = True
    refinement_coverage_threshold: float = 0.7
    evidence_max_chars: int = 8000
    evidence_item_max_chars: int = 2000
    trace_steps: bool = True
    trace_max_chars: int = 300


SolverFn = Callable[
    [str, str, bool],
    Awaitable[tuple[str, dict[str, dict[str, int]] | None, list[Any]]],
]
UsageFn = Callable[[dict[str, dict[str, int]] | None], None]
BudgetFn = Callable[[], bool]


log = structlog.get_logger("anymind.sop")


def _build_graph(
    sop: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    nodes: dict[str, dict[str, Any]] = {}
    for nd in sop.get("nodes", []):
        if isinstance(nd, dict) and isinstance(nd.get("id"), str):
            nodes[nd["id"]] = nd

    preds: dict[str, set[str]] = {nid: set() for nid in nodes.keys()}
    succs: dict[str, set[str]] = {nid: set() for nid in nodes.keys()}
    for ed in sop.get("edges", []):
        if not isinstance(ed, dict):
            continue
        src = ed.get("source") or ed.get("src") or ed.get("from")
        dst = ed.get("target") or ed.get("dst") or ed.get("to")
        if src in nodes and dst in nodes and src != dst:
            succs[src].add(dst)
            preds[dst].add(src)

    return nodes, preds, succs


def _node_type(node: dict[str, Any]) -> str:
    t = node.get("type")
    return str(t or "").lower().strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_upstream_context(
    *,
    node_id: str,
    predecessors: list[str],
    node_results: dict[str, dict[str, Any]],
    max_chars: int,
) -> str:
    if not predecessors:
        return ""

    per_node = max(1, int(max_chars / max(1, len(predecessors))))
    upstream: list[dict[str, Any]] = []
    for pid in predecessors:
        res = node_results.get(pid, {})
        content = str(res.get("content") or "").strip()
        evidence_ids: list[str] = []
        for record in res.get("evidence") or []:
            if hasattr(record, "id"):
                evidence_ids.append(str(record.id))
            elif isinstance(record, dict) and record.get("id"):
                evidence_ids.append(str(record.get("id")))
        upstream.append(
            {
                "node_id": pid,
                "status": res.get("status"),
                "content": _truncate(content, per_node),
                "evidence_ids": evidence_ids,
            }
        )

    rendered = json.dumps(
        {"upstream": upstream}, ensure_ascii=False, indent=2, default=str
    )
    return _truncate(rendered, max_chars)


def _select_final_answer(
    nodes: dict[str, dict[str, Any]], node_results: dict[str, dict[str, Any]]
) -> str:
    output_nodes = [nid for nid, nd in nodes.items() if _node_type(nd) == "output"]
    for nid in output_nodes:
        content = str(node_results.get(nid, {}).get("content") or "").strip()
        if content:
            return content
    for nid in reversed(list(nodes.keys())):
        content = str(node_results.get(nid, {}).get("content") or "").strip()
        if content:
            return content
    return ""


async def execute_sop(
    *,
    sop: dict[str, Any],
    execution_id: str,
    config: SopExecutionConfig,
    solver: SolverFn,
    record_usage: UsageFn,
    budget_exhausted: BudgetFn,
    ledger: EvidenceLedger | None,
    model_client: Any,
    model_name: str,
) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    nodes, preds, succs = _build_graph(sop)
    node_ids = list(nodes.keys())

    pending: set[str] = set(node_ids)
    completed: set[str] = set()
    node_results: dict[str, dict[str, Any]] = {}

    ready: set[str] = {nid for nid in node_ids if not preds.get(nid)}

    start = time.perf_counter()
    round_idx = 0

    while pending and not budget_exhausted():
        if not ready:
            remaining = sorted(pending)
            raise RuntimeError(
                f"SOP execution stalled (cycle or unmet deps). Remaining={remaining}"
            )

        round_idx += 1
        batch = sorted(list(ready))[: max(1, int(config.max_concurrency))]
        for nid in batch:
            ready.discard(nid)

        log.info(
            "sop_round_start",
            execution_id=execution_id,
            round=round_idx,
            batch=batch,
            pending=len(pending),
        )

        async def _run_node(nid: str) -> tuple[str, dict[str, Any]]:
            node = nodes[nid]
            ntype = _node_type(node)
            question = get_node_question(node)
            allow_tools = node.get("allow_tools")
            if allow_tools is None:
                allow_tools = True
            allow_tools = bool(allow_tools)

            if ntype in {"parallel"}:
                return nid, {
                    "status": "ok",
                    "content": "",
                    "node_type": ntype,
                    "algorithm": "none",
                    "skipped": True,
                }

            if ntype == "input":
                explicit = get_node_question(node, allow_fallback=False)
                if not explicit:
                    return nid, {
                        "status": "ok",
                        "content": "",
                        "node_type": ntype,
                        "algorithm": "none",
                        "skipped": True,
                    }
                question = explicit

            if ntype == "input" and not question:
                return nid, {
                    "status": "ok",
                    "content": "",
                    "node_type": ntype,
                    "algorithm": "none",
                }

            if budget_exhausted():
                return nid, {
                    "status": "failed",
                    "content": "",
                    "node_type": ntype,
                    "algorithm": "none",
                    "error": "budget_exhausted",
                }

            brain, lvl, score, features = select_brain_for_question(question)
            predecessors = sorted(list(preds.get(nid, set())))
            ctx = _format_upstream_context(
                node_id=nid,
                predecessors=predecessors,
                node_results=node_results,
                max_chars=config.node_context_max_chars,
            )

            def _build_query(question_text: str) -> str:
                parts: list[str] = [question_text]
                if ctx:
                    parts.append("Upstream context:\n" + ctx)
                return "\n\n".join(parts)

            log.info(
                "sop_node_start",
                execution_id=execution_id,
                node_id=nid,
                node_type=ntype,
                algorithm=brain,
                ambiguity_level=lvl,
                ambiguity_score=round(float(score), 4),
                features=features,
            )

            ledger_start = len(ledger.all()) if ledger is not None else 0

            async def _invoke_once(
                question_text: str,
            ) -> tuple[str, dict[str, dict[str, int]] | None]:
                response, usage_metadata, _ = await solver(
                    brain, _build_query(question_text), allow_tools
                )
                record_usage(usage_metadata)
                return response, usage_metadata

            refinement_enabled = bool(config.refinement_enabled) and ntype not in {
                "input",
                "parallel",
            }
            if not refinement_enabled:
                answer, _ = await _invoke_once(question)
                result: dict[str, Any] = {
                    "status": "ok" if answer else "failed",
                    "content": answer,
                    "node_type": ntype,
                    "algorithm": brain,
                    "ambiguity_level": lvl,
                    "ambiguity_score": float(score),
                }
            else:
                current_question = question
                seen_refinements: set[str] = set()
                history: list[dict[str, Any]] = []
                best_answer = ""
                best_coverage = -1.0
                exit_reason = "budget_exhausted"

                while True:
                    if budget_exhausted():
                        exit_reason = "budget_exhausted"
                        break

                    answer, _ = await _invoke_once(current_question)

                    gate, gate_usage = await evaluate_answer(
                        question=current_question,
                        answer=answer,
                        context={
                            "node_id": nid,
                            "node_type": ntype,
                            "upstream_context": ctx,
                        },
                        model_client=model_client,
                    )
                    for usage in gate_usage:
                        if usage:
                            record_usage({model_name: usage})

                    coverage = float(gate.get("coverage", 0.0))
                    sufficiency = str(
                        gate.get("sufficiency", "refine") or "refine"
                    ).lower()
                    refined_follow_up = str(
                        gate.get("refined_follow_up", "") or ""
                    ).strip()

                    history.append(
                        {
                            "question": current_question,
                            "answer": answer,
                            "coverage": coverage,
                            "sufficiency": sufficiency,
                        }
                    )

                    if coverage >= best_coverage:
                        best_coverage = coverage
                        best_answer = answer

                    if sufficiency == "stop":
                        exit_reason = "coverage_stop"
                        break

                    if best_coverage >= float(config.refinement_coverage_threshold):
                        exit_reason = "coverage_threshold"
                        break

                    if not refined_follow_up:
                        exit_reason = "no_refinement"
                        break

                    lowered = refined_follow_up.lower()
                    if lowered in seen_refinements:
                        exit_reason = "repeat_refinement"
                        break
                    seen_refinements.add(lowered)
                    current_question = refined_follow_up

                final_answer = (best_answer or answer).strip()
                result = {
                    "status": "ok" if final_answer else "failed",
                    "content": final_answer,
                    "node_type": ntype,
                    "algorithm": brain,
                    "ambiguity_level": lvl,
                    "ambiguity_score": float(score),
                    "refinement": {
                        "enabled": True,
                        "coverage": max(
                            0.0, float(best_coverage if best_coverage >= 0 else 0.0)
                        ),
                        "exit_reason": exit_reason,
                        "threshold": float(config.refinement_coverage_threshold),
                        "history": history,
                    },
                }

            node_evidence = []
            if ledger is not None:
                node_evidence = ledger.all()[ledger_start:]
            result["evidence"] = node_evidence

            log.info(
                "sop_node_end",
                execution_id=execution_id,
                node_id=nid,
                status=result.get("status"),
                content_len=len(str(result.get("content") or "")),
                content_preview=(
                    _truncate(
                        str(result.get("content") or ""),
                        config.node_output_preview_chars,
                    )
                    if config.trace_steps
                    else None
                ),
                refinement_exit=(
                    result.get("refinement", {}).get("exit_reason")
                    if isinstance(result.get("refinement"), dict)
                    else None
                ),
            )

            return nid, result

        results = await asyncio.gather(*[_run_node(nid) for nid in batch])

        for nid, result in results:
            node_results[nid] = result
            pending.discard(nid)
            completed.add(nid)

            for child in succs.get(nid, set()):
                if child in pending:
                    if preds[child].issubset(completed):
                        ready.add(child)

    elapsed = time.perf_counter() - start
    final_answer = _select_final_answer(nodes, node_results)
    ok_count = sum(1 for r in node_results.values() if r.get("status") == "ok")
    failed_count = sum(1 for r in node_results.values() if r.get("status") != "ok")

    metrics = {
        "execution_id": execution_id,
        "nodes": len(node_ids),
        "ok": ok_count,
        "failed": failed_count,
        "elapsed_s": round(elapsed, 3),
        "budget_exhausted": bool(budget_exhausted()),
    }

    log.info("sop_execution_complete", **metrics)
    return node_results, final_answer, metrics
