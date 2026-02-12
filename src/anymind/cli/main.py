from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from anymind.config.loader import load_model_config, load_model_config_from_path
from anymind.config.schemas import ModelConfig
from anymind.runtime.logging import configure_logging
from anymind.runtime.orchestrator import BudgetExceededError, Orchestrator

app = typer.Typer(add_completion=False, invoke_without_command=True)
console = Console()


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _format_evidence(evidence: list[dict[str, object]]) -> str:
    max_total = int(os.getenv("ANYMIND_EVIDENCE_MAX_CHARS", "8000"))
    max_item = int(os.getenv("ANYMIND_EVIDENCE_ITEM_MAX_CHARS", "2000"))
    lines: list[str] = []
    total = 0
    for record in evidence:
        record_id = str(record.get("id", "")).strip()
        tool = str(record.get("tool", "")).strip()
        content = str(record.get("content", "")).strip()
        if max_item > 0:
            content = _truncate_text(content, max_item)
        label = f"[{record_id}] {tool}".strip()
        if label == "[]":
            label = "[evidence]"
        line = f"{label}: {content}" if content else label
        addition = len(line) + (1 if lines else 0)
        if max_total > 0 and total + addition > max_total:
            remaining = max_total - total
            if remaining > 0:
                lines.append(_truncate_text(line, remaining))
            break
        lines.append(line)
        total += addition
    return "\n".join(lines)


def _run_chat(
    *,
    agent: str,
    config_path: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    temperature: Optional[float],
    tools_policy: Optional[str],
    tools_enabled: Optional[bool],
    thread_id: Optional[str],
    log_level: str,
    query: Optional[str],
) -> None:
    active_thread_id = thread_id or f"{agent}-{uuid.uuid4().hex}"
    configure_logging(
        log_level,
        log_path=os.getenv("ANYMIND_LOG_PATH"),
        run_id=active_thread_id,
    )

    async def _run() -> None:
        orchestrator = Orchestrator()
        if config_path:
            model_config = load_model_config_from_path(Path(config_path))
        else:
            model_config = load_model_config()
        model_config = _apply_overrides(
            model_config,
            model=model,
            provider=provider,
            temperature=temperature,
            tools_policy=tools_policy,
            tools_enabled=tools_enabled,
            thread_id=thread_id,
        )
        session = await orchestrator.create_session(
            agent_name=agent, model_config=model_config
        )
        try:
            if query:
                try:
                    result = await orchestrator.run_turn(
                        session, user_input=query, thread_id=active_thread_id
                    )
                except BudgetExceededError as exc:
                    console.print(str(exc), style="bold red")
                else:
                    console.print(
                        Panel(
                            result["response"],
                            title="assistant",
                            border_style="blue",
                        )
                    )
                    evidence = result.get("evidence") or []
                    if evidence:
                        console.print(
                            Panel(
                                _format_evidence(evidence),
                                title="evidence",
                                border_style="magenta",
                            )
                        )
            else:
                console.print(
                    Panel(
                        Text(
                            "AnyMind chat ready. Type /exit to quit.",
                            style="bold green",
                        ),
                        title="AnyMind",
                        border_style="green",
                    )
                )
                while True:
                    user_input = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in {"/exit", "/quit"}:
                        break
                    try:
                        result = await orchestrator.run_turn(
                            session, user_input=user_input, thread_id=active_thread_id
                        )
                    except BudgetExceededError as exc:
                        console.print(str(exc), style="bold red")
                        break
                    console.print(
                        Panel(
                            result["response"],
                            title="assistant",
                            border_style="blue",
                        )
                    )
                    evidence = result.get("evidence") or []
                    if evidence:
                        console.print(
                            Panel(
                                _format_evidence(evidence),
                                title="evidence",
                                border_style="magenta",
                            )
                        )
        finally:
            summary = orchestrator.session_summary(session)
            if summary["models"]:
                table = Table(
                    title="Session Summary",
                    show_header=True,
                    header_style="bold",
                )
                table.add_column("Model")
                table.add_column("Input Tokens")
                table.add_column("Output Tokens")
                table.add_column("Total Tokens")
                table.add_column(f"Input Cost ({summary['currency']})")
                table.add_column(f"Output Cost ({summary['currency']})")
                table.add_column(f"Total Cost ({summary['currency']})")
                for model_name, totals in summary["models"].items():
                    table.add_row(
                        model_name,
                        str(totals["input_tokens"]),
                        str(totals["output_tokens"]),
                        str(totals["total_tokens"]),
                        f"{totals['input_cost']:.6f}",
                        f"{totals['output_cost']:.6f}",
                        f"{totals['total_cost']:.6f}",
                    )
                overall = summary["total"]
                table.add_row(
                    "TOTAL",
                    str(overall["input_tokens"]),
                    str(overall["output_tokens"]),
                    str(overall["total_tokens"]),
                    f"{overall['input_cost']:.6f}",
                    f"{overall['output_cost']:.6f}",
                    f"{overall['total_cost']:.6f}",
                )
                console.print(table)
            else:
                console.print("No usage recorded.", style="dim")
            await session.close()

    asyncio.run(_run())


@app.callback()
def main(
    ctx: typer.Context,
    agent: str = typer.Option("research_agent", "--agent"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    model: Optional[str] = typer.Option(None, "--model"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    tools_policy: Optional[str] = typer.Option(None, "--tools-policy"),
    tools_enabled: Optional[bool] = typer.Option(None, "--tools-enabled/--no-tools"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    query: Optional[str] = typer.Option(None, "--query", "-q"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    _run_chat(
        agent=agent,
        config_path=config_path,
        model=model,
        provider=provider,
        temperature=temperature,
        tools_policy=tools_policy,
        tools_enabled=tools_enabled,
        thread_id=thread_id,
        log_level=log_level,
        query=query,
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Start the FastAPI server (Swagger at /docs)."""
    from anymind.api.app import create_app
    import uvicorn

    app_instance = create_app()
    uvicorn.run(app_instance, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    app()


def _apply_overrides(
    model_config: ModelConfig,
    *,
    model: Optional[str],
    provider: Optional[str],
    temperature: Optional[float],
    tools_policy: Optional[str],
    tools_enabled: Optional[bool],
    thread_id: Optional[str],
) -> ModelConfig:
    if model is not None:
        model_config.model = model
    if provider is not None:
        model_config.model_provider = provider
    if temperature is not None:
        params = dict(model_config.model_parameters)
        params["temperature"] = temperature
        model_config.model_parameters = params
    if tools_policy is not None:
        model_config.tools_policy = tools_policy
    if tools_enabled is not None:
        model_config.tools_enabled = tools_enabled
    if thread_id is not None:
        model_config.thread_id = thread_id
    return model_config
