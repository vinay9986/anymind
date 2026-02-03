from __future__ import annotations

import asyncio
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

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def chat(
    agent: str = typer.Option("chat_agent", "--agent"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    model: Optional[str] = typer.Option(None, "--model"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    tools_policy: Optional[str] = typer.Option(None, "--tools-policy"),
    tools_enabled: Optional[bool] = typer.Option(None, "--tools-enabled/--no-tools"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run an interactive chat session."""
    configure_logging(log_level)

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
                        session, user_input=user_input, thread_id=thread_id
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
                if result.get("tokens"):
                    table = Table(
                        title="Usage (Tokens)", show_header=True, header_style="bold"
                    )
                    table.add_column("Model")
                    table.add_column("Input Tokens")
                    table.add_column("Output Tokens")
                    table.add_column("Total Tokens")
                    for model_name, totals in result["tokens"].items():
                        table.add_row(
                            model_name,
                            str(totals["input_tokens"]),
                            str(totals["output_tokens"]),
                            str(totals["total_tokens"]),
                        )
                    console.print(table)
        finally:
            await session.close()

    asyncio.run(_run())


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
