from __future__ import annotations

import os
from typing import Any, Callable, Mapping


def internet_search_enabled(env: Mapping[str, str] | None = None) -> bool:
    environment = os.environ if env is None else env
    kagi_api_key = str(environment.get("KAGI_API_KEY", "") or "").strip()
    scrapfly_api_key = str(environment.get("SCRAPFLY_API_KEY", "") or "").strip()
    scrapfly_secret_arn = str(
        environment.get("SCRAPFLY_API_KEY_SECRET_ARN", "") or ""
    ).strip()
    return bool(kagi_api_key and (scrapfly_api_key or scrapfly_secret_arn))


def register_core_tools(
    mcp: Any,
    *,
    env: Mapping[str, str] | None = None,
    current_time_tool: Callable[..., Any] | None = None,
    internet_search_tool: Callable[..., Any] | None = None,
    pdf_extract_text_tool: Callable[..., Any] | None = None,
) -> None:
    """Register core tools on a FastMCP server."""
    if current_time_tool is None:
        from anymind.tools.core_time import current_time as current_time_tool

    if pdf_extract_text_tool is None:
        from anymind.tools.core_pdf import pdf_extract_text as pdf_extract_text_tool

    mcp.tool()(current_time_tool)
    if internet_search_enabled(env):
        if internet_search_tool is None:
            from anymind.tools.core_search import (
                internet_search as internet_search_tool,
            )
        mcp.tool()(internet_search_tool)
    mcp.tool()(pdf_extract_text_tool)
