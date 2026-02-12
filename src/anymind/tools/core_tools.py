from __future__ import annotations

from typing import Any

from anymind.tools.core_pdf import pdf_extract_text
from anymind.tools.core_search import internet_search
from anymind.tools.core_time import current_time


def register_core_tools(mcp: Any) -> None:
    """Register core tools on a FastMCP server."""
    mcp.tool()(current_time)
    mcp.tool()(internet_search)
    mcp.tool()(pdf_extract_text)
