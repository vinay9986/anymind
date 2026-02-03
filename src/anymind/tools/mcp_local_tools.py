from datetime import datetime

from mcp.server.fastmcp import FastMCP

from anymind.tools.core_tools import register_core_tools

mcp = FastMCP("local-tools")

register_core_tools(mcp)


@mcp.tool()
def get_current_time() -> str:
    """Return the current local time in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
