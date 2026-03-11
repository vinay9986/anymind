import os

from mcp.server.fastmcp import FastMCP

from anymind.runtime.logging import configure_logging
from anymind.tools.core_tools import register_core_tools

_log_path = os.getenv("ANYMIND_LOG_PATH")
_log_dir = os.getenv("ANYMIND_LOG_DIR")
if _log_path or _log_dir:
    configure_logging(
        os.getenv("ANYMIND_LOG_LEVEL", "INFO"),
        log_path=_log_path,
        run_id=os.getenv("ANYMIND_LOG_RUN_ID"),
    )

mcp = FastMCP("local-tools")

register_core_tools(mcp)


if __name__ == "__main__":
    mcp.run()
