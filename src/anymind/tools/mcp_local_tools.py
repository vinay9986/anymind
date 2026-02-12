import os

from mcp.server.fastmcp import FastMCP

from anymind.tools.core_pdf import pdf_extract_text
from anymind.tools.core_search import internet_search
from anymind.tools.core_time import current_time
from anymind.runtime.logging import configure_logging

_log_path = os.getenv("ANYMIND_LOG_PATH")
_log_dir = os.getenv("ANYMIND_LOG_DIR")
if _log_path or _log_dir:
    configure_logging(
        os.getenv("ANYMIND_LOG_LEVEL", "INFO"),
        log_path=_log_path,
        run_id=os.getenv("ANYMIND_LOG_RUN_ID"),
    )

mcp = FastMCP("local-tools")

mcp.tool()(current_time)
mcp.tool()(internet_search)
mcp.tool()(pdf_extract_text)


if __name__ == "__main__":
    mcp.run()
