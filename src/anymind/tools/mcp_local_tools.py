from mcp.server.fastmcp import FastMCP

from anymind.tools.core_tools import register_core_tools

mcp = FastMCP("local-tools")

register_core_tools(mcp)


if __name__ == "__main__":
    mcp.run()
