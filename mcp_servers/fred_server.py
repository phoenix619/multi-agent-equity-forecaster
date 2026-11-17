"""MCP server: FRED macro series (read-only). Run: python mcp_servers/fred_server.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from forecaster.data import fred

mcp = FastMCP("fred")


@mcp.tool()
def get_macro_series(series_id: str, as_of: str) -> dict:
    """A FRED series (for example DCOILWTICO oil, DGS10 10-year yield, CPIAUCSL
    inflation) as known on as_of, with 30 and 90 day percent changes."""
    return fred.get_series(series_id, as_of)


@mcp.tool()
def get_macro_snapshot(as_of: str) -> dict:
    """Oil, 10-year yield, fed funds rate and CPI as of a date."""
    return fred.macro_snapshot(as_of)


if __name__ == "__main__":
    mcp.run()
