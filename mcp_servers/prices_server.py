"""MCP server: daily prices (read-only). Run: python mcp_servers/prices_server.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from forecaster.data import prices

mcp = FastMCP("prices")


@mcp.tool()
def get_price_history(ticker: str, as_of: str, days: int = 400) -> dict:
    """Daily adjusted close and volume ending on as_of (YYYY-MM-DD). Never includes
    data after as_of."""
    df = prices.get_prices(ticker, as_of, days)
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return {"ticker": ticker, "as_of": as_of,
            "dates": df["Date"].tolist(), "close": df["Close"].round(4).tolist(),
            "volume": df["Volume"].tolist()}


if __name__ == "__main__":
    mcp.run()
