"""MCP server: SEC EDGAR (read-only). Run: python mcp_servers/edgar_server.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from forecaster import rag, tools
from forecaster.data import edgar

mcp = FastMCP("edgar")


@mcp.tool()
def get_financials(ticker: str, as_of: str) -> dict:
    """Latest quarterly and annual revenue, margins, debt and cash flow for a
    ticker, using only filings filed on or before as_of (YYYY-MM-DD). Ratios are
    computed in code. Each figure includes form, period and accession for citation."""
    return tools._trim_financials(edgar.get_financials(ticker, as_of))


@mcp.tool()
def list_filings(ticker: str, as_of: str, limit: int = 4) -> list:
    """Recent 10-K and 10-Q filings for a ticker filed on or before as_of."""
    return edgar.list_filings(ticker, as_of, limit=limit)


@mcp.tool()
def search_filings(ticker: str, as_of: str, query: str) -> dict:
    """Most relevant, non-redundant passages from the latest 10-K and 10-Q (risk factors and
    MD&A) filed on or before as_of, each with filing, accession and section for citation."""
    return rag.search_filings(ticker, as_of, query)


@mcp.resource("filing://{ticker}/{as_of}")
def latest_filings_resource(ticker: str, as_of: str) -> str:
    """Latest filing URLs for a ticker as of a date."""
    rows = edgar.list_filings(ticker, as_of, limit=3)
    return "\n".join(f"{r['form']} {r['period']} filed {r['filed']} {r['url']}" for r in rows)


if __name__ == "__main__":
    mcp.run()
