"""Check that the MCP route returns exactly what the direct Python route returns.

    python scripts/test_mcp.py [TICKER] [AS_OF]

Starts the three MCP servers as child processes, calls each tool both ways, and compares.
search_filings needs Ollama with the embedding model; it is skipped if that fails.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import mcp_client, tools

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
as_of = sys.argv[2] if len(sys.argv) > 2 else "2026-11-17"

t0 = time.time()
mcp_client.toolbox()          # starts the three server processes now (otherwise lazily)
mcp = tools.make_tools(as_of, backend="mcp")
print(f"MCP servers started in {time.time() - t0:.1f}s")
direct = tools.make_tools(as_of, backend="direct")
CALLS = [("get_financials", {"ticker": ticker}), ("list_filings", {"ticker": ticker}),
         ("get_technicals", {"ticker": ticker}), ("get_macro", {"ticker": ticker}),
         ("search_filings", {"ticker": ticker, "query": "principal risks"})]
norm = lambda o: json.loads(json.dumps(o, default=str, sort_keys=True))
bad = 0
for name, args in CALLS:
    try:
        a, b = norm(direct[name](**args)), norm(mcp[name](**args))
    except Exception as e:
        print(f"{name:15s} SKIPPED/ERROR  {type(e).__name__}: {str(e)[:100]}")
        bad += name != "search_filings"
        continue
    same = a == b
    bad += not same
    print(f"{name:15s} {'identical' if same else 'DIFFERENT'}  ({len(json.dumps(b))} chars)")
    if not same:
        print("   direct:", json.dumps(a)[:200], "\n   mcp:   ", json.dumps(b)[:200])
# the model can never pick the date: a wrong as_of in the arguments is dropped
try:
    guard = mcp["get_technicals"](ticker=ticker, as_of="2099-01-01")
    ok = norm(guard) == norm(direct["get_technicals"](ticker=ticker))
except Exception as e:
    ok = False
    print("guard error:", type(e).__name__, str(e)[:100])
print("as_of guard:", "ok" if ok else "FAILED")
sys.exit(1 if bad or not ok else 0)
