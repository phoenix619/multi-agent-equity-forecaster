"""Call the tools through MCP servers over stdio, with the as-of date injected by code.

Three servers (mcp_servers/): edgar (get_financials, list_filings, search_filings),
prices (get_technicals) and fred (get_macro). They start once per Python process and are
reused. The model supplies only the tool arguments in TOOL_SCHEMAS (ticker, query); this
module adds as_of, so the model cannot read data from the future through MCP either.

The servers run as child processes with this process's environment, so TLS settings
such as REQUESTS_CA_BUNDLE carry over.
"""
import asyncio
import atexit
import json
import os
import sys
import threading

from .config import ROOT

SERVERS = {"edgar": "edgar_server.py", "prices": "prices_server.py", "fred": "fred_server.py"}
# model-facing tool -> (server, MCP tool name)
ROUTES = {"get_financials": ("edgar", "get_financials"),
          "list_filings": ("edgar", "list_filings"),
          "search_filings": ("edgar", "search_filings"),
          "get_technicals": ("prices", "get_technicals"),
          "get_macro": ("fred", "get_macro")}
CALL_TIMEOUT = 900


def decode(result):
    """Turn an MCP tool result into plain Python data (raises on tool errors)."""
    text = "".join(getattr(c, "text", "") for c in (result.content or []))
    if getattr(result, "isError", False):
        raise RuntimeError(text or "MCP tool error")
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc["result"] if set(sc) == {"result"} else sc
    parts = [getattr(c, "text", "") for c in (result.content or [])]
    if len(parts) == 1:
        try:
            return json.loads(parts[0])
        except ValueError:
            return parts[0]
    return [json.loads(p) for p in parts]          # a list result arrives as one block per item


class Toolbox:
    """Owns an event loop in a background thread and one session per server."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._sessions, self._stack = {}, None
        self._run(self._connect())

    def _run(self, coro, timeout=CALL_TIMEOUT):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self):
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        self._stack = AsyncExitStack()
        for name, script in SERVERS.items():
            params = StdioServerParameters(command=sys.executable,
                                           args=[str(ROOT / "mcp_servers" / script)],
                                           env=dict(os.environ))
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[name] = session

    def call(self, server: str, tool: str, args: dict):
        async def go():
            return await self._sessions[server].call_tool(tool, args)
        return decode(self._run(go()))

    def close(self):
        if self._stack is not None:
            try:
                self._run(self._stack.aclose(), timeout=15)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


_BOX = None
_LOCK = threading.Lock()


def toolbox() -> Toolbox:
    global _BOX
    with _LOCK:
        if _BOX is None:
            _BOX = Toolbox()
            atexit.register(_BOX.close)
        return _BOX


def make_mcp_tools(as_of: str, box=None) -> dict:
    """{tool_name: callable(**model_args)} that call MCP with as_of added by code."""
    def bind(name):
        server, mcp_tool = ROUTES[name]

        def call(**args):
            args = {k: v for k, v in args.items() if k != "as_of"}   # never model-chosen
            return (box or toolbox()).call(server, mcp_tool, dict(args, as_of=as_of))
        return call
    return {name: bind(name) for name in ROUTES}
