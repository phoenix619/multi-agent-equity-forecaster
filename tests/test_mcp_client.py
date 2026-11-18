import json
from types import SimpleNamespace as NS

import pytest

from forecaster import mcp_client, tools


def _result(texts=(), structured=None, error=False):
    return NS(content=[NS(text=t) for t in texts], structuredContent=structured, isError=error)


def test_decode_dict_text():
    assert mcp_client.decode(_result([json.dumps({"a": 1})])) == {"a": 1}


def test_decode_structured_dict_and_wrapped_list():
    assert mcp_client.decode(_result(["{}"], structured={"a": 2})) == {"a": 2}
    assert mcp_client.decode(_result(["x"], structured={"result": [1, 2]})) == [1, 2]


def test_decode_list_as_one_block_per_item():
    out = mcp_client.decode(_result([json.dumps({"i": 1}), json.dumps({"i": 2})]))
    assert out == [{"i": 1}, {"i": 2}]


def test_decode_error_raises():
    with pytest.raises(RuntimeError):
        mcp_client.decode(_result(["boom"], error=True))


class FakeBox:
    def __init__(self):
        self.calls = []

    def call(self, server, tool, args):
        self.calls.append((server, tool, args))
        return {"ok": True}


def test_as_of_is_injected_by_code_and_model_value_is_ignored():
    box = FakeBox()
    t = mcp_client.make_mcp_tools("2026-01-12", box=box)
    t["get_financials"](ticker="AAPL", as_of="2030-01-01")       # model tries to pick a date
    t["search_filings"](ticker="AAPL", query="risks")
    assert box.calls[0] == ("edgar", "get_financials", {"ticker": "AAPL", "as_of": "2026-01-12"})
    assert box.calls[1] == ("edgar", "search_filings",
                            {"ticker": "AAPL", "query": "risks", "as_of": "2026-01-12"})


def test_every_model_tool_has_an_mcp_route():
    names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert names == set(mcp_client.ROUTES) == set(tools.make_direct_tools("2026-01-12"))


def test_baseline_schemas_unchanged_by_new_tool():
    assert [s["function"]["name"] for s in tools.BASELINE_SCHEMAS] == [
        "get_financials", "list_filings", "get_technicals", "get_macro"]


def test_make_tools_selects_backend(monkeypatch):
    marker = {"mcp": True}
    monkeypatch.setattr(mcp_client, "make_mcp_tools", lambda as_of: marker)
    assert tools.make_tools("2026-01-12", backend="mcp") is marker
    monkeypatch.setenv("TOOL_BACKEND", "mcp")
    assert tools.make_tools("2026-01-12") is marker
    monkeypatch.setenv("TOOL_BACKEND", "direct")
    assert "get_financials" in tools.make_tools("2026-01-12")
