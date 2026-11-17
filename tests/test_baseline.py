import json

from forecaster import baseline


def test_rating_bands():
    r = baseline.rating_from_probability
    assert r(0.75) == "Strong Buy" and r(0.70) == "Strong Buy"
    assert r(0.60) == "Moderate Buy"
    assert r(0.50) == "Hold" and r(0.42) == "Hold"
    assert r(0.35) == "Moderate Sell"
    assert r(0.10) == "Strong Sell"


def test_parse_json_variants():
    assert baseline._parse_json('{"a": 1}') == {"a": 1}
    assert baseline._parse_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert baseline._parse_json("no json here") is None
    assert baseline._parse_json("{broken") is None
    assert baseline._parse_json(None) is None


def _tool(name, result=None, ok=True):
    return {"type": "tool", "tool": name, "ok": ok,
            "result": json.dumps(result) if result is not None else None}


def test_gaps_flag_missing_and_failed_tools():
    trace = [_tool("get_financials", {"quarter": {}, "annual": {}}),
             _tool("get_technicals", ok=False)]
    gaps = baseline.coverage_gaps(trace, "2026-11-16")
    assert "get_technicals returned an error" in gaps
    assert "list_filings was not consulted" in gaps
    assert "get_macro was not consulted" in gaps
    assert any(g.startswith("quarter figures not reported") for g in gaps)
    assert set(baseline.UNAVAILABLE) <= set(gaps)


def test_gaps_flag_stale_quarter():
    q = {"revenue": {"period_end": "2026-01-31"}, "net_income": {}, "operating_cash_flow": {},
         "long_term_debt": {}, "equity": {}}
    fin = {"quarter": q, "annual": {k: {} for k in q}}
    trace = [_tool(n, fin if n == "get_financials" else {}) for n in baseline.EXPECTED_TOOLS]
    gaps = baseline.coverage_gaps(trace, "2026-11-16")
    assert any("days before the as-of date" in g for g in gaps)


def test_gaps_clean_when_all_present():
    q = {"revenue": {"period_end": "2026-09-30"}, "net_income": {}, "operating_cash_flow": {},
         "long_term_debt": {}, "equity": {}}
    fin = {"quarter": q, "annual": {k: {} for k in q}}
    trace = [_tool(n, fin if n == "get_financials" else {}) for n in baseline.EXPECTED_TOOLS]
    gaps = baseline.coverage_gaps(trace, "2026-11-16")
    assert gaps == list(baseline.UNAVAILABLE)
