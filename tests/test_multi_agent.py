"""Multi-agent pipeline tests with a fake LLM and fake tools (no network, no Ollama).

The node functions are driven by a tiny sequential runner that mimics the graph's
state merging; one extra test runs the real LangGraph if it is installed.
"""
import json

import pytest

from forecaster import llm, multi_agent as ma

FIN = {"quarter": {"ratios": {"net_margin": 0.2761},
                   "revenue": {"value": 102466000000, "period_end": "2026-09-30"}},
       "annual": {"ratios": {}}}
TECH = {"rsi14": 55.0, "signal": "ok", "paper_trade": {"entry_zone": [100.0, 102.0],
                                                       "stop_loss": 95.0, "target": 110.0}}
GOOD = {"probability": 0.58, "rationale": "Margins are steady.",
        "evidence": [{"claim": "Net margin is 27.6%", "source": "get_financials"}], "risks": []}
BAD = {"probability": 0.6, "rationale": "Growth is strong.",
       "evidence": [{"claim": "Net margin is 27.6%", "source": "get_financials"},
                    {"claim": "Revenue grew 18.4%", "source": "get_financials"}], "risks": []}
VERY_BAD = {"probability": 0.7, "rationale": "Up 11.1%, 12.2%, 13.3%, 14.4%.",
            "evidence": [{"claim": "x", "source": "s"}], "risks": []}


SEARCH = {"results": [{"text": "Liquidity: we held $2.5 billion in cash and equivalents and "
                               "expect supplier concentration to remain a risk.",
                       "source": "10-K 0001-26-000015 Risk Factors (period 2025-12-31)",
                       "similarity": 0.8}]}


def _resp(content=None, tool_calls=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


class FakeLLM:
    def __init__(self, drafts, critic=None):
        self.drafts, self.critic, self.roles = list(drafts), critic, []

    def __call__(self, role, messages, tools=None, **kw):
        self.roles.append(role)
        if role == "specialist":
            if any(m["role"] == "tool" for m in messages):
                return _resp(json.dumps({"summary": "s", "gaps": ["junk gap"], "findings": [
                    {"claim": "ok", "source": "t", "direction": "neutral", "strength": 0.5}]}))
            name = tools[0]["function"]["name"]
            return _resp(None, [{"id": "1", "function": {
                "name": name, "arguments": json.dumps({"ticker": "AAA"})}}])
        if role == "strategist":
            d = self.drafts.pop(0)
            return _resp(d if isinstance(d, str) else json.dumps(d))
        if role == "critic":
            return _resp(json.dumps(self.critic or {"verdict": "accept", "issues": []}))
        raise AssertionError(role)


def _patch(monkeypatch, fake):
    monkeypatch.setattr(llm, "chat", fake)
    monkeypatch.setattr(ma, "make_tools", lambda as_of: {
        "get_financials": lambda ticker: FIN, "list_filings": lambda ticker: [],
        "get_technicals": lambda ticker: TECH, "get_macro": lambda ticker: {"series": {}},
        "search_filings": lambda ticker, query="": SEARCH})


def drive(variant):
    """Sequential stand-in for the graph, using the real node and route functions."""
    cfg = ma.VARIANTS[variant]
    s = {"ticker": "AAA", "as_of": "2026-11-16", "attempts": 0, "started": 0.0,
         "specialists": {}, "tool_log": [], "trace": [], "calls": 0}

    def apply(upd):
        for k, v in (upd or {}).items():
            if k in ("tool_log", "trace"):
                s[k] = s[k] + v
            elif k == "calls":
                s[k] += v
            elif k == "specialists":
                s[k] = {**s[k], **v}
            else:
                s[k] = v

    s["variant"] = variant
    apply(ma.orchestrate_node(s))
    for name in ma.fan_out(variant):
        apply(ma.stub_node(name, s) if name in ma.STUBS else ma.specialist_node(name, s))
    apply(ma.strategist_node(s))
    if cfg["critic"] and ma.after_strategist(s) == "critic":
        apply(ma.critic_node(s))
        apply(ma.verify_node(s))
        while ma.after_verify(s) == "revise":
            apply(ma.revise_node(s))
            apply(ma.verify_node(s))
    apply(ma.finalize_node(s, variant))
    return s["final"]


def test_specialists_variant_measures_but_does_not_route(monkeypatch):
    fake = FakeLLM([BAD])
    _patch(monkeypatch, fake)
    out = drive("multi_specialists")
    f = out["forecast"]
    assert "critic" not in fake.roles
    assert f["probability"] == 0.6 and f["rating"] == "Moderate Buy"
    assert f["verification"]["status"] == "orange" and not f["verification"]["routed"]
    assert len(f["evidence"]) == 2                      # nothing dropped without routing
    assert f["paper_trade"]["stop_loss"] == 95.0
    assert any("analyst" in g.lower() for g in f["gaps"])
    assert "junk gap" not in f["gaps"]                  # model-written gaps are not trusted
    assert any(t.get("type") == "timing" for t in out["trace"])
    assert out["model_calls"] == 3 * 2 + 1 and out["tool_calls"] == 3


def test_green_path_no_revision(monkeypatch):
    fake = FakeLLM([GOOD])
    _patch(monkeypatch, fake)
    f = drive("multi_critic_verifier")["forecast"]
    assert f["verification"]["status"] == "green" and f["revisions"] == 0
    assert fake.roles.count("strategist") == 1


def test_orange_revised_to_green(monkeypatch):
    fake = FakeLLM([BAD, GOOD])
    _patch(monkeypatch, fake)
    f = drive("multi_critic_verifier")["forecast"]
    assert f["revisions"] == 1 and f["verification"]["status"] == "green"
    assert f["probability"] == 0.58


def test_orange_after_revision_drops_unverified_claims(monkeypatch):
    _patch(monkeypatch, FakeLLM([BAD, BAD]))
    f = drive("multi_critic_verifier")["forecast"]
    assert f["revisions"] == 1
    assert [e["claim"] for e in f["evidence"]] == ["Net margin is 27.6%"]


def test_red_after_revision_abstains(monkeypatch):
    _patch(monkeypatch, FakeLLM([VERY_BAD, VERY_BAD]))
    f = drive("multi_critic_verifier")["forecast"]
    assert f["verification"]["abstained"] and f["probability"] == 0.5
    assert f["rating"] == "Hold" and f["probability_raw"] == 0.7


def test_critic_revise_triggers_revision_even_when_green(monkeypatch):
    critic = {"verdict": "revise", "suggested_probability": 0.5,
              "issues": [{"type": "overconfident", "detail": "too high"}]}
    fake = FakeLLM([GOOD, dict(GOOD, probability=0.5)], critic)
    _patch(monkeypatch, fake)
    f = drive("multi_critic_verifier")["forecast"]
    assert f["revisions"] == 1 and f["probability"] == 0.5


def test_invalid_strategist_json_gives_error_not_crash(monkeypatch):
    _patch(monkeypatch, FakeLLM(["not json"]))
    out = drive("multi_critic_verifier")
    assert out["forecast"] is None and out["error"]


def test_probability_is_clamped(monkeypatch):
    _patch(monkeypatch, FakeLLM([dict(GOOD, probability=0.99)]))
    f = drive("multi_specialists")["forecast"]
    assert f["probability"] == 0.95 and f["probability_raw"] == 0.99


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        ma.run("AAA", "2026-11-16", variant="nope")


def test_real_langgraph_end_to_end(monkeypatch):
    pytest.importorskip("langgraph")
    _patch(monkeypatch, FakeLLM([BAD, GOOD]))
    ma._GRAPHS.clear()
    out = ma.run("AAA", "2026-11-16", "multi_critic_verifier")
    assert out["forecast"]["revisions"] == 1
    assert out["forecast"]["verification"]["status"] == "green"
    assert set(out["forecast"]["specialists"]) == {"fundamentals", "macro", "technical",
                                                   "analyst_news"}


def test_critic_failure_is_visible_not_silent(monkeypatch):
    class BrokenCritic(FakeLLM):
        def __call__(self, role, messages, tools=None, **kw):
            if role == "critic":
                self.roles.append(role)
                return _resp("")
            return super().__call__(role, messages, tools, **kw)

    _patch(monkeypatch, BrokenCritic([GOOD]))
    out = drive("multi_critic_verifier")
    assert out["forecast"]["critic"]["failed"] is True
    review = [t for t in out["trace"] if t.get("type") == "review"][0]
    assert review["failed"] is True


def test_rag_variant_adds_filings_specialist_and_accepts_filing_numbers(monkeypatch):
    draft = {"probability": 0.55, "rationale": "Cash is ample.",
             "evidence": [{"claim": "The company held $2.5 billion in cash", "source": "10-K"}],
             "risks": [{"risk": "Supplier concentration remains a risk", "source": "10-K"}]}
    _patch(monkeypatch, FakeLLM([draft]))
    out = drive("multi_rag")
    f = out["forecast"]
    assert "filings" in f["specialists"] and "filings" in ma.fan_out("multi_rag")
    assert "filings" not in ma.fan_out("multi_critic_verifier")
    assert f["verification"]["status"] == "green"                  # $2.5 billion is in the filing text
    assert not any(g.startswith("Text of filing risk factors") for g in f["gaps"])


def test_other_variants_keep_the_filing_text_gap(monkeypatch):
    _patch(monkeypatch, FakeLLM([GOOD]))
    f = drive("multi_critic_verifier")["forecast"]
    assert any(g.startswith("Text of filing risk factors") for g in f["gaps"])


def test_repeated_search_calls_are_merged():
    state = {"tool_log": [
        {"tool": "search_filings", "ok": True, "obs": {"results": [{"text": "a"}]}},
        {"tool": "search_filings", "ok": True, "obs": {"results": [{"text": "b"}]}},
        {"tool": "get_macro", "ok": False, "obs": {"error": "x"}}]}
    out = ma._tool_results(state)
    assert [r["text"] for r in out["search_filings"]["results"]] == ["a", "b"]
    assert "get_macro" not in out
