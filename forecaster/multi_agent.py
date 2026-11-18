"""Variants 2 and 3: a LangGraph team of agents.

    START -> orchestrate -> [fundamentals | macro | technical | analyst_news] -> strategist
          variant 2 (multi_specialists):        strategist -> finalize
          variant 3 (multi_critic_verifier):    strategist -> critic -> verify -> (revise -> verify) -> finalize

Specialists (gemma4) call tools in parallel and report findings with sources. The
Strategist (gpt-oss) turns the findings into a probability. In variant 3 a Critic
(a different model family from the Strategist) challenges the draft and a
deterministic Verifier checks every number against the tool results; one revision is
allowed, then the guardrail routes: green -> finish, orange -> drop unverified claims,
red -> abstain at 50%.

The as-of date is fixed by code in every tool. The rating, coverage gaps and trade
levels are computed in code, never written by a model.
"""
import json
import operator
import time
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from . import llm, verifier
from .agent import _parse_json, react
from .baseline import PROB_MAX, PROB_MIN, coverage_gaps, rating_from_probability
from .config import HORIZON_TRADING_DAYS, MODELS
from .tools import TOOL_SCHEMAS, make_tools

VARIANTS = {
    "multi_specialists": {"critic": False, "route": False},
    "multi_critic_verifier": {"critic": True, "route": True},
    # + filing text retrieval (RAG over 10-K/10-Q risk factors and MD&A)
    "multi_rag": {"critic": True, "route": True, "filings": True},
}

# name -> (tools it may call, focus)
SPECIALISTS = {
    "fundamentals": (["get_financials", "list_filings"],
                     "profitability, cash flow and leverage from SEC filings. Call both "
                     "tools. Report the latest quarter and the latest fiscal year as "
                     "separate findings, each labelled with its fiscal_period, and cite "
                     "the most recent filing from list_filings"),
    "macro": (["get_macro"], "which macro series matter for this company and how they moved"),
    "technical": (["get_technicals"], "trend, momentum, volatility and entry timing"),
}
# Added only in the RAG variant: reads the text of the filings.
FILINGS_SPECIALIST = {"filings": (
    ["search_filings"],
    "risks, litigation, liquidity and management commentary found in the TEXT of the latest "
    "10-K and 10-Q. Call search_filings two or three times with different specific queries "
    "(for example 'principal risks', 'liquidity and capital resources', 'outlook and trends'). "
    "Report each finding as a short paraphrase or quote and cite its source field exactly "
    "(form, accession, section)")}
ALL_SPECIALISTS = {**SPECIALISTS, **FILINGS_SPECIALIST}
STEPS = {"filings": 5}   # tool-call rounds allowed per specialist (default 3)

# Slots in the design with no data source built yet; they report a gap, no model call.
STUBS = {"analyst_news": "No analyst-rating or news data source is connected in this build."}

SPECIALIST_SYSTEM = """You are the {role} specialist on an equity research team. Your \
focus: {focus}. Data cutoff {as_of}; tools only return data known on that date.
Call your tools for the ticker, then report only what the tool results show.
- Quote figures exactly as the tools return them; do not recompute or round. Use the \
fiscal_period label the tool gives; never mix quarterly and annual figures in one claim.
- Every finding names its source (form and accession, or tool name and field).
- direction is bullish, bearish or neutral for outperforming the S&P 500 over \
{horizon} trading days; findings about entry timing are "timing".
- Cite only macro series the tool returned, with its why_it_matters text.
- Under gaps, list only evidence missing from YOUR tools' results (an error or an empty \
result). Do not comment on other specialists' areas. Never invent numbers.
Reply with ONLY this JSON:
{{"summary": "<2 sentences>", "findings": [{{"claim": "...", "source": "...", \
"direction": "bullish|bearish|neutral|timing", "strength": <0-1>}}], "gaps": ["..."]}}"""

STRATEGIST_SYSTEM = """You are the strategist on an equity research team. Combine the \
specialists' findings into the probability that the stock outperforms the S&P 500 over \
the next {horizon} trading days. Data cutoff {as_of}.
- The base rate is about 50%. Stay near it unless the evidence is clear and consistent.
- Technicals inform timing, not direction.
- Name the period next to every financial figure (for example "FY2026 Q3" or "FY2025 FY"). \
Never put a quarterly figure and an annual figure in the same sentence or compare them.
- Use only facts in the findings. Quote figures exactly as given; do not compute new \
numbers, ratios or growth rates.
- Cite each claim with the source from the finding it came from.
- Risks must come from a specific finding and cite its source; no generic risks. An \
empty list is fine.
- Do not write a rating, coverage gaps or trade levels; the system adds them.
Reply with ONLY this JSON:
{{"probability": <0-1>, "rationale": "<2-3 sentences>", \
"evidence": [{{"claim": "...", "source": "..."}}], \
"risks": [{{"risk": "...", "source": "..."}}]}}"""

CRITIC_SYSTEM = """You are a skeptical reviewer of an equity forecast. You did not \
write it. Check the draft against the specialists' findings and flag:
- unsupported: a claim, number or source not in the findings
- overconfident: probability far from 50% for weak or mixed evidence
- ignored: a material bearish or bullish finding the draft leaves out
- misuse: technicals used as direction, or quarterly and annual figures mixed in one \
statement (for example a quarterly margin next to annual revenue)
Always name the draft's weakest claim, even if you accept it. Choose "revise" when that \
weakness is a real error (one of the four kinds above), and "accept" only when you found \
none. Be specific and brief; keep your thinking short. Reply with ONLY this JSON:
{"weakest_claim": "...", "verdict": "accept" or "revise", "issues": [{"type": \
"unsupported|overconfident|ignored|misuse", "detail": "..."}], "suggested_probability": \
<0-1 or null>}"""


# gemma4's thinking can run past 20k tokens on the Critic prompt and never answer.
# Set to None to let it think; "none" asks Ollama to switch thinking off.
CRITIC_EFFORT = "none"
CRITIC_MAX_TOKENS = 4000


def _timing(agent: str, started: float) -> dict:
    return {"agent": agent, "type": "timing", "seconds": round(time.time() - started, 1)}


def _merge(a, b):
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    ticker: str
    as_of: str
    variant: str
    started: float
    specialists: Annotated[dict, _merge]
    tool_log: Annotated[list, operator.add]     # full tool observations
    trace: Annotated[list, operator.add]
    calls: Annotated[int, operator.add]
    messages: list
    draft: dict
    critic: dict
    verification: dict
    attempts: int
    error: dict
    final: dict


# ---------------------------------------------------------------- helpers
def fan_out(variant: str) -> list:
    """Agents that run in parallel before the Strategist."""
    extra = list(FILINGS_SPECIALIST) if VARIANTS[variant].get("filings") else []
    return list(SPECIALISTS) + extra + list(STUBS)


def _tool_results(state: dict) -> dict:
    """Latest result per tool; repeated search_filings calls are merged, not overwritten."""
    out = {}
    for r in state.get("tool_log", []):
        if not r["ok"]:
            continue
        if r["tool"] == "search_filings":
            out.setdefault("search_filings", {"results": []})["results"].extend(
                r["obs"].get("results", []))
        else:
            out[r["tool"]] = r["obs"]
    return out


def _findings_json(state: dict) -> str:
    return json.dumps(state.get("specialists", {}), default=str)[:12000]


def _strategist_messages(state: dict) -> list:
    system = STRATEGIST_SYSTEM.format(horizon=HORIZON_TRADING_DAYS, as_of=state["as_of"])
    if VARIANTS.get(state.get("variant"), {}).get("filings"):
        system += ("\nFindings from filing text carry a source such as "
                   "'10-K <accession> Risk Factors'. Prefer them for risks, and cite that source.")
    user = (f"Ticker {state['ticker']}, as of {state['as_of']}.\n"
            f"Specialist findings (JSON):\n{_findings_json(state)}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _ask_strategist(messages: list):
    resp = llm.chat("strategist", messages, max_tokens=4000)
    if "error" in resp:
        return None, "", resp["error"]
    text = resp["choices"][0]["message"].get("content") or ""
    return _parse_json(text), text, None


# ---------------------------------------------------------------- nodes
def orchestrate_node(state: State) -> dict:
    plan = fan_out(state.get("variant") or "multi_critic_verifier")
    return {"trace": [{"agent": "orchestrator", "type": "plan", "specialists": plan,
                       "note": "all specialists run in parallel; as-of date fixed by code"}]}


def specialist_node(name: str, state: State) -> dict:
    t0 = time.time()
    tool_names, focus = ALL_SPECIALISTS[name]
    system = SPECIALIST_SYSTEM.format(role=name, focus=focus, as_of=state["as_of"],
                                      horizon=HORIZON_TRADING_DAYS)
    out = react("specialist", system, f"Analyze {state['ticker']}.", tool_names,
                make_tools(state["as_of"]), TOOL_SCHEMAS, agent=name,
                max_steps=STEPS.get(name, 3))
    findings = out["final"] or {
        "summary": "", "findings": [],
        "gaps": [f"{name} specialist produced no valid answer: "
                 f"{(out['error'] or {}).get('message', 'unknown')}"]}
    return {"specialists": {name: findings}, "tool_log": out["results"],
            "trace": out["trace"] + [_timing(name, t0)], "calls": out["calls"]}


def stub_node(name: str, state: State) -> dict:
    return {"specialists": {name: {"summary": STUBS[name], "findings": [],
                                   "gaps": [STUBS[name]]}},
            "trace": [{"agent": name, "type": "stub", "note": STUBS[name]}]}


def strategist_node(state: State) -> dict:
    t0 = time.time()
    messages = _strategist_messages(state)
    draft, text, err = _ask_strategist(messages)
    upd = {"calls": 1, "trace": [{"agent": "strategist", "type": "draft",
                                  "content": text[:2000]}, _timing("strategist", t0)]}
    if err or draft is None:
        upd["error"] = err or {"message": "strategist answer was not valid JSON"}
        return upd
    upd["draft"] = draft
    upd["messages"] = messages + [{"role": "assistant", "content": text}]
    return upd


def critic_node(state: State) -> dict:
    t0 = time.time()
    user = (f"Specialist findings:\n{_findings_json(state)}\n\n"
            f"Draft forecast:\n{json.dumps(state['draft'], default=str)}")
    resp = llm.chat("critic", [{"role": "system", "content": CRITIC_SYSTEM},
                               {"role": "user", "content": user}],
                    max_tokens=CRITIC_MAX_TOKENS, reasoning_effort=CRITIC_EFFORT,
                    escalate=False)
    crit = None if "error" in resp else _parse_json(
        resp["choices"][0]["message"].get("content"))
    failed = not isinstance(crit, dict)
    crit = crit if not failed else {"verdict": "accept", "issues": [], "failed": True,
                                    "note": "critic produced no valid answer; defaulted to accept"}
    crit["verdict"] = "revise" if crit.get("verdict") == "revise" else "accept"
    crit["issues"] = [i for i in (crit.get("issues") or []) if isinstance(i, dict)]
    return {"critic": crit, "calls": 1,
            "trace": [{"agent": "critic", "type": "review", "verdict": crit["verdict"],
                       "issues": crit["issues"], "failed": failed,
                       "weakest_claim": crit.get("weakest_claim")}, _timing("critic", t0)]}


def verify_node(state: State) -> dict:
    ver = verifier.verify_forecast(state["draft"], _tool_results(state))
    return {"verification": ver,
            "trace": [{"agent": "verifier", "type": "check", **ver}]}


def needs_revision(state: State) -> bool:
    if state.get("attempts", 0) >= 1:
        return False
    crit = state.get("critic") or {}
    flagged = crit.get("verdict") == "revise" and crit.get("issues")
    return state["verification"]["status"] != verifier.GREEN or bool(flagged)


def after_strategist(state: State) -> str:
    return "finalize" if state.get("error") or not state.get("draft") else "critic"


def after_verify(state: State) -> str:
    return "revise" if needs_revision(state) else "finalize"


def revise_node(state: State) -> dict:
    t0 = time.time()
    parts = []
    crit = state.get("critic") or {}
    if crit.get("issues"):
        parts.append("Reviewer issues:\n" + "\n".join(
            f"- {i.get('type')}: {i.get('detail')}" for i in crit["issues"]))
        if crit.get("suggested_probability") is not None:
            parts.append(f"Reviewer suggests a probability near {crit['suggested_probability']}.")
    bad = state["verification"]["unverified"]
    if bad:
        parts.append("These numbers match no tool result. Quote them exactly as the "
                     "findings give them, or remove the claim: "
                     + ", ".join(f"{u['number']} ({u['where']})" for u in bad))
    messages = state["messages"] + [{"role": "user", "content":
                                     "Revise your forecast.\n" + "\n".join(parts)
                                     + "\nDo not mention the reviewer, the issues or these "
                                       "instructions in your answer; just write the improved "
                                       "forecast. Reply with ONLY the same JSON."}]
    draft, text, err = _ask_strategist(messages)
    upd = {"calls": 1, "attempts": state.get("attempts", 0) + 1,
           "trace": [{"agent": "strategist", "type": "revision", "content": text[:2000]},
                     _timing("strategist_revision", t0)]}
    if draft is not None:
        upd["draft"] = draft
    else:
        upd["trace"][0]["note"] = "revision was not valid JSON; kept the first draft"
    return upd


def finalize_node(state: State, variant: str) -> dict:
    return {"final": assemble(state, variant)}


# ---------------------------------------------------------------- assembly
def assemble(state: dict, variant: str) -> dict:
    cfg = VARIANTS[variant]
    as_of = state["as_of"]
    draft, error = state.get("draft"), state.get("error")
    forecast = None
    if draft is not None:
        # Routed variants already verified the latest draft; otherwise measure it now.
        ver = (state["verification"] if state.get("verification")
               else verifier.verify_forecast(draft, _tool_results(state)))
        draft = dict(draft)
        abstained = False
        if cfg["route"]:
            if ver["status"] == verifier.RED:
                abstained = True
                draft = {"probability": 0.5, "probability_raw": draft.get("probability"),
                         "rationale": "Abstained: the Verifier could not tie the draft's "
                                      "figures to tool results (red).",
                         "evidence": [], "risks": []}
            elif ver["status"] == verifier.ORANGE:
                draft = verifier.drop_unverified(draft, ver["unverified"])
                ver = dict(ver, dropped_items=len({u["where"] for u in ver["unverified"]}),
                           note="unverified indices refer to the draft before dropping")
        try:
            p = float(draft.get("probability"))
            draft.setdefault("probability_raw", p)
            draft["probability"] = round(min(max(p, PROB_MIN), PROB_MAX), 3)
        except (TypeError, ValueError):
            draft, error = None, {"message": "probability missing or not a number"}
    if draft is not None:
        draft["rating"] = rating_from_probability(draft["probability"])
        risks = draft.get("risks") or []
        kept = [r for r in risks if isinstance(r, dict) and r.get("risk") and r.get("source")]
        draft["risks"] = kept
        draft["risks_dropped_unsourced"] = len(risks) - len(kept)
        gap_trace = [{"type": "tool", "tool": r["tool"], "ok": r["ok"],
                      "result": json.dumps(r["obs"], default=str)}
                     for r in state.get("tool_log", [])]
        draft["gaps"] = coverage_gaps(gap_trace, as_of)
        # Gaps stay code-computed. Specialists' own gap notes remain in
        # draft["specialists"] for inspection but are not trusted as evidence.
        tech = _tool_results(state).get("get_technicals", {})
        if tech.get("paper_trade"):
            draft["paper_trade"] = dict(tech["paper_trade"], signal=tech.get("signal"))
        draft["verification"] = {**ver, "routed": cfg["route"], "abstained": abstained}
        draft["critic"] = state.get("critic")
        draft["revisions"] = state.get("attempts", 0)
        draft["specialists"] = state.get("specialists")
        forecast = draft

    trace = list(state.get("trace", []))
    return {"ticker": state["ticker"], "as_of": as_of, "variant": variant,
            "model": f"{MODELS['strategist']} + {MODELS['specialist']}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "forecast": forecast, "error": error,
            "model_calls": state.get("calls", 0),
            "tool_calls": len(state.get("tool_log", [])),
            "seconds": round(time.time() - state.get("started", time.time()), 1),
            "trace": trace,
            "disclaimer": "Research tool, not financial advice. No trades executed."}


# ---------------------------------------------------------------- graph
_GRAPHS = {}


def build_graph(variant: str):
    from langgraph.graph import END, START, StateGraph
    cfg = VARIANTS[variant]
    g = StateGraph(State)
    g.add_node("orchestrate", orchestrate_node)
    fan = fan_out(variant)

    def _specialist(name):
        def node(state: State) -> dict:
            return specialist_node(name, state)
        return node

    def _stub(name):
        def node(state: State) -> dict:
            return stub_node(name, state)
        return node

    def finalize(state: State) -> dict:
        return finalize_node(state, variant)

    for name in fan:
        if name in STUBS:
            continue
        g.add_node(name, _specialist(name))
    for name in STUBS:
        g.add_node(name, _stub(name))
    g.add_node("strategist", strategist_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "orchestrate")
    for name in fan:
        g.add_edge("orchestrate", name)
    g.add_edge(fan, "strategist")
    if cfg["critic"]:
        g.add_node("critic", critic_node)
        g.add_node("verify", verify_node)
        g.add_node("revise", revise_node)
        g.add_conditional_edges("strategist", after_strategist,
                                {"critic": "critic", "finalize": "finalize"})
        g.add_edge("critic", "verify")
        g.add_conditional_edges("verify", after_verify,
                                {"revise": "revise", "finalize": "finalize"})
        g.add_edge("revise", "verify")
    else:
        g.add_edge("strategist", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run(ticker: str, as_of: str, variant: str = "multi_critic_verifier") -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose from {list(VARIANTS)}")
    if variant not in _GRAPHS:
        _GRAPHS[variant] = build_graph(variant)
    out = _GRAPHS[variant].invoke({"ticker": ticker.upper(), "as_of": as_of,
                                   "variant": variant, "attempts": 0,
                                   "started": time.time()})
    return out["final"]
