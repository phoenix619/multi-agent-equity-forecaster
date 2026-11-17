"""Variant 1: a single ReAct agent with tools (the ablation baseline).

One model plans, calls tools, reads results and answers with a probability that
the stock outperforms the S&P 500 over the next 20 trading days. Every tool call
and result is recorded in a trace.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import llm
from .config import HORIZON_TRADING_DAYS, ROOT
from .tools import TOOL_SCHEMAS, make_tools

MAX_STEPS = 8
PROB_MIN, PROB_MAX = 0.05, 0.95

SYSTEM = f"""You are an equity research agent. Estimate the probability that a US \
stock outperforms the S&P 500 over the next {HORIZON_TRADING_DAYS} trading days, using tools.

Rules:
- Today's data cutoff is {{as_of}}. Tools only return data known on that date.
- Gather evidence with tools first: fundamentals (get_financials), filings \
(list_filings), technical timing (get_technicals), macro (get_macro). Think about \
which tool closes each gap. If a tool errors or returns nothing, try another or \
record the gap. Never invent numbers.
- The base rate is about 50%. Stay near it unless the evidence is clear.
- Technicals inform timing, not direction.
- Cite every claim with its source, such as "10-K FY2026 accession ...", \
"get_technicals", or "get_macro treasury_10y".
- Quote figures exactly as the tools return them; do not recompute or round ratios. \
Use the fiscal_period label the tool gives (companies' fiscal years differ from the \
calendar); do not infer quarter names from dates. Quarterly and annual figures are \
different periods: never mix them in one claim.
- Macro: cite only series that get_macro returned, and state why it matters for this \
company using its why_it_matters text. If no macro series is relevant, say macro is \
not a factor. Never cite excluded series.
- Risks: list only risks that a specific tool result supports (for example a high \
debt ratio, a stretched technical signal, or a relevant macro move) and give that \
result as the source. Do not list generic risks that apply to every company. An empty \
list is fine.
- Do not write rating, gaps or trade levels; the system adds them.

When you have enough evidence, reply with ONLY this JSON (no other text):
{{"probability": <0-1>, "rationale": "<2-3 sentences>", \
"evidence": [{{"claim": "...", "source": "..."}}], \
"risks": [{{"risk": "...", "source": "..."}}]}}
"""

# Fixed rating rule (provisional thresholds, recalibrated later from results).
RATING_BANDS = [(0.70, "Strong Buy"), (0.58, "Moderate Buy"), (0.42, "Hold"),
                (0.30, "Moderate Sell"), (0.0, "Strong Sell")]

# Evidence sources this variant does not have, reported as gaps by code.
UNAVAILABLE = [
    "Analyst rating changes: no analyst tool in this variant",
    "News and event headlines: no news tool in this variant",
    "Text of filing risk factors: no filing retrieval in this variant",
]
EXPECTED_TOOLS = ["get_financials", "list_filings", "get_technicals", "get_macro"]


def rating_from_probability(p: float) -> str:
    for floor, label in RATING_BANDS:
        if p >= floor:
            return label
    return RATING_BANDS[-1][1]


def _tool_results(trace: list) -> dict:
    out = {}
    for t in trace:
        if t["type"] == "tool" and t.get("result"):
            try:
                out[t["tool"]] = json.loads(t["result"])
            except ValueError:
                pass
    return out


def coverage_gaps(trace: list, as_of: str) -> list:
    """Gaps computed from what actually happened, not written by the model."""
    gaps = list(UNAVAILABLE)
    called = {t["tool"]: t for t in trace if t["type"] == "tool"}
    for name in EXPECTED_TOOLS:
        if name not in called:
            gaps.append(f"{name} was not consulted")
        elif not called[name]["ok"]:
            gaps.append(f"{name} returned an error")
    fin = _tool_results(trace).get("get_financials")
    if fin:
        need = {"quarter": ("revenue", "net_income", "operating_cash_flow",
                            "long_term_debt", "equity"),
                "annual": ("revenue", "net_income", "operating_cash_flow")}
        for kind, keys in need.items():
            missing = [k for k in keys if k not in fin.get(kind, {})]
            if missing:
                gaps.append(f"{kind} figures not reported: {', '.join(missing)}")
        fresh = fin.get("freshness") or {}
        end = fin.get("quarter", {}).get("revenue", {}).get("period_end")
        if fresh.get("lag"):
            lf = fresh["latest_filing"]
            gaps.append(f"newest filing ({lf['form']} for period ended {lf['period']}, filed "
                        f"{lf['filed']}) is not in SEC structured data; figures use the period "
                        f"ended {fresh['latest_period_in_structured_data']}")
        elif end:
            age = (datetime.strptime(as_of, "%Y-%m-%d") - datetime.strptime(end, "%Y-%m-%d")).days
            if age > 135:
                gaps.append(f"latest quarterly period ended {age} days before the as-of date")
    return gaps


def _parse_json(text: str):
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def _clean_message(msg: dict) -> dict:
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def run(ticker: str, as_of: str, role: str = "orchestrator", max_steps: int = MAX_STEPS) -> dict:
    tools = make_tools(as_of)
    messages = [
        {"role": "system", "content": SYSTEM.replace("{as_of}", as_of)},
        {"role": "user", "content": f"Analyze {ticker.upper()} as of {as_of}."},
    ]
    trace, started = [], time.time()
    result, error, model_calls = None, None, 0

    for step in range(1, max_steps + 1):
        resp = llm.chat(role, messages, tools=TOOL_SCHEMAS, max_tokens=4000)
        model_calls += 1
        if "error" in resp:
            error = resp["error"]
            break
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            result = _parse_json(msg.get("content"))
            trace.append({"step": step, "type": "final", "content": (msg.get("content") or "")[:2000]})
            if result is None:
                error = {"message": "final answer was not valid JSON"}
            break
        messages.append(_clean_message(msg))
        for c in calls:
            name = c["function"]["name"]
            try:
                args = c["function"]["arguments"]
                args = json.loads(args) if isinstance(args, str) else (args or {})
                obs = tools[name](**args) if name in tools else {"error": f"unknown tool {name}"}
            except Exception as e:  # recovery: the model sees the error and can adapt
                obs = {"error": f"{type(e).__name__}: {e}"}
            payload = json.dumps(obs, default=str)
            trace.append({"step": step, "type": "tool", "tool": name,
                          "args": args if isinstance(args, dict) else {},
                          "ok": not (isinstance(obs, dict) and "error" in obs),
                          "result_chars": len(payload),
                          "result": payload[:6000]})
            messages.append({"role": "tool", "tool_call_id": c.get("id", f"call_{step}"),
                             "content": payload[:6000]})
    else:
        # Step budget used up: ask for the final answer without tools.
        messages.append({"role": "user", "content":
                         "Step limit reached. Give your final JSON answer now using only "
                         "the evidence gathered; list missing evidence under gaps."})
        resp = llm.chat(role, messages, max_tokens=4000)
        model_calls += 1
        if "error" not in resp:
            result = _parse_json(resp["choices"][0]["message"].get("content"))
            trace.append({"step": max_steps + 1, "type": "forced_final"})
        if result is None:
            error = {"message": "no valid final answer after step limit"}

    if result is not None:
        try:
            p = float(result.get("probability"))
            result["probability_raw"] = p
            result["probability"] = round(min(max(p, PROB_MIN), PROB_MAX), 3)
        except (TypeError, ValueError):
            result, error = None, {"message": "probability missing or not a number"}
    if result is not None:
        result["rating"] = rating_from_probability(result["probability"])
        # Keep only risks that name a source; generic unsourced risks are dropped.
        risks = result.get("risks") or []
        kept = [r for r in risks if isinstance(r, dict) and r.get("risk") and r.get("source")]
        result["risks"] = kept
        result["risks_dropped_unsourced"] = len(risks) - len(kept)
        result["gaps"] = coverage_gaps(trace, as_of)
        tech = _tool_results(trace).get("get_technicals", {})
        if tech.get("paper_trade"):
            result["paper_trade"] = dict(tech["paper_trade"], signal=tech.get("signal"))

    out = {"ticker": ticker.upper(), "as_of": as_of, "variant": "single_agent_react",
           "model": llm.MODELS.get(role, role),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "forecast": result, "error": error, "model_calls": model_calls,
           "tool_calls": sum(1 for t in trace if t["type"] == "tool"),
           "seconds": round(time.time() - started, 1), "trace": trace,
           "disclaimer": "Research tool, not financial advice. No trades executed."}
    return out


def save_trace(result: dict) -> Path:
    d = ROOT / "logs"
    d.mkdir(exist_ok=True)
    p = d / f"{result['ticker']}_{result['as_of']}_{result['variant']}.json"
    p.write_text(json.dumps(result, indent=2, default=str))
    return p


def log_forecast(result: dict, folder: str = "forecasts", name: str = "live.jsonl",
                 mode: str = "live") -> Path:
    """Append a timestamped forecast to a forecast log (live by default).

    mode is "live" (made before outcomes exist) or "backtest" (past as-of date,
    created_at is the run time, not the as-of date)."""
    d = ROOT / "eval" / folder
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    row = {k: result[k] for k in ("ticker", "as_of", "variant", "model", "created_at")}
    row["mode"] = mode
    row["probability"] = result["forecast"]["probability"] if result["forecast"] else None
    row["rating"] = result["forecast"].get("rating") if result["forecast"] else None
    ver = (result["forecast"] or {}).get("verification")
    if ver:  # multi-agent variants: guardrail outcome, for the ablation analysis
        row["verification"] = ver["status"]
        row["unverified"] = len(ver["unverified"])
        row["abstained"] = ver.get("abstained", False)
        row["revisions"] = result["forecast"].get("revisions", 0)
        crit = result["forecast"].get("critic") or {}
        row["critic_verdict"] = crit.get("verdict")
        row["critic_failed"] = bool(crit.get("failed"))
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return p
