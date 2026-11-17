"""Test local Ollama models for tool calling and structured output.

Prerequisites (one time):
    ollama pull gpt-oss:20b
    ollama pull gemma4:12b
    (Ollama must be running; the macOS app starts it, or run: ollama serve)

Run from the equity-forecaster folder:
    python scripts/test_local.py                       # gpt-oss:20b, gemma4:12b
    python scripts/test_local.py gemma4:26b            # one model

Same three checks as the cloud test: tool call, follow-up that uses the tool
result, JSON-only reply. Nothing leaves your laptop. The first request per model
is slow because the model loads into memory.
"""
import json
import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:11434/v1"
DEFAULT_MODELS = ["gpt-oss:20b", "gemma4:12b"]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_closing_price",
        "description": "Get a stock's closing price on a date (YYYY-MM-DD).",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "date": {"type": "string"}},
            "required": ["ticker", "date"],
        },
    },
}]


def post(body):
    start = time.time()
    try:
        r = requests.post(f"{BASE}/chat/completions", json=body, timeout=600,
                          headers={"Authorization": "Bearer ollama"})
    except requests.exceptions.ConnectionError:
        sys.exit("Cannot reach Ollama at localhost:11434. Start the Ollama app "
                 "or run `ollama serve`.")
    secs = round(time.time() - start, 1)
    try:
        return r.json(), secs
    except ValueError:
        return {"error": {"message": r.text[:300]}}, secs


def test_model(model):
    res = {"model": model, "tool_call": False, "args_ok": False,
           "followup_ok": False, "json_ok": False, "seconds": [], "notes": []}
    msgs = [{"role": "user", "content":
             "What was AAPL's closing price on 2026-11-13? Use the tool."}]
    data, t = post({"model": model, "messages": msgs, "tools": TOOLS,
                    "max_tokens": 3000, "temperature": 0.2})
    res["seconds"].append(t)
    if "error" in data:
        res["notes"].append(f"turn 1 error: {data['error']}")
        return res
    msg = data["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    res["tool_call"] = bool(calls)
    if calls:
        fn = calls[0]["function"]
        try:
            args = fn["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            res["args_ok"] = (fn["name"] == "get_closing_price"
                              and str(args.get("ticker", "")).upper() == "AAPL"
                              and args.get("date") == "2026-11-13")
        except (ValueError, KeyError, AttributeError):
            res["notes"].append("tool arguments were not valid")
        msgs.append({"role": "assistant", "content": msg.get("content") or "",
                     "tool_calls": calls})
        msgs.append({"role": "tool", "tool_call_id": calls[0].get("id", "call_1"),
                     "content": json.dumps({"close": 271.35})})
        d2, t2 = post({"model": model, "messages": msgs, "tools": TOOLS,
                       "max_tokens": 3000, "temperature": 0.2})
        res["seconds"].append(t2)
        if "error" in d2:
            res["notes"].append(f"turn 2 error: {d2['error']}")
        else:
            res["followup_ok"] = "271.35" in (d2["choices"][0]["message"].get("content") or "")
    else:
        res["notes"].append("no tool call in turn 1")

    jm = [{"role": "user", "content":
           'Reply with ONLY a JSON object: {"direction": "bullish" or "bearish" or "neutral", '
           '"strength": number 0-1, "claim": short string}. '
           "Evidence: revenue grew 8% year over year."}]
    d3, t3 = post({"model": model, "messages": jm, "max_tokens": 3000,
                   "temperature": 0.2})
    res["seconds"].append(t3)
    if "error" in d3:
        res["notes"].append(f"json error: {d3['error']}")
    else:
        text = (d3["choices"][0]["message"].get("content") or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            res["json_ok"] = {"direction", "strength", "claim"} <= set(json.loads(text))
        except ValueError:
            res["notes"].append("JSON reply did not parse")
    return res


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    results = []
    for m in models:
        print(f"\nTesting {m} (first call loads the model, may take a minute)")
        r = test_model(m)
        results.append(r)
        print(f"  tool call: {r['tool_call']}  args ok: {r['args_ok']}  "
              f"follow-up ok: {r['followup_ok']}  json ok: {r['json_ok']}  "
              f"seconds: {r['seconds']}")
        for n in r["notes"]:
            print("  note:", n)
    out = Path(__file__).resolve().parent / "local_test_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nSaved", out)


if __name__ == "__main__":
    main()
