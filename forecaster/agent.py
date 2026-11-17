"""A small ReAct loop shared by the specialist agents.

The model may call only the tools it is given. The as-of date is baked into the
tool callables, so the model cannot choose or change it. Full tool results are
returned in memory (for the Verifier); the trace keeps a truncated copy.
"""
import json

from . import llm


def _parse_json(text):
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def _assistant(msg: dict) -> dict:
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def react(role: str, system: str, user: str, tool_names: list, tools: dict,
          schemas: list, agent: str, max_steps: int = 3, max_tokens: int = 3000) -> dict:
    """Run one agent. Returns {final, trace, results, calls, error}.

    results is a list of {tool, ok, obs} with the full (untruncated) observations.
    """
    allowed = [s for s in schemas if s["function"]["name"] in tool_names]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    trace, results, calls, final, error = [], [], 0, None, None

    for step in range(1, max_steps + 1):
        resp = llm.chat(role, messages, tools=allowed, max_tokens=max_tokens)
        calls += 1
        if "error" in resp:
            error = resp["error"]
            break
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            final = _parse_json(msg.get("content"))
            trace.append({"agent": agent, "step": step, "type": "final",
                          "content": (msg.get("content") or "")[:2000]})
            if final is None:
                error = {"message": "final answer was not valid JSON"}
            break
        messages.append(_assistant(msg))
        for c in tool_calls:
            name = c["function"]["name"]
            args = {}
            try:
                raw = c["function"]["arguments"]
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if name not in tool_names or name not in tools:
                    obs = {"error": f"tool {name} is not available to this agent"}
                else:
                    obs = tools[name](**args)
            except Exception as e:  # the model sees the error and can adapt
                obs = {"error": f"{type(e).__name__}: {e}"}
            payload = json.dumps(obs, default=str)
            ok = not (isinstance(obs, dict) and "error" in obs)
            results.append({"tool": name, "ok": ok, "obs": json.loads(payload)})
            trace.append({"agent": agent, "step": step, "type": "tool", "tool": name,
                          "args": args if isinstance(args, dict) else {}, "ok": ok,
                          "result_chars": len(payload), "result": payload[:6000]})
            messages.append({"role": "tool", "tool_call_id": c.get("id", f"call_{step}"),
                             "content": payload[:6000]})
    else:
        messages.append({"role": "user", "content":
                         "Step limit reached. Give your final JSON answer now using "
                         "only the evidence gathered."})
        resp = llm.chat(role, messages, max_tokens=max_tokens)
        calls += 1
        if "error" not in resp:
            final = _parse_json(resp["choices"][0]["message"].get("content"))
            trace.append({"agent": agent, "step": max_steps + 1, "type": "forced_final"})
        if final is None:
            error = {"message": "no valid final answer after step limit"}
    return {"final": final, "trace": trace, "results": results, "calls": calls, "error": error}
