"""Find a Critic setting where gemma4 answers (valid JSON) quickly.

Run from the equity-forecaster folder with Ollama running, after one multi-agent run
exists in logs/ (for example: python scripts/run_baseline.py NVDA --variant multi_critic_verifier):

    python scripts/test_thinking.py

Tries the same Critic prompt with different thinking settings and prints seconds,
finish reason, tokens and whether the reply was valid JSON.
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import multi_agent as ma
from forecaster.agent import _parse_json
from forecaster.config import LLM_BASE_URL, MODELS, ROOT

log = sorted((ROOT / "logs").glob("*_multi_critic_verifier.json"))
if not log:
    sys.exit("Run one multi-agent forecast first (see the docstring).")
f = json.loads(log[-1].read_text())["forecast"]
draft = {k: f[k] for k in ("probability", "rationale", "evidence", "risks")}
user = (f"Specialist findings:\n{json.dumps(f['specialists'])[:12000]}\n\n"
        f"Draft forecast:\n{json.dumps(draft)}")
messages = [{"role": "system", "content": ma.CRITIC_SYSTEM}, {"role": "user", "content": user}]
model = MODELS["critic"]
native = LLM_BASE_URL.replace("/v1", "") + "/api/chat"

TESTS = [
    ("/v1 default (thinking on), 4000 tokens", "v1", {"max_tokens": 4000}),
    ('/v1 reasoning_effort="none"', "v1", {"max_tokens": 4000, "reasoning_effort": "none"}),
    ('/v1 reasoning_effort="low"', "v1", {"max_tokens": 4000, "reasoning_effort": "low"}),
    ("native /api/chat think=false", "native", {"think": False}),
]
for name, kind, extra in TESTS:
    t0 = time.time()
    try:
        if kind == "v1":
            r = requests.post(f"{LLM_BASE_URL}/chat/completions", timeout=900,
                              json={"model": model, "messages": messages,
                                    "temperature": 0.2, **extra}).json()
            ch, msg = r["choices"][0], r["choices"][0]["message"]
            finish, content = ch.get("finish_reason"), msg.get("content") or ""
            toks = (r.get("usage") or {}).get("completion_tokens")
            think = len(msg.get("reasoning") or "")
        else:
            r = requests.post(native, timeout=900,
                              json={"model": model, "messages": messages, "stream": False,
                                    "options": {"temperature": 0.2, "num_predict": 4000},
                                    **extra}).json()
            msg = r.get("message", {})
            finish, content = r.get("done_reason"), msg.get("content") or ""
            toks, think = r.get("eval_count"), len(msg.get("thinking") or "")
        parsed = _parse_json(content)
        print(f"{name}\n   {time.time() - t0:.0f}s  finish={finish}  tokens={toks}  "
              f"thinking_chars={think}  valid_json={parsed is not None}"
              f"  verdict={(parsed or {}).get('verdict')}")
    except Exception as e:
        print(f"{name}\n   ERROR {type(e).__name__}: {str(e)[:120]}")
