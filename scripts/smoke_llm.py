"""Check that forecaster.llm.chat works for every role, and that caching works.
Run from the equity-forecaster folder:  python scripts/smoke_llm.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster.config import MODELS
from forecaster.llm import chat

for role in sorted(set(MODELS)):
    msgs = [{"role": "user", "content": "Reply with exactly: ready"}]
    t = time.time()
    r = chat(role, msgs, max_tokens=500)
    first = round(time.time() - t, 1)
    if "error" in r:
        print(f"{role:13s} {MODELS[role]:14s} ERROR: {r['error']}")
        continue
    text = (r["choices"][0]["message"].get("content") or "").strip()
    t = time.time()
    chat(role, msgs, max_tokens=500)
    cached = round(time.time() - t, 2)
    print(f"{role:13s} {MODELS[role]:14s} reply={text[:30]!r} "
          f"first={first}s cached={cached}s")
