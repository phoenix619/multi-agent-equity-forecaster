"""Chat client for any OpenAI-compatible endpoint (local Ollama by default).

Responses are cached on disk, so reruns are free and deterministic. An optional
daily budget (DAILY_CALL_BUDGET > 0) caps real requests, for hosted endpoints.
"""
import json
import time
from datetime import datetime, timezone

import requests

from . import cache
from .config import (CACHE_DIR, DAILY_CALL_BUDGET, LLM_API_KEY, LLM_BASE_URL,
                     MODELS)


class BudgetExceeded(RuntimeError):
    """Raised when today's request budget is used up."""


def _usage_path():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "usage.json"


def calls_today() -> int:
    p = _usage_path()
    if not p.exists():
        return 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return json.loads(p.read_text()).get(today, 0)


def _count_call() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p = _usage_path()
    data = json.loads(p.read_text()) if p.exists() else {}
    data[today] = data.get(today, 0) + 1
    p.write_text(json.dumps(data))


def chat(role_or_model: str, messages: list, tools: list | None = None,
         temperature: float = 0.2, max_tokens: int = 3000,
         use_cache: bool = True, retries: int = 3) -> dict:
    """Send a chat completion. role_or_model is a key of MODELS or a model name.

    Cache hits cost nothing. Returns the full response JSON; check for an
    "error" key. Reasoning models spend tokens thinking, so max_tokens is
    generous by default.
    """
    model = MODELS.get(role_or_model, role_or_model)
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools

    key = cache.make_key(body)
    if use_cache:
        hit = cache.get("llm", key)
        if hit is not None:
            return hit

    if DAILY_CALL_BUDGET and calls_today() >= DAILY_CALL_BUDGET:
        raise BudgetExceeded(
            f"Daily budget of {DAILY_CALL_BUDGET} requests used up (UTC day).")

    headers = {"Authorization": f"Bearer {LLM_API_KEY}",
               "Content-Type": "application/json"}
    last = {"error": {"message": "no attempt made"}}
    for attempt in range(retries):
        try:
            r = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers,
                              json=body, timeout=600)
        except requests.exceptions.ConnectionError:
            return {"error": {"message": f"Cannot reach {LLM_BASE_URL}. "
                                         "Is Ollama running?"}}
        except requests.exceptions.Timeout:
            last = {"error": {"message": "request timed out"}}
            continue
        if DAILY_CALL_BUDGET:
            _count_call()
        if r.status_code == 429:
            time.sleep(int(float(r.headers.get("Retry-After", 15 * (attempt + 1)))))
            last = {"error": {"message": "rate limited"}}
            continue
        try:
            last = r.json()
        except ValueError:
            last = {"error": {"message": r.text[:300]}}
        if "error" not in last and use_cache:
            cache.put("llm", key, last)
        return last
    return last
