"""Tiny JSON disk cache so reruns cost nothing and respect rate limits."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import CACHE_DIR


def make_key(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _path(namespace: str, key: str) -> Path:
    d = CACHE_DIR / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def get(namespace: str, key: str, ttl: Optional[float] = None):
    p = _path(namespace, key)
    if not p.exists():
        return None
    if ttl is not None and time.time() - p.stat().st_mtime > ttl:
        return None
    return json.loads(p.read_text())


def put(namespace: str, key: str, value: Any) -> None:
    _path(namespace, key).write_text(json.dumps(value, default=str))


def cached(namespace: str, key_obj: Any, fn: Callable[[], Any],
           ttl: Optional[float] = None):
    """Return the cached value for key_obj, or compute with fn() and store it."""
    key = make_key(key_obj)
    hit = get(namespace, key, ttl)
    if hit is not None:
        return hit
    value = fn()
    put(namespace, key, value)
    return value
