import os
import time

from forecaster import cache


def test_cached_computes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return {"v": len(calls)}

    assert cache.cached("ns", {"a": 1}, fn) == {"v": 1}
    assert cache.cached("ns", {"a": 1}, fn) == {"v": 1}
    assert len(calls) == 1
    assert cache.cached("ns", {"a": 2}, fn) == {"v": 2}


def test_key_is_order_independent():
    assert cache.make_key({"a": 1, "b": 2}) == cache.make_key({"b": 2, "a": 1})


def test_ttl_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.put("ns", "k", {"x": 1})
    assert cache.get("ns", "k", ttl=60) == {"x": 1}
    p = tmp_path / "ns" / "k.json"
    old = time.time() - 120
    os.utime(p, (old, old))
    assert cache.get("ns", "k", ttl=60) is None
    assert cache.get("ns", "k") == {"x": 1}
