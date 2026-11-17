import json

from forecaster import cache, llm


class FakeResp:
    status_code = 200
    headers = {}

    def __init__(self, content, finish):
        self._d = {"choices": [{"finish_reason": finish,
                                "message": {"role": "assistant", "content": content}}]}
        self.text = json.dumps(self._d)

    def json(self):
        return self._d


def _setup(monkeypatch, tmp_path, replies):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append(json["max_tokens"])
        return replies.pop(0)

    monkeypatch.setattr(llm.requests, "post", post)
    return sent


def test_truncated_detection():
    empty = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
    full = {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
    tool = {"choices": [{"finish_reason": "length", "message": {"content": "", "tool_calls": [1]}}]}
    assert llm.truncated(empty) and not llm.truncated(full) and not llm.truncated(tool)


def test_empty_length_answer_is_retried_with_more_tokens(monkeypatch, tmp_path):
    sent = _setup(monkeypatch, tmp_path, [FakeResp("", "length"), FakeResp('{"ok": 1}', "stop")])
    out = llm.chat("m", [{"role": "user", "content": "hi"}], max_tokens=3000)
    assert out["choices"][0]["message"]["content"] == '{"ok": 1}'
    assert sent == [3000, 6000]


def test_truncated_answer_is_not_cached_and_old_ones_are_ignored(monkeypatch, tmp_path):
    sent = _setup(monkeypatch, tmp_path, [FakeResp("", "length")] * 3)
    llm.chat("m", [{"role": "user", "content": "x"}], max_tokens=3000, retries=3)
    assert not list(tmp_path.rglob("*.json"))          # nothing cached
    sent2 = _setup(monkeypatch, tmp_path, [FakeResp("good", "stop")])
    key = cache.make_key({"model": "m", "messages": [{"role": "user", "content": "y"}],
                          "temperature": 0.2, "max_tokens": 3000})
    cache.put("llm", key, {"choices": [{"finish_reason": "length", "message": {"content": ""}}]})
    out = llm.chat("m", [{"role": "user", "content": "y"}], max_tokens=3000)
    assert out["choices"][0]["message"]["content"] == "good" and sent2 == [3000]


def test_reasoning_effort_sent_and_escalation_can_be_disabled(monkeypatch, tmp_path):
    bodies = []
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(dict(json))
        return FakeResp("", "length")

    monkeypatch.setattr(llm.requests, "post", post)
    out = llm.chat("m", [{"role": "user", "content": "z"}], max_tokens=4000,
                   reasoning_effort="none", escalate=False, retries=3)
    assert llm.truncated(out)
    assert all(b["reasoning_effort"] == "none" and b["max_tokens"] == 4000 for b in bodies)
