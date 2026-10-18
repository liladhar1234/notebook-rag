"""LLM client retries transient failures only; a bad key must fail immediately."""
from __future__ import annotations

import pytest

from notebook_rag.generation.llm_client import OpenAICompatClient, _is_transient


class _Err(Exception):
    def __init__(self, status_code=None):
        super().__init__(f"error {status_code}")
        self.status_code = status_code


class APIConnectionError(Exception):  # same class name as the openai/anthropic SDK errors
    pass


def _client_that_raises(monkeypatch, exc_factory, ok_after=None):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)  # no real waiting
    client = OpenAICompatClient(model="m", api_key="k", base_url="http://localhost:9")
    calls = {"n": 0}

    class _Completions:
        def create(self, **_):
            calls["n"] += 1
            if ok_after is not None and calls["n"] > ok_after:
                msg = type("M", (), {"content": "fine"})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()
            raise exc_factory()

    client._client = type("X", (), {"chat": type("Y", (), {"completions": _Completions()})()})()
    return client, calls


def test_classification():
    assert all(_is_transient(_Err(code)) for code in (429, 500, 502, 503, 504))
    assert not any(_is_transient(_Err(code)) for code in (400, 401, 403, 404))
    assert _is_transient(APIConnectionError())
    assert not _is_transient(ValueError("boom"))


def test_bad_key_is_not_retried(monkeypatch):
    client, calls = _client_that_raises(monkeypatch, lambda: _Err(401))
    with pytest.raises(_Err):
        client.complete("s", "u")
    assert calls["n"] == 1


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    client, calls = _client_that_raises(monkeypatch, lambda: _Err(429), ok_after=2)
    assert client.complete("s", "u") == "fine" and calls["n"] == 3


def test_rate_limit_gives_up_after_three_attempts(monkeypatch):
    client, calls = _client_that_raises(monkeypatch, lambda: _Err(429))
    with pytest.raises(_Err):
        client.complete("s", "u")
    assert calls["n"] == 3
