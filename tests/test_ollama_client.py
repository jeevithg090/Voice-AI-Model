import json

import pytest
import requests

from src.llm import ollama_client


class FakeResponse:
    def __init__(self, payload=None, lines=None, status=200):
        self._payload = payload or {}
        self._lines = lines or []
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_embed_text(monkeypatch):
    calls = {}

    def fake_post(url, json=None, timeout=None, **_):
        calls["url"], calls["json"] = url, json
        return FakeResponse({"embedding": [0.1, 0.2]})

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    assert ollama_client.embed_text("hi") == [0.1, 0.2]
    assert calls["url"].endswith("/api/embeddings")
    assert calls["json"]["prompt"] == "hi"


def test_embed_text_propagates_errors(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(ollama_client.requests, "post", boom)
    with pytest.raises(requests.exceptions.RequestException):
        ollama_client.embed_text("hi")


def test_chat_stream_yields_tokens(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}).encode(),
        b"",
        json.dumps({"message": {"content": "lo"}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True}).encode(),
    ]
    monkeypatch.setattr(ollama_client.requests, "post", lambda *a, **k: FakeResponse(lines=lines))
    tokens = list(ollama_client.chat_stream(messages=[{"role": "user", "content": "x"}]))
    assert "".join(tokens) == "Hello"


def test_check_status_when_server_down(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(ollama_client.requests, "get", boom)
    status = ollama_client.check_ollama_status()
    assert status["server_running"] is False


def test_chat_stream_error_yields_nothing(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(ollama_client.requests, "post", boom)
    assert list(ollama_client.chat_stream(messages=[])) == []
    assert list(ollama_client.generate_response_stream("hi")) == []
