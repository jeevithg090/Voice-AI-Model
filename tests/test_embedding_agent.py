import pytest

from src.llm import embedding_agent as ea
from src.llm.semantic_cache import SemanticCache


@pytest.fixture
def agent(monkeypatch, tmp_path):
    cache = SemanticCache(dimension=3, cache_file=str(tmp_path / "c.json"), use_cosine_only=True)
    monkeypatch.setattr(ea, "cache", cache)
    monkeypatch.setattr(ea.context_manager, "ping", lambda: False)  # Redis down
    monkeypatch.setattr(ea, "embed_text", lambda text: [1.0, 0.0, 0.0])
    return ea, cache


def _stream(tokens):
    def fake_chat_stream(messages, max_tokens, temperature):
        fake_chat_stream.messages = messages
        yield from tokens

    return fake_chat_stream


def test_streaming_llm_response_is_cached(agent, monkeypatch):
    ea, cache = agent
    fake = _stream(["Namaste", "!"])
    monkeypatch.setattr("src.llm.ollama_client.chat_stream", fake)

    out = ea.process_with_context("hello", session_id="abc", system_prompt="SYS", stream=True)
    assert out["source"] == "llm" and out["session_id"] == "temp_session"
    assert "".join(out["response_generator"]) == "Namaste!"
    assert fake.messages == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]

    again = ea.process_with_context("hello", session_id="abc", system_prompt="SYS", stream=True)
    assert again["source"] == "semantic-cache"
    assert "".join(again["response_generator"]) == "Namaste!"


def test_empty_llm_stream_is_not_cached(agent, monkeypatch):
    ea, cache = agent
    monkeypatch.setattr("src.llm.ollama_client.chat_stream", _stream([]))
    out = ea.process_with_context("hello", stream=True, system_prompt="SYS")
    assert "".join(out["response_generator"]) == ""
    assert cache.entries == []


def test_embedding_failure_falls_back_to_llm(agent, monkeypatch):
    ea, _ = agent

    def boom(_):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(ea, "embed_text", boom)
    monkeypatch.setattr("src.llm.ollama_client.chat_stream", _stream(["ok"]))
    out = ea.process_with_context("hi", stream=True, system_prompt="SYS")
    assert "".join(out["response_generator"]) == "ok"


def test_default_prompt_switches_to_minimal_indic_prompt(agent, monkeypatch):
    ea, _ = agent
    fake = _stream(["सही"])
    monkeypatch.setattr("src.llm.ollama_client.chat_stream", fake)
    out = ea.process_with_context(
        "मदद", stream=True, metadata={"detected_language": "hi", "sustained_distress": "True"}
    )
    list(out["response_generator"])
    system = fake.messages[0]["content"]
    assert "Reply ONLY in Hindi script" in system
    assert "USER IS IN DISTRESS" in system
