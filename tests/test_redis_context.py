import fakeredis
import pytest

from src.llm import redis_context
from src.llm.redis_context import RedisContextManager


@pytest.fixture
def ctx():
    mgr = RedisContextManager(max_history=4, summarize_threshold=2)
    mgr.client = fakeredis.FakeRedis(decode_responses=True)
    return mgr


def test_session_lifecycle(ctx):
    assert ctx.ping() is True
    sid = ctx.create_session(language="hi", metadata={"channel": "webrtc"})
    session = ctx.get_session(sid)
    assert session["language"] == "hi"
    assert session["history"] == []
    assert ctx.client.ttl(f"session:{sid}") > 0

    assert ctx.update_language(sid, "ta") is True
    assert ctx.get_session(sid)["language"] == "ta"
    assert ctx.clear_session(sid) is True
    assert ctx.get_session(sid) is None
    assert ctx.add_turn(sid, "user", "x") is False


def test_history_is_trimmed_without_summary(ctx):
    sid = ctx.create_session()
    for i in range(10):
        ctx.add_turn(sid, "user", f"m{i}")
    history = ctx.get_context(sid)
    assert len(history) <= ctx.max_history * 2
    assert history[-1]["content"] == "m9"
    assert ctx.get_session(sid)["needs_summarization"] is True


def test_context_for_llm_includes_summary(ctx):
    sid = ctx.create_session()
    ctx.add_turn(sid, "user", "hello")
    session = ctx.get_session(sid)
    session["summary"] = "Caller asked about fever."
    ctx.client.set(f"session:{sid}", __import__("json").dumps(session))

    messages = ctx.get_context_for_llm(sid, system_prompt="SYS")
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert "fever" in messages[1]["content"]
    assert messages[-1] == {"role": "user", "content": "hello"}
    assert ctx.get_context_for_llm("missing") == []


def test_summarization_trigger(ctx, monkeypatch):
    submitted = []
    monkeypatch.setattr(redis_context._executor, "submit", lambda fn, *a: submitted.append(a))
    sid = ctx.create_session()
    for i in range(2):
        ctx.add_turn(sid, "user", f"m{i}")
    assert ctx.trigger_summarization_during_stt(sid) is False  # too little history

    for i in range(4):
        ctx.add_turn(sid, "assistant", f"r{i}")
    assert ctx.is_summarization_needed(sid) is True
    assert ctx.trigger_summarization_during_stt(sid) is True
    assert submitted and submitted[0][0] == sid
    assert ctx.trigger_summarization_during_stt(sid) is False  # already running


def test_ping_false_when_unreachable():
    mgr = RedisContextManager(host="127.0.0.1", port=9)
    assert mgr.ping() is False
