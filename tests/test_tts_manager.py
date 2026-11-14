import asyncio
import dataclasses

import pytest

from src.tts import tts_manager as tm


@pytest.fixture
def manager(tmp_path):
    mgr = tm.TTSManager()
    mgr.models_dir = str(tmp_path)
    return mgr


@pytest.mark.parametrize(("lang", "engine"), [("kn", "edge"), ("en", "piper"), ("ta", "piper"), ("ml", "edge")])
def test_engine_routing_auto(manager, monkeypatch, lang, engine):
    monkeypatch.setattr(tm, "settings", dataclasses.replace(tm.settings, TTS_BACKEND="auto"))
    assert manager.get_engine(lang) == engine


@pytest.mark.parametrize("backend", ["piper", "edge"])
def test_engine_routing_forced(manager, monkeypatch, backend):
    monkeypatch.setattr(tm, "settings", dataclasses.replace(tm.settings, TTS_BACKEND=backend))
    assert manager.get_engine("kn") == backend


def test_pick_piper_model_prefers_language_then_english(manager, tmp_path):
    assert manager._pick_piper_model("ta") is None
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"x")
    assert manager._pick_piper_model("ta").endswith("en_US-lessac-medium.onnx")
    (tmp_path / "ta_IN-iitm-female-s1-medium.onnx").write_bytes(b"x")
    assert manager._pick_piper_model("ta").endswith("ta_IN-iitm-female-s1-medium.onnx")


def test_speak_falls_back_from_piper_to_edge(manager, monkeypatch):
    async def no_piper(text, lang):
        return None

    async def edge(text, lang):
        return b"edge-audio"

    monkeypatch.setattr(manager, "get_engine", lambda lang: "piper")
    monkeypatch.setattr(manager, "_speak_piper", no_piper)
    monkeypatch.setattr(manager, "_speak_edge", edge)
    assert asyncio.run(manager.speak("hello", "en")) == b"edge-audio"


def test_speak_edge_falls_back_to_english_piper(manager, monkeypatch):
    seen = {}

    async def edge_fail(text, lang):
        return None

    async def piper(text, lang):
        seen["lang"] = lang
        return b"piper-audio"

    monkeypatch.setattr(manager, "get_engine", lambda lang: "edge")
    monkeypatch.setattr(manager, "_speak_edge", edge_fail)
    monkeypatch.setattr(manager, "_speak_piper", piper)
    assert asyncio.run(manager.speak("ನಮಸ್ಕಾರ", "kn")) == b"piper-audio"
    assert seen["lang"] == "en"
