from src.llm.filler_manager import (
    BACKCHANNELS,
    FILLERS,
    FillerBackchannelManager,
    FillerConfig,
    FillerType,
    get_enhanced_system_prompt,
)


def test_filler_uses_session_language_and_falls_back():
    mgr = FillerBackchannelManager()
    mgr.set_session_language("hi")
    assert mgr.get_filler() in FILLERS["hi"]["thinking"]
    assert mgr.get_filler(language="xx") in FILLERS["en"]["thinking"]
    mgr.set_session_language("unsupported")
    assert mgr.get_stats()["session_language"] == "hi"


def test_filler_budget_per_turn():
    mgr = FillerBackchannelManager(FillerConfig(min_processing_time_ms=1000, max_fillers_per_turn=1))
    assert mgr.should_use_filler(500) is False
    assert mgr.should_use_filler(1500) is True
    mgr.get_filler(filler_type=FillerType.HESITATION)
    assert mgr.should_use_filler(5000) is False
    mgr.reset_turn()
    assert mgr.should_use_filler(5000) is True


def test_backchannel_phrase_and_prompt():
    mgr = FillerBackchannelManager()
    assert mgr.get_backchannel("ta") in BACKCHANNELS["ta"]
    prompt = get_enhanced_system_prompt("BASE", "en")
    assert prompt.startswith("BASE")
