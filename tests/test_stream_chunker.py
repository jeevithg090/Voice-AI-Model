from src.llm.stream_chunker import StreamChunker


def feed(chunker, tokens):
    out = []
    for tok in tokens:
        out.extend(chunker.process(tok))
    tail = chunker.flush()
    if tail:
        out.append(tail)
    return out


def test_splits_on_sentence_punctuation():
    c = StreamChunker(min_words=50, flush_ms=0)
    chunks = feed(c, ["Hello there, how are you today? ", "I am doing well, thank you."])
    assert chunks == ["Hello there, how are you today?", "I am doing well, thank you."]


def test_word_count_fallback_without_punctuation():
    c = StreamChunker(min_words=4, flush_ms=0, min_chars=1)
    chunks = feed(c, [w + " " for w in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]])
    assert chunks == ["one two three four", "five six seven eight", "nine"]


def test_tiny_fragments_are_merged_forward():
    c = StreamChunker(min_words=50, flush_ms=0, min_chars=14)
    chunks = feed(c, ["Ok. ", "Sure. ", "That works perfectly for me."])
    assert chunks == ["Ok. Sure. That works perfectly for me."]


def test_indic_danda_is_a_boundary():
    c = StreamChunker(min_words=50, flush_ms=0, min_chars=2)
    chunks = feed(c, ["नमस्ते, आप कैसे हैं। ", "मैं ठीक हूँ।"])
    assert chunks == ["नमस्ते, आप कैसे हैं।", "मैं ठीक हूँ।"]


def test_time_based_flush(monkeypatch):
    import src.llm.stream_chunker as mod

    now = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: now[0])
    c = StreamChunker(min_words=50, flush_ms=500, min_chars=5)
    assert list(c.process("slow stream text")) == []
    now[0] += 1.0
    assert list(c.process(" more")) == ["slow stream text more"]


def test_flush_on_empty_returns_none():
    assert StreamChunker().flush() is None
