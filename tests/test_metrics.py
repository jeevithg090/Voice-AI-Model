from src.realtime.metrics import Metrics


def test_counters_never_go_negative():
    m = Metrics()
    m.inc("active_calls")
    m.dec("active_calls")
    m.dec("active_calls")
    assert m.get("active_calls") == 0
    m.inc("custom", 5)
    assert m.get("custom") == 5


def test_latency_window_and_stats():
    m = Metrics(window=3)
    for v in (10, 20, 30, 40):
        m.record_latency("stt_ms", v)
    stats = m.latency_stats()["stt_ms"]
    assert stats["count"] == 3  # oldest value dropped
    assert stats["avg_ms"] == 30.0
    assert stats["p50_ms"] == 30.0
    assert 38.0 <= stats["p95_ms"] <= 40.0


def test_snapshot_shape():
    m = Metrics()
    m.inc("calls_total")
    m.record_latency("tts_ms", 5)
    snap = m.snapshot()
    assert snap["calls_total"] == 1
    assert snap["latencies"] == {"tts_ms": [5.0]}
    assert snap["latency_stats"]["tts_ms"]["count"] == 1
