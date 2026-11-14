"""Thread-safe in-process counters and rolling latency windows for /metrics."""

from __future__ import annotations

import statistics
import threading
from collections import deque
from typing import Any

import numpy as np


class Metrics:
    def __init__(self, window: int = 200) -> None:
        self.window = max(1, int(window))
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "calls_total": 0,
            "active_calls": 0,
            "utterances_total": 0,
        }
        self._latencies: dict[str, deque[float]] = {}

    def inc(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta

    def dec(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = max(0, self._counters.get(name, 0) - delta)

    def get(self, name: str, default: int = 0) -> int:
        with self._lock:
            return self._counters.get(name, default)

    def record_latency(self, name: str, value_ms: float) -> None:
        with self._lock:
            bucket = self._latencies.get(name)
            if bucket is None:
                bucket = self._latencies[name] = deque(maxlen=self.window)
            bucket.append(float(value_ms))

    def latency_stats(self) -> dict[str, dict[str, float]]:
        with self._lock:
            snapshot = {name: list(vals) for name, vals in self._latencies.items() if vals}
        return {
            name: {
                "count": len(vals),
                "avg_ms": float(sum(vals) / len(vals)),
                "p50_ms": float(statistics.median(vals)),
                "p95_ms": float(np.percentile(vals, 95)),
            }
            for name, vals in snapshot.items()
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = dict(self._counters)
            data["latencies"] = {name: list(vals) for name, vals in self._latencies.items()}
        data["latency_stats"] = self.latency_stats()
        return data
