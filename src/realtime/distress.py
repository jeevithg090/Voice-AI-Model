"""Sustained-distress tracking for the acoustic emergency classifier.

The classifier produces a probability for each short audio window. A single
spike is not enough to escalate; distress is considered *sustained* when at
least ``min_hits`` windows exceed ``threshold`` within ``window_s`` seconds.
Once triggered, the state latches for ``latch_s`` seconds so the ASR/LLM path
(which runs a few seconds later, after the utterance ends) can still see it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class DistressTracker:
    def __init__(
        self,
        window_s: float = 5.0,
        min_hits: int = 3,
        threshold: float = 0.7,
        latch_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.window_s = float(window_s)
        self.min_hits = int(min_hits)
        self.threshold = float(threshold)
        self.latch_s = float(latch_s) if latch_s is not None else max(10.0, 2 * self.window_s)
        self._clock = clock
        self._events: deque[float] = deque()
        self._sustained_at: float | None = None
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0] > self.window_s:
            self._events.popleft()

    def observe(self, probability: float) -> bool:
        """Record one classifier output. Returns True when this observation
        newly triggers sustained distress."""
        now = self._clock()
        with self._lock:
            self._prune(now)
            if probability > self.threshold:
                self._events.append(now)
            if len(self._events) >= self.min_hits:
                self._events.clear()  # avoid re-triggering on every window
                self._sustained_at = now
                return True
            return False

    @property
    def hit_count(self) -> int:
        with self._lock:
            self._prune(self._clock())
            return len(self._events)

    def is_sustained(self) -> bool:
        """True while a recent sustained-distress trigger is still latched."""
        with self._lock:
            if self._sustained_at is None:
                return False
            if self._clock() - self._sustained_at > self.latch_s:
                self._sustained_at = None
                return False
            return True

    def consume(self) -> bool:
        """Return the latched state and clear it (use once a handover was issued)."""
        with self._lock:
            sustained = self._sustained_at is not None and (self._clock() - self._sustained_at <= self.latch_s)
            self._sustained_at = None
            return sustained

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._sustained_at = None
