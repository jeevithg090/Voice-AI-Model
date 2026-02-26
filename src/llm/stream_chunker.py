import re
import time
from typing import Generator, Optional


class StreamChunker:
    """
    Streaming text chunker for low-latency TTS.
    Emits on punctuation, word-count threshold, or time-based flush.
    """

    def __init__(self, min_words: int = 4, flush_ms: int = 800, min_chars: int = 14):
        self.buffer = ""
        self.min_words = max(1, int(min_words))
        self.flush_ms = max(0, int(flush_ms))
        self.min_chars = max(1, int(min_chars))
        # Include Indic danda and newline for multilingual sentence boundaries.
        self.delimiters = re.compile(r"([.?!|\u0964\n]+)")
        self._pending_short = ""
        self._last_emit_ts = time.monotonic()

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\S+", text or ""))

    def _is_tiny(self, text: str) -> bool:
        if not text:
            return True
        words = self._word_count(text)
        return len(text) < self.min_chars and words < self.min_words

    def _materialize(self, candidate: str) -> Optional[str]:
        chunk = (candidate or "").strip()
        if not chunk:
            return None
        if self._pending_short:
            chunk = f"{self._pending_short} {chunk}".strip()
            self._pending_short = ""
        if self._is_tiny(chunk):
            self._pending_short = chunk
            return None
        self._last_emit_ts = time.monotonic()
        return chunk

    def process(self, token: str) -> Generator[str, None, None]:
        self.buffer += token or ""

        # 1) Sentence/phrase boundaries by punctuation.
        while True:
            match = self.delimiters.search(self.buffer)
            if not match:
                break
            end_idx = match.end()
            candidate = self.buffer[:end_idx]
            remainder = self.buffer[end_idx:]
            # Keep waiting if this punctuation fragment is still too tiny and no trailing text yet.
            if self._is_tiny(candidate.strip()) and not remainder.strip():
                break
            self.buffer = remainder
            emitted = self._materialize(candidate)
            if emitted:
                yield emitted

        # 2) Low-latency fallback: emit every N words when punctuation is delayed.
        while True:
            words = list(re.finditer(r"\S+\s*", self.buffer))
            if len(words) < self.min_words:
                break
            end_idx = words[self.min_words - 1].end()
            candidate = self.buffer[:end_idx]
            self.buffer = self.buffer[end_idx:]
            emitted = self._materialize(candidate)
            if emitted:
                yield emitted

        # 3) Time-based flush for slow token streams.
        if self.flush_ms > 0 and self.buffer.strip():
            elapsed_ms = (time.monotonic() - self._last_emit_ts) * 1000.0
            if elapsed_ms >= self.flush_ms and len(self.buffer.strip()) >= self.min_chars:
                candidate = self.buffer
                self.buffer = ""
                emitted = self._materialize(candidate)
                if emitted:
                    yield emitted

    def flush(self) -> Optional[str]:
        remaining = self.buffer.strip()
        self.buffer = ""
        if self._pending_short:
            remaining = f"{self._pending_short} {remaining}".strip() if remaining else self._pending_short
            self._pending_short = ""
        if not remaining:
            return None
        self._last_emit_ts = time.monotonic()
        return remaining
