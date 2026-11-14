"""Central runtime configuration, read from environment variables.

Values are resolved once at import time. A local ``.env`` file is loaded
automatically when ``python-dotenv`` is installed (it never overrides
variables already present in the process environment).
"""

import os
from dataclasses import dataclass

try:  # Optional: load a local .env for development convenience.
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover - dotenv is optional
    pass


def _env(name: str, default=None):
    value = os.getenv(name)
    return default if value is None else value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_list(name: str, default: str = "") -> tuple:
    """Parse a comma-separated env var into a tuple of non-empty, stripped items."""
    raw = os.getenv(name)
    if raw is None:
        raw = default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = _env("LLM_MODEL", "llama3.2:3b")
    SUMMARIZE_MODEL: str = _env("SUMMARIZE_MODEL", "llama3.2:1b")
    EMBED_MODEL: str = _env("EMBED_MODEL", "nomic-embed-text")

    # Speech Models
    WHISPER_MODEL: str = _env("WHISPER_MODEL", "openai/whisper-small")
    LID_MODEL: str = _env("LID_MODEL", "facebook/mms-lid-256")

    # TTS Configuration
    TTS_BACKEND: str = _env("TTS_BACKEND", "auto")
    PIPER_MODELS_DIR: str = _env("PIPER_MODELS_DIR", "src/tts/piper_models")
    PIPER_PATH: str = _env("PIPER_PATH", "")
    EDGE_TTS_VOICE_TA: str = _env("EDGE_TTS_VOICE_TA", "ta-IN-PallaviNeural")
    EDGE_TTS_VOICE_KN: str = _env("EDGE_TTS_VOICE_KN", "kn-IN-SapnaNeural")
    EDGE_TTS_VOICE_EN: str = _env("EDGE_TTS_VOICE_EN", "en-US-AriaNeural")
    EDGE_TTS_VOICE_HI: str = _env("EDGE_TTS_VOICE_HI", "hi-IN-SwaraNeural")
    EDGE_TTS_VOICE_TE: str = _env("EDGE_TTS_VOICE_TE", "te-IN-MohanNeural")
    EDGE_TTS_RATE: str = _env("EDGE_TTS_RATE", "+0%")
    EDGE_TTS_VOLUME: str = _env("EDGE_TTS_VOLUME", "+0%")
    EDGE_TTS_PITCH: str = _env("EDGE_TTS_PITCH", "+0Hz")

    # Audio Processing
    SILENCE_THRESHOLD: float = _env_float("SILENCE_THRESHOLD", 0.01)
    SILENCE_DURATION: float = _env_float("SILENCE_DURATION", 2.0)
    MIN_SPEECH_DURATION: float = _env_float("MIN_SPEECH_DURATION", 1.0)

    # Feature Flags
    ENABLE_TWILIO: bool = _env_bool("ENABLE_TWILIO", True)
    ENABLE_LID_AUTODETECT: bool = _env_bool("ENABLE_LID_AUTODETECT", True)

    # Handover & Metrics
    HANDOVER_MODE: str = _env("HANDOVER_MODE", "phone")
    METRICS_WINDOW: int = _env_int("METRICS_WINDOW", 200)
    STREAM_CHUNK_MIN_WORDS: int = _env_int("STREAM_CHUNK_MIN_WORDS", 4)
    STREAM_CHUNK_MAX_QUEUE: int = _env_int("STREAM_CHUNK_MAX_QUEUE", 8)
    STREAM_CHUNK_FLUSH_MS: int = _env_int("STREAM_CHUNK_FLUSH_MS", 800)
    VOICE_TURN_STREAM_ENABLED: bool = _env_bool("VOICE_TURN_STREAM_ENABLED", True)
    VOICE_TURN_MAX_TOKENS: int = _env_int("VOICE_TURN_MAX_TOKENS", 80)

    # Redis Configuration
    REDIS_HOST: str = _env("REDIS_HOST", "localhost")
    REDIS_PORT: int = _env_int("REDIS_PORT", 6379)
    REDIS_DB: int = _env_int("REDIS_DB", 0)
    REDIS_SESSION_TTL: int = _env_int("REDIS_SESSION_TTL", 1800)

    # Server Configuration
    SERVER_HOST: str = _env("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = _env_int("SERVER_PORT", 8080)
    # Comma-separated list of allowed CORS origins ("*" allows any origin).
    CORS_ORIGINS: tuple = _env_list("CORS_ORIGINS", "*")
    # Maximum accepted upload size for /voice-turn, /voice-turn-stream, /detect-lang.
    MAX_UPLOAD_MB: int = _env_int("MAX_UPLOAD_MB", 20)

    # Admin panel / API protection (HTTP Basic auth). Empty password = auth disabled.
    ADMIN_USERNAME: str = _env("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = _env("ADMIN_PASSWORD", "")
    ADMIN_DB_PATH: str = _env("ADMIN_DB_PATH", "data/admin/voice_agent.db")
    CONVERSATION_MEDIA_DIR: str = _env("CONVERSATION_MEDIA_DIR", "data/conversation_media")

    # WebRTC ICE configuration. TURN is optional but recommended behind NAT.
    STUN_URLS: tuple = _env_list(
        "STUN_URLS", "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302"
    )
    TURN_URLS: tuple = _env_list("TURN_URLS", "")
    TURN_USERNAME: str = _env("TURN_USERNAME", "")
    TURN_CREDENTIAL: str = _env("TURN_CREDENTIAL", "")

    # Semantic cache persistence file.
    SEMANTIC_CACHE_PATH: str = _env("SEMANTIC_CACHE_PATH", "semantic_cache.json")

    # Emergency Detection
    ENABLE_EMERGENCY_DETECTION: bool = _env_bool("ENABLE_EMERGENCY_DETECTION", True)
    MIMI_MODEL: str = _env("MIMI_MODEL", "kyutai/mimi")
    EMERGENCY_CLASSIFIER_PATH: str = _env(
        "EMERGENCY_CLASSIFIER_PATH", "models/emergency_classifier.pt"
    )
    DISTRESS_WINDOW: int = _env_int("DISTRESS_WINDOW", 5)
    DISTRESS_MIN_HITS: int = _env_int("DISTRESS_MIN_HITS", 3)
    EMERGENCY_THRESHOLD: float = _env_float("EMERGENCY_THRESHOLD", 0.7)


settings = Settings()
