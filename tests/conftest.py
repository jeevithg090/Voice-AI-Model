"""Shared test setup.

Environment variables are set *before* any ``src`` module is imported because
``src.config.settings`` is resolved once at import time.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="voice-ai-tests-"))
os.environ.setdefault("SEMANTIC_CACHE_PATH", str(_TMP / "semantic_cache.json"))
os.environ.setdefault("ADMIN_DB_PATH", str(_TMP / "admin.db"))
os.environ.setdefault("CONVERSATION_MEDIA_DIR", str(_TMP / "media"))
# Never reach real services from unit tests.
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("REDIS_HOST", "127.0.0.1")
os.environ.setdefault("REDIS_PORT", "9")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
