"""
Redis Context Manager for Voice AI Pipeline
Manages conversation context with key-value storage, TTL, and async summarization.
"""

import redis
import json
import uuid
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import settings for configuration
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Thread pool for async summarization
_executor = ThreadPoolExecutor(max_workers=2)


def _generate_summary(history: List[Dict], ollama_url: str = None) -> str:
    """
    Generate a summary of conversation history using Ollama.
    This runs in a background thread.
    """
    import requests

    if ollama_url is None:
        ollama_url = settings.OLLAMA_BASE_URL

    # Build conversation text for summarization
    conv_text = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in history]
    )

    prompt = f"""Summarize this medical conversation in 2-3 sentences. 
Focus on: patient symptoms, age, medical conditions, and key concerns.

Conversation:
{conv_text}

Summary:"""

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": settings.SUMMARIZE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 100, "temperature": 0.3},
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"[WARN] Summarization failed: {e}")
        # Fallback: simple extraction of key info
        return f"Previous context: {conv_text[:200]}..."


class RedisContextManager:
    """
    Manages conversation context in Redis for multi-turn dialogue.
    Features N-1 async summarization to prevent info loss during trimming.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        session_ttl: int = None,  # 30 minutes default
        max_history: int = 8,  # Keep last N turns (pairs)
        prefix: str = "session",
        summarize_threshold: int = 6,  # Trigger summarization at N-threshold
    ):
        """
        Initialize Redis connection.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            session_ttl: Session expiry in seconds
            max_history: Maximum conversation turns to keep
            prefix: Key prefix for session data
            summarize_threshold: Number of turns to keep when summarizing
        """
        redis_host = settings.REDIS_HOST if host is None else host
        redis_port = settings.REDIS_PORT if port is None else port
        redis_db = settings.REDIS_DB if db is None else db
        ttl = settings.REDIS_SESSION_TTL if session_ttl is None else session_ttl

        self.client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
            # Fast-fail when Redis is unavailable so voice turns don't stall.
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
            retry_on_timeout=False,
        )
        self.session_ttl = ttl
        self.max_history = max_history
        self.prefix = prefix
        self.summarize_threshold = summarize_threshold
        self._summarizing_sessions = set()  # Track sessions being summarized

    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.prefix}:{session_id}"

    def ping(self) -> bool:
        """Check if Redis is connected."""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    def create_session(self, language: str = "en", metadata: Dict = None) -> str:
        """
        Create a new conversation session.
        """
        session_id = str(uuid.uuid4())[:8]
        key = self._get_key(session_id)

        session_data = {
            "history": [],
            "summary": None,  # Will be populated by async summarization
            "language": language,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "metadata": metadata or {},
            "summarization_pending": False,
        }

        self.client.setex(key, self.session_ttl, json.dumps(session_data))
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get full session data."""
        key = self._get_key(session_id)
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def get_context(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for LLM context."""
        session = self.get_session(session_id)
        if session:
            return session.get("history", [])
        return []

    def _async_summarize(self, session_id: str, history_to_summarize: List[Dict]):
        """
        Background task to summarize old conversation turns.
        Called at N-1 threshold to prepare for upcoming trim.
        """
        try:
            # Generate summary
            summary = _generate_summary(history_to_summarize)

            # Update session with summary
            session = self.get_session(session_id)
            if session:
                session["summary"] = summary
                session["summarization_pending"] = False
                key = self._get_key(session_id)
                self.client.setex(key, self.session_ttl, json.dumps(session))
                print(
                    f"[INFO] Async summarization complete for {session_id}: {summary[:50]}..."
                )
        except Exception as e:
            print(f"[ERROR] Async summarization failed: {e}")
        finally:
            self._summarizing_sessions.discard(session_id)

    def add_turn(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a conversation turn. Does NOT auto-trigger summarization.
        Summarization is triggered via trigger_summarization_during_stt() by STT layer.

        At N turns: Uses summary if available + trims old messages
        """
        session = self.get_session(session_id)
        if not session:
            return False

        # Add new message
        session["history"].append({"role": role, "content": content})

        history_len = len(session["history"])
        max_messages = self.max_history * 2  # pairs (user + assistant)

        # Check if we need summarization but don't have it
        if history_len >= max_messages - 4 and not session.get("summary"):
            session["needs_summarization"] = True

        # ===== N THRESHOLD: Apply trim with summary =====
        if history_len >= max_messages:
            if session.get("summary"):
                # Keep summary + recent turns
                keep_count = self.summarize_threshold * 2
                session["history"] = session["history"][-keep_count:]
                print(
                    f"[INFO] Trimmed with summary for {session_id}, keeping {keep_count} messages"
                )
            else:
                # Fallback: simple trim if summarization not ready
                session["history"] = session["history"][-max_messages:]
                print(f"[WARN] Simple trim for {session_id}, summary not ready")

        # Update timestamps and save
        session["last_active"] = datetime.now().isoformat()
        key = self._get_key(session_id)
        self.client.setex(key, self.session_ttl, json.dumps(session))

        return True

    def trigger_summarization_during_stt(self, session_id: str) -> bool:
        """
        VAD-aware summarization trigger. Call this DURING STT processing
        (when user is speaking) to ensure summarization runs when Ollama is idle.

        This is the key method for zero-latency-impact summarization!

        Call timing:
            User speaks → VAD detects → STT starts → CALL THIS METHOD

        Returns:
            True if summarization was triggered, False if not needed/already running
        """
        session = self.get_session(session_id)
        if not session:
            return False

        # Skip if already summarizing
        if session_id in self._summarizing_sessions:
            return False

        # Skip if summary already exists
        if session.get("summary"):
            return False

        # Skip if not enough history to summarize
        history_len = len(session.get("history", []))
        min_history_for_summary = 4  # At least 2 pairs before summarizing

        if history_len < min_history_for_summary:
            return False

        # Check if approaching limit (N-4 or more)
        max_messages = self.max_history * 2
        if history_len >= max_messages - 4 or session.get("needs_summarization"):
            # Good time to summarize - STT is processing, Ollama is idle!
            summarize_count = min(
                history_len // 2, 6
            )  # Summarize first half, max 6 messages
            history_to_summarize = session["history"][:summarize_count]

            session["summarization_pending"] = True
            session["needs_summarization"] = False
            self._summarizing_sessions.add(session_id)

            # Save the pending flag
            key = self._get_key(session_id)
            self.client.setex(key, self.session_ttl, json.dumps(session))

            # Submit to thread pool (non-blocking)
            _executor.submit(self._async_summarize, session_id, history_to_summarize)
            print(f"[VAD] Triggered summarization during STT window for {session_id}")
            return True

        return False

    def is_summarization_needed(self, session_id: str) -> bool:
        """
        Check if session needs summarization.
        STT layer can use this to decide if it should trigger summarization.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        if session.get("summary"):
            return False  # Already have summary

        if session_id in self._summarizing_sessions:
            return False  # Already in progress

        history_len = len(session.get("history", []))
        max_messages = self.max_history * 2

        return history_len >= max_messages - 4 or session.get(
            "needs_summarization", False
        )

    def get_context_for_llm(
        self, session_id: str, system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """
        Get formatted context for Ollama chat API.
        Includes summary as context if available.
        """
        session = self.get_session(session_id)
        if not session:
            return []

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Inject summary as context if available
        if session.get("summary"):
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {session['summary']}",
                }
            )

        # Add recent conversation history
        messages.extend(session.get("history", []))

        return messages

    def update_language(self, session_id: str, language: str) -> bool:
        """Update detected language for session."""
        session = self.get_session(session_id)
        if not session:
            return False
        session["language"] = language
        key = self._get_key(session_id)
        self.client.setex(key, self.session_ttl, json.dumps(session))
        return True

    def clear_session(self, session_id: str) -> bool:
        """Delete a session (end conversation)."""
        key = self._get_key(session_id)
        self._summarizing_sessions.discard(session_id)
        return self.client.delete(key) > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis connection stats."""
        try:
            info = self.client.info("clients")
            keys = self.client.keys(f"{self.prefix}:*")
            return {
                "connected": True,
                "active_sessions": len(keys),
                "connected_clients": info.get("connected_clients", 0),
                "summarizing": len(self._summarizing_sessions),
            }
        except redis.ConnectionError:
            return {"connected": False}
