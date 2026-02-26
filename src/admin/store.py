import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_load(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(value or {}, ensure_ascii=False)
    except Exception:
        return "{}"


class AdminStore:
    """
    Durable storage for admin configuration + conversation history.
    Uses SQLite by default to keep setup friction low.
    """

    def __init__(self, db_path: Optional[str] = None, media_root: Optional[str] = None):
        self.db_path = db_path or os.getenv("ADMIN_DB_PATH", "data/admin/voice_agent.db")
        self.media_root = media_root or os.getenv("CONVERSATION_MEDIA_DIR", "data/conversation_media")
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.media_root, exist_ok=True)

        self._init_db()
        self._seed_defaults()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        ddl = """
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            system_prompt TEXT NOT NULL,
            defaults_json TEXT NOT NULL DEFAULT '{}',
            is_builtin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            template_id INTEGER,
            config_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (template_id) REFERENCES templates(id)
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            agent_id INTEGER,
            language TEXT DEFAULT 'en',
            status TEXT NOT NULL DEFAULT 'open',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (agent_id) REFERENCES agents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);

        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL DEFAULT '',
            language TEXT DEFAULT 'en',
            audio_path TEXT,
            audio_mime TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id);
        """
        with self._lock:
            with self._conn() as conn:
                conn.executescript(ddl)
                conn.commit()

    def _seed_defaults(self) -> None:
        default_templates = [
            {
                "name": "Emergency Response",
                "description": "Calm, concise emergency-first assistant for urgent support.",
                "system_prompt": (
                    "You are {{assistant_name}}, a multilingual voice support assistant. "
                    "Your domain is {{domain}}. Keep tone {{tone}} and concise for spoken calls.\n"
                    "Always reply in the caller language: {{detected_language}}. "
                    "If confidence is low, ask preferred language politely.\n"
                    "Primary goal: {{objective}}.\n"
                    "Safety policy: {{safety_policy}}.\n"
                    "Escalation policy: {{escalation_policy}}.\n"
                    "Never fabricate facts. If unsure, state uncertainty and offer human handover."
                ),
                "defaults": {
                    "assistant_name": "Asha",
                    "domain": "Emergency and citizen support",
                    "tone": "calm, empathetic, and clear",
                    "objective": "Collect key details quickly and guide the caller safely",
                    "safety_policy": "Prioritize life-safety and direct emergency escalation when needed",
                    "escalation_policy": "If distress is detected or query is high risk, suggest immediate human escalation",
                    "temperature": 0.5,
                    "max_tokens": 220,
                    "similarity_threshold": 0.85,
                },
                "is_builtin": 1,
            },
            {
                "name": "Customer Support",
                "description": "Transactional support assistant for product and service questions.",
                "system_prompt": (
                    "You are {{assistant_name}}, a multilingual customer support voice agent for {{domain}}.\n"
                    "Reply in {{detected_language}} with short spoken sentences.\n"
                    "Objective: {{objective}}.\n"
                    "Tone: {{tone}}.\n"
                    "Policy: {{safety_policy}}.\n"
                    "If policy constraints block action, explain clearly and offer next best step."
                ),
                "defaults": {
                    "assistant_name": "Mitra",
                    "domain": "Customer support",
                    "tone": "professional and friendly",
                    "objective": "Resolve caller issue in minimum turns",
                    "safety_policy": "Do not expose sensitive account data; verify identity when needed",
                    "escalation_policy": "Escalate to human for billing, account lock, or policy exceptions",
                    "temperature": 0.6,
                    "max_tokens": 260,
                    "similarity_threshold": 0.84,
                },
                "is_builtin": 1,
            },
            {
                "name": "Healthcare Intake",
                "description": "Structured intake assistant for symptom collection and routing.",
                "system_prompt": (
                    "You are {{assistant_name}}, a voice intake assistant for {{domain}}.\n"
                    "Use {{detected_language}} only.\n"
                    "Tone: {{tone}}. Objective: {{objective}}.\n"
                    "Collect structured facts: age, symptoms, duration, severity, known conditions.\n"
                    "Do not diagnose definitively. Provide safe guidance and escalation per {{escalation_policy}}."
                ),
                "defaults": {
                    "assistant_name": "Sanjeev",
                    "domain": "Healthcare triage",
                    "tone": "reassuring and neutral",
                    "objective": "Collect clinically useful structured intake and route safely",
                    "safety_policy": "Avoid diagnosis claims; provide caution and escalation guidance",
                    "escalation_policy": "Escalate urgent symptoms immediately",
                    "temperature": 0.45,
                    "max_tokens": 240,
                    "similarity_threshold": 0.86,
                },
                "is_builtin": 1,
            },
        ]

        default_agent_name = "Default Voice Agent"

        with self._lock:
            with self._conn() as conn:
                now = _utc_now()
                for tpl in default_templates:
                    existing = conn.execute(
                        "SELECT id FROM templates WHERE name = ?",
                        (tpl["name"],),
                    ).fetchone()
                    if existing:
                        continue
                    conn.execute(
                        """
                        INSERT INTO templates (
                            name, description, system_prompt, defaults_json,
                            is_builtin, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tpl["name"],
                            tpl["description"],
                            tpl["system_prompt"],
                            _json_dump(tpl["defaults"]),
                            tpl["is_builtin"],
                            now,
                            now,
                        ),
                    )

                agent_row = conn.execute("SELECT id, is_active FROM agents WHERE name = ?", (default_agent_name,)).fetchone()
                if not agent_row:
                    template_row = conn.execute(
                        "SELECT id FROM templates WHERE name = ?",
                        ("Emergency Response",),
                    ).fetchone()
                    template_id = int(template_row["id"]) if template_row else None
                    conn.execute(
                        """
                        INSERT INTO agents (
                            name, description, template_id, config_json,
                            is_active, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            default_agent_name,
                            "Default production agent profile.",
                            template_id,
                            _json_dump(
                                {
                                    "assistant_name": "Asha",
                                    "tone": "calm and practical",
                                    "objective": "Help quickly with short spoken responses",
                                    "temperature": 0.55,
                                    "max_tokens": 230,
                                }
                            ),
                            1,
                            now,
                            now,
                        ),
                    )

                active_count = conn.execute("SELECT COUNT(*) AS c FROM agents WHERE is_active = 1").fetchone()["c"]
                if active_count == 0:
                    first_agent = conn.execute("SELECT id FROM agents ORDER BY id ASC LIMIT 1").fetchone()
                    if first_agent:
                        conn.execute("UPDATE agents SET is_active = 1 WHERE id = ?", (int(first_agent["id"]),))

                conn.commit()

    def _normalize_template_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "description": row["description"] or "",
            "system_prompt": row["system_prompt"],
            "defaults": _json_load(row["defaults_json"], {}),
            "is_builtin": bool(row["is_builtin"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _normalize_agent_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = {
            "id": int(row["id"]),
            "name": row["name"],
            "description": row["description"] or "",
            "template_id": int(row["template_id"]) if row["template_id"] is not None else None,
            "config": _json_load(row["config_json"], {}),
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if "template_name" in row.keys():
            data["template_name"] = row["template_name"]
        return data

    def list_templates(self) -> List[Dict[str, Any]]:
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute("SELECT * FROM templates ORDER BY id DESC").fetchall()
        return [self._normalize_template_row(r) for r in rows]

    def get_template(self, template_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._conn() as conn:
                row = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
        if not row:
            return None
        return self._normalize_template_row(row)

    def create_template(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = (payload.get("name") or "").strip()
        if not name:
            raise ValueError("Template name is required")

        system_prompt = (payload.get("system_prompt") or "").strip()
        if not system_prompt:
            raise ValueError("system_prompt is required")

        description = (payload.get("description") or "").strip()
        defaults = payload.get("defaults") or {}
        if not isinstance(defaults, dict):
            defaults = {}

        now = _utc_now()
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO templates (
                        name, description, system_prompt, defaults_json,
                        is_builtin, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (name, description, system_prompt, _json_dump(defaults), now, now),
                )
                conn.commit()
                template_id = cur.lastrowid
                row = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
        return self._normalize_template_row(row)

    def update_template(self, template_id: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {
            "name": "name",
            "description": "description",
            "system_prompt": "system_prompt",
            "defaults": "defaults_json",
        }

        updates: List[str] = []
        values: List[Any] = []
        for key, col in allowed.items():
            if key not in payload:
                continue
            val = payload[key]
            if key == "defaults":
                if not isinstance(val, dict):
                    val = {}
                val = _json_dump(val)
            elif isinstance(val, str):
                val = val.strip()
            updates.append(f"{col} = ?")
            values.append(val)

        if not updates:
            return self.get_template(template_id)

        updates.append("updated_at = ?")
        values.append(_utc_now())
        values.append(template_id)

        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE templates SET {', '.join(updates)} WHERE id = ?",
                    tuple(values),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()

        if not row:
            return None
        return self._normalize_template_row(row)

    def list_agents(self) -> List[Dict[str, Any]]:
        sql = """
        SELECT a.*, t.name AS template_name
        FROM agents a
        LEFT JOIN templates t ON t.id = a.template_id
        ORDER BY a.id DESC
        """
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(sql).fetchall()
        return [self._normalize_agent_row(r) for r in rows]

    def get_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT a.*, t.name AS template_name
        FROM agents a
        LEFT JOIN templates t ON t.id = a.template_id
        WHERE a.id = ?
        """
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(sql, (agent_id,)).fetchone()
        if not row:
            return None
        return self._normalize_agent_row(row)

    def get_active_agent(self) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT a.*, t.name AS template_name
        FROM agents a
        LEFT JOIN templates t ON t.id = a.template_id
        WHERE a.is_active = 1
        ORDER BY a.updated_at DESC
        LIMIT 1
        """
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(sql).fetchone()
        if not row:
            return None
        return self._normalize_agent_row(row)

    def create_agent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = (payload.get("name") or "").strip()
        if not name:
            raise ValueError("Agent name is required")

        description = (payload.get("description") or "").strip()
        template_id = payload.get("template_id")
        if template_id is not None:
            try:
                template_id = int(template_id)
            except Exception as exc:
                raise ValueError("template_id must be an integer") from exc

        config = payload.get("config") or {}
        if not isinstance(config, dict):
            config = {}

        now = _utc_now()
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO agents (
                        name, description, template_id, config_json,
                        is_active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (name, description, template_id, _json_dump(config), now, now),
                )
                conn.commit()
                agent_id = cur.lastrowid
        agent = self.get_agent(int(agent_id))
        if not agent:
            raise RuntimeError("Failed to load created agent")
        return agent

    def update_agent(self, agent_id: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {
            "name": "name",
            "description": "description",
            "template_id": "template_id",
            "config": "config_json",
        }

        updates: List[str] = []
        values: List[Any] = []
        for key, col in allowed.items():
            if key not in payload:
                continue
            val = payload[key]
            if key == "template_id":
                if val is None:
                    pass
                else:
                    val = int(val)
            elif key == "config":
                if not isinstance(val, dict):
                    val = {}
                val = _json_dump(val)
            elif isinstance(val, str):
                val = val.strip()
            updates.append(f"{col} = ?")
            values.append(val)

        if not updates:
            return self.get_agent(agent_id)

        updates.append("updated_at = ?")
        values.append(_utc_now())
        values.append(agent_id)

        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE agents SET {', '.join(updates)} WHERE id = ?",
                    tuple(values),
                )
                conn.commit()

        return self.get_agent(agent_id)

    def set_active_agent(self, agent_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._conn() as conn:
                exists = conn.execute("SELECT id FROM agents WHERE id = ?", (agent_id,)).fetchone()
                if not exists:
                    return None
                conn.execute("UPDATE agents SET is_active = 0")
                conn.execute(
                    "UPDATE agents SET is_active = 1, updated_at = ? WHERE id = ?",
                    (_utc_now(), agent_id),
                )
                conn.commit()
        return self.get_agent(agent_id)

    def _merge_metadata(self, base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(base or {})
        if isinstance(extra, dict):
            merged.update(extra)
        return merged

    def start_or_get_conversation(
        self,
        session_id: str,
        channel: str,
        agent_id: Optional[int] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session_id = (session_id or "").strip() or str(uuid.uuid4())[:8]
        channel = (channel or "voice-turn").strip() or "voice-turn"

        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM conversations
                    WHERE session_id = ? AND channel = ? AND status = 'open'
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (session_id, channel),
                ).fetchone()

                now = _utc_now()
                if row:
                    row_meta = _json_load(row["metadata_json"], {})
                    merged_meta = self._merge_metadata(row_meta, metadata)
                    conn.execute(
                        """
                        UPDATE conversations
                        SET updated_at = ?,
                            language = ?,
                            agent_id = COALESCE(?, agent_id),
                            metadata_json = ?
                        WHERE id = ?
                        """,
                        (
                            now,
                            language or row["language"] or "en",
                            agent_id,
                            _json_dump(merged_meta),
                            int(row["id"]),
                        ),
                    )
                    conn.commit()
                    conv_id = int(row["id"])
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO conversations (
                            session_id, channel, agent_id, language, status,
                            metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
                        """,
                        (
                            session_id,
                            channel,
                            agent_id,
                            language or "en",
                            _json_dump(metadata or {}),
                            now,
                            now,
                        ),
                    )
                    conn.commit()
                    conv_id = int(cur.lastrowid)

                conv = conn.execute(
                    """
                    SELECT c.*, a.name AS agent_name
                    FROM conversations c
                    LEFT JOIN agents a ON a.id = c.agent_id
                    WHERE c.id = ?
                    """,
                    (conv_id,),
                ).fetchone()

        return {
            "id": int(conv["id"]),
            "session_id": conv["session_id"],
            "channel": conv["channel"],
            "agent_id": int(conv["agent_id"]) if conv["agent_id"] is not None else None,
            "agent_name": conv["agent_name"],
            "language": conv["language"] or "en",
            "status": conv["status"],
            "metadata": _json_load(conv["metadata_json"], {}),
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
        }

    def add_turn(
        self,
        conversation_id: int,
        role: str,
        text: str,
        language: str = "en",
        audio_path: Optional[str] = None,
        audio_mime: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now()
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO turns (
                        conversation_id, role, text, language,
                        audio_path, audio_mime, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        role,
                        text or "",
                        language or "en",
                        audio_path,
                        audio_mime,
                        _json_dump(metadata or {}),
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ?, language = ? WHERE id = ?",
                    (now, language or "en", conversation_id),
                )
                conn.commit()
                turn_id = int(cur.lastrowid)
                row = conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()

        return {
            "id": int(row["id"]),
            "conversation_id": int(row["conversation_id"]),
            "role": row["role"],
            "text": row["text"],
            "language": row["language"],
            "audio_path": row["audio_path"],
            "audio_mime": row["audio_mime"],
            "metadata": _json_load(row["metadata_json"], {}),
            "created_at": row["created_at"],
        }

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))

        sql = """
        SELECT
            c.*,
            a.name AS agent_name,
            (SELECT COUNT(1) FROM turns t WHERE t.conversation_id = c.id) AS turn_count,
            (SELECT text FROM turns t WHERE t.conversation_id = c.id ORDER BY t.id DESC LIMIT 1) AS last_text,
            (SELECT created_at FROM turns t WHERE t.conversation_id = c.id ORDER BY t.id DESC LIMIT 1) AS last_turn_at
        FROM conversations c
        LEFT JOIN agents a ON a.id = c.agent_id
        ORDER BY c.updated_at DESC
        LIMIT ? OFFSET ?
        """

        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(sql, (limit, offset)).fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": int(row["id"]),
                    "session_id": row["session_id"],
                    "channel": row["channel"],
                    "agent_id": int(row["agent_id"]) if row["agent_id"] is not None else None,
                    "agent_name": row["agent_name"],
                    "language": row["language"] or "en",
                    "status": row["status"],
                    "metadata": _json_load(row["metadata_json"], {}),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "turn_count": int(row["turn_count"] or 0),
                    "last_text": row["last_text"] or "",
                    "last_turn_at": row["last_turn_at"],
                }
            )
        return out

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._conn() as conn:
                conv = conn.execute(
                    """
                    SELECT c.*, a.name AS agent_name
                    FROM conversations c
                    LEFT JOIN agents a ON a.id = c.agent_id
                    WHERE c.id = ?
                    """,
                    (conversation_id,),
                ).fetchone()
                if not conv:
                    return None

                turns = conn.execute(
                    "SELECT * FROM turns WHERE conversation_id = ? ORDER BY id ASC",
                    (conversation_id,),
                ).fetchall()

        return {
            "id": int(conv["id"]),
            "session_id": conv["session_id"],
            "channel": conv["channel"],
            "agent_id": int(conv["agent_id"]) if conv["agent_id"] is not None else None,
            "agent_name": conv["agent_name"],
            "language": conv["language"] or "en",
            "status": conv["status"],
            "metadata": _json_load(conv["metadata_json"], {}),
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "turns": [
                {
                    "id": int(t["id"]),
                    "conversation_id": int(t["conversation_id"]),
                    "role": t["role"],
                    "text": t["text"],
                    "language": t["language"] or "en",
                    "audio_path": t["audio_path"],
                    "audio_mime": t["audio_mime"],
                    "metadata": _json_load(t["metadata_json"], {}),
                    "created_at": t["created_at"],
                }
                for t in turns
            ],
        }

    def save_audio_bytes(self, conversation_id: int, role: str, audio_bytes: bytes, ext: str = "wav") -> Optional[str]:
        if not audio_bytes:
            return None
        rel_dir = f"conv_{conversation_id}"
        abs_dir = os.path.join(self.media_root, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        file_name = f"{ts}_{role}_{uuid.uuid4().hex[:8]}.{ext.strip('.') or 'wav'}"
        rel_path = os.path.join(rel_dir, file_name)
        abs_path = os.path.join(self.media_root, rel_path)

        with open(abs_path, "wb") as f:
            f.write(audio_bytes)
        return rel_path

    def save_audio_array(
        self,
        conversation_id: int,
        role: str,
        audio: np.ndarray,
        sample_rate: int,
        subtype: str = "PCM_16",
    ) -> Optional[str]:
        if audio is None:
            return None
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio)
        if audio.size == 0:
            return None

        rel_dir = f"conv_{conversation_id}"
        abs_dir = os.path.join(self.media_root, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        file_name = f"{ts}_{role}_{uuid.uuid4().hex[:8]}.wav"
        rel_path = os.path.join(rel_dir, file_name)
        abs_path = os.path.join(self.media_root, rel_path)

        sf.write(abs_path, audio.astype(np.float32), sample_rate, format="WAV", subtype=subtype)
        return rel_path
