"""Seed the admin database with clearly labelled demo conversations.

Useful for trying the admin console (and for the README screenshots) without
running real calls. Nothing here touches models, Redis or Ollama.

    python scripts/seed_demo_data.py                     # uses ADMIN_DB_PATH
    ADMIN_DB_PATH=/tmp/demo.db python scripts/seed_demo_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.admin.store import AdminStore
from src.config import settings

DEMO_CONVERSATIONS = [
    {
        "session_id": "demo-kiosk-1",
        "channel": "voice-turn",
        "language": "en",
        "turns": [
            ("user", "Where is the outpatient registration counter?"),
            (
                "assistant",
                "Outpatient registration is on the ground floor, to the left of the main entrance. "
                "Please bring a photo ID.",
            ),
            ("user", "What time does it close today?"),
            ("assistant", "It is open until 5 pm today. Would you like directions to the pharmacy as well?"),
        ],
    },
    {
        "session_id": "demo-hindi-1",
        "channel": "webrtc",
        "language": "hi",
        "turns": [
            ("user", "मुझे कॉलेज में दाखिले की प्रक्रिया के बारे में जानना है।"),
            ("assistant", "ज़रूर। पहले ऑनलाइन फ़ॉर्म भरें, फिर दस्तावेज़ सत्यापन के लिए कार्यालय आएँ।"),
        ],
    },
    {
        "session_id": "demo-tamil-1",
        "channel": "media-stream",
        "language": "ta",
        "turns": [
            ("user", "என் லேப்டாப் சார்ஜ் ஆகவில்லை."),
            ("assistant", "சார்ஜர் கேபிள் சரியாக இணைக்கப்பட்டுள்ளதா என்று பாருங்கள். வேறு சாக்கெட்டில் முயற்சிக்கவும்."),
        ],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=settings.ADMIN_DB_PATH, help="SQLite path (default: ADMIN_DB_PATH)")
    parser.add_argument("--media", default=settings.CONVERSATION_MEDIA_DIR, help="recordings directory")
    args = parser.parse_args()

    store = AdminStore(db_path=args.db, media_root=args.media)
    active = store.get_active_agent()
    agent_id = active["id"] if active else None

    for convo in DEMO_CONVERSATIONS:
        record = store.start_or_get_conversation(
            session_id=convo["session_id"],
            channel=convo["channel"],
            agent_id=agent_id,
            language=convo["language"],
            metadata={"demo": True},
        )
        for role, text in convo["turns"]:
            store.add_turn(record["id"], role, text, language=convo["language"], metadata={"demo": True})
        print(f"seeded conversation #{record['id']} ({convo['session_id']}, {len(convo['turns'])} turns)")

    print(f"done: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
