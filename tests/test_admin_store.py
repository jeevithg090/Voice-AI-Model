import os
import sqlite3

import numpy as np
import pytest
import soundfile as sf

from src.admin.store import AdminStore


@pytest.fixture
def store(tmp_path):
    return AdminStore(db_path=str(tmp_path / "db" / "admin.db"), media_root=str(tmp_path / "media"))


def test_seeds_builtin_templates_and_active_agent(store):
    names = {t["name"] for t in store.list_templates()}
    assert {"Emergency Response", "Customer Support", "Healthcare Intake"} <= names
    active = store.get_active_agent()
    assert active is not None and active["is_active"] is True


def test_seeding_is_idempotent(store, tmp_path):
    again = AdminStore(db_path=store.db_path, media_root=store.media_root)
    assert len(again.list_templates()) == len(store.list_templates())
    assert len(again.list_agents()) == len(store.list_agents())


def test_template_crud_and_validation(store):
    with pytest.raises(ValueError):
        store.create_template({"name": "", "system_prompt": "x"})
    with pytest.raises(ValueError):
        store.create_template({"name": "T", "system_prompt": "  "})

    tpl = store.create_template({"name": " Clinic ", "system_prompt": "Hi {{x}}", "defaults": "not-a-dict"})
    assert tpl["name"] == "Clinic"
    assert tpl["defaults"] == {}

    updated = store.update_template(tpl["id"], {"description": "  d  ", "defaults": {"x": 1}})
    assert updated["description"] == "d"
    assert updated["defaults"] == {"x": 1}
    assert store.update_template(99999, {"name": "nope"}) is None

    with pytest.raises(sqlite3.IntegrityError):
        store.create_template({"name": "Clinic", "system_prompt": "dup"})


def test_agent_lifecycle_and_activation(store):
    tpl = store.list_templates()[0]
    with pytest.raises(ValueError):
        store.create_agent({"name": "x", "template_id": "abc"})

    agent = store.create_agent({"name": "Desk", "template_id": tpl["id"], "config": {"tone": "warm"}})
    assert agent["template_name"] == tpl["name"]
    assert agent["is_active"] is False

    activated = store.set_active_agent(agent["id"])
    assert activated["is_active"] is True
    assert store.get_active_agent()["id"] == agent["id"]
    assert sum(a["is_active"] for a in store.list_agents()) == 1
    assert store.set_active_agent(424242) is None

    renamed = store.update_agent(agent["id"], {"name": " Front Desk ", "config": {"tone": "brisk"}})
    assert renamed["name"] == "Front Desk"
    assert renamed["config"] == {"tone": "brisk"}


def test_conversation_turns_and_listing(store):
    conv = store.start_or_get_conversation("sess1", "webrtc", language="hi", metadata={"a": 1})
    same = store.start_or_get_conversation("sess1", "webrtc", metadata={"b": 2})
    assert same["id"] == conv["id"]
    assert same["metadata"] == {"a": 1, "b": 2}
    other_channel = store.start_or_get_conversation("sess1", "voice-turn")
    assert other_channel["id"] != conv["id"]

    store.add_turn(conv["id"], "user", "namaste", language="hi")
    store.add_turn(conv["id"], "assistant", "kaise madad karun?", language="hi")

    full = store.get_conversation(conv["id"])
    assert [t["role"] for t in full["turns"]] == ["user", "assistant"]
    listed = {c["id"]: c for c in store.list_conversations(limit=10)}
    assert listed[conv["id"]]["turn_count"] == 2
    assert listed[conv["id"]]["last_text"] == "kaise madad karun?"
    assert store.get_conversation(123456) is None


def test_audio_saving_is_sandboxed(store):
    rel = store.save_audio_bytes(7, "../../etc/evil", b"RIFFdata", ext="../sh")
    assert rel.startswith("conv_7" + os.sep)
    assert ".." not in rel
    assert os.path.isfile(os.path.join(store.media_root, rel))
    assert store.save_audio_bytes(7, "user", b"") is None

    rel_wav = store.save_audio_array(7, "user", np.zeros(4800, dtype=np.float32), 48000)
    data, sr = sf.read(os.path.join(store.media_root, rel_wav))
    assert sr == 48000 and len(data) == 4800
    assert store.save_audio_array(7, "user", np.array([]), 48000) is None
