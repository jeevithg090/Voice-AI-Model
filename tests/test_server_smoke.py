"""Smoke tests for the full Flask server.

They import ``src.realtime.signaling_server`` with no models available
(HF_HUB_OFFLINE=1, no emergency checkpoint) and verify the server degrades
gracefully: it boots, reports not-ready, serves UIs and the admin API.
Requires the full runtime stack (torch, transformers, aiortc, librosa, ...).
"""

import dataclasses
import io

import numpy as np
import pytest

pytestmark = pytest.mark.server

for _mod in ("torch", "transformers", "aiortc", "librosa", "flask_sock", "gevent"):
    pytest.importorskip(_mod)


@pytest.fixture(scope="module")
def server():
    from src.realtime import signaling_server

    return signaling_server


@pytest.fixture
def client(server):
    server.app.config["TESTING"] = True
    return server.app.test_client()


@pytest.fixture
def with_admin_password(server, monkeypatch):
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, ADMIN_PASSWORD="pw"))


def test_health_and_readiness_without_models(client):
    health = client.get("/healthz")
    assert health.status_code == 200
    body = health.get_json()
    assert body["status"] == "ok"
    assert body["redis"] is False
    assert body["ollama"]["server_running"] is False
    assert "emergency_detection" in body

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.get_json()["ready"] is False


def test_model_guarded_endpoints_return_503(client):
    assert client.post("/voice-turn").status_code == 503
    assert client.post("/voice-turn-stream").status_code == 503
    assert client.post("/offer", json={"sdp": "x", "type": "offer"}).status_code == 503
    assert client.post("/detect-lang").status_code == 503


@pytest.mark.parametrize("path", ["/", "/client", "/call", "/push-to-talk", "/lid-test"])
def test_ui_pages_served(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert b"<title>" in resp.data
    assert b'name="viewport"' in resp.data


def test_ice_config_and_metrics(client):
    ice = client.get("/ice-config").get_json()
    assert ice["iceServers"] and all("urls" in s for s in ice["iceServers"])
    metrics = client.get("/metrics").get_json()
    assert {"calls_total", "active_calls", "utterances_total", "latency_stats"} <= metrics.keys()


def test_agent_types_creates_purpose_agents(client):
    types = client.get("/agent-types").get_json()["agent_types"]
    assert {t["type"] for t in types} == {"hospital_kiosk", "college_admission", "laptop_support"}
    assert all(t["agent_id"] for t in types)


def test_admin_requires_auth_when_password_set(client, with_admin_password):
    assert client.get("/admin/agents").status_code == 401
    assert client.get("/admin").status_code == 401
    bad = client.get("/admin/agents", auth=("admin", "nope"))
    assert bad.status_code == 401
    good = client.get("/admin/agents", auth=("admin", "pw"))
    assert good.status_code == 200
    assert good.get_json()["agents"]


def test_admin_crud_roundtrip(client):
    created = client.post("/admin/templates", json={"name": "Smoke T", "system_prompt": "Hello"})
    assert created.status_code == 201
    tid = created.get_json()["template"]["id"]
    assert client.post("/admin/templates", json={"name": "No prompt"}).status_code == 400

    agent = client.post("/admin/agents", json={"name": "Smoke A", "template_id": tid}).get_json()["agent"]
    assert client.post(f"/admin/agents/{agent['id']}/activate").status_code == 200
    assert client.get("/admin/active-agent").get_json()["agent"]["id"] == agent["id"]
    assert client.get("/admin/agents/999999").status_code == 404
    assert client.get("/admin/conversations").status_code == 200


def test_recordings_cannot_escape_media_root(client):
    resp = client.get("/admin/recordings/../admin.db")
    assert resp.status_code == 404


def test_upload_size_limit(client, server, monkeypatch):
    monkeypatch.setitem(server.app.config, "MAX_CONTENT_LENGTH", 1024)
    monkeypatch.setattr(server, "whisper_model", object())
    monkeypatch.setattr(server, "whisper_processor", object())
    data = {"audio": (io.BytesIO(b"\0" * 4096), "a.wav")}
    resp = client.post("/voice-turn", data=data, content_type="multipart/form-data")
    assert resp.status_code == 413


def test_uploaded_distress_assessment_uses_detector(server, monkeypatch):
    class FakeDetector:
        sample_rate = 24000

        def predict(self, audio):
            return 0.99

    audio = np.zeros(server.INPUT_SR * 3, dtype=np.float32)
    monkeypatch.setattr(server, "emergency_detector", None)
    assert server.assess_uploaded_distress(audio) is False
    monkeypatch.setattr(server, "emergency_detector", FakeDetector())
    assert server.assess_uploaded_distress(audio) is True
    assert server.assess_uploaded_distress(audio[:100]) is False
