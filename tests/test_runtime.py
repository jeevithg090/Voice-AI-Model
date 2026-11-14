from src.admin.runtime import FALLBACK_SYSTEM_PROMPT, render_template, resolve_runtime_profile
from src.admin.store import AdminStore


class EmptyStore:
    def get_agent(self, _id):
        return None

    def get_active_agent(self):
        return None


def test_render_template_placeholders():
    assert render_template("Hi {{ name }}, {{missing}}!", {"name": "Asha"}) == "Hi Asha, !"
    assert render_template("{{x}}", {"x": None}) == ""
    assert render_template("", {"x": 1}) == ""


def test_default_profile_without_agents():
    profile = resolve_runtime_profile(EmptyStore(), None, "en")
    assert profile["agent_id"] is None
    assert profile["system_prompt"] == FALLBACK_SYSTEM_PROMPT


def test_profile_merges_template_agent_and_clamps(tmp_path):
    store = AdminStore(db_path=str(tmp_path / "a.db"), media_root=str(tmp_path / "m"))
    tpl = store.create_template(
        {
            "name": "Kiosk",
            "system_prompt": "I am {{assistant_name}} speaking {{detected_language_name}} about {{domain}}.",
            "defaults": {"domain": "health", "temperature": 0.3},
        }
    )
    agent = store.create_agent(
        {
            "name": "Asha",
            "template_id": tpl["id"],
            "config": {
                "temperature": 9,
                "max_tokens": 5000,
                "similarity_threshold": "nan",
                "additional_instructions": "Be brief.",
            },
        }
    )
    profile = resolve_runtime_profile(store, agent["id"], "ta")
    assert profile["system_prompt"].startswith("I am Asha speaking Tamil about health.")
    assert profile["system_prompt"].endswith("Additional instructions:\nBe brief.")
    assert profile["temperature"] == 1.5
    assert profile["max_tokens"] == 1024
    assert profile["similarity_threshold"] == 0.85
    assert profile["template_name"] == "Kiosk"


def test_unknown_agent_falls_back_to_active(tmp_path):
    store = AdminStore(db_path=str(tmp_path / "a.db"), media_root=str(tmp_path / "m"))
    active = store.get_active_agent()
    profile = resolve_runtime_profile(store, 99999, "en")
    assert profile["agent_id"] == active["id"]
