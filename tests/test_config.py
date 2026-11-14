from src import config


def test_env_parsers(monkeypatch):
    monkeypatch.setenv("X_BOOL", " Yes ")
    monkeypatch.setenv("X_INT", "notanint")
    monkeypatch.setenv("X_FLOAT", "0.25")
    monkeypatch.setenv("X_LIST", " a, ,b ,c ")
    assert config._env_bool("X_BOOL") is True
    assert config._env_bool("MISSING_BOOL", default=True) is True
    assert config._env_int("X_INT", 7) == 7
    assert config._env_float("X_FLOAT", 1.0) == 0.25
    assert config._env_list("X_LIST") == ("a", "b", "c")
    assert config._env_list("MISSING_LIST", "x,y") == ("x", "y")


def test_settings_defaults_are_sane():
    s = config.settings
    assert s.SERVER_PORT > 0
    assert s.DISTRESS_MIN_HITS >= 1
    assert 0.0 < s.EMERGENCY_THRESHOLD <= 1.0
    assert s.MAX_UPLOAD_MB > 0
