from src.realtime.security import build_ice_servers, check_basic_auth


def test_ice_servers_stun_only_by_default():
    servers = build_ice_servers(["stun:a:1", ""], [], "", "")
    assert servers == [{"urls": "stun:a:1"}]


def test_ice_servers_turn_requires_credentials():
    assert build_ice_servers([], ["turn:t:3478"], "user", "") == []
    servers = build_ice_servers(["stun:a:1"], ["turn:t:3478", "turns:t:5349"], "u", "p")
    assert servers[1:] == [
        {"urls": "turn:t:3478", "username": "u", "credential": "p"},
        {"urls": "turns:t:5349", "username": "u", "credential": "p"},
    ]


def test_basic_auth():
    assert check_basic_auth(None, None, "admin", "") is True  # disabled
    assert check_basic_auth(None, None, "admin", "pw") is False
    assert check_basic_auth("admin", "wrong", "admin", "pw") is False
    assert check_basic_auth("root", "pw", "admin", "pw") is False
    assert check_basic_auth("admin", "pw", "admin", "pw") is True
