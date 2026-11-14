"""Small HTTP security helpers: ICE config and admin Basic auth."""

from __future__ import annotations

import hmac
from collections.abc import Iterable
from typing import Any


def build_ice_servers(
    stun_urls: Iterable[str],
    turn_urls: Iterable[str],
    turn_username: str = "",
    turn_credential: str = "",
) -> list[dict[str, Any]]:
    """Build a browser-style ``iceServers`` list from configuration.

    TURN entries are only included when URLs *and* credentials are provided.
    """
    servers: list[dict[str, Any]] = [{"urls": url} for url in stun_urls if url]
    turn_urls = [u for u in turn_urls if u]
    if turn_urls and turn_username and turn_credential:
        servers.extend({"urls": url, "username": turn_username, "credential": turn_credential} for url in turn_urls)
    return servers


def check_basic_auth(
    provided_username: str | None,
    provided_password: str | None,
    expected_username: str,
    expected_password: str,
) -> bool:
    """Constant-time comparison of Basic-auth credentials.

    Returns True when auth is disabled (``expected_password`` empty).
    """
    if not expected_password:
        return True
    if provided_username is None or provided_password is None:
        return False
    user_ok = hmac.compare_digest(provided_username.encode(), expected_username.encode())
    pass_ok = hmac.compare_digest(provided_password.encode(), expected_password.encode())
    return user_ok and pass_ok
