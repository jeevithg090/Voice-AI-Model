import pytest

from src.realtime.language import (
    HANDOVER_MESSAGES,
    KIOSK_HANDOVER_MESSAGES,
    augment_voice_turn_prompt,
    get_handover_message,
    is_uncertain_response,
    normalize_language_code,
    voice_turn_max_tokens,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hin", "hi"),
        ("tam", "ta"),
        ("eng", "en"),
        ("kn", "kn"),
        (" TEL ", "te"),
        ("", "en"),
        (None, "en"),
        ("auto", "en"),
        ("xyz", "en"),
    ],
)
def test_normalize_language_code(raw, expected):
    assert normalize_language_code(raw) == expected


def test_handover_message_modes_and_fallback():
    assert get_handover_message("hi") == HANDOVER_MESSAGES["hi"]
    assert get_handover_message("ta", mode="kiosk") == KIOSK_HANDOVER_MESSAGES["ta"]
    assert get_handover_message("ml") == HANDOVER_MESSAGES["en"]


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("", "en", True),
        ("   ", "en", True),
        (None, "en", True),
        ("I'm not sure about that.", "en", True),
        ("Sorry, I DO NOT KNOW.", "en", True),
        ("Your appointment is at 5pm.", "en", False),
        ("मुझे नहीं पता", "hi", True),
        ("आपका स्वागत है", "hi", False),
        ("I'm not sure", "ml", False),  # no phrase list for Malayalam
    ],
)
def test_is_uncertain_response(text, lang, expected):
    assert is_uncertain_response(text, lang) is expected


def test_voice_turn_max_tokens_clamps():
    assert voice_turn_max_tokens(256, 80) == 80
    assert voice_turn_max_tokens(40, 80) == 40
    assert voice_turn_max_tokens(0, 80) == 80
    assert voice_turn_max_tokens(None, 0) == 1


def test_augment_voice_turn_prompt_adds_language_rules():
    prompt = augment_voice_turn_prompt("Base prompt.\n\n", "ta")
    assert prompt.startswith("Base prompt.\n\n[VOICE TURN RULES]")
    assert "Respond only in Tamil (ta)" in prompt
    assert "Tamil script only" in prompt

    fallback = augment_voice_turn_prompt(None, "zz")
    assert "Respond only in English (zz)" in fallback
