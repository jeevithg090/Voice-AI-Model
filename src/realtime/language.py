"""Language helpers shared by the realtime server.

Pure functions with no model dependencies so they can be unit-tested.
"""

from __future__ import annotations

# MMS-LID (ISO 639-3) -> short code used throughout the pipeline.
LANGUAGE_CODE_MAP: dict[str, str] = {
    "hin": "hi",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "ben": "bn",
    "guj": "gu",
    "mal": "ml",
    "mar": "mr",
    "pan": "pa",
    "ori": "or",
    "asm": "as",
    "urd": "ur",
    "eng": "en",
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "mr": "Marathi",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}

INDIAN_LANGUAGES: frozenset[str] = frozenset(
    {"hin", "tam", "tel", "kan", "ben", "guj", "mal", "mar", "pan", "ori", "asm", "urd"}
)

HANDOVER_MESSAGES: dict[str, str] = {
    "en": "I will connect you to a human for further help.",
    "hi": "मैं आपको आगे की मदद के लिए मानव से जोड़ रहा हूँ।",
    "ta": "மேலும் உதவிக்காக நான் உங்களை மனித உதவியுடன் இணைக்கிறேன்.",
    "te": "ఇంకా సహాయం కోసం నేను మిమ్మల్ని మనిషితో కలుపుతున్నాను.",
    "kn": "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ ನಾನು ನಿಮ್ಮನ್ನು ಮಾನವರೊಂದಿಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
}

KIOSK_HANDOVER_MESSAGES: dict[str, str] = {
    "en": "Please visit the help desk for further assistance.",
    "hi": "कृपया आगे की मदद के लिए सहायता डेस्क पर जाएँ।",
    "ta": "மேலும் உதவிக்காக உதவி மேசையை அணுகவும்.",
    "te": "ఇంకా సహాయం కోసం సహాయ డెస్క్ వద్దకు వెళ్లండి.",
    "kn": "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ ಸಹಾಯ ಮೇಜಿಗೆ ಭೇಟಿ ನೀಡಿ.",
}

UNCERTAINTY_PHRASES: dict[str, list[str]] = {
    "en": ["i'm not sure", "i do not know", "i don't know", "cannot help", "unable to answer"],
    "hi": ["मुझे नहीं पता", "मैं निश्चित नहीं हूँ", "माफ़ कीजिए, पता नहीं"],
    "ta": ["எனக்குத் தெரியவில்லை", "நான் உறுதி இல்லை", "மன்னிக்கவும், தெரியவில்லை"],
    "te": ["నాకు తెలియదు", "నాకు ఖచ్చితంగా తెలియదు", "క్షమించండి, తెలియదు"],
    "kn": ["ನನಗೆ ತಿಳಿದಿಲ್ಲ", "ನಾನು ಖಚಿತವಿಲ್ಲ", "ಕ್ಷಮಿಸಿ, ತಿಳಿದಿಲ್ಲ"],
}

_SCRIPT_HINTS: dict[str, str] = {
    "en": "Use English script only.",
    "hi": "Use Devanagari script only.",
    "ta": "Use Tamil script only.",
    "te": "Use Telugu script only.",
    "kn": "Use Kannada script only.",
}


def normalize_language_code(code: str | None) -> str:
    """Map an MMS-LID / ISO 639-3 code (or a short code) to a short code.

    Unknown or empty values fall back to English.
    """
    if not code:
        return "en"
    code = code.strip().lower()
    if code in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[code]
    if code in LANGUAGE_CODE_MAP.values():
        return code
    return "en"


def get_handover_message(language_code: str, mode: str = "phone") -> str:
    """Return the localized human-handover message for ``phone`` or ``kiosk`` mode."""
    messages = KIOSK_HANDOVER_MESSAGES if mode == "kiosk" else HANDOVER_MESSAGES
    return messages.get(language_code, messages["en"])


def is_uncertain_response(text: str | None, language_code: str) -> bool:
    """True when the LLM reply is empty or contains a known 'I don't know' phrase."""
    if not text or not text.strip():
        return True
    lower = text.lower()
    return any(p in lower for p in UNCERTAINTY_PHRASES.get(language_code, []))


def voice_turn_max_tokens(profile_max_tokens: int | None, configured_max: int) -> int:
    """Clamp the agent profile's max tokens to the server-wide voice-turn ceiling."""
    configured = max(1, int(configured_max))
    if profile_max_tokens and profile_max_tokens > 0:
        return min(int(profile_max_tokens), configured)
    return configured


def augment_voice_turn_prompt(system_prompt: str | None, language_code: str) -> str:
    """Append strict language/brevity rules used for single-turn voice replies."""
    lang = LANGUAGE_NAMES.get(language_code, LANGUAGE_NAMES["en"])
    script_hint = _SCRIPT_HINTS.get(language_code, f"Use only {lang} language script.")
    return (
        f"{(system_prompt or '').rstrip()}\n\n"
        "[VOICE TURN RULES]\n"
        f"- Respond only in {lang} ({language_code}). {script_hint}\n"
        "- Keep it concise for speech output (1-2 short sentences).\n"
        "- Do not switch languages unless the user explicitly asks.\n"
    )
