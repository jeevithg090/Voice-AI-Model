import math
import re
from typing import Any, Dict, Optional


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ur": "Urdu",
    "as": "Assamese",
    "or": "Odia",
}


FALLBACK_SYSTEM_PROMPT = (
    "You are a multilingual voice assistant. "
    "Reply in the user's language, keep responses short and speech-friendly, "
    "avoid hallucination, and offer human handover when uncertain."
)


_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def render_template(template_text: str, variables: Dict[str, Any]) -> str:
    if not template_text:
        return ""

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        value = variables.get(key, "")
        if value is None:
            return ""
        return str(value)

    return _PLACEHOLDER_PATTERN.sub(_replace, template_text)


def resolve_runtime_profile(
    store,
    requested_agent_id: Optional[int],
    language_code: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    requested_agent_id = int(requested_agent_id) if requested_agent_id is not None else None

    agent = None
    if requested_agent_id is not None:
        agent = store.get_agent(requested_agent_id)

    if agent is None:
        agent = store.get_active_agent()

    default_profile = {
        "agent_id": None,
        "agent_name": "Built-in Default",
        "template_id": None,
        "template_name": None,
        "system_prompt": FALLBACK_SYSTEM_PROMPT,
        "temperature": 0.7,
        "max_tokens": 256,
        "similarity_threshold": 0.85,
    }

    if agent is None:
        return default_profile

    template = store.get_template(agent["template_id"]) if agent.get("template_id") else None
    template_defaults = dict(template.get("defaults", {})) if template else {}
    agent_config = dict(agent.get("config", {}))

    merged = {**template_defaults, **agent_config}
    merged.setdefault("assistant_name", agent["name"])
    merged.setdefault("detected_language", language_code or "en")
    merged.setdefault("detected_language_name", LANGUAGE_NAMES.get(language_code or "en", "English"))
    merged.setdefault("language_confidence", metadata.get("language_confidence", "0.99"))
    merged.setdefault("emergency_probability", metadata.get("emergency_probability", "Low"))
    merged.setdefault("sustained_distress", metadata.get("sustained_distress", "False"))
    merged.setdefault("processing_latency_ms", metadata.get("processing_latency_ms", "0"))

    base_prompt = template["system_prompt"] if template else FALLBACK_SYSTEM_PROMPT
    system_prompt = render_template(base_prompt, merged).strip()
    if not system_prompt:
        system_prompt = FALLBACK_SYSTEM_PROMPT

    extra = (agent_config.get("additional_instructions") or "").strip()
    if extra:
        system_prompt = f"{system_prompt}\n\nAdditional instructions:\n{extra}"

    temperature = _clamp(_safe_float(merged.get("temperature", 0.7), 0.7), 0.0, 1.5)
    max_tokens = max(64, min(_safe_int(merged.get("max_tokens", 256), 256), 1024))
    similarity_threshold = _clamp(_safe_float(merged.get("similarity_threshold", 0.85), 0.85), 0.0, 1.0)

    return {
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "template_id": template["id"] if template else None,
        "template_name": template["name"] if template else None,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "similarity_threshold": similarity_threshold,
    }
