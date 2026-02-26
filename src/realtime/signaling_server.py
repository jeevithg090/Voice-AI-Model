# import asyncio
# import threading
# from flask import Flask, request, jsonify
# from aiortc import RTCPeerConnection, RTCSessionDescription
# import numpy as np
# import librosa

# from flask_cors import CORS

# import torch

# from transformers import MimiModel, AutoFeatureExtractor
# from src.models.emergency_classifier import MimiEmergencyClassifier
# from collections import deque
# import time

# from src.models.language_detector import IndicLIDDetector
# lid = IndicLIDDetector()
# current_language = "unknown"

# INPUT_SR = 48000

# EMERGENCY_WINDOW = int(INPUT_SR * 5)   # 0.5 sec
# LANG_WINDOW = int(INPUT_SR * 1.5) 
# DISTRESS_WINDOW = 5        # seconds to track
# MIN_HITS = 3               # how many positives required
# THRESHOLD = 0.7

# distress_events = deque()
# # -----------------------
# # Setup Async Loop in Background
# # -----------------------
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)

# def run_loop():
#     loop.run_forever()

# threading.Thread(target=run_loop, daemon=True).start()

# # -----------------------
# # Flask App
# # -----------------------
# app = Flask(__name__)

# CORS(app) 
# pcs = set()

# device = "mps" if torch.backends.mps.is_available() else "cpu"

# print("Loading Mimi...")
# mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
# fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
# TARGET_SR = fe.sampling_rate

# print("Loading classifier...")
# clf = MimiEmergencyClassifier().to(device)
# clf.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
# clf.eval()


# # -----------------------
# # Audio Processing
# # -----------------------
# async def process_audio_track(track):
#     global current_language

#     print("🎙 Audio track started")

#     buffer = np.array([], dtype=np.float32)
#     last_lang_detect_samples = 0

#     while True:
#         frame = await track.recv()
#         pcm = frame.to_ndarray()

#         if pcm.ndim > 1:
#             pcm = pcm.mean(axis=0)

#         pcm = pcm.astype(np.float32) / 32768.0
#         buffer = np.concatenate([buffer, pcm])

#         # print("Frame shape:", pcm.shape, "Buffer length:", len(buffer))

#         # 🌍 Language Detection (every 1.5 sec)
#         if len(buffer) - last_lang_detect_samples >= LANG_WINDOW:
#             segment = buffer[last_lang_detect_samples:last_lang_detect_samples + LANG_WINDOW]

#             try:
#                 lang, conf = lid.detect_from_audio(segment, sr=INPUT_SR)
#                 current_language = lang
#                 print(f"🌍 Detected language: {lang} ({conf:.2f})")
#             except Exception as e:
#                 print("LID error:", e)

#             last_lang_detect_samples = len(buffer)

#         # 🚑 Emergency Detection (every 0.5 sec)
#         if len(buffer) >= EMERGENCY_WINDOW:
#             chunk = buffer[:EMERGENCY_WINDOW]
#             buffer = buffer[EMERGENCY_WINDOW:]

#             await run_emergency_detection(chunk)


# async def run_emergency_detection(audio_chunk):
#     global distress_events

#     audio_16k = librosa.resample(audio_chunk, orig_sr=48000, target_sr=TARGET_SR)
#     inputs = fe(raw_audio=audio_16k, sampling_rate=TARGET_SR, return_tensors="pt").to(device)

#     with torch.no_grad():
#         codes = mimi.encode(inputs["input_values"]).audio_codes
#         prob = clf(codes).item()

#     print(f"🚑 Emergency probability: {prob:.3f}")

#     now = time.time()

#     # Remove old events outside window
#     while distress_events and now - distress_events[0] > DISTRESS_WINDOW:
#         distress_events.popleft()

#     # Add new event if above threshold
#     if prob > THRESHOLD:
#         distress_events.append(now)

#     # Check sustained distress
#     if len(distress_events) >= MIN_HITS:
#         print("🚨 SUSTAINED DISTRESS DETECTED 🚨")
#         distress_events.clear()  # prevent repeat spam



# # -----------------------
# # WebRTC Offer Handling
# # -----------------------
# @app.route("/offer", methods=["POST"])
# def offer():
#     params = request.json
#     offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

#     pc = RTCPeerConnection()
#     pcs.add(pc)

#     @pc.on("track")
#     def on_track(track):
#         if track.kind == "audio":
#             print("🎧 Receiving audio from browser")
#             asyncio.run_coroutine_threadsafe(process_audio_track(track), loop)

#     async def handle_offer():
#         await pc.setRemoteDescription(offer)
#         answer = await pc.createAnswer()
#         await pc.setLocalDescription(answer)
#         return pc.localDescription

#     future = asyncio.run_coroutine_threadsafe(handle_offer(), loop)
#     local_desc = future.result()

#     return jsonify({"sdp": local_desc.sdp, "type": local_desc.type})


# # -----------------------
# # Start Server
# # -----------------------
# if __name__ == "__main__":
#     print("🚀 Signaling + Audio Processing Server Running")
#     app.run("0.0.0.0", 8080)



import asyncio
import concurrent.futures
import threading
import time
import os
import io
import json
import base64
import audioop
import statistics
import queue
from collections import deque
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_sock import Sock
import gevent
from gevent.queue import Queue
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
import numpy as np
import librosa
import soundfile as sf
import av

from src.config import settings
from src.llm.embedding_agent import process_with_context, context_manager as llm_context
from src.llm.stream_chunker import StreamChunker
from src.tts.tts_manager import TTSManager
from src.realtime.audio_track import TTSAudioTrack
from src.realtime.twilio_track import TwilioInputTrack
from src.llm.ollama_client import check_ollama_status
from src.admin import AdminStore, resolve_runtime_profile

tts_manager = TTSManager()
admin_store = AdminStore()

# -----------------------
# Metrics
# -----------------------
metrics_lock = threading.Lock()
metrics: Dict[str, Any] = {
    "calls_total": 0,
    "active_calls": 0,
    "utterances_total": 0,
    "latencies": {},
}

# Purpose-specific runtime profiles for upload/web clients.
AGENT_TYPE_PROFILES: Dict[str, Dict[str, Any]] = {
    "hospital_kiosk": {
        "label": "Hospital Kiosk",
        "template_name": "Healthcare Intake",
        "description": "Front-desk triage and guidance for hospital visitors.",
        "config": {
            "assistant_name": "Nurse Asha",
            "tone": "calm, empathetic, and structured",
            "objective": "Collect symptoms and route patients to the right counter quickly",
            "additional_instructions": (
                "Ask short follow-up questions for symptoms, duration, and urgency. "
                "Do not diagnose. Escalate emergency signs immediately."
            ),
            "temperature": 0.45,
            "max_tokens": 220,
            "similarity_threshold": 0.86,
        },
    },
    "college_admission": {
        "label": "College Admission",
        "template_name": "Customer Support",
        "description": "Admissions helpdesk for application, fees, and document guidance.",
        "config": {
            "assistant_name": "Admission Mitra",
            "tone": "clear, polite, and informative",
            "objective": "Guide applicants through eligibility, deadlines, and document checklist",
            "additional_instructions": (
                "If exact institute policy is unknown, state assumption clearly and "
                "give the next best step to confirm with admissions office."
            ),
            "temperature": 0.55,
            "max_tokens": 240,
            "similarity_threshold": 0.84,
        },
    },
    "laptop_support": {
        "label": "Laptop Customer Support",
        "template_name": "Customer Support",
        "description": "Troubleshooting and support for laptop issues.",
        "config": {
            "assistant_name": "Tech Mitra",
            "tone": "friendly and step-by-step",
            "objective": "Troubleshoot laptop issues quickly with safe diagnostic steps",
            "additional_instructions": (
                "Ask model, OS, and error details first. Prefer low-risk steps. "
                "For hardware risk or data-loss risk, recommend authorized technician."
            ),
            "temperature": 0.5,
            "max_tokens": 240,
            "similarity_threshold": 0.84,
        },
    },
}
_purpose_agent_lock = threading.Lock()
_purpose_agent_cache: Dict[str, int] = {}


def _latency_bucket(name: str):
    with metrics_lock:
        if name not in metrics["latencies"]:
            metrics["latencies"][name] = deque(maxlen=settings.METRICS_WINDOW)
        return metrics["latencies"][name]


def record_latency(name: str, value_ms: float):
    bucket = _latency_bucket(name)
    with metrics_lock:
        bucket.append(value_ms)


def inc_metric(name: str, delta: int = 1):
    with metrics_lock:
        metrics[name] = metrics.get(name, 0) + delta


def dec_metric(name: str, delta: int = 1):
    with metrics_lock:
        metrics[name] = max(0, metrics.get(name, 0) - delta)


def get_latency_stats() -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    # Copy under lock, then compute stats outside lock.
    with metrics_lock:
        latency_snapshot = {
            name: list(values)
            for name, values in metrics["latencies"].items()
            if values
        }
    for name, vals in latency_snapshot.items():
        stats[name] = {
            "count": len(vals),
            "avg_ms": float(sum(vals) / len(vals)),
            "p50_ms": float(statistics.median(vals)),
            "p95_ms": float(np.percentile(vals, 95)),
        }
    return stats


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        value_str = str(value).strip()
        if not value_str:
            return None
        return int(value_str)
    except Exception:
        return None


def _to_audio_url(audio_path: Optional[str]) -> Optional[str]:
    if not audio_path:
        return None
    normalized = str(audio_path).replace("\\\\", "/").lstrip("/")
    return f"/admin/recordings/{normalized}"


def _template_id_by_name(template_name: str) -> Optional[int]:
    name = (template_name or "").strip().lower()
    if not name:
        return None
    try:
        for template in admin_store.list_templates():
            if (template.get("name") or "").strip().lower() == name:
                return int(template["id"])
    except Exception as e:
        print(f"⚠️ Failed to read templates for purpose routing: {e}")
    return None


def _ensure_agent_for_type(agent_type: str) -> Optional[int]:
    normalized = (agent_type or "").strip().lower()
    if normalized not in AGENT_TYPE_PROFILES:
        return None

    with _purpose_agent_lock:
        cached = _purpose_agent_cache.get(normalized)
        if cached is not None:
            return cached

        profile = AGENT_TYPE_PROFILES[normalized]
        target_name = profile["label"]
        try:
            for agent in admin_store.list_agents():
                if (agent.get("name") or "").strip().lower() == target_name.lower():
                    agent_id = int(agent["id"])
                    _purpose_agent_cache[normalized] = agent_id
                    return agent_id

            template_id = _template_id_by_name(profile["template_name"])
            created = admin_store.create_agent(
                {
                    "name": target_name,
                    "description": profile["description"],
                    "template_id": template_id,
                    "config": profile["config"],
                }
            )
            agent_id = int(created["id"])
            _purpose_agent_cache[normalized] = agent_id
            print(f"✅ Created purpose agent '{target_name}' (id={agent_id})")
            return agent_id
        except Exception as e:
            print(f"⚠️ Failed to ensure purpose agent '{target_name}': {e}")
            return None


def _resolve_requested_agent(
    requested_agent_id: Optional[int],
    requested_agent_type: str,
) -> Optional[int]:
    if requested_agent_id is not None:
        return requested_agent_id
    return _ensure_agent_for_type(requested_agent_type)


def _warm_purpose_agents() -> None:
    for agent_type in AGENT_TYPE_PROFILES.keys():
        _ensure_agent_for_type(agent_type)

import torch

from transformers import MimiModel, AutoFeatureExtractor
from src.models.emergency_classifier import MimiEmergencyClassifier

DISTRESS_WINDOW = 5        # seconds to track
MIN_HITS = 3               # how many positives required
THRESHOLD = 0.7
# -----------------------
# Whisper Endpointing State
# -----------------------
SILENCE_THRESHOLD = settings.SILENCE_THRESHOLD
SILENCE_DURATION = settings.SILENCE_DURATION
MIN_SPEECH_DURATION = settings.MIN_SPEECH_DURATION

distress_events = deque()
# -----------------------
# Setup Async Loop in Background
# -----------------------
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_loop():
    loop.run_forever()

threading.Thread(target=run_loop, daemon=True).start()

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)
sock = Sock(app)

# Enable CORS for cross-origin requests (local frontend → remote GPU backend)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

pcs = set()

# ICE Servers with TURN support for NAT traversal
ICE_SERVERS = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
    # Free TURN servers for NAT traversal (frontend on local PC ↔ backend on DigitalOcean)
    RTCIceServer(
        urls="turn:openrelay.metered.ca:80",
        username="openrelayproject",
        credential="openrelayproject",
    ),
    RTCIceServer(
        urls="turn:openrelay.metered.ca:443",
        username="openrelayproject",
        credential="openrelayproject",
    ),
]
RTC_CONFIG = RTCConfiguration(iceServers=ICE_SERVERS)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Loading Mimi...")
mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
TARGET_SR = fe.sampling_rate

print("Loading classifier...")
clf = MimiEmergencyClassifier().to(device)
clf.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
clf.eval()



from transformers import WhisperProcessor, WhisperForConditionalGeneration

whisper_model = None
whisper_processor = None

def load_whisper():
    global whisper_model, whisper_processor
    print("🔄 Loading Whisper model in background...")
    try:
        whisper_processor = WhisperProcessor.from_pretrained(settings.WHISPER_MODEL)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(settings.WHISPER_MODEL).to(device)
        whisper_model.eval()

        # Warm-up to reduce first-token latency
        dummy_audio = np.zeros(16000, dtype=np.float32)
        inputs = whisper_processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            whisper_model.generate(inputs.input_features.to(device), max_new_tokens=1)

        print("✅ Whisper ASR ready!")
    except Exception as e:
        print(f"❌ Whisper load failed: {e}")
        whisper_model = None
        whisper_processor = None


threading.Thread(target=load_whisper, daemon=True).start()

# -----------------------
# MMS-LID Language Detection (uses raw audio - no extra normalization)
# -----------------------
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor as LIDFeatureExtractor

INPUT_SR = 48000
LID_TARGET_SR = 16000  # MMS-LID expects 16kHz
LANG_WINDOW = int(INPUT_SR * 1.5)  # 1.5 sec of audio for language detection

current_language = "unknown"

# Language code mappings
LANGUAGE_CODE_MAP = {
    'hin': 'hi', 'tam': 'ta', 'tel': 'te', 'kan': 'kn',
    'ben': 'bn', 'guj': 'gu', 'mal': 'ml', 'mar': 'mr',
    'pan': 'pa', 'ori': 'or', 'asm': 'as', 'urd': 'ur',
    'eng': 'en',
}

LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada',
    'bn': 'Bengali', 'gu': 'Gujarati', 'ml': 'Malayalam', 'mr': 'Marathi',
    'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'ur': 'Urdu',
}

INDIAN_LANGUAGES = {'hin', 'tam', 'tel', 'kan', 'ben', 'guj', 'mal', 'mar', 'pan', 'ori', 'asm', 'urd'}

lid_model = None
lid_processor = None
lid_id2label = None

def load_lid_model():
    global lid_model, lid_processor, lid_id2label
    print("🔄 Loading MMS-LID language detector in background...")
    
    lid_processor = LIDFeatureExtractor.from_pretrained(settings.LID_MODEL)
    lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(settings.LID_MODEL)
    lid_model = lid_model.to(device)
    lid_model.eval()
    lid_id2label = lid_model.config.id2label
    
    print(f"✅ MMS-LID ready! ({len(lid_id2label)} languages)")

threading.Thread(target=load_lid_model, daemon=True).start()


def normalize_language_code(code: str) -> str:
    if not code:
        return "en"
    if code in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[code]
    if code in LANGUAGE_CODE_MAP.values():
        return code
    return "en"


def detect_language_from_audio(audio_chunk_48k: np.ndarray) -> Optional[str]:
    if not settings.ENABLE_LID_AUTODETECT or lid_model is None or lid_processor is None:
        return None
    try:
        audio_16k = librosa.resample(audio_chunk_48k, orig_sr=INPUT_SR, target_sr=LID_TARGET_SR)
        inputs = lid_processor(audio_16k, sampling_rate=LID_TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = lid_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        candidates = []
        for i, p in enumerate(probs):
            code = lid_id2label[i]
            if code in INDIAN_LANGUAGES or code == "eng":
                candidates.append((code, float(p)))

        if not candidates:
            return None

        best_code, best_prob = max(candidates, key=lambda x: x[1])
        if best_prob < 0.4:
            return None

        return normalize_language_code(best_code)
    except Exception as e:
        print(f"⚠️ LID detection failed: {e}")
        return None


HANDOVER_MESSAGES = {
    "en": "I will connect you to a human for further help.",
    "hi": "मैं आपको आगे की मदद के लिए मानव से जोड़ रहा हूँ।",
    "ta": "மேலும் உதவிக்காக நான் உங்களை மனித உதவியுடன் இணைக்கிறேன்.",
    "te": "ఇంకా సహాయం కోసం నేను మిమ్మల్ని మనిషితో కలుపుతున్నాను.",
    "kn": "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ ನಾನು ನಿಮ್ಮನ್ನು ಮಾನವರೊಂದಿಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
}


UNCERTAINTY_PHRASES = {
    "en": ["i'm not sure", "i do not know", "cannot help", "unable to answer"],
    "hi": ["मुझे नहीं पता", "मैं निश्चित नहीं हूँ", "माफ़ कीजिए, पता नहीं"],
    "ta": ["எனக்குத் தெரியவில்லை", "நான் உறுதி இல்லை", "மன்னிக்கவும், தெரியவில்லை"],
    "te": ["నాకు తెలియదు", "నాకు ఖచ్చితంగా తెలియదు", "క్షమించండి, తెలియదు"],
    "kn": ["ನನಗೆ ತಿಳಿದಿಲ್ಲ", "ನಾನು ಖಚಿತವಿಲ್ಲ", "ಕ್ಷಮಿಸಿ, ತಿಳಿದಿಲ್ಲ"],
}


def get_handover_message(language_code: str) -> str:
    message = HANDOVER_MESSAGES.get(language_code, HANDOVER_MESSAGES["en"])
    if settings.HANDOVER_MODE == "kiosk":
        kiosk_messages = {
            "en": "Please visit the help desk for further assistance.",
            "hi": "कृपया आगे की मदद के लिए सहायता डेस्क पर जाएँ।",
            "ta": "மேலும் உதவிக்காக உதவி மேசையை அணுகவும்.",
            "te": "ఇంకా సహాయం కోసం సహాయ డెస్క్ వద్దకు వెళ్లండి.",
            "kn": "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ ಸಹಾಯ ಮೇಜಿಗೆ ಭೇಟಿ ನೀಡಿ.",
        }
        message = kiosk_messages.get(language_code, kiosk_messages["en"])
    return message


def is_uncertain_response(text: str, language_code: str) -> bool:
    if not text:
        return True
    phrases = UNCERTAINTY_PHRASES.get(language_code, [])
    lower = text.lower()
    return any(p in lower for p in phrases)


def _voice_turn_max_tokens(profile_max_tokens: int) -> int:
    configured = max(1, int(settings.VOICE_TURN_MAX_TOKENS))
    if profile_max_tokens and profile_max_tokens > 0:
        return min(profile_max_tokens, configured)
    return configured


def _augment_voice_turn_prompt(system_prompt: str, language_code: str) -> str:
    lang = LANGUAGE_NAMES.get(language_code, LANGUAGE_NAMES["en"])
    script_hint = {
        "en": "Use English script only.",
        "hi": "Use Devanagari script only.",
        "ta": "Use Tamil script only.",
        "te": "Use Telugu script only.",
        "kn": "Use Kannada script only.",
    }.get(language_code, f"Use only {lang} language script.")

    return (
        f"{(system_prompt or '').rstrip()}\n\n"
        "[VOICE TURN RULES]\n"
        f"- Respond only in {lang} ({language_code}). {script_hint}\n"
        "- Keep it concise for speech output (1-2 short sentences).\n"
        "- Do not switch languages unless the user explicitly asks.\n"
    )


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

# -----------------------
# Audio Processing
# -----------------------
import io # Ensure io is imported

async def process_audio_track(
    track,
    target_language="eng",
    output_track=None,
    twilio_output_sender=None,
    auto_detect: bool = False,
    agent_id: Optional[int] = None,
):
    language_code = normalize_language_code(target_language)
    if auto_detect or target_language in ["auto", "", None]:
        language_code = "en"
    print(f"🎙 Audio track started (Language: {language_code}, auto_detect={auto_detect})")

    # Create LLM Session for this track
    try:
        session_id = llm_context.create_session(language=language_code)
        print(f"🆕 Started LLM Session: {session_id}")
    except Exception as e:
        print(f"⚠️ Failed to create Redis session: {e}")
        session_id = "temp_session"

    conversation_id = None
    try:
        channel = "websocket" if isinstance(track, TwilioInputTrack) else "webrtc"
        conversation = admin_store.start_or_get_conversation(
            session_id=session_id,
            channel=channel,
            agent_id=agent_id,
            language=language_code,
            metadata={"auto_detect": auto_detect},
        )
        conversation_id = conversation["id"]
    except Exception as e:
        print(f"⚠️ Failed to initialize conversation record: {e}")


    # Create local buffer for this track session (don't use global asr_buffer safely multiple calls)
    # Actually, using global is bad for multiple calls. Let's make it local.
    local_asr_buffer = np.array([], dtype=np.float32)
    local_speech_start = None
    local_last_voice = None

    buffer = np.array([], dtype=np.float32)
    lid_buffer = np.array([], dtype=np.float32)
    last_lid_samples = 0

    # Base endpointing values from env.
    silence_threshold = settings.SILENCE_THRESHOLD
    silence_duration = settings.SILENCE_DURATION
    min_speech_duration = settings.MIN_SPEECH_DURATION
    input_gain = 1.0

    # WebSocket μ-law path often has lower amplitude than direct WebRTC PCM.
    if isinstance(track, TwilioInputTrack):
        silence_threshold = min(silence_threshold, 0.0025)
        silence_duration = min(silence_duration, 1.0)
        min_speech_duration = min(min_speech_duration, 0.35)
        input_gain = 2.5

    print(
        f"🎚 Endpointing config: threshold={silence_threshold:.4f}, "
        f"silence={silence_duration:.2f}s, min_speech={min_speech_duration:.2f}s, "
        f"gain={input_gain:.2f}"
    )

    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray()

        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)

        pcm = pcm.astype(np.float32) / 32768.0
        if input_gain != 1.0:
            pcm = np.clip(pcm * input_gain, -1.0, 1.0)

        # ---------------- EMERGENCY DETECTION ----------------
        buffer = np.concatenate([buffer, pcm])

        if len(buffer) >= TARGET_SR:
            chunk = buffer[:TARGET_SR]
            buffer = buffer[TARGET_SR:]
            await run_emergency_detection(chunk)

        # ---------------- LID AUTO-DETECT ----------------
        if auto_detect:
            lid_buffer = np.concatenate([lid_buffer, pcm])
            if len(lid_buffer) - last_lid_samples >= LANG_WINDOW:
                segment = lid_buffer[last_lid_samples:last_lid_samples + LANG_WINDOW]
                detected = detect_language_from_audio(segment)
                if detected and detected != language_code:
                    language_code = detected
                    try:
                        llm_context.update_language(session_id, language_code)
                    except Exception:
                        pass
                    print(f"🌍 Auto-detected language: {language_code}")
                last_lid_samples = len(lid_buffer)

        # ---------------- WHISPER ENDPOINTING ----------------
        energy = np.mean(np.abs(pcm))
        now = time.time()
        if isinstance(track, TwilioInputTrack):
            if not hasattr(process_audio_track, "_ws_energy_tick"):
                process_audio_track._ws_energy_tick = 0
            process_audio_track._ws_energy_tick += 1
            tick = process_audio_track._ws_energy_tick
            if tick <= 10 or tick % 200 == 0:
                print(f"🔎 WS frame energy={energy:.6f} (threshold={silence_threshold:.6f})")

        if energy > silence_threshold:
            # Speech detected
            if local_speech_start is None:
                local_speech_start = now
                print("🗣 Speech started")

            local_last_voice = now
            local_asr_buffer = np.concatenate([local_asr_buffer, pcm])

        else:
            # Silence frame
            if local_speech_start is not None and local_last_voice is not None:
                silence_time = now - local_last_voice

                if silence_time > silence_duration:
                    speech_length = local_last_voice - local_speech_start

                    if speech_length > min_speech_duration and len(local_asr_buffer) > 0:
                        print(f"🛑 Speech ended → Sending to Whisper ({language_code})")
                        chunk = local_asr_buffer.copy()
                        resolved_agent_id = getattr(track, "agent_id", None) or agent_id
                        asyncio.create_task(run_whisper_asr(
                            chunk, 
                            language_code=language_code, 
                            session_id=session_id,
                            output_track=output_track,
                            twilio_output_sender=twilio_output_sender,
                            conversation_id=conversation_id,
                            agent_id=resolved_agent_id,
                            user_audio_chunk=chunk,
                        ))

                    # Reset state
                    local_asr_buffer = np.array([], dtype=np.float32)
                    local_speech_start = None
                    local_last_voice = None


# async def run_whisper_asr(audio_chunk_48k):
#     global whisper_model

#     if whisper_model is None:
#         return

#     print("🧠 Running Whisper ASR...")

#     audio_16k = librosa.resample(audio_chunk_48k, orig_sr=48000, target_sr=16000)
#     audio_16k = audio_16k.astype(np.float32)

#     # Boost mic level a bit
#     audio_16k *= 2.5
#     audio_16k = np.clip(audio_16k, -1.0, 1.0)

#     segments, info = whisper_model.transcribe(
#         audio_16k,
#         beam_size=1,
#         vad_filter=True,
#         vad_parameters=dict(min_silence_duration_ms=300)
#     )

#     text = " ".join(seg.text for seg in segments).strip()

#     if text:
#         print(f"📝 ASR: {text}")
#     else:
#         print("🤐 (No speech detected)")

async def run_whisper_asr(
    audio_chunk_48k,
    language_code="eng",
    session_id="temp_session",
    output_track=None,
    twilio_output_sender=None,
    conversation_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    user_audio_chunk: Optional[np.ndarray] = None,
):
    global whisper_model, whisper_processor

    if whisper_model is None or whisper_processor is None:
        print("⏳ Whisper still loading...")
        return

    whisper_lang = normalize_language_code(language_code)
    print(f"🧠 Running Whisper ASR on full utterance (Lang: {language_code} -> {whisper_lang})...")

    stt_start = time.perf_counter()
    audio_16k = librosa.resample(audio_chunk_48k, orig_sr=48000, target_sr=16000)
    audio_np = audio_16k.astype(np.float32)

    audio_np *= 2.0
    audio_np = np.clip(audio_np, -1.0, 1.0)

    inputs = whisper_processor(audio_np, sampling_rate=16000, return_tensors="pt")
    try:
        forced_ids = whisper_processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
    except Exception:
        forced_ids = None
    with torch.no_grad():
        generate_kwargs = {"forced_decoder_ids": forced_ids} if forced_ids else {}
        generated_ids = whisper_model.generate(
            inputs.input_features.to(device),
            **generate_kwargs
        )
    text = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    record_latency("stt_ms", (time.perf_counter() - stt_start) * 1000)

    if text:
        print(f"📝 ASR Final: {text}")
        inc_metric("utterances_total", 1)

        # --- LLM INTEGRATION (STREAMING) ---
        print(f"🤖 Sending to LLM (Session: {session_id})...")

        sustained_distress = len(distress_events) >= MIN_HITS
        metadata = {
            "detected_language": whisper_lang,
            "language_confidence": "0.99",
            "emergency_probability": "High" if sustained_distress else "Low",
            "sustained_distress": str(sustained_distress),
            "processing_latency_ms": "0"
        }
        runtime_profile = resolve_runtime_profile(
            admin_store,
            requested_agent_id=agent_id,
            language_code=whisper_lang,
            metadata=metadata,
        )

        if conversation_id:
            try:
                user_audio_path = admin_store.save_audio_array(
                    conversation_id=conversation_id,
                    role="user",
                    audio=user_audio_chunk if user_audio_chunk is not None else audio_chunk_48k,
                    sample_rate=INPUT_SR,
                )
                admin_store.add_turn(
                    conversation_id=conversation_id,
                    role="user",
                    text=text,
                    language=whisper_lang,
                    audio_path=user_audio_path,
                    audio_mime="audio/wav" if user_audio_path else None,
                    metadata={"session_id": session_id, "agent_id": runtime_profile["agent_id"]},
                )
            except Exception as e:
                print(f"⚠️ Failed to persist user turn: {e}")

        def persist_assistant_turn(final_text: str):
            if not final_text or not conversation_id:
                return
            try:
                archive_audio = asyncio.run_coroutine_threadsafe(
                    tts_manager.speak(final_text, whisper_lang),
                    loop,
                ).result(timeout=120) or b""
                assistant_audio_path = admin_store.save_audio_bytes(
                    conversation_id=conversation_id,
                    role="assistant",
                    audio_bytes=archive_audio,
                    ext="wav",
                ) if archive_audio else None
                admin_store.add_turn(
                    conversation_id=conversation_id,
                    role="assistant",
                    text=final_text,
                    language=whisper_lang,
                    audio_path=assistant_audio_path,
                    audio_mime="audio/wav" if assistant_audio_path else None,
                    metadata={"session_id": session_id, "agent_id": runtime_profile["agent_id"]},
                )
            except Exception as e:
                print(f"⚠️ Failed to persist assistant turn: {e}")

        # Emergency handover shortcut
        if sustained_distress:
            handover_text = get_handover_message(whisper_lang)
            asyncio.run_coroutine_threadsafe(
                tts_manager.speak(handover_text, whisper_lang), loop
            )
            persist_assistant_turn(handover_text)
            return

        def stream_worker():
            llm_start = time.perf_counter()
            full_response = ""
            first_token_time = None
            try:
                llm_result = process_with_context(
                    prompt=text,
                    session_id=session_id,
                    system_prompt=runtime_profile["system_prompt"],
                    similarity_threshold=runtime_profile["similarity_threshold"],
                    metadata=metadata,
                    stream=True,
                    temperature=runtime_profile["temperature"],
                    max_tokens=runtime_profile["max_tokens"],
                )

                print("🌊 LLM Stream Started...")
                chunker = StreamChunker(
                    min_words=settings.STREAM_CHUNK_MIN_WORDS,
                    flush_ms=settings.STREAM_CHUNK_FLUSH_MS,
                )
                chunk_count = 0

                async def run_tts_and_send(text_chunk, lang):
                    tts_start = time.perf_counter()
                    audio_bytes = await tts_manager.speak(text_chunk, lang)
                    record_latency("tts_ms", (time.perf_counter() - tts_start) * 1000)
                    if audio_bytes:
                        print(f"🔊 Generated {len(audio_bytes)} bytes of audio for: '{text_chunk[:10]}...'")

                        if output_track:
                            try:
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(None, output_track.add_audio_bytes, audio_bytes)
                            except Exception as e:
                                print(f"❌ Failed to add audio to track: {e}")

                        if twilio_output_sender:
                            await twilio_output_sender(audio_bytes)
                    else:
                        print("⚠️ TTS Synthesis Failed")

                if not llm_result.get("stream"):
                    response_text = llm_result["response"]
                    full_response = response_text
                    print(f"✅ [Cache] Response: {response_text}")
                    asyncio.run_coroutine_threadsafe(
                        run_tts_and_send(response_text, whisper_lang), loop
                    )
                    if is_uncertain_response(response_text, whisper_lang):
                        handover_text = get_handover_message(whisper_lang)
                        asyncio.run_coroutine_threadsafe(
                            run_tts_and_send(handover_text, whisper_lang), loop
                        )
                    persist_assistant_turn(full_response)
                    return

                generator = llm_result["response_generator"]

                adaptive_first_chunk = True
                for token in generator:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        record_latency("llm_ttft_ms", (first_token_time - llm_start) * 1000)
                    full_response += token
                    for chunk in chunker.process(token):
                        if adaptive_first_chunk:
                            adaptive_first_chunk = False
                            chunker.min_words = max(chunker.min_words, settings.STREAM_CHUNK_MIN_WORDS * 2)
                        chunk_count += 1
                        print(f"🗣️  [Chunk #{chunk_count}] {chunk}")
                        asyncio.run_coroutine_threadsafe(
                            run_tts_and_send(chunk, whisper_lang), loop
                        )

                last_chunk = chunker.flush()
                if last_chunk:
                    chunk_count += 1
                    print(f"🗣️  [Chunk #{chunk_count}] {last_chunk}")
                    asyncio.run_coroutine_threadsafe(
                        run_tts_and_send(last_chunk, whisper_lang), loop
                    )

                record_latency("llm_total_ms", (time.perf_counter() - llm_start) * 1000)
                print(f"✅ Stream Complete ({chunk_count} chunks)")
                persist_assistant_turn(full_response)

                if is_uncertain_response(full_response, whisper_lang):
                    handover_text = get_handover_message(whisper_lang)
                    asyncio.run_coroutine_threadsafe(
                        tts_manager.speak(handover_text, whisper_lang), loop
                    )

            except Exception as e:
                print(f"❌ LLM Stream Error: {e}")

        threading.Thread(target=stream_worker, daemon=True).start()

    else:
        print("🤐 No speech detected")

async def run_emergency_detection(audio_chunk):
    global distress_events

    audio_16k = librosa.resample(audio_chunk, orig_sr=48000, target_sr=TARGET_SR)
    inputs = fe(raw_audio=audio_16k, sampling_rate=TARGET_SR, return_tensors="pt").to(device)

    with torch.no_grad():
        codes = mimi.encode(inputs["input_values"]).audio_codes
        prob = clf(codes).item()

    print(f"🚑 Emergency probability: {prob:.3f}")

    now = time.time()

    # Remove old events outside window
    while distress_events and now - distress_events[0] > DISTRESS_WINDOW:
        distress_events.popleft()

    # Add new event if above threshold
    if prob > THRESHOLD:
        distress_events.append(now)

    # Check sustained distress
    if len(distress_events) >= MIN_HITS:
        print("🚨 SUSTAINED DISTRESS DETECTED 🚨")
        distress_events.clear()  # prevent repeat spam


def _decode_audio_bytes(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """
    Decode WAV/MP3 bytes into float32 PCM at target_sr.
    """
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    except Exception:
        container = av.open(io.BytesIO(audio_bytes))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        pcm = bytearray()
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                pcm += bytes(resampled.planes[0])
        if not pcm:
            raise
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def wav_bytes_to_mulaw_chunks(audio_bytes: bytes, target_sr: int = 8000, frame_ms: int = 20):
    """
    Convert WAV/MP3 bytes to μ-law base64 chunks for Twilio Media Streams.
    """
    audio = _decode_audio_bytes(audio_bytes, target_sr)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16).tobytes()
    mulaw = audioop.lin2ulaw(pcm16, 2)
    chunk_size = int(target_sr * (frame_ms / 1000.0))  # samples per frame
    for i in range(0, len(mulaw), chunk_size):
        chunk = mulaw[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk += b"\xff" * (chunk_size - len(chunk))
        yield base64.b64encode(chunk).decode("utf-8")


def run_single_turn_pipeline(
    audio_chunk_48k: np.ndarray,
    language_code: str,
    session_id: str,
    requested_agent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Non-streaming fallback pipeline used by /voice-turn:
    Audio upload -> ASR -> LLM -> TTS -> JSON payload.
    """
    turn_start = time.perf_counter()
    whisper_lang = normalize_language_code(language_code)

    # ASR
    stt_start = time.perf_counter()
    audio_16k = librosa.resample(audio_chunk_48k, orig_sr=INPUT_SR, target_sr=16000).astype(np.float32)
    audio_16k = np.clip(audio_16k * 2.0, -1.0, 1.0)
    inputs = whisper_processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    try:
        forced_ids = whisper_processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
    except Exception:
        forced_ids = None
    with torch.no_grad():
        generate_kwargs = {"forced_decoder_ids": forced_ids} if forced_ids else {}
        generated_ids = whisper_model.generate(
            inputs.input_features.to(device),
            **generate_kwargs
        )
    transcript = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    stt_ms = (time.perf_counter() - stt_start) * 1000
    record_latency("stt_ms", stt_ms)

    if not transcript:
        turn_total_ms = (time.perf_counter() - turn_start) * 1000
        record_latency("turn_total_ms", turn_total_ms)
        return {
            "transcript": "",
            "response": "",
            "language": whisper_lang,
            "audio_bytes": b"",
            "timings_ms": {
                "stt_ms": round(stt_ms, 2),
                "llm_ttft_ms": 0.0,
                "llm_total_ms": 0.0,
                "tts_first_chunk_ms": 0.0,
                "tts_total_ms": 0.0,
                "turn_total_ms": round(turn_total_ms, 2),
            },
        }

    inc_metric("utterances_total", 1)
    print(f"📝 [Upload] ASR Final: {transcript}")

    sustained_distress = len(distress_events) >= MIN_HITS
    metadata = {
        "detected_language": whisper_lang,
        "language_confidence": "0.99",
        "emergency_probability": "High" if sustained_distress else "Low",
        "sustained_distress": str(sustained_distress),
        "processing_latency_ms": f"{stt_ms:.1f}",
    }
    runtime_profile = resolve_runtime_profile(
        admin_store,
        requested_agent_id=requested_agent_id,
        language_code=whisper_lang,
        metadata=metadata,
    )
    constrained_prompt = _augment_voice_turn_prompt(runtime_profile["system_prompt"], whisper_lang)
    max_tokens = _voice_turn_max_tokens(runtime_profile.get("max_tokens", 0))

    llm_chunks = []
    response_text = ""
    llm_ttft_ms = 0.0
    llm_total_ms = 0.0

    if sustained_distress:
        response_text = get_handover_message(whisper_lang)
        llm_chunks = [response_text]
    else:
        llm_start = time.perf_counter()
        llm_result = process_with_context(
            prompt=transcript,
            session_id=session_id,
            system_prompt=constrained_prompt,
            similarity_threshold=runtime_profile["similarity_threshold"],
            metadata=metadata,
            stream=True,
            temperature=runtime_profile["temperature"],
            max_tokens=max_tokens,
        )

        if isinstance(llm_result, dict) and llm_result.get("stream"):
            chunker = StreamChunker(
                min_words=settings.STREAM_CHUNK_MIN_WORDS,
                flush_ms=settings.STREAM_CHUNK_FLUSH_MS,
            )
            first_token_time = None
            adaptive_first_chunk = True
            for token in llm_result["response_generator"]:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    llm_ttft_ms = (first_token_time - llm_start) * 1000
                    record_latency("llm_ttft_ms", llm_ttft_ms)
                response_text += token
                for chunk in chunker.process(token):
                    if adaptive_first_chunk:
                        adaptive_first_chunk = False
                        chunker.min_words = max(chunker.min_words, settings.STREAM_CHUNK_MIN_WORDS * 2)
                    llm_chunks.append(chunk)
            tail = chunker.flush()
            if tail:
                llm_chunks.append(tail)
        else:
            if isinstance(llm_result, dict):
                response_text = (llm_result.get("response") or "").strip()
            else:
                response_text = str(llm_result).strip()
            if response_text:
                llm_chunks = [response_text]

        llm_total_ms = (time.perf_counter() - llm_start) * 1000
        record_latency("llm_total_ms", llm_total_ms)
        response_text = (response_text or "").strip()
        if not response_text:
            response_text = get_handover_message(whisper_lang)
            llm_chunks = [response_text]
        elif is_uncertain_response(response_text, whisper_lang):
            handover = get_handover_message(whisper_lang)
            response_text = f"{response_text} {handover}".strip()
            llm_chunks.append(handover)

    tts_stage_start = time.perf_counter()
    tts_first_chunk_ms = 0.0
    tts_futures = []
    for text_chunk in llm_chunks:
        tts_futures.append(
            asyncio.run_coroutine_threadsafe(
                tts_manager.speak(text_chunk, whisper_lang),
                loop,
            )
        )

    pcm_parts = []
    for f in tts_futures:
        part = f.result(timeout=120) or b""
        if part:
            if tts_first_chunk_ms <= 0.0:
                tts_first_chunk_ms = (time.perf_counter() - tts_stage_start) * 1000
            pcm_parts.append(_decode_audio_bytes(part, INPUT_SR))

    if pcm_parts:
        merged = np.concatenate(pcm_parts, axis=0).astype(np.float32)
        wav_buf = io.BytesIO()
        sf.write(wav_buf, merged, INPUT_SR, format="WAV", subtype="PCM_16")
        audio_bytes = wav_buf.getvalue()
    else:
        audio_bytes = b""

    tts_total_ms = (time.perf_counter() - tts_stage_start) * 1000
    if tts_first_chunk_ms <= 0.0 and tts_total_ms > 0:
        tts_first_chunk_ms = tts_total_ms
    record_latency("tts_ms", tts_total_ms)
    record_latency("tts_first_chunk_ms", tts_first_chunk_ms)
    record_latency("tts_total_ms", tts_total_ms)
    turn_total_ms = (time.perf_counter() - turn_start) * 1000
    record_latency("turn_total_ms", turn_total_ms)

    if len(llm_chunks) > 12:
        print(f"⚠️ [Upload] Large chunk count: {len(llm_chunks)} chunks")
    if len(audio_bytes) > 500_000:
        print(f"⚠️ [Upload] Large reply payload: {len(audio_bytes)} bytes")
    print(f"🔊 [Upload] Generated {len(audio_bytes)} bytes from {len(llm_chunks)} chunks")

    return {
        "transcript": transcript,
        "response": response_text,
        "language": whisper_lang,
        "audio_bytes": audio_bytes,
        "agent_id": runtime_profile["agent_id"],
        "agent_name": runtime_profile["agent_name"],
        "timings_ms": {
            "stt_ms": round(stt_ms, 2),
            "llm_ttft_ms": round(llm_ttft_ms, 2),
            "llm_total_ms": round(llm_total_ms, 2),
            "tts_first_chunk_ms": round(tts_first_chunk_ms, 2),
            "tts_total_ms": round(tts_total_ms, 2),
            "turn_total_ms": round(turn_total_ms, 2),
        },
    }


@app.route("/voice-turn", methods=["POST"])
def voice_turn():
    """
    HTTP fallback for unreliable streaming setups:
    client records one utterance, uploads it, and receives synthesized reply audio.
    """
    if whisper_model is None or whisper_processor is None:
        return jsonify({"error": "Server models not ready yet. Please wait."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file field"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "Uploaded audio file is empty"}), 400

    requested_language = (request.form.get("language", "eng") or "eng").strip()
    requested_agent_id = _parse_optional_int(request.form.get("agent_id"))
    requested_agent_type = (request.form.get("agent_type") or "").strip().lower()
    resolved_agent_id = _resolve_requested_agent(requested_agent_id, requested_agent_type)
    auto_detect_flag = (request.form.get("auto_detect", "false") or "false").strip().lower()
    auto_detect = auto_detect_flag in {"1", "true", "yes", "on"} or requested_language in {"auto", ""}
    language_code = normalize_language_code(requested_language)

    try:
        audio_48k = _decode_audio_bytes(audio_bytes, INPUT_SR)
    except Exception as e:
        return jsonify({"error": f"Could not decode uploaded audio: {e}"}), 400

    if audio_48k.size == 0:
        return jsonify({"error": "Decoded audio is empty"}), 400

    if auto_detect:
        detected = detect_language_from_audio(audio_48k)
        if detected:
            language_code = detected
            print(f"🌍 [Upload] Auto-detected language: {language_code}")

    session_id = (request.form.get("session_id") or "").strip()
    if not session_id:
        try:
            session_id = llm_context.create_session(language=language_code)
        except Exception:
            session_id = "temp_session"
    else:
        try:
            llm_context.update_language(session_id, language_code)
        except Exception:
            pass

    conversation_id = None
    conversation_agent_id = resolved_agent_id
    try:
        conv = admin_store.start_or_get_conversation(
            session_id=session_id,
            channel="voice-turn",
            agent_id=resolved_agent_id,
            language=language_code,
            metadata={
                "requested_language": requested_language,
                "auto_detect": auto_detect,
                "requested_agent_type": requested_agent_type,
            },
        )
        conversation_id = conv["id"]
        conversation_agent_id = conv.get("agent_id") or resolved_agent_id
    except Exception as e:
        print(f"⚠️ Failed to initialize voice-turn conversation: {e}")

    try:
        result = run_single_turn_pipeline(
            audio_48k,
            language_code,
            session_id,
            requested_agent_id=resolved_agent_id,
        )
    except concurrent.futures.TimeoutError:
        return jsonify({"error": "TTS timed out"}), 504
    except Exception as e:
        print(f"❌ /voice-turn failed: {e}")
        return jsonify({"error": f"Voice turn failed: {e}"}), 500

    if conversation_id:
        try:
            user_audio_path = admin_store.save_audio_array(
                conversation_id=conversation_id,
                role="user",
                audio=audio_48k,
                sample_rate=INPUT_SR,
            )
            transcript_text = (result.get("transcript") or "").strip()
            if transcript_text or user_audio_path:
                admin_store.add_turn(
                    conversation_id=conversation_id,
                    role="user",
                    text=transcript_text,
                    language=result.get("language") or language_code,
                    audio_path=user_audio_path,
                    audio_mime="audio/wav" if user_audio_path else None,
                    metadata={
                        "session_id": session_id,
                        "agent_id": conversation_agent_id,
                        "agent_type": requested_agent_type or None,
                    },
                )

            assistant_audio_path = admin_store.save_audio_bytes(
                conversation_id=conversation_id,
                role="assistant",
                audio_bytes=result.get("audio_bytes") or b"",
                ext="wav",
            )
            response_text = (result.get("response") or "").strip()
            if response_text or assistant_audio_path:
                admin_store.add_turn(
                    conversation_id=conversation_id,
                    role="assistant",
                    text=response_text,
                    language=result.get("language") or language_code,
                    audio_path=assistant_audio_path,
                    audio_mime="audio/wav" if assistant_audio_path else None,
                    metadata={
                        "session_id": session_id,
                        "agent_id": result.get("agent_id"),
                        "agent_type": requested_agent_type or None,
                    },
                )
        except Exception as e:
            print(f"⚠️ Failed to persist voice-turn conversation: {e}")

    audio_b64 = ""
    if result["audio_bytes"]:
        audio_b64 = base64.b64encode(result["audio_bytes"]).decode("ascii")
        if len(audio_b64) > 700_000:
            print(f"⚠️ [Upload] Large base64 payload to browser: {len(audio_b64)} chars")

    return jsonify({
        "ok": True,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "agent_id": result.get("agent_id"),
        "agent_name": result.get("agent_name"),
        "agent_type": requested_agent_type or None,
        "language": result["language"],
        "transcript": result["transcript"],
        "response": result["response"],
        "audio_b64": audio_b64,
        "audio_mime": "audio/wav",
        "timings_ms": result.get("timings_ms", {}),
    })


@app.route("/voice-turn-stream", methods=["POST"])
def voice_turn_stream():
    """
    Streaming voice-turn endpoint (SSE):
    emits transcript + chunked LLM text + chunked TTS audio.
    """
    if not settings.VOICE_TURN_STREAM_ENABLED:
        return jsonify({"error": "Streaming voice-turn is disabled"}), 404

    if whisper_model is None or whisper_processor is None:
        return jsonify({"error": "Server models not ready yet. Please wait."}), 503
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file field"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error": "Uploaded audio file is empty"}), 400

    requested_language = (request.form.get("language", "eng") or "eng").strip()
    requested_agent_id = _parse_optional_int(request.form.get("agent_id"))
    requested_agent_type = (request.form.get("agent_type") or "").strip().lower()
    resolved_agent_id = _resolve_requested_agent(requested_agent_id, requested_agent_type)
    auto_detect_flag = (request.form.get("auto_detect", "false") or "false").strip().lower()
    auto_detect = auto_detect_flag in {"1", "true", "yes", "on"} or requested_language in {"auto", ""}
    language_code = normalize_language_code(requested_language)

    try:
        audio_48k = _decode_audio_bytes(audio_bytes, INPUT_SR)
    except Exception as e:
        return jsonify({"error": f"Could not decode uploaded audio: {e}"}), 400
    if audio_48k.size == 0:
        return jsonify({"error": "Decoded audio is empty"}), 400

    if auto_detect:
        detected = detect_language_from_audio(audio_48k)
        if detected:
            language_code = detected
            print(f"🌍 [Stream] Auto-detected language: {language_code}")

    session_id = (request.form.get("session_id") or "").strip()
    if not session_id:
        try:
            session_id = llm_context.create_session(language=language_code)
        except Exception:
            session_id = "temp_session"
    else:
        try:
            llm_context.update_language(session_id, language_code)
        except Exception:
            pass

    conversation_id = None
    conversation_agent_id = resolved_agent_id
    try:
        conv = admin_store.start_or_get_conversation(
            session_id=session_id,
            channel="voice-turn-stream",
            agent_id=resolved_agent_id,
            language=language_code,
            metadata={
                "requested_language": requested_language,
                "auto_detect": auto_detect,
                "requested_agent_type": requested_agent_type,
            },
        )
        conversation_id = conv["id"]
        conversation_agent_id = conv.get("agent_id") or resolved_agent_id
    except Exception as e:
        print(f"⚠️ Failed to initialize voice-turn-stream conversation: {e}")

    @stream_with_context
    def event_stream():
        turn_start = time.perf_counter()
        whisper_lang = normalize_language_code(language_code)
        meta_agent_name = None
        if conversation_agent_id:
            try:
                existing_agent = admin_store.get_agent(conversation_agent_id)
                if existing_agent:
                    meta_agent_name = existing_agent.get("name")
            except Exception:
                meta_agent_name = None
        yield _sse_event(
            "meta",
            {
                "session_id": session_id,
                "language": whisper_lang,
                "agent_id": conversation_agent_id,
                "agent_name": meta_agent_name,
            },
        )

        stt_ms = 0.0
        llm_ttft_ms = 0.0
        llm_total_ms = 0.0
        tts_first_chunk_ms = 0.0
        tts_total_ms = 0.0
        chunk_count = 0
        queue_max_depth = 0
        transcript = ""
        response_text = ""
        runtime_profile = None
        pcm_parts = []

        text_queue: "queue.Queue[Optional[tuple[int, str]]]" = queue.Queue(
            maxsize=max(1, int(settings.STREAM_CHUNK_MAX_QUEUE))
        )
        audio_event_queue: "queue.Queue[tuple[str, Dict[str, Any]]]" = queue.Queue()
        worker_done = threading.Event()
        worker_state: Dict[str, Any] = {
            "tts_total_ms": 0.0,
            "tts_first_chunk_ms": 0.0,
        }
        worker_lock = threading.Lock()

        def emit_audio_events(wait_timeout: float = 0.0):
            while True:
                try:
                    evt, payload = audio_event_queue.get(timeout=wait_timeout)
                except queue.Empty:
                    break
                wait_timeout = 0.0
                if evt == "__worker_done__":
                    continue
                yield _sse_event(evt, payload)

        def tts_worker():
            while True:
                item = text_queue.get()
                try:
                    if item is None:
                        return
                    idx, text_chunk = item
                    synth_start = time.perf_counter()
                    try:
                        audio_part = asyncio.run_coroutine_threadsafe(
                            tts_manager.speak(text_chunk, whisper_lang),
                            loop,
                        ).result(timeout=120) or b""
                    except Exception as e:
                        audio_event_queue.put(("error", {"message": f"TTS chunk {idx} failed: {e}"}))
                        continue

                    synth_ms = (time.perf_counter() - synth_start) * 1000
                    with worker_lock:
                        worker_state["tts_total_ms"] += synth_ms

                    if not audio_part:
                        audio_event_queue.put(("error", {"message": f"TTS returned empty audio for chunk {idx}"}))
                        continue

                    with worker_lock:
                        if worker_state["tts_first_chunk_ms"] <= 0.0:
                            worker_state["tts_first_chunk_ms"] = (time.perf_counter() - turn_start) * 1000

                    if len(audio_part) > 500_000:
                        print(f"⚠️ [Stream] Large audio chunk idx={idx}, bytes={len(audio_part)}")

                    try:
                        pcm_parts.append(_decode_audio_bytes(audio_part, INPUT_SR))
                    except Exception as decode_err:
                        print(f"⚠️ [Stream] Could not decode chunk {idx} for persistence: {decode_err}")

                    audio_event_queue.put(
                        (
                            "audio_chunk",
                            {
                                "index": idx,
                                "audio_b64": base64.b64encode(audio_part).decode("ascii"),
                                "audio_mime": "audio/wav",
                            },
                        )
                    )
                finally:
                    text_queue.task_done()

        worker_thread = threading.Thread(target=tts_worker, daemon=True)
        worker_thread.start()

        try:
            stt_start = time.perf_counter()
            audio_16k = librosa.resample(audio_48k, orig_sr=INPUT_SR, target_sr=16000).astype(np.float32)
            audio_16k = np.clip(audio_16k * 2.0, -1.0, 1.0)
            inputs = whisper_processor(audio_16k, sampling_rate=16000, return_tensors="pt")
            try:
                forced_ids = whisper_processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
            except Exception:
                forced_ids = None
            with torch.no_grad():
                generate_kwargs = {"forced_decoder_ids": forced_ids} if forced_ids else {}
                generated_ids = whisper_model.generate(inputs.input_features.to(device), **generate_kwargs)
            transcript = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            stt_ms = (time.perf_counter() - stt_start) * 1000
            record_latency("stt_ms", stt_ms)

            yield _sse_event("transcript", {"text": transcript})
            if not transcript:
                turn_total_ms = (time.perf_counter() - turn_start) * 1000
                record_latency("turn_total_ms", turn_total_ms)
                yield _sse_event(
                    "done",
                    {
                        "response": "",
                        "timings_ms": {
                            "stt_ms": round(stt_ms, 2),
                            "llm_ttft_ms": 0.0,
                            "llm_total_ms": 0.0,
                            "tts_first_chunk_ms": 0.0,
                            "tts_total_ms": 0.0,
                            "turn_total_ms": round(turn_total_ms, 2),
                        },
                        "chunk_count": 0,
                    },
                )
                return

            inc_metric("utterances_total", 1)
            sustained_distress = len(distress_events) >= MIN_HITS
            metadata = {
                "detected_language": whisper_lang,
                "language_confidence": "0.99",
                "emergency_probability": "High" if sustained_distress else "Low",
                "sustained_distress": str(sustained_distress),
                "processing_latency_ms": f"{stt_ms:.1f}",
            }
            runtime_profile = resolve_runtime_profile(
                admin_store,
                requested_agent_id=resolved_agent_id,
                language_code=whisper_lang,
                metadata=metadata,
            )

            if sustained_distress:
                response_text = get_handover_message(whisper_lang)
                yield _sse_event("llm_chunk", {"index": chunk_count, "text": response_text})
                text_queue.put((chunk_count, response_text))
                chunk_count += 1
                queue_max_depth = max(queue_max_depth, text_queue.qsize())
            else:
                llm_start = time.perf_counter()
                llm_result = process_with_context(
                    prompt=transcript,
                    session_id=session_id,
                    system_prompt=_augment_voice_turn_prompt(runtime_profile["system_prompt"], whisper_lang),
                    similarity_threshold=runtime_profile["similarity_threshold"],
                    metadata=metadata,
                    stream=True,
                    temperature=runtime_profile["temperature"],
                    max_tokens=_voice_turn_max_tokens(runtime_profile.get("max_tokens", 0)),
                )

                if isinstance(llm_result, dict) and llm_result.get("stream"):
                    chunker = StreamChunker(
                        min_words=settings.STREAM_CHUNK_MIN_WORDS,
                        flush_ms=settings.STREAM_CHUNK_FLUSH_MS,
                    )
                    first_token_time = None

                    adaptive_first_chunk = True
                    for token in llm_result["response_generator"]:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            llm_ttft_ms = (first_token_time - llm_start) * 1000
                            record_latency("llm_ttft_ms", llm_ttft_ms)
                        response_text += token
                        for chunk in chunker.process(token):
                            if adaptive_first_chunk:
                                adaptive_first_chunk = False
                                chunker.min_words = max(chunker.min_words, settings.STREAM_CHUNK_MIN_WORDS * 2)
                            yield _sse_event("llm_chunk", {"index": chunk_count, "text": chunk})
                            while True:
                                try:
                                    text_queue.put_nowait((chunk_count, chunk))
                                    break
                                except queue.Full:
                                    print("⚠️ [Stream] TTS queue full; waiting for worker...")
                                    for event in emit_audio_events(wait_timeout=0.05):
                                        yield event
                                    time.sleep(0.01)
                            chunk_count += 1
                            queue_max_depth = max(queue_max_depth, text_queue.qsize())
                            for event in emit_audio_events():
                                yield event

                    tail = chunker.flush()
                    if tail:
                        yield _sse_event("llm_chunk", {"index": chunk_count, "text": tail})
                        text_queue.put((chunk_count, tail))
                        chunk_count += 1
                        queue_max_depth = max(queue_max_depth, text_queue.qsize())
                else:
                    response_text = (llm_result.get("response") or "").strip() if isinstance(llm_result, dict) else str(llm_result).strip()
                    if response_text:
                        yield _sse_event("llm_chunk", {"index": chunk_count, "text": response_text})
                        text_queue.put((chunk_count, response_text))
                        chunk_count += 1
                        queue_max_depth = max(queue_max_depth, text_queue.qsize())

                llm_total_ms = (time.perf_counter() - llm_start) * 1000
                record_latency("llm_total_ms", llm_total_ms)

                response_text = (response_text or "").strip()
                if not response_text:
                    response_text = get_handover_message(whisper_lang)
                    yield _sse_event("llm_chunk", {"index": chunk_count, "text": response_text})
                    text_queue.put((chunk_count, response_text))
                    chunk_count += 1
                    queue_max_depth = max(queue_max_depth, text_queue.qsize())
                elif is_uncertain_response(response_text, whisper_lang):
                    handover = get_handover_message(whisper_lang)
                    response_text = f"{response_text} {handover}".strip()
                    yield _sse_event("llm_chunk", {"index": chunk_count, "text": handover})
                    text_queue.put((chunk_count, handover))
                    chunk_count += 1
                    queue_max_depth = max(queue_max_depth, text_queue.qsize())

            text_queue.put(None)
            while True:
                emitted = False
                for event in emit_audio_events(wait_timeout=0.1):
                    emitted = True
                    yield event
                if not worker_thread.is_alive() and audio_event_queue.empty():
                    break
                if not emitted:
                    yield ": keepalive\n\n"
            worker_done.set()

            with worker_lock:
                tts_first_chunk_ms = worker_state["tts_first_chunk_ms"]
                tts_total_ms = worker_state["tts_total_ms"]
            if tts_first_chunk_ms > 0:
                record_latency("tts_first_chunk_ms", tts_first_chunk_ms)
            if tts_total_ms > 0:
                record_latency("tts_total_ms", tts_total_ms)
                record_latency("tts_ms", tts_total_ms)

            turn_total_ms = (time.perf_counter() - turn_start) * 1000
            record_latency("turn_total_ms", turn_total_ms)

            if chunk_count > 12:
                print(f"⚠️ [Stream] Large chunk count: {chunk_count}")
            if queue_max_depth >= settings.STREAM_CHUNK_MAX_QUEUE:
                print(f"⚠️ [Stream] Queue depth reached limit: {queue_max_depth}")

            assistant_audio_path = None
            if conversation_id:
                try:
                    user_audio_path = admin_store.save_audio_array(
                        conversation_id=conversation_id,
                        role="user",
                        audio=audio_48k,
                        sample_rate=INPUT_SR,
                    )
                    if transcript or user_audio_path:
                        admin_store.add_turn(
                            conversation_id=conversation_id,
                            role="user",
                            text=transcript,
                            language=whisper_lang,
                            audio_path=user_audio_path,
                            audio_mime="audio/wav" if user_audio_path else None,
                            metadata={
                                "session_id": session_id,
                                "agent_id": runtime_profile["agent_id"] if runtime_profile else conversation_agent_id,
                                "agent_type": requested_agent_type or None,
                            },
                        )

                    if pcm_parts:
                        merged = np.concatenate(pcm_parts, axis=0).astype(np.float32)
                        wav_buf = io.BytesIO()
                        sf.write(wav_buf, merged, INPUT_SR, format="WAV", subtype="PCM_16")
                        assistant_audio_path = admin_store.save_audio_bytes(
                            conversation_id=conversation_id,
                            role="assistant",
                            audio_bytes=wav_buf.getvalue(),
                            ext="wav",
                        )

                    if response_text or assistant_audio_path:
                        admin_store.add_turn(
                            conversation_id=conversation_id,
                            role="assistant",
                            text=response_text,
                            language=whisper_lang,
                            audio_path=assistant_audio_path,
                            audio_mime="audio/wav" if assistant_audio_path else None,
                            metadata={
                                "session_id": session_id,
                                "agent_id": runtime_profile["agent_id"] if runtime_profile else conversation_agent_id,
                                "agent_type": requested_agent_type or None,
                            },
                        )
                except Exception as persist_err:
                    print(f"⚠️ Failed to persist voice-turn-stream conversation: {persist_err}")

            yield _sse_event(
                "done",
                {
                    "response": response_text,
                    "timings_ms": {
                        "stt_ms": round(stt_ms, 2),
                        "llm_ttft_ms": round(llm_ttft_ms, 2),
                        "llm_total_ms": round(llm_total_ms, 2),
                        "tts_first_chunk_ms": round(tts_first_chunk_ms, 2),
                        "tts_total_ms": round(tts_total_ms, 2),
                        "turn_total_ms": round(turn_total_ms, 2),
                    },
                    "chunk_count": chunk_count,
                },
            )
        except Exception as e:
            print(f"❌ /voice-turn-stream failed: {e}")
            yield _sse_event("error", {"message": f"Voice turn stream failed: {e}"})
            turn_total_ms = (time.perf_counter() - turn_start) * 1000
            record_latency("turn_total_ms", turn_total_ms)
            yield _sse_event(
                "done",
                {
                    "response": response_text,
                    "timings_ms": {
                        "stt_ms": round(stt_ms, 2),
                        "llm_ttft_ms": round(llm_ttft_ms, 2),
                        "llm_total_ms": round(llm_total_ms, 2),
                        "tts_first_chunk_ms": round(tts_first_chunk_ms, 2),
                        "tts_total_ms": round(tts_total_ms, 2),
                        "turn_total_ms": round(turn_total_ms, 2),
                    },
                    "chunk_count": chunk_count,
                },
            )
        finally:
            if not worker_done.is_set():
                pushed = False
                for _ in range(5):
                    try:
                        text_queue.put(None, timeout=0.2)
                        pushed = True
                        break
                    except queue.Full:
                        continue
                if not pushed:
                    print("⚠️ [Stream] Could not stop TTS worker cleanly (queue still full)")

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



# -----------------------
# WebRTC Offer Handling
# -----------------------
@sock.route("/media-stream")
def media_stream(ws):
    if not settings.ENABLE_TWILIO:
        ws.close()
        return

    inc_metric("calls_total", 1)
    inc_metric("active_calls", 1)

    input_track = TwilioInputTrack(loop=loop)
    call_language = "auto"
    input_track.agent_id = None

    # Send on the same greenlet to avoid WS frame corruption.
    out_queue: Queue = Queue()

    def sender_loop():
        try:
            while True:
                message = out_queue.get()
                if message is None:
                    break
                ws.send(json.dumps(message))
        except Exception as e:
            print(f"❌ Media stream sender error: {e}")

    sender_greenlet = gevent.spawn(sender_loop)

    async def twilio_output_sender(audio_bytes: bytes):
        for payload in wav_bytes_to_mulaw_chunks(audio_bytes):
            out_queue.put({"event": "media", "media": {"payload": payload}})

    asyncio.run_coroutine_threadsafe(
        process_audio_track(
            input_track,
            target_language=call_language,
            output_track=None,
            twilio_output_sender=twilio_output_sender,
            auto_detect=settings.ENABLE_LID_AUTODETECT,
            agent_id=input_track.agent_id,
        ),
        loop
    )

    try:
        while True:
            message = ws.receive()
            if message is None:
                break
            event = json.loads(message)
            if event.get("event") == "start":
                params = event.get("start", {}).get("customParameters", {})
                lang = params.get("language")
                if lang:
                    call_language = lang
                agent_id = _parse_optional_int(params.get("agent_id"))
                if agent_id is not None:
                    input_track.agent_id = agent_id
            elif event.get("event") == "media":
                payload = event["media"]["payload"]
                if not isinstance(payload, (str, list)):
                    print(f"⚠️ Media payload type={type(payload)} value={payload}")
                input_track.add_mulaw_chunk(payload)
            elif event.get("event") == "stop":
                break
    except Exception as e:
        print(f"❌ Media stream error: {e}")
    finally:
        out_queue.put(None)
        try:
            sender_greenlet.kill()
        except Exception:
            pass
        dec_metric("active_calls", 1)


@app.route("/offer", methods=["POST"])
def offer():
    # Readiness guard: reject calls if models aren't loaded yet
    if whisper_model is None or whisper_processor is None:
        return jsonify({"error": "Server models not ready yet. Please wait."}), 503
    
    params = request.json or {}
    sdp = params.get("sdp")
    sdp_type = params.get("type")
    if not sdp or not sdp_type:
        return jsonify({"error": "Missing SDP"}), 400

    client_language = params.get("language", "eng")
    auto_detect = bool(params.get("auto_detect", False)) or client_language in ["auto", "", None]
    requested_agent_id = _parse_optional_int(params.get("agent_id"))

    inc_metric("calls_total", 1)
    inc_metric("active_calls", 1)

    async def handle_offer():
        offer_desc = RTCSessionDescription(sdp=sdp, type=sdp_type)

        pc = RTCPeerConnection(configuration=RTC_CONFIG)
        pcs.add(pc)

        # Create Speaker Track
        output_track = TTSAudioTrack(loop=loop)
        pc.addTrack(output_track)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                print(f"🎧 Receiving audio from browser (Lang: {client_language}, auto={auto_detect})")
                asyncio.run_coroutine_threadsafe(
                    process_audio_track(
                        track,
                        client_language,
                        output_track=output_track,
                        auto_detect=auto_detect,
                        agent_id=requested_agent_id,
                    ),
                    loop
                )

        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            if pc.iceConnectionState in ["failed", "closed", "disconnected"]:
                try:
                    await pc.close()
                finally:
                    pcs.discard(pc)
                    dec_metric("active_calls", 1)

        await pc.setRemoteDescription(offer_desc)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Non-trickle ICE: wait briefly so answer SDP includes candidates.
        if pc.iceGatheringState != "complete":
            for _ in range(80):
                await asyncio.sleep(0.1)
                if pc.iceGatheringState == "complete":
                    break

        return pc.localDescription

    try:
        future = asyncio.run_coroutine_threadsafe(handle_offer(), loop)
        local_desc = future.result(timeout=25)
    except concurrent.futures.TimeoutError:
        dec_metric("active_calls", 1)
        return jsonify({"error": "Offer timed out while gathering ICE"}), 504
    except Exception as e:
        dec_metric("active_calls", 1)
        return jsonify({"error": f"Offer failed: {e}"}), 500

    return jsonify({"sdp": local_desc.sdp, "type": local_desc.type})


# -----------------------
# Admin API
# -----------------------
def _conversation_with_urls(conversation: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(conversation)
    turns = []
    for turn in conversation.get("turns", []):
        item = dict(turn)
        item["audio_url"] = _to_audio_url(item.get("audio_path"))
        turns.append(item)
    enriched["turns"] = turns
    return enriched


@app.route("/admin")
def admin_panel():
    return open("src/realtime/admin_panel.html").read()


@app.route("/admin/templates", methods=["GET"])
def admin_list_templates():
    return jsonify({"templates": admin_store.list_templates()})


@app.route("/admin/templates", methods=["POST"])
def admin_create_template():
    payload = request.get_json(silent=True) or {}
    try:
        template = admin_store.create_template(payload)
        return jsonify({"template": template}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin/templates/<int:template_id>", methods=["PUT"])
def admin_update_template(template_id: int):
    payload = request.get_json(silent=True) or {}
    try:
        template = admin_store.update_template(template_id, payload)
        if template is None:
            return jsonify({"error": "Template not found"}), 404
        return jsonify({"template": template})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin/agents", methods=["GET"])
def admin_list_agents():
    return jsonify({"agents": admin_store.list_agents()})


@app.route("/admin/agents", methods=["POST"])
def admin_create_agent():
    payload = request.get_json(silent=True) or {}
    try:
        agent = admin_store.create_agent(payload)
        return jsonify({"agent": agent}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin/agents/<int:agent_id>", methods=["GET"])
def admin_get_agent(agent_id: int):
    agent = admin_store.get_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"agent": agent})


@app.route("/admin/agents/<int:agent_id>", methods=["PUT"])
def admin_update_agent(agent_id: int):
    payload = request.get_json(silent=True) or {}
    try:
        agent = admin_store.update_agent(agent_id, payload)
        if agent is None:
            return jsonify({"error": "Agent not found"}), 404
        return jsonify({"agent": agent})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/admin/agents/<int:agent_id>/activate", methods=["POST"])
def admin_activate_agent(agent_id: int):
    agent = admin_store.set_active_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"agent": agent})


@app.route("/admin/active-agent", methods=["GET"])
def admin_active_agent():
    return jsonify({"agent": admin_store.get_active_agent()})


@app.route("/admin/conversations", methods=["GET"])
def admin_list_conversations():
    limit = _parse_optional_int(request.args.get("limit")) or 50
    offset = _parse_optional_int(request.args.get("offset")) or 0
    conversations = admin_store.list_conversations(limit=limit, offset=offset)
    return jsonify({"conversations": conversations})


@app.route("/admin/conversations/<int:conversation_id>", methods=["GET"])
def admin_get_conversation(conversation_id: int):
    conversation = admin_store.get_conversation(conversation_id)
    if not conversation:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"conversation": _conversation_with_urls(conversation)})


@app.route("/admin/recordings/<path:recording_path>", methods=["GET"])
def admin_get_recording(recording_path: str):
    return send_from_directory(admin_store.media_root, recording_path, as_attachment=False)


# -----------------------
# Start Server
# -----------------------
# -----------------------
# Start Server
# -----------------------
@app.route("/")
def client_root():
    return open("src/realtime/client.html").read()

@app.route("/client")
def client_page():
    return open("src/realtime/client.html").read()

@app.route("/call")
def call_interface():
    """Serve the new production call interface."""
    try:
        return open("src/realtime/call_interface.html").read()
    except FileNotFoundError:
        return "Call interface not yet created. Use /client for now.", 404


@app.route("/push-to-talk")
def push_to_talk_interface():
    """Serve press-and-hold upload interface for ngrok demos."""
    try:
        return open("src/realtime/push_to_talk.html").read()
    except FileNotFoundError:
        return "Push-to-talk interface not found.", 404


@app.route("/agent-types", methods=["GET"])
def list_agent_types():
    _warm_purpose_agents()
    payload = []
    for agent_type, profile in AGENT_TYPE_PROFILES.items():
        payload.append(
            {
                "type": agent_type,
                "label": profile["label"],
                "description": profile["description"],
                "agent_id": _purpose_agent_cache.get(agent_type),
            }
        )
    return jsonify({"agent_types": payload})

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

@app.route("/lid-test")
def lid_test_page():
    return open("src/realtime/lid_test.html").read()

@app.route("/detect-lang", methods=["POST"])
def detect_lang_endpoint():
    global lid_model, lid_processor, lid_id2label

    if lid_model is None or lid_processor is None:
        return jsonify({"error": "LID model not ready"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    file = request.files["audio"]
    temp_path = f"lid_debug/upload_{int(time.time())}.wav"
    os.makedirs("lid_debug", exist_ok=True)
    file.save(temp_path)

    try:
        # Load exactly like Colab
        waveform, sr = librosa.load(temp_path, sr=16000, mono=True)
        
        # Energy check
        energy = np.mean(np.abs(waveform))
        
        # Inference
        inputs = lid_processor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = lid_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        ranked = sorted(
            [(lid_id2label[i], probs[i].item()) for i in range(len(probs))],
            key=lambda x: x[1],
            reverse=True
        )

        # Build top lists
        top_global_list = [{"code": r[0], "prob": r[1]} for r in ranked[:3]]

        # Filter for Indian
        indian_results = []
        for lang, p in ranked:
            if lang in INDIAN_LANGUAGES:
                indian_results.append({"code": lang, "prob": p, "name": LANGUAGE_NAMES.get(LANGUAGE_CODE_MAP.get(lang, lang), lang)})

        # Choose highest-confidence overall language (top_global[0]) if available, else fallback to top_indian
        selected_lang = None
        selected_prob = None
        if top_global_list:
            selected_lang = top_global_list[0]["code"]
            selected_prob = top_global_list[0]["prob"]
        elif indian_results:
            selected_lang = indian_results[0]["code"]
            selected_prob = indian_results[0]["prob"]

        # Persist selection to server state for this process
        try:
            if selected_lang is not None:
                # set current_language to selected (store raw model code)
                current_language = selected_lang
        except Exception:
            # ignore if current_language not used elsewhere
            pass

        # Log the full result prefixed with Done! and a concise line with chosen language
        result_payload = {
            "status": "success",
            "energy": float(energy),
            "top_indian": indian_results[:3],
            "top_global": top_global_list,
            "duration": len(waveform)/16000
        }

        print("Done!", result_payload)
        if selected_lang is not None:
            print(f"🌍 Language detected (highest global confidence): {selected_lang} (prob={selected_prob:.4f})")

        return jsonify(result_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/ice-config")
def ice_config():
    """Return ICE server configuration for WebRTC clients."""
    ice_servers_json = [
        {"urls": srv.urls} if not hasattr(srv, 'username') else
        {"urls": srv.urls, "username": srv.username, "credential": srv.credential}
        for srv in ICE_SERVERS
    ]
    return jsonify({"iceServers": ice_servers_json})


@app.route("/ready")
def ready():
    """Check if all models are loaded and ready to accept calls."""
    whisper_ready = whisper_model is not None and whisper_processor is not None
    lid_ready = lid_model is not None and lid_processor is not None
    
    all_ready = whisper_ready and lid_ready
    
    return jsonify({
        "ready": all_ready,
        "whisper_ready": whisper_ready,
        "lid_ready": lid_ready,
        "voice_turn_stream_enabled": settings.VOICE_TURN_STREAM_ENABLED,
        "device": device
    }), 200 if all_ready else 503


@app.route("/healthz")
def healthz():
    active_agent = admin_store.get_active_agent()
    status = {
        "status": "ok",
        "device": device,
        "whisper_ready": whisper_model is not None,
        "lid_ready": lid_model is not None,
        "tts_backend": settings.TTS_BACKEND,
        "redis": llm_context.ping(),
        "ollama": check_ollama_status(),
        "active_calls": metrics.get("active_calls", 0),
        "active_agent": active_agent["name"] if active_agent else None,
        "voice_turn_stream_enabled": settings.VOICE_TURN_STREAM_ENABLED,
    }
    return jsonify(status)


@app.route("/metrics")
def metrics_endpoint():
    with metrics_lock:
        snapshot = {
            "calls_total": metrics.get("calls_total", 0),
            "active_calls": metrics.get("active_calls", 0),
            "utterances_total": metrics.get("utterances_total", 0),
            "latencies": {
                name: list(values)
                for name, values in metrics.get("latencies", {}).items()
            },
        }
    snapshot["latency_stats"] = get_latency_stats()
    return jsonify(snapshot)

if __name__ == "__main__":
    print("🚀 Signaling + Audio Processing Server Running")
    _warm_purpose_agents()
    app.run("0.0.0.0", 8080)
